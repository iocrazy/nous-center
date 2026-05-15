"""Lane J test infrastructure: mock image/TTS runner subprocess (spec §5.6).

Wraps Lane C's runner_main() + FakeAdapter + RunnerClient into a configurable
(crash/slow/fail-load) pytest fixture with managed start/stop lifecycle.
Integration tests use this to drive the full "main process + real runner
subprocess + complete IPC" loop without GPUs or real models.

Lane C's `runner_main` signature accepts only `models_yaml_path` +
`fake_adapter`, NOT arbitrary adapter kwargs. FakeAdapter pulls its
crash/slow/fail-load knobs from `ModelSpec.params` in models.yaml. So
this fixture writes a per-test temp models.yaml with the requested
params and points runner_main at it.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.runner.client import RunnerClient
from src.runner.runner_process import runner_main

_SPAWN = mp.get_context("spawn")


_DEFAULT_MODEL_KEY = "fake-img"


def _write_models_yaml(
    dir_path: Path,
    *,
    model_key: str = _DEFAULT_MODEL_KEY,
    model_type: str = "image",
    params: dict[str, Any] | None = None,
) -> Path:
    """Write a temp models.yaml exposing one FakeAdapter spec with given params."""
    models_yaml = {
        "models": [
            {
                "id": model_key,
                "type": model_type,
                "adapter": "src.runner.fake_adapter.FakeAdapter",
                "paths": {"main": f"/fake/{model_key}"},
                "vram_mb": 0,
                "resident": False,
                "params": params or {},
            }
        ]
    }
    path = Path(dir_path) / "fake_runner_models.yaml"
    path.write_text(yaml.safe_dump(models_yaml, sort_keys=False))
    return path


@dataclass
class FakeRunnerHandle:
    """Handle for one fake runner subprocess + main-side RunnerClient."""

    group_id: str
    gpus: list[int]
    models_yaml_path: Path
    model_key: str = _DEFAULT_MODEL_KEY
    _process: mp.process.BaseProcess | None = field(default=None, repr=False)
    client: RunnerClient | None = field(default=None, repr=False)
    _parent_conn: Any = field(default=None, repr=False)

    async def start(self) -> None:
        """Spawn the runner subprocess, build RunnerClient, wait for Ready."""
        parent_conn, child_conn = _SPAWN.Pipe()
        self._parent_conn = parent_conn
        self._process = _SPAWN.Process(
            target=runner_main,
            args=(self.group_id, self.gpus, child_conn),
            kwargs={
                "models_yaml_path": str(self.models_yaml_path),
                "fake_adapter": True,
            },
            daemon=True,
            name=f"fake-runner-{self.group_id}",
        )
        self._process.start()
        child_conn.close()  # main side does not hold the child end
        self.client = RunnerClient(
            parent_conn, runner_id=f"fake-runner-{self.group_id}"
        )
        await self.client.start()  # waits for Ready handshake (ready_timeout=30s)

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def kill(self) -> None:
        """SIGKILL the subprocess — simulate a hard crash."""
        if self._process is not None and self._process.is_alive():
            self._process.kill()

    async def stop(self) -> None:
        """Graceful shutdown: close client → terminate subprocess → join."""
        if self.client is not None:
            try:
                await self.client.close()
            except Exception:
                pass
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2.0)


@pytest.fixture
def fake_runner(tmp_path):
    """Factory fixture for FakeRunnerHandle instances.

    Usage::

        runner = fake_runner(group_id="image", gpus=[2])
        await runner.start()
        # ... drive runner.client ...
        await runner.stop()

    Or pass crash/slow/fail-load knobs (forwarded into ModelSpec.params,
    which FakeAdapter consumes)::

        runner = fake_runner(
            group_id="image", gpus=[2],
            crash_on_node=True,    # FakeAdapter.crash_on_infer
            slow_seconds=0.5,      # FakeAdapter.infer_seconds
            fail_load=True,        # FakeAdapter.fail_load
        )

    The fixture teardown SIGKILLs any handle whose .stop() was not awaited
    (e.g. on test failure mid-await), so tests don't leak subprocesses.
    """
    created: list[FakeRunnerHandle] = []

    def _factory(
        *,
        group_id: str,
        gpus: list[int],
        model_key: str = _DEFAULT_MODEL_KEY,
        model_type: str = "image",
        crash_on_node: bool = False,
        slow_seconds: float = 0.0,
        fail_load: bool = False,
    ) -> FakeRunnerHandle:
        params: dict[str, Any] = {}
        if crash_on_node:
            params["crash_on_infer"] = True
        if slow_seconds > 0:
            params["infer_seconds"] = slow_seconds
        if fail_load:
            params["fail_load"] = True
        # FakeAdapter has infer_seconds default 0.01 — set 0 when caller did
        # not override, to keep happy-path tests fast.
        params.setdefault("infer_seconds", 0.0)

        # Per-handle subdir so multiple handles in one test don't collide.
        sub = tmp_path / f"runner-{len(created)}-{group_id}"
        sub.mkdir(exist_ok=True)
        yaml_path = _write_models_yaml(
            sub, model_key=model_key, model_type=model_type, params=params,
        )

        handle = FakeRunnerHandle(
            group_id=group_id,
            gpus=gpus,
            models_yaml_path=yaml_path,
            model_key=model_key,
        )
        created.append(handle)
        return handle

    yield _factory

    # Teardown — kill any survivor and reap.
    for h in created:
        if h._process is not None and h._process.is_alive():
            h._process.kill()
            h._process.join(timeout=2.0)
