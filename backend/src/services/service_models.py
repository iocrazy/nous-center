"""Display-only enumeration of the models / components a published service
depends on, for the service overview UI ("有多少模型 + 对应加载情况").

This is **purely static** ref extraction from a frozen workflow snapshot —
NOT model management (it deliberately does not touch the registry /
ModelManager, sidestepping the unified-model-mgmt gap). Live load-state is
overlaid client-side: components matched by file against the component-state
registry, engines matched by key against /api/v1/engines. Matching by file
(not the full file|device|dtype|lora state_key) keeps the overview robust to
device/dtype/lora resolution details — the question the overview answers is
"is this model loaded at all", not "loaded with which exact knobs".

Snapshot shape (see workflow_publish._build_snapshot):
    {"nodes": {"<id>": {"class_type": <type>, "inputs": <node.data>}}}
Older / editor shape (list of {id, type, data}) is also tolerated.
"""
from __future__ import annotations

import os
from typing import Any, Iterator

# flux2 single-file component loaders → role. These carry a `file` (abs path)
# in their inputs; checkpoint loads a whole-model dir but is still one ref.
_COMPONENT_ROLE_BY_TYPE: dict[str, str] = {
    "flux2_load_diffusion_model": "diffusion_models",
    "flux2_load_vae": "vae",
    "flux2_load_checkpoint": "checkpoint",
}


def _iter_nodes(snapshot: dict | None) -> Iterator[dict]:
    """Yield node dicts from either snapshot shape (dict-of-id or list)."""
    nodes = (snapshot or {}).get("nodes")
    if isinstance(nodes, dict):
        for node in nodes.values():
            if isinstance(node, dict):
                yield node
    elif isinstance(nodes, list):
        for node in nodes:
            if isinstance(node, dict):
                yield node


def _node_type(node: dict) -> str:
    return str(node.get("class_type") or node.get("type") or "").strip()


def _node_inputs(node: dict) -> dict:
    # published snapshot stores node.data under "inputs"; editor uses "data".
    v = node.get("inputs")
    if isinstance(v, dict):
        return v
    v = node.get("data")
    return v if isinstance(v, dict) else {}


def _label_for_file(path: str) -> str:
    return os.path.basename(path.rstrip("/")) or path


def extract_service_models(snapshot: dict | None) -> list[dict[str, Any]]:
    """Return the distinct model/component refs a snapshot depends on.

    Each ref: {kind: 'component'|'engine', role, label, file, engine_key}.
    Order follows first appearance; dedup by file (components) / key (engines).
    """
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(ref: dict[str, Any], dedup_key: str) -> None:
        if dedup_key in seen:
            return
        seen.add(dedup_key)
        refs.append(ref)

    for node in _iter_nodes(snapshot):
        ntype = _node_type(node)
        ntlow = ntype.lower()
        inp = _node_inputs(node)

        # --- CLIP: dynamic multi-encoder (clips=[{file,...}]) or back-compat single file ---
        if ntype == "flux2_load_clip":
            files: list[str] = []
            clips = inp.get("clips")
            if isinstance(clips, list):
                files = [c["file"] for c in clips if isinstance(c, dict) and c.get("file")]
            elif inp.get("file"):
                files = [inp["file"]]
            for f in files:
                add(
                    {"kind": "component", "role": "clip", "label": _label_for_file(f),
                     "file": f, "engine_key": None},
                    f"component:{f}",
                )
            continue

        # --- single-file components: diffusion / vae / checkpoint ---
        role = _COMPONENT_ROLE_BY_TYPE.get(ntype)
        if role:
            f = inp.get("file")
            if f:
                add(
                    {"kind": "component", "role": role, "label": _label_for_file(f),
                     "file": f, "engine_key": None},
                    f"component:{f}",
                )
            continue

        # --- registry engines: editor `llm` (model_key) / `tts_engine` (engine) /
        #     trivial quick-provision `LLMEngine`/`TTSEngine`/`VLEngine` (engine) ---
        engine_key = inp.get("model_key") or inp.get("engine")
        if engine_key and (ntlow == "llm" or ntlow == "tts_engine"
                           or ntype.endswith("Engine") or "model_key" in inp):
            role_guess = (
                "llm" if ntlow == "llm" or ntype.startswith("LLM") or "model_key" in inp
                else "tts" if ntlow == "tts_engine" or ntype.startswith("TTS")
                else None
            )
            add(
                {"kind": "engine", "role": role_guess, "label": str(engine_key),
                 "file": None, "engine_key": str(engine_key)},
                f"engine:{engine_key}",
            )
            continue

    return refs
