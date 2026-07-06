"""Alembic baseline 守门:fresh DB `upgrade head` 后 schema 必须与 models 零 diff。

这是引入 alembic 的安全基石 —— 证明 baseline 迁移 == Base.metadata(生产 schema)。
也顺带守未来:谁改了 model 却忘了生成对应迁移,`alembic check` 会检出 diff → 本测试红,
逼你补迁移。

用 subprocess 跑真实 alembic CLI(最忠实,就是运维会敲的命令),对临时 sqlite 库。
"""

import os
import subprocess
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent


def _run_alembic(args: list[str], db_path: Path) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "DATABASE_URL": f"sqlite+aiosqlite:///{db_path}",
        "CUDA_VISIBLE_DEVICES": "",
    }
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=_BACKEND,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_baseline_upgrade_then_zero_diff(tmp_path):
    db = tmp_path / "alembic_baseline.db"

    up = _run_alembic(["upgrade", "head"], db)
    assert up.returncode == 0, f"upgrade head failed:\n{up.stdout}\n{up.stderr}"

    check = _run_alembic(["check"], db)
    combined = check.stdout + check.stderr
    # check 返回 0 且明确"无新操作" = upgrade 后的库 schema 与 models 完全一致。
    assert check.returncode == 0, f"alembic check found drift (baseline != models):\n{combined}"
    assert "No new upgrade operations detected" in combined, combined
