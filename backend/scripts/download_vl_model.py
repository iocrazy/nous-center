#!/usr/bin/env python3
"""Download a vision-language model into LOCAL_MODELS_PATH/llm/<repo>.

Usage:
  python scripts/download_vl_model.py                                 # default Qwen2.5-VL-7B-Instruct
  python scripts/download_vl_model.py Qwen/Qwen2.5-VL-32B-Instruct    # any HF repo

Uses huggingface_hub.snapshot_download. Skips files already present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND))

from src.config import get_settings  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("repo", nargs="?", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--revision", default=None)
    args = p.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Run: uv pip install huggingface_hub", file=sys.stderr)
        return 1

    root = Path(get_settings().LOCAL_MODELS_PATH) / "llm"
    target_name = args.repo.split("/")[-1]
    target = root / target_name
    target.mkdir(parents=True, exist_ok=True)

    print(f"→ Downloading {args.repo} → {target}")
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(target),
        revision=args.revision,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"✓ Done. Update configs/models.yaml path → llm/{target_name} if different.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
