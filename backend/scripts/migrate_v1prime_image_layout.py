"""V1' P0 — reorganize nous/image/ to ComfyUI-style 4 subdirs.

Layout transition:

    BEFORE                                  AFTER
    nous/image/                             nous/image/
    +- Flux2-klein-9B/                      +- diffusers/
    |  +- transformer/quantized/*.safe.     |  +- Flux2-klein-9B/
    +- ERNIE-Image/                         |  +- ERNIE-Image/
    +- vae/                                 +- diffusion_models/
    +- ._____temp/      (purged)            |  +- Flux2-Klein-9B-True-v2-{bf16,fp8mixed,...}.safetensors
                                            +- text_encoders/
                                            +- vae/

Idempotent. Default is dry-run; pass --apply to actually move files.

Run from backend/:
    python scripts/migrate_v1prime_image_layout.py            # dry-run
    python scripts/migrate_v1prime_image_layout.py --apply    # execute
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Order matters: mv top-level dirs first so quantized/ ends up under diffusers/
TOP_LEVEL_MOVES = [
    ("Flux2-klein-9B", "diffusers/Flux2-klein-9B"),
    ("ERNIE-Image", "diffusers/ERNIE-Image"),
]

# After top-level moves, quantized/ lives under diffusers/Flux2-klein-9B/transformer/quantized/
QUANTIZED_SRC_DIR = "diffusers/Flux2-klein-9B/transformer/quantized"
QUANTIZED_DST_DIR = "diffusion_models"

NEW_SUBDIRS = ["diffusers", "diffusion_models", "text_encoders", "vae"]
PURGE_DIRS = ["._____temp"]


def _resolve_image_root() -> Path:
    env_path = os.environ.get("LOCAL_MODELS_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve() / "image"
    # Fallback: read from backend/.env
    here = Path(__file__).resolve().parents[1]
    env_file = here / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("LOCAL_MODELS_PATH="):
                return Path(line.split("=", 1)[1].strip()).expanduser().resolve() / "image"
    raise SystemExit("LOCAL_MODELS_PATH not set in env or backend/.env")


def _plan_mkdirs(image_root: Path) -> list[tuple[str, Path]]:
    return [
        ("MKDIR", image_root / sub)
        for sub in NEW_SUBDIRS
        if not (image_root / sub).is_dir()
    ]


def _plan_top_level_moves(image_root: Path) -> list[tuple[str, Path, Path]]:
    plan: list[tuple[str, Path, Path]] = []
    for src_rel, dst_rel in TOP_LEVEL_MOVES:
        src = image_root / src_rel
        dst = image_root / dst_rel
        if dst.exists():
            plan.append(("SKIP", src, dst))
        elif src.exists():
            plan.append(("MOVE", src, dst))
        else:
            plan.append(("MISS", src, dst))
    return plan


def _plan_quantized_moves(image_root: Path) -> list[tuple[str, Path, Path]]:
    plan: list[tuple[str, Path, Path]] = []
    src_dir = image_root / QUANTIZED_SRC_DIR
    dst_dir = image_root / QUANTIZED_DST_DIR
    # Check both pre-move and post-move source location
    candidate_dirs = [
        src_dir,
        image_root / "Flux2-klein-9B" / "transformer" / "quantized",
    ]
    for cand in candidate_dirs:
        if cand.is_dir():
            files = sorted(list(cand.glob("*.safetensors")) + list(cand.glob("*.gguf")))
            for f in files:
                dst = dst_dir / f.name
                if dst.exists():
                    plan.append(("SKIP", f, dst))
                else:
                    plan.append(("MOVE", f, dst))
            break
    return plan


def _plan_quantized_dir_cleanup(image_root: Path) -> list[tuple[str, Path]]:
    """Only rmdir if the directory is fully empty.

    wikeeyang's quantized/ ships with metadata (LICENSE.md, README.md,
    Sample-V2.jpg, workflow JSON) that we keep in place — those files belong
    with the original BFL diffusers tree, not with the relocated weights.
    """
    plan: list[tuple[str, Path]] = []
    for cand in [
        image_root / QUANTIZED_SRC_DIR,
        image_root / "Flux2-klein-9B" / "transformer" / "quantized",
    ]:
        if cand.is_dir() and not any(cand.iterdir()):
            plan.append(("RMDIR", cand))
    return plan


def _plan_purge(image_root: Path) -> list[tuple[str, Path]]:
    return [
        ("PURGE", image_root / d)
        for d in PURGE_DIRS
        if (image_root / d).exists()
    ]


def _print_plan(image_root: Path, mkdirs, top_moves, quant_moves, dir_cleanup, purges):
    print(f"image root: {image_root}\n")
    print("=== mkdir ===")
    for action, p in mkdirs:
        print(f"  {action} {p.relative_to(image_root)}")
    if not mkdirs:
        print("  (all subdirs already exist)")
    print("\n=== top-level moves ===")
    for action, src, dst in top_moves:
        marker = "->" if action == "MOVE" else ".."
        print(f"  {action} {src.relative_to(image_root)} {marker} {dst.relative_to(image_root)}")
    print("\n=== quantized file moves ===")
    if not quant_moves:
        print("  (no quantized files to move)")
    for action, src, dst in quant_moves:
        marker = "->" if action == "MOVE" else ".."
        try:
            src_disp = src.relative_to(image_root)
        except ValueError:
            src_disp = src
        print(f"  {action} {src_disp} {marker} {dst.relative_to(image_root)}")
    print("\n=== empty dir cleanup ===")
    for action, p in dir_cleanup:
        print(f"  {action} {p.relative_to(image_root)}")
    if not dir_cleanup:
        print("  (none)")
    print("\n=== purge ===")
    for action, p in purges:
        print(f"  {action} {p.relative_to(image_root)}")
    if not purges:
        print("  (none)")


def _apply(image_root: Path, mkdirs, top_moves, _quant_moves, _dir_cleanup, purges):
    for _, p in mkdirs:
        p.mkdir(parents=True, exist_ok=True)
        print(f"  mkdir {p.relative_to(image_root)}")
    for action, src, dst in top_moves:
        if action == "MOVE":
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"  moved {src.relative_to(image_root)} -> {dst.relative_to(image_root)}")
    # Recompute quant moves AFTER top-level moves: BFL is now at diffusers/Flux2-klein-9B/
    # and the quantized files travelled with it.
    for action, src, dst in _plan_quantized_moves(image_root):
        if action == "MOVE":
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            try:
                src_disp = src.relative_to(image_root)
            except ValueError:
                src_disp = src
            print(f"  moved {src_disp} -> {dst.relative_to(image_root)}")
    for _, p in _plan_quantized_dir_cleanup(image_root):
        p.rmdir()
        print(f"  rmdir {p.relative_to(image_root)}")
    for _, p in purges:
        shutil.rmtree(p)
        print(f"  purged {p.relative_to(image_root)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="actually perform the moves (default: dry-run)")
    ap.add_argument("--root", type=str, default=None, help="override image root path")
    args = ap.parse_args(argv)

    image_root = Path(args.root).resolve() if args.root else _resolve_image_root()
    if not image_root.is_dir():
        print(f"ERROR: image root not found: {image_root}", file=sys.stderr)
        return 2

    mkdirs = _plan_mkdirs(image_root)
    top_moves = _plan_top_level_moves(image_root)
    quant_moves = _plan_quantized_moves(image_root)
    dir_cleanup = _plan_quantized_dir_cleanup(image_root)
    purges = _plan_purge(image_root)

    _print_plan(image_root, mkdirs, top_moves, quant_moves, dir_cleanup, purges)

    if not args.apply:
        print("\n[dry-run] pass --apply to execute.")
        return 0

    print("\n[apply] executing...")
    _apply(image_root, mkdirs, top_moves, quant_moves, dir_cleanup, purges)
    print("[apply] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
