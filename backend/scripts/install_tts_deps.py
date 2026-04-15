#!/usr/bin/env python3
"""Install TTS engine deps from the command line.

Usage:
  python scripts/install_tts_deps.py                # show status of all
  python scripts/install_tts_deps.py cosyvoice2     # install one
  python scripts/install_tts_deps.py --all          # install all
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running from anywhere — make `src.*` importable
_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND))

from src.services.tts_deps import install, is_installed, list_manifest  # noqa: E402


async def _run_install(name: str) -> int:
    print(f"\n→ Installing {name} ...")

    async def echo(line: str):
        print(f"  {line}")

    ok, log = await install(name, on_log=echo)
    if ok:
        print(f"✓ {name}: installed")
        return 0
    print(f"✗ {name}: install failed")
    return 1


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("engine", nargs="?", help="engine name to install")
    p.add_argument("--all", action="store_true", help="install all engines")
    args = p.parse_args()

    manifest = list_manifest()

    if not args.engine and not args.all:
        # Status mode
        print(f"{'engine':<20} {'installed':<10} {'pip':<60} note")
        print("-" * 110)
        for name, dep in manifest.items():
            specs = ",".join(dep["pip_specs"]) or "(none)"
            print(f"{name:<20} {str(dep['installed']):<10} {specs[:58]:<60} {dep['note'][:40]}")
        return 0

    targets = list(manifest.keys()) if args.all else [args.engine]
    if args.engine and args.engine not in manifest:
        print(f"unknown engine: {args.engine}", file=sys.stderr)
        print(f"available: {', '.join(manifest.keys())}", file=sys.stderr)
        return 2

    rc = 0
    for name in targets:
        if is_installed(name) and not args.all:
            print(f"skip {name}: already installed")
            continue
        rc |= await _run_install(name)
    return rc


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
