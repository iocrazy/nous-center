"""Node package management — list, install (zip/git), uninstall, install deps."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from nodes import get_all_definitions, get_packages, scan_packages

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/nodes", tags=["nodes"])

_PACKAGE_DIR = Path(__file__).resolve().parents[3] / "nodes"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,40}$", re.IGNORECASE)


def _resolve_pkg_dir(name: str) -> Path:
    """Return the on-disk path for a package, validating the name."""
    if not _NAME_RE.match(name):
        raise HTTPException(400, detail="invalid package name")
    return _PACKAGE_DIR / name


@router.get("/packages")
async def list_packages():
    return get_packages()


@router.get("/definitions")
async def list_node_definitions():
    return get_all_definitions()


@router.post("/scan")
async def rescan_packages():
    pkgs = scan_packages()
    return {"count": len(pkgs), "packages": list(pkgs.keys())}


# ---------- install / uninstall ---------- #

@router.post("/packages/install_zip")
async def install_package_zip(file: UploadFile = File(...), name: str | None = Form(None)):
    """Install a node package from an uploaded .zip.

    The archive must contain a top-level dir with `node.yaml`. If `name` is
    omitted, the top-level dir name is used.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".zip",):
        raise HTTPException(400, detail="upload must be a .zip file")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        zip_path = td_path / "pkg.zip"
        with zip_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

        extract_dir = td_path / "extracted"
        extract_dir.mkdir()
        try:
            with zipfile.ZipFile(zip_path) as zf:
                # Reject zip-slip: any member starting with .. or absolute
                for m in zf.namelist():
                    if m.startswith(("/", "..")) or ".." in Path(m).parts:
                        raise HTTPException(400, detail=f"unsafe path in archive: {m}")
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            raise HTTPException(400, detail="bad zip archive")

        # Find the package root (must contain node.yaml; allow nested-by-one)
        candidates = list(extract_dir.glob("**/node.yaml"))
        if not candidates:
            raise HTTPException(400, detail="archive contains no node.yaml")
        pkg_root = candidates[0].parent

        pkg_name = name or pkg_root.name
        if not _NAME_RE.match(pkg_name):
            raise HTTPException(400, detail="invalid package name")

        target = _PACKAGE_DIR / pkg_name
        if target.exists():
            raise HTTPException(409, detail=f"package '{pkg_name}' already installed")
        shutil.copytree(pkg_root, target)

    pkgs = scan_packages()
    return {"installed": pkg_name, "package_count": len(pkgs)}


@router.post("/packages/install_git")
async def install_package_git(repo_url: str = Form(...), name: str | None = Form(None)):
    """Clone a node package from a git URL."""
    if not repo_url.startswith(("https://", "git@")):
        raise HTTPException(400, detail="repo_url must be https:// or git@")
    pkg_name = name or repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    if not _NAME_RE.match(pkg_name):
        raise HTTPException(400, detail="cannot derive valid package name; pass name=...")
    target = _PACKAGE_DIR / pkg_name
    if target.exists():
        raise HTTPException(409, detail=f"package '{pkg_name}' already installed")

    # Clone into a temp dir first so we can validate before placing
    with tempfile.TemporaryDirectory() as td:
        clone_dir = Path(td) / "clone"
        proc = await asyncio.create_subprocess_exec(
            "git", "clone", "--depth", "1", repo_url, str(clone_dir),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        rc = await proc.wait()
        if rc != 0:
            out = (await proc.stdout.read()).decode(errors="replace") if proc.stdout else ""
            raise HTTPException(400, detail=f"git clone failed: {out[-300:]}")
        # Validate
        candidates = list(clone_dir.glob("**/node.yaml"))
        if not candidates:
            raise HTTPException(400, detail="repo contains no node.yaml")
        pkg_root = candidates[0].parent
        # Strip .git so the installed copy is a plain dir
        shutil.rmtree(pkg_root / ".git", ignore_errors=True)
        shutil.copytree(pkg_root, target)

    pkgs = scan_packages()
    return {"installed": pkg_name, "package_count": len(pkgs)}


@router.delete("/packages/{name}")
async def uninstall_package(name: str):
    pkg_dir = _resolve_pkg_dir(name)
    if not pkg_dir.exists():
        raise HTTPException(404, detail=f"package '{name}' not installed")
    shutil.rmtree(pkg_dir)
    pkgs = scan_packages()
    return {"uninstalled": name, "package_count": len(pkgs)}


@router.post("/packages/{name}/install_deps")
async def install_package_deps(name: str):
    """Run pip install -r requirements.txt for the package, if present."""
    pkg_dir = _resolve_pkg_dir(name)
    req = pkg_dir / "requirements.txt"
    if not pkg_dir.exists():
        raise HTTPException(404, detail=f"package '{name}' not installed")
    if not req.exists():
        return {"name": name, "status": "no_requirements", "log": ""}

    py = sys.executable
    uv = shutil.which("uv")
    cmd = [uv, "pip", "install", "--python", py, "-r", str(req)] if uv else [
        py, "-m", "pip", "install", "-r", str(req)
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    out_bytes = await proc.stdout.read() if proc.stdout else b""
    rc = await proc.wait()
    log = out_bytes.decode(errors="replace")
    if rc != 0:
        raise HTTPException(500, detail=f"deps install failed: {log[-500:]}")
    return {"name": name, "status": "installed", "log": log[-500:]}
