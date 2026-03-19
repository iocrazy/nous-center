from fastapi import APIRouter

from nodes import get_all_definitions, get_packages, scan_packages

router = APIRouter(prefix="/api/v1/nodes", tags=["nodes"])


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
