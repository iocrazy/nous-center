"""Log query and frontend error reporting endpoints."""
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from src.api.deps_admin import require_admin
from src.services.log_db import query_logs, insert_frontend_log

router = APIRouter(prefix="/api/v1/logs", tags=["logs"])


class FrontendLogReport(BaseModel):
    type: str
    message: str
    page: str = ""
    stack: str | None = None


@router.get("/requests", dependencies=[Depends(require_admin)])
async def get_request_logs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    search: str | None = None,
    method: str | None = None,
    status: str | None = None,
    since: str | None = None,
):
    return query_logs(table="request_logs", limit=limit, offset=offset, search=search, method=method, status=status, since=since)


@router.get("/app", dependencies=[Depends(require_admin)])
async def get_app_logs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    search: str | None = None,
    level: str | None = None,
    since: str | None = None,
):
    return query_logs(table="app_logs", limit=limit, offset=offset, search=search, level=level, since=since)


@router.get("/frontend", dependencies=[Depends(require_admin)])
async def get_frontend_logs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    search: str | None = None,
    type: str | None = None,
    since: str | None = None,
):
    return query_logs(table="frontend_logs", limit=limit, offset=offset, search=search, type_filter=type, since=since)


@router.post("/frontend", status_code=201)
async def report_frontend_log(body: FrontendLogReport):
    insert_frontend_log(type=body.type, message=body.message, page=body.page, stack=body.stack)
    return {"status": "recorded"}


@router.get("/audit", dependencies=[Depends(require_admin)])
async def get_audit_logs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    search: str | None = None,
    since: str | None = None,
):
    return query_logs(table="audit_logs", limit=limit, offset=offset, search=search, since=since)
