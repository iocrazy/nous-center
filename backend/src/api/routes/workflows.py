"""Workflow CRUD routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_async_session
from src.models.schemas import WorkflowCreate, WorkflowUpdate, WorkflowOut
from src.models.workflow import Workflow

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])


@router.post("", response_model=WorkflowOut, status_code=201)
async def create_workflow(
    body: WorkflowCreate,
    session: AsyncSession = Depends(get_async_session),
):
    wf = Workflow(**body.model_dump())
    session.add(wf)
    await session.commit()
    await session.refresh(wf)
    return wf


@router.get("", response_model=list[WorkflowOut])
async def list_workflows(
    is_template: bool | None = None,
    session: AsyncSession = Depends(get_async_session),
):
    stmt = select(Workflow).order_by(Workflow.updated_at.desc())
    if is_template is not None:
        stmt = stmt.where(Workflow.is_template == is_template)
    result = await session.execute(stmt)
    return result.scalars().all()


@router.get("/{workflow_id}", response_model=WorkflowOut)
async def get_workflow(
    workflow_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    wf = await session.get(Workflow, workflow_id)
    if not wf:
        raise HTTPException(404, "Workflow not found")
    return wf


@router.patch("/{workflow_id}", response_model=WorkflowOut)
async def update_workflow(
    workflow_id: int,
    body: WorkflowUpdate,
    session: AsyncSession = Depends(get_async_session),
):
    wf = await session.get(Workflow, workflow_id)
    if not wf:
        raise HTTPException(404, "Workflow not found")
    for key, value in body.model_dump(exclude_unset=True).items():
        setattr(wf, key, value)
    await session.commit()
    await session.refresh(wf)
    return wf


@router.delete("/{workflow_id}", status_code=204)
async def delete_workflow(
    workflow_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    wf = await session.get(Workflow, workflow_id)
    if not wf:
        raise HTTPException(404, "Workflow not found")
    await session.delete(wf)
    await session.commit()
