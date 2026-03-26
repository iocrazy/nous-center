"""Agent CRUD routes (file-based, no DB)."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from src.api.deps_admin import require_admin
from src.services import agent_manager

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])


class AgentCreate(BaseModel):
    name: str
    display_name: str | None = None


class AgentUpdate(BaseModel):
    display_name: str | None = None
    model: str | None = None
    skills: list[str] | None = None
    status: str | None = None


@router.post("", status_code=201, dependencies=[Depends(require_admin)])
def create_agent(body: AgentCreate):
    try:
        return agent_manager.create_agent(body.name, body.display_name)
    except FileExistsError as e:
        raise HTTPException(409, str(e))


@router.get("")
def list_agents():
    return agent_manager.list_agents()


@router.get("/{name}")
def get_agent(name: str):
    try:
        return agent_manager.get_agent(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@router.patch("/{name}", dependencies=[Depends(require_admin)])
def update_agent(name: str, body: AgentUpdate):
    updates = body.model_dump(exclude_unset=True)
    try:
        return agent_manager.update_agent(name, updates)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@router.delete("/{name}", status_code=204, dependencies=[Depends(require_admin)])
def delete_agent(name: str):
    try:
        agent_manager.delete_agent(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@router.put("/{name}/prompts/{filename}", dependencies=[Depends(require_admin)])
async def save_prompt(name: str, filename: str, request: Request):
    content = (await request.body()).decode("utf-8")
    try:
        agent_manager.save_prompt(name, filename, content)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"ok": True}
