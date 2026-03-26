"""Skill CRUD routes (file-based, no DB)."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from src.api.deps_admin import require_admin
from src.services import skill_manager

router = APIRouter(prefix="/api/v1/skills", tags=["skills"])


class SkillCreate(BaseModel):
    name: str
    description: str = ""
    body: str = ""


@router.post("", status_code=201, dependencies=[Depends(require_admin)])
def create_skill(body: SkillCreate):
    try:
        return skill_manager.create_skill(body.name, body.description, body.body)
    except FileExistsError as e:
        raise HTTPException(409, str(e))


@router.get("")
def list_skills():
    return skill_manager.list_skills()


@router.get("/{name}")
def get_skill(name: str):
    try:
        return skill_manager.get_skill(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@router.put("/{name}", dependencies=[Depends(require_admin)])
async def update_skill(name: str, request: Request):
    raw_content = (await request.body()).decode("utf-8")
    try:
        return skill_manager.update_skill(name, raw_content)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@router.delete("/{name}", status_code=204, dependencies=[Depends(require_admin)])
def delete_skill(name: str):
    try:
        skill_manager.delete_skill(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
