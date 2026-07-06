"""python_exec 安全默认关闭(安全 P1):AST 黑名单本质可绕(getattr(__builtins__,
'__imp'+'ort__') 等),故默认禁用整个节点,operator 显式设 NOUS_ENABLE_PYTHON_EXEC=1
才放行。经已发布工作流+granted key 可达 → 等价 RCE,能读 .env。"""
import pytest

from src.services.skill_tools import _execute_python


@pytest.mark.asyncio
async def test_python_exec_disabled_by_default(monkeypatch):
    monkeypatch.delenv("NOUS_ENABLE_PYTHON_EXEC", raising=False)
    out = await _execute_python("print(1)")
    assert "disabled" in out.lower() or "禁用" in out


@pytest.mark.asyncio
async def test_python_exec_disabled_when_flag_not_1(monkeypatch):
    monkeypatch.setenv("NOUS_ENABLE_PYTHON_EXEC", "0")
    out = await _execute_python("print(1)")
    assert "disabled" in out.lower() or "禁用" in out


@pytest.mark.asyncio
async def test_python_exec_runs_when_enabled(monkeypatch):
    monkeypatch.setenv("NOUS_ENABLE_PYTHON_EXEC", "1")
    out = await _execute_python("print(6*7)")
    assert "42" in out
