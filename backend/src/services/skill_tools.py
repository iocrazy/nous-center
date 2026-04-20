"""Convert Skills (SKILL.md) to OpenAI tool schemas and execute them."""

import json
import logging

from src.services import skill_manager

logger = logging.getLogger(__name__)


def skills_to_tools(skill_names: list[str]) -> list[dict]:
    """Convert a list of skill names to OpenAI tools format."""
    tools = []
    for name in skill_names:
        try:
            skill = skill_manager.get_skill(name)
        except FileNotFoundError:
            continue
        if not skill:
            continue

        # Parse parameters from SKILL.md body
        params = _extract_params(skill.get("body", ""))

        tool = {
            "type": "function",
            "function": {
                "name": name.replace("-", "_"),
                "description": skill.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": params,
                },
            },
        }
        tools.append(tool)

    # Always add code execution tool
    tools.append({
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Execute Python code in a sandboxed environment. "
                "Use this for data processing, file generation, calculations, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    },
                },
                "required": ["code"],
            },
        },
    })

    return tools


def _extract_params(body: str) -> dict:
    """Extract parameter definitions from SKILL.md body text.

    Looks for a ``## 参数`` or ``## Parameters`` section containing lines
    formatted as ``- name: description``.
    """
    params: dict[str, dict[str, str]] = {}
    in_params = False
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("## 参数") or stripped.startswith("## Parameters"):
            in_params = True
            continue
        if in_params and stripped.startswith("## "):
            break
        if in_params and stripped.startswith("- "):
            # Parse "- name: description" format
            parts = stripped[2:].split(":", 1)
            if len(parts) == 2:
                param_name = parts[0].strip()
                param_desc = parts[1].strip()
                params[param_name] = {
                    "type": "string",
                    "description": param_desc,
                }
    return params


def skill_tool_schema() -> dict:
    """Return the OpenAI function-tool schema for the Skill tool.

    This is the single tool that replaces per-skill function generation
    (cf. ``skills_to_tools``). The model calls ``Skill(skill=<name>, args=?)``
    and receives the SKILL.md body as the tool_result.
    """
    return {
        "type": "function",
        "function": {
            "name": "Skill",
            "description": "Load a local skill definition and its instructions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "description": "Skill name from <available_skills>.",
                    },
                    "args": {
                        "type": "string",
                        "description": "Optional arguments to pass to the skill.",
                    },
                },
                "required": ["skill"],
            },
        },
    }


async def _execute_skill_tool(args: dict | None) -> str:
    """Execute the Skill tool. Returns a JSON string for ``tool_result``."""
    args = args or {}  # model may send null
    name = (args.get("skill") or "").strip()
    if not name:
        return json.dumps({"error": "skill name required"})
    try:
        sk = skill_manager.get_skill(name)
    except FileNotFoundError:
        return json.dumps({"error": f"unknown skill: {name}"})
    return json.dumps(
        {
            "skill": name,
            "description": sk.get("description", ""),
            "prompt": sk.get("body", ""),
            "args": args.get("args"),
        },
        ensure_ascii=False,
    )


async def execute_tool(tool_name: str, arguments: str | dict | None) -> str:
    """Execute a tool by name with given arguments."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"input": arguments}

    # Lazy-readable Skill tool (Claude Code style): dispatched before
    # execute_python so a skill named "execute_python" can't shadow it.
    if tool_name == "Skill":
        return await _execute_skill_tool(arguments if isinstance(arguments, dict) else None)

    # Built-in tools
    if tool_name == "execute_python":
        return await _execute_python(arguments.get("code", ""))

    # Skill-based tools (mapped from SKILL.md)
    # For now, these are informational — the model uses the skill's instructions
    # to generate code, then calls execute_python
    return f"Skill '{tool_name}' executed with args: {arguments}"


# Dangerous modules/builtins that should not appear in sandboxed code
_BLOCKED_IMPORTS = {
    "os", "subprocess", "shutil", "sys", "importlib",
    "ctypes", "signal", "socket", "http", "urllib",
    "pathlib",  # can traverse filesystem
    "pickle",  # arbitrary code execution via deserialization
    "multiprocessing",  # can spawn unrestricted subprocesses
    "code",  # interactive interpreter, allows arbitrary eval
    "pty",  # pseudo-terminal, can spawn unrestricted shells
}
_BLOCKED_BUILTINS = {"exec", "eval", "compile", "__import__", "breakpoint"}


def _check_code_safety(code: str) -> str | None:
    """Return an error message if code contains blocked patterns, else None."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _BLOCKED_IMPORTS:
                    return f"安全限制: 禁止导入 '{alias.name}'"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in _BLOCKED_IMPORTS:
                    return f"安全限制: 禁止导入 '{node.module}'"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_BUILTINS:
                return f"安全限制: 禁止调用 '{node.func.id}()'"
            if isinstance(node.func, ast.Attribute) and node.func.attr == "system":
                return "安全限制: 禁止调用 os.system()"
    return None


async def _execute_python(code: str) -> str:
    """Execute Python code in a subprocess sandbox with safety checks."""
    import asyncio
    import os
    import tempfile

    if not code.strip():
        return "Error: empty code"

    # Safety check before execution
    safety_error = _check_code_safety(code)
    if safety_error:
        return safety_error

    # Write code to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        # Execute with timeout (30 seconds)
        proc = await asyncio.create_subprocess_exec(
            "python3",
            tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/tmp",
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            return "Error: execution timed out (30s)"

        output = stdout.decode("utf-8", errors="replace")
        errors = stderr.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            return f"Error (exit code {proc.returncode}):\n{errors}\n{output}"

        result = output.strip()
        if errors.strip():
            result += f"\n[stderr]: {errors.strip()}"

        return result if result else "(no output)"
    finally:
        os.unlink(tmp_path)
