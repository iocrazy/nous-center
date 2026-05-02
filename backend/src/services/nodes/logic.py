"""Logic-layer invokable nodes: prompt template, agent, python code, if/else.

Migrated from workflow_executor._exec_prompt_template / _exec_agent /
_exec_python_code / _exec_if_else as part of Wave 1 Task 4.4. Bodies are
copied verbatim — no refactor.
"""

from __future__ import annotations

import re

from src.services.nodes.registry import register


@register("prompt_template")
class PromptTemplateNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        """Replace {variable} placeholders in a template with input values."""
        template = data.get("template", "")
        result = template
        for key, value in inputs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return {"text": result}


@register("agent")
class AgentNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        """Execute agent with multi-turn tool call loop."""
        # Import via workflow_executor module so tests that patch
        # src.services.workflow_executor.agent_manager (legacy seam) keep
        # working after the class-dispatch migration (Wave 1 Task 4.5).
        from src.services import workflow_executor as we
        from src.services.llm_service import call_llm_with_tools
        from src.services.skill_tools import skills_to_tools, execute_tool
        from src.services.nodes.llm import _validate_llm_url
        from src.services.workflow_executor import ExecutionError
        from src.config import get_settings

        agent_manager = we.agent_manager

        agent_name = data.get("agent_name", "")
        input_text = inputs.get("text", "")
        if not agent_name:
            raise ExecutionError("Agent 节点未选择 Agent")
        if not input_text:
            raise ExecutionError("Agent 节点缺少输入")

        agent = agent_manager.get_agent(agent_name)
        if not agent:
            raise ExecutionError(f"Agent '{agent_name}' 不存在")

        # Assemble system prompt from MD files
        prompts = agent.get("prompts", {})
        system_parts: list[str] = []
        for fname in ["IDENTITY.md", "SOUL.md", "AGENT.md"]:
            content = prompts.get(fname, "").strip()
            if content:
                system_parts.append(content)
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        # Get tools from agent's skills
        agent_skills = agent.get("skills", [])
        tools = skills_to_tools(agent_skills)

        # LLM config
        model_config = agent.get("model", {})
        settings = get_settings()
        base_url = model_config.get("base_url") or settings.VLLM_BASE_URL
        model = model_config.get("model") or model_config.get("engine_key") or ""
        api_key = model_config.get("api_key") or model_config.get("fallback_api")

        _validate_llm_url(base_url)

        # Build allowed tool names set for whitelist validation
        allowed_tools = {t["function"]["name"] for t in tools} if tools else set()

        # Multi-turn loop (max 10 iterations to prevent infinite loops)
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": input_text})

        max_iterations = 10
        response: dict = {}
        for _i in range(max_iterations):
            response = await call_llm_with_tools(
                messages=messages,
                base_url=base_url,
                model=model,
                api_key=api_key,
                tools=tools if tools else None,
            )

            # Check if response has tool calls
            if response.get("tool_calls"):
                # Append assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": response.get("content", ""),
                    "tool_calls": response["tool_calls"],
                })

                for tool_call in response["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]

                    # Whitelist validation: reject unknown tools
                    if tool_name not in allowed_tools:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": f"Error: unknown tool '{tool_name}'",
                        })
                        continue

                    try:
                        result = await execute_tool(tool_name, tool_args)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": str(result),
                        })
                    except Exception as e:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": f"Error: {e}",
                        })
            else:
                # No tool calls — return final response
                return {"text": response.get("content", "")}

        # Max iterations reached
        return {"text": response.get("content", "Agent 达到最大迭代次数")}


@register("python_exec")
class PythonCodeNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        """Execute Python code from the node."""
        from src.services.skill_tools import _execute_python

        code = data.get("code", "")
        if not code and inputs.get("text"):
            code = inputs["text"]
        result = await _execute_python(code)
        return {"text": result}


@register("if_else")
class IfElseNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        """Conditional branching based on match_type."""
        from src.services.workflow_executor import ExecutionError

        condition = data.get("condition", "")
        match_type = data.get("match_type", "contains")
        text = inputs.get("text", "")

        if match_type == "contains":
            matched = condition in text
        elif match_type == "equals":
            matched = text == condition
        elif match_type == "regex":
            try:
                matched = bool(re.search(condition, text))
            except re.error:
                raise ExecutionError(f"无效的正则表达式: {condition}")
        elif match_type == "not_empty":
            matched = bool(text.strip())
        else:
            matched = False

        return {"true": text if matched else "", "false": text if not matched else ""}
