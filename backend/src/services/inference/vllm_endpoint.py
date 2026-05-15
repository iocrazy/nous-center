"""vLLM base-URL 查找 —— compat 路由 / workflow executor 直连 vLLM HTTP 的唯一真相源。

spec §4.5 D6/D8「inline 执行点改道清单」：openai/anthropic/ollama/responses compat
路由 + workflow executor 的 llm 节点都直连 vLLM 的 HTTP 端口，零 per-token pipe 开销。

本仓库的 compat 路由本来就走 HTTP→vLLM（各处 `getattr(adapter, "base_url")`），
Lane E 把那段重复的「get_adapter → 检查 is_loaded → 取 base_url → 检查空值」收敛到这里。

注意：Lane 0 删掉的旧 `get_llm_base_url`（model_scheduler.py:233）是零调用方死代码，
本函数是面向 `model_manager` 持有的 VLLMAdapter 的全新查找，不是它的复活。
"""
from __future__ import annotations

from typing import Any


class VLLMNotLoaded(RuntimeError):
    """目标 LLM engine 未加载（adapter 缺失 / is_loaded=False / model_manager 不可用）。

    调用方（compat 路由）应映射为 HTTP 503。
    """


class VLLMNoEndpoint(RuntimeError):
    """LLM engine 已加载但没有 HTTP 推理端点（base_url 为空）。

    调用方应映射为 HTTP 500 —— 这是不该发生的状态（vLLM 加载成功必有端口）。
    """


def get_vllm_base_url(model_manager: Any, engine_name: str) -> str:
    """返回 *engine_name* 对应 vLLM 实例的 HTTP base_url。

    Parameters
    ----------
    model_manager:
        `app.state.model_manager`（services.model_manager.ModelManager）。
        允许为 None —— app.state 尚未初始化时直接抛 VLLMNotLoaded。
    engine_name:
        模型 / engine 标识（ServiceInstance.source_name 或 source_id）。

    Raises
    ------
    VLLMNotLoaded:  model_manager 不可用，或 engine 未加载。
    VLLMNoEndpoint: engine 已加载但 base_url 为空。
    """
    if model_manager is None:
        raise VLLMNotLoaded("model_manager 不可用（app.state 未初始化）")

    adapter = model_manager.get_adapter(engine_name)
    if adapter is None or not getattr(adapter, "is_loaded", False):
        raise VLLMNotLoaded(
            f"模型 '{engine_name}' 未加载 —— 请在模型管理页加载后重试"
        )

    base_url = getattr(adapter, "base_url", None)
    if not base_url:
        raise VLLMNoEndpoint(
            f"模型 '{engine_name}' 已加载但没有 HTTP 推理端点（base_url 为空）"
        )
    return base_url
