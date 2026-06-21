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


async def ensure_vllm_base_url(model_manager: Any, engine_name: str) -> str:
    """`get_vllm_base_url` 的**按需懒加载**版:engine 未加载 → `await load_model`
    (复用 model_manager 现有 `_lock_for(model_id)` 防并发首调 + 显存守卫/驱逐)再解析。

    让已发布服务「发现即能调」——客户端首次调用冷模型不再吃 503「请先手动加载」,
    而是自动加载后服务(首调有加载延迟:小模型几秒、大模型几十秒)。加载失败
    (未知模型 / OOM 装不下)冒泡成 VLLMNotLoaded → 调用方仍映射 503,不比原来差。

    spec §4.5 D6/D8 的 base-URL 查找统一走这里(auto-load 变体)。
    """
    if model_manager is None:
        raise VLLMNotLoaded("model_manager 不可用（app.state 未初始化）")

    adapter = model_manager.get_adapter(engine_name)
    if adapter is None or not getattr(adapter, "is_loaded", False):
        try:
            # load_model 内 _lock_for(model_id) 串行化并发首调:第一个真加载,
            # 其余等锁后命中 is_loaded 直接返回,不会重复 spawn。
            await model_manager.load_model(engine_name)
        except Exception as e:  # noqa: BLE001 — 任何加载失败都收敛成「未加载」语义
            raise VLLMNotLoaded(
                f"模型 '{engine_name}' 自动加载失败：{e}"
            ) from e

    # 复用同步解析做最终校验(is_loaded 仍 False / base_url 空 → 各自抛对应异常)。
    return get_vllm_base_url(model_manager, engine_name)
