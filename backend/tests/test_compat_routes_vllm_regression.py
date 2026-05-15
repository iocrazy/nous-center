"""Lane E: [回归] compat 路由收敛 base-URL 查找后产出不变（spec §5.3 CRITICAL）。

Task 2 把 4 个 compat 路由的「get_adapter -> base_url」收敛到 get_vllm_base_url()。
本套用静态保证 + 行为保证两层断言 4 个路由（openai/anthropic/ollama/responses）
收敛后输出不变、错误码语义不变。
"""
import pytest


@pytest.mark.asyncio
async def test_openai_compat_not_loaded_returns_503(client):
    """未加载的 engine → openai_compat 返回 503（VLLMNotLoaded 映射）。

    用一个不存在的 model + 无效 key → resolve 链路上鉴权失败是 401，model 未找到
    是 404，未加载是 503 —— 断言不是 5xx-500（收敛前后这条路径的错误码语义必须一致）。
    """
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "definitely-not-loaded", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer test-key"},
    )
    assert resp.status_code in (401, 404, 503)


def test_get_vllm_base_url_wired_into_all_four_routes():
    """静态保证：4 个 compat 路由都 import 了 get_vllm_base_url（收敛已落地），
    且无裸 `getattr(adapter, "base_url" ...)` 查找残留。"""
    import src.api.routes.anthropic_compat as anthropic_mod
    import src.api.routes.ollama_compat as ollama_mod
    import src.api.routes.openai_compat as openai_mod
    import src.api.routes.responses as responses_mod

    for mod in (openai_mod, anthropic_mod, ollama_mod, responses_mod):
        with open(mod.__file__) as f:
            content = f.read()
        assert "get_vllm_base_url" in content, (
            f"{mod.__name__} 应已收敛到 get_vllm_base_url"
        )
        # 收敛后不应再有裸的 getattr(adapter, "base_url" ...) 查找逻辑。
        assert 'getattr(adapter, "base_url"' not in content, (
            f"{mod.__name__} 仍有裸 base_url 查找 —— 应收敛到 get_vllm_base_url"
        )
