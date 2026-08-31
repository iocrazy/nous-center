"""风格清单代理路由 —— 把 sidecar 的 ComfyUI-Easy-Use 风格数据转给前端。

为什么要代理而不是让前端直连 sidecar:sidecar 只监听内网(`NOUS_COMFY_URL`,
生产是 127.0.0.1:8888),浏览器根本够不着;而且 `/api/v1/*` 这条线已经有 admin
鉴权,直连等于绕过。

数据源 `GET <sidecar>/easyuse/prompt/styles?name=<包>` 现成返回
`{name, name_cn, thumbnail, prompt, negative_prompt}`,缩略图对 fooocus 包是
GitHub raw 外链、对 krea2 包是 sidecar 本地文件 —— 两种都是 URL,前端直接用。
"""
import pytest

import src.api.routes.comfy_templates as ct_mod

STYLES = [
    {"name": "Fooocus Enhance", "name_cn": "Fooocus-优化增强",
     "thumbnail": "https://raw.githubusercontent.com/x/fooocus_enhance.jpg",
     "prompt": "...", "negative_prompt": "..."},
    {"name": "sai-anime", "name_cn": "SAI-动漫",
     "thumbnail": "https://raw.githubusercontent.com/x/sai-anime.jpg"},
    {"name": "no-thumb"},
]


class FakeClient:
    def __init__(self, styles=None, packs=None, boom=False):
        self._styles = STYLES if styles is None else styles
        self._packs = packs if packs is not None else ["fooocus_styles", "krea2_397styles-anime_动漫"]
        self._boom = boom

    async def styles(self, pack: str):
        if self._boom:
            from src.services.comfy.client import ComfyError
            raise ComfyError("sidecar 掉线")
        assert pack, "pack 必须透传给 sidecar"
        return self._styles

    async def style_packs(self):
        if self._boom:
            from src.services.comfy.client import ComfyError
            raise ComfyError("sidecar 掉线")
        return self._packs


@pytest.mark.asyncio
async def test_styles_proxied_and_normalised(client, monkeypatch):
    """sidecar 的 name/name_cn/thumbnail → 统一成 mapping 直接能吃的
    {value,label,image} 形 —— 前端拿到就能塞进 exposed_params.options。"""
    monkeypatch.setattr(ct_mod, "get_client", lambda: FakeClient())
    r = await client.get("/api/v1/comfy/styles", params={"pack": "fooocus_styles"})
    assert r.status_code == 200, r.text
    opts = r.json()["options"]
    assert opts[0] == {"value": "Fooocus Enhance", "label": "Fooocus-优化增强",
                       "image": "https://raw.githubusercontent.com/x/fooocus_enhance.jpg"}
    # 没有 name_cn / thumbnail 的项:只留 value,不编造键
    assert opts[2] == {"value": "no-thumb"}


@pytest.mark.asyncio
async def test_styles_requires_pack(client, monkeypatch):
    """缺 pack 走全局 RequestValidationError handler(main.py:1159),它把 FastAPI
    的 422 统一成 OpenAI 兼容信封的 400 + code=validation_error —— 不是 422。"""
    monkeypatch.setattr(ct_mod, "get_client", lambda: FakeClient())
    r = await client.get("/api/v1/comfy/styles")
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "validation_error"


@pytest.mark.asyncio
async def test_style_packs_listed(client, monkeypatch):
    monkeypatch.setattr(ct_mod, "get_client", lambda: FakeClient())
    r = await client.get("/api/v1/comfy/style-packs")
    assert r.status_code == 200
    assert "fooocus_styles" in r.json()["packs"]


@pytest.mark.asyncio
async def test_sidecar_down_degrades_to_502_not_500(client, monkeypatch):
    """sidecar 掉线是可预期的(单元默认 disabled),要给可读的 502,不是裸 500。"""
    monkeypatch.setattr(ct_mod, "get_client", lambda: FakeClient(boom=True))
    r = await client.get("/api/v1/comfy/styles", params={"pack": "fooocus_styles"})
    assert r.status_code == 502, r.text
