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


# ---------- 缩略图代理 ----------


class ImageClient:
    """styles() 给 krea2 那种相对路径缩略图;style_image() 回一坨假 JPEG。"""

    def __init__(self, boom=False, thumbnail=None):
        self.boom = boom
        self.thumbnail = thumbnail
        self.seen: list[str] = []

    async def styles(self, pack: str, *, timeout: float = 15.0):
        thumb = self.thumbnail or (
            "/easyuse/prompt/styles/image?path=./samples/摄影__Photography.jpg")
        return [{"name": "Photography", "name_cn": "摄影", "thumbnail": thumb},
                {"name": "Fooocus Enhance", "name_cn": "Fooocus-优化增强",
                 "thumbnail": "https://raw.githubusercontent.com/x/a.jpg"}]

    async def style_image(self, path: str):
        self.seen.append(path)
        if self.boom:
            from src.services.comfy.client import ComfyError
            raise ComfyError("sidecar 掉线")
        return b"\xff\xd8\xff-fake-jpeg", "image/jpeg"


@pytest.mark.asyncio
async def test_relative_thumbnail_rewritten_to_proxy(client, monkeypatch):
    """krea2 那批包的 thumbnail 是 **sidecar 侧相对路径** —— 浏览器按 nous 的 origin
    解析必 404。归一时抠出 `path=` 改写成走代理;绝对外链原样透传。"""
    monkeypatch.setattr(ct_mod, "get_client", lambda: ImageClient())
    opts = (await client.get("/api/v1/comfy/styles", params={"pack": "krea2"})).json()["options"]

    assert opts[0]["image"].startswith("/api/v1/comfy/style-image?path=")
    # 传的是**文件路径**,不是整条 sidecar URL —— 代理端点没有"转发任意 src"这回事
    assert "easyuse" not in opts[0]["image"]
    assert opts[1]["image"] == "https://raw.githubusercontent.com/x/a.jpg", "外链不动"


@pytest.mark.asyncio
async def test_thumbnail_without_path_is_dropped(client, monkeypatch):
    """相对地址里没有 path= → 代理不了,干脆不给 image(前端退回文字卡片),
    而不是吐一个必然 404 的地址。"""
    monkeypatch.setattr(ct_mod, "get_client",
                        lambda: ImageClient(thumbnail="/easyuse/whatever.jpg"))
    opts = (await client.get("/api/v1/comfy/styles", params={"pack": "krea2"})).json()["options"]
    assert "image" not in opts[0]


@pytest.mark.asyncio
async def test_thumbnail_with_dotdot_is_dropped(client, monkeypatch):
    monkeypatch.setattr(ct_mod, "get_client", lambda: ImageClient(
        thumbnail="/easyuse/prompt/styles/image?path=../../../etc/passwd"))
    opts = (await client.get("/api/v1/comfy/styles", params={"pack": "krea2"})).json()["options"]
    assert "image" not in opts[0]


@pytest.mark.asyncio
async def test_style_image_proxied(client, monkeypatch):
    fake = ImageClient()
    monkeypatch.setattr(ct_mod, "get_client", lambda: fake)
    r = await client.get("/api/v1/comfy/style-image",
                         params={"path": "./samples/摄影__Photography.jpg"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("image/jpeg")
    # private 而不是 public:这个端点要 admin 鉴权,public 会让 cloudflared /
    # 中间缓存把字节回给没鉴权的人。
    assert r.headers.get("cache-control") == "private, max-age=3600"
    assert fake.seen == ["./samples/摄影__Photography.jpg"]


@pytest.mark.asyncio
async def test_style_image_rejects_dotdot(client, monkeypatch):
    """`..` 段直接 400 —— sidecar 自己会拿这个路径去找文件,别让它接到能跳出
    styles 目录的东西。"""
    fake = ImageClient()
    monkeypatch.setattr(ct_mod, "get_client", lambda: fake)
    for bad in ["../../etc/passwd", "./samples/../../secret.png",
                "..\\..\\windows\\win.ini"]:
        r = await client.get("/api/v1/comfy/style-image", params={"path": bad})
        assert r.status_code == 400, (bad, r.text)
    assert fake.seen == [], "被拒的请求一次都不该打到 sidecar"


@pytest.mark.asyncio
async def test_style_image_cannot_reach_other_sidecar_routes(client, monkeypatch):
    """回归(实测过的安全绕过):早先那版收整条 `src` 再转发、靠
    `startswith("/easyuse/")` 把关,而 httpx 合并相对 URL 时会**归一化点段** ——
    `/easyuse/../history` 到 sidecar 就成了 `/history`,于是这个端点等于"拿 admin
    身份任意打 sidecar GET"。现在参数是文件路径、路由写死,`src` 根本不存在。
    """
    fake = ImageClient()
    monkeypatch.setattr(ct_mod, "get_client", lambda: fake)

    # 老的绕过形状:src 参数已不被接受(缺 path → 422 参数校验)
    r = await client.get("/api/v1/comfy/style-image", params={"src": "/easyuse/../history"})
    assert r.status_code in (400, 422), r.text

    # 就算把同样的串塞进 path,也只会被当成文件名传给缩略图路由,且 `..` 先被拦下
    r2 = await client.get("/api/v1/comfy/style-image", params={"path": "/easyuse/../history"})
    assert r2.status_code == 400, r2.text
    assert fake.seen == []


@pytest.mark.asyncio
async def test_style_image_path_with_hash_and_space(client, monkeypatch):
    """含 `#`/空格/`%` 的文件名要原样送到 sidecar。

    早先当 URL 字符串拼给 httpx 时,`#` 会被当 fragment 把查询截断
    (`path=./samples/Neon #3 50%.jpg` → `path=./samples/Neon%20`),图必取不回来。
    改走 `params=` 之后由 httpx 负责转义,值不再被二次解析。
    """
    fake = ImageClient()
    monkeypatch.setattr(ct_mod, "get_client", lambda: fake)
    weird = "./samples/Neon #3 50% + more.jpg"
    r = await client.get("/api/v1/comfy/style-image", params={"path": weird})
    assert r.status_code == 200, r.text
    assert fake.seen == [weird], "路径要一字不差地到达 client"


@pytest.mark.asyncio
async def test_style_image_sidecar_down_is_502(client, monkeypatch):
    monkeypatch.setattr(ct_mod, "get_client", lambda: ImageClient(boom=True))
    r = await client.get("/api/v1/comfy/style-image", params={"path": "x.jpg"})
    assert r.status_code == 502, r.text


def test_client_style_image_hits_fixed_route_with_params():
    """单测 client 那一层:URL 由写死的路由 + params 组装,不受 path 内容影响。"""
    from src.services.comfy.client import STYLE_IMAGE_ROUTE, ComfyClient

    c = ComfyClient(base_url="http://sidecar:8888")
    for evil in ["/easyuse/../history", "../../view?filename=secret.png"]:
        url = str(c._client.build_request(
            "GET", STYLE_IMAGE_ROUTE, params={"path": evil}).url)
        assert url.startswith("http://sidecar:8888/easyuse/prompt/styles/image?"), url
