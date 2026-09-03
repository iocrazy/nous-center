"""ComfyUI sidecar HTTP 客户端(spec §5)。所有请求 trust_env=False。"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.parse

import httpx


def _base_url() -> str:
    return os.getenv("NOUS_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")


# ComfyUI-Easy-Use 的风格缩略图路由。代理端点只打这一条(见 ComfyClient.style_image)。
STYLE_IMAGE_ROUTE = "/easyuse/prompt/styles/image"


class ComfyError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def translate_prompt_error(status: int, body: str) -> str:
    """ComfyUI /prompt 校验 payload → 可操作中文(仿 IC comfy_prompt_error_message)。"""
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return f"ComfyUI 请求失败(HTTP {status}):{body[:300] or '未知错误'}"
    parts: list[str] = []
    top = (data.get("error") or {}).get("message") or ""
    for node_id, ne in (data.get("node_errors") or {}).items():
        ct = ne.get("class_type") or ""
        for err in ne.get("errors") or []:
            inp = (err.get("extra_info") or {}).get("input_name") or ""
            parts.append(f"节点 {node_id}({ct}) 输入 {inp}:{err.get('message', '')}")
    detail = ";".join(parts) or top
    return f"ComfyUI 拒绝了工作流(HTTP {status}):{detail[:600] or '校验失败'}。请检查模板字段映射。"


class ComfyClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or _base_url()).rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url, trust_env=False,
            timeout=float(os.getenv("NOUS_COMFY_DOWNLOAD_TIMEOUT", "120")))

    async def health(self) -> dict:
        try:
            r = await self._client.get("/queue", timeout=3)
            q = r.json()
            depth = len(q.get("queue_running", [])) + len(q.get("queue_pending", []))
            ver = ""
            try:
                ver = (await self._client.get("/system_stats", timeout=3)).json() \
                    .get("system", {}).get("comfyui_version", "")
            except (httpx.HTTPError, ValueError):
                pass
            return {"online": True, "queue_depth": depth, "version": ver}
        except (httpx.HTTPError, ValueError):
            return {"online": False, "queue_depth": 0, "version": ""}

    async def object_info(self) -> dict:
        return (await self._client.get("/object_info", timeout=15)).json()

    async def styles(self, pack: str, *, timeout: float = 15.0) -> list[dict]:
        """ComfyUI-Easy-Use 的风格清单(`/easyuse/prompt/styles?name=<包>`)。

        返回项形如 `{name, name_cn, thumbnail, prompt, negative_prompt}`。thumbnail
        对 fooocus_styles 是 GitHub raw 外链(浏览器直接能加载),对 krea2 那批包是
        **sidecar 侧的相对路径**(`/easyuse/prompt/styles/image?path=…`)—— 后者浏览器
        会按 nous 的 origin 解析必 404,由路由层 `_style_to_option` 改写成走
        `/api/v1/comfy/style-image` 代理(见 comfy_templates.py)。
        """
        r = await self._client.get(
            "/easyuse/prompt/styles", params={"name": pack}, timeout=timeout)
        if r.status_code != 200:
            raise ComfyError(f"读取风格清单失败(HTTP {r.status_code})")
        data = r.json()
        return data if isinstance(data, list) else []

    async def style_image(self, path: str) -> tuple[bytes, str]:
        """取一张风格缩略图。`path` 是 sidecar 缩略图 URL 里 `path=` 那个值
        (如 `./samples/x.jpg`),**不是**整条 URL。

        krea2 那批风格包的 `thumbnail` 是**相对路径**(fooocus 包才是 GitHub 外链)——
        浏览器拿到会按 nous 自己的 origin 解析,必 404。所以要经后端代理一手。

        路由**写死**在这里、参数走 `params=`,绝不把调用方给的字符串当 URL 拼:
          · httpx 合并相对 URL 时会归一化点段 —— `/easyuse/../history` 直接变成
            `/history`(实测),任何"前缀白名单"都挡不住,代理会退化成"拿 admin
            身份任意打 sidecar GET"的跳板;
          · 当 URL 拼还会把文件名里的 `#` 当 fragment、`%`/`+` 当转义,
            `./samples/Neon #3.jpg` 这种名字的图直接取不回来。
        """
        r = await self._client.get(
            STYLE_IMAGE_ROUTE, params={"path": path}, timeout=30)
        if r.status_code != 200:
            raise ComfyError(f"读取风格缩略图失败(HTTP {r.status_code})")
        return r.content, r.headers.get("content-type", "image/jpeg")

    async def style_packs(self) -> list[str]:
        """可选的风格包清单 —— 从 object_info 的 `easy stylesSelector.styles` combo 取。

        没装 ComfyUI-Easy-Use(或节点改名)时返回空列表,由路由层决定怎么呈现。
        """
        oi = await self.object_info()
        try:
            return list(oi["easy stylesSelector"]["input"]["required"]["styles"][1]["options"])
        except (KeyError, IndexError, TypeError):
            return []

    async def system_stats(self) -> dict:
        """VRAM/设备快照,同 health() 降级——瞬断不应打断健康面板渲染。"""
        try:
            return (await self._client.get("/system_stats", timeout=3)).json()
        except (httpx.HTTPError, ValueError):
            return {}

    async def free(self, *, unload_models: bool = True, free_memory: bool = True) -> None:
        """用户触发的显存释放——失败必须冒泡(不像 health/system_stats 静默降级)。"""
        r = await self._client.post(
            "/free",
            json={"unload_models": unload_models, "free_memory": free_memory},
            timeout=30,
        )
        if r.status_code != 200:
            raise ComfyError(f"释放显存失败(HTTP {r.status_code})")

    async def upload_image(self, filename: str, content: bytes, mime: str = "image/png") -> str:
        r = await self._client.post(
            "/upload/image", files={"image": (filename, content, mime)}, timeout=60)
        if r.status_code != 200:
            raise ComfyError(f"上传参考素材失败(HTTP {r.status_code})")
        return r.json().get("name", filename)

    async def submit(self, graph: dict) -> str:
        r = await self._client.post("/prompt", json={"prompt": graph}, timeout=30)
        if r.status_code != 200:
            raise ComfyError(translate_prompt_error(r.status_code, r.text), status_code=422)
        return r.json()["prompt_id"]

    async def wait(self, prompt_id: str, *, timeout_s: float, interval_s: float = 2.0) -> dict:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout_s
        while True:
            try:
                res = (await self._client.get(f"/history/{prompt_id}", timeout=10)).json()
                if prompt_id in res:
                    return res[prompt_id]
            except (httpx.HTTPError, ValueError):
                pass  # sidecar 瞬断:继续轮询直到超时
            if loop.time() >= deadline:
                raise ComfyError("ComfyUI 渲染超时(NOUS_COMFY_TIMEOUT)。注意:ComfyUI 侧任务可能仍在运行。")
            await asyncio.sleep(interval_s)

    async def download(self, item: dict) -> bytes:
        qs = urllib.parse.urlencode({
            "filename": item["filename"], "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output")})
        r = await self._client.get(f"/view?{qs}")
        if r.status_code != 200:
            raise ComfyError(f"下载产物失败(HTTP {r.status_code}):{item['filename']}")
        return r.content

    async def interrupt(self) -> None:
        try:
            await self._client.post("/interrupt", timeout=5)
        except httpx.HTTPError:
            pass  # 尽力而为:sidecar 掉线时取消不应抛错


# I4 fix:进程级懒单例——`comfy_bridge.py` / `comfy_templates.py` 此前各自的
# `get_client()` 都是「每次调用现建一个 ComfyClient」,每建一个就新开一个
# httpx.AsyncClient(+ 连接池),旧的从不关闭,纯泄漏。改成单进程共享一个实例。
# 两个模块各自 `from ... import get_comfy_client as get_client` 保留各自的
# `get_client` 名字(测试按模块 monkeypatch.setattr(mod, "get_client", ...) 的
# 惯例不变——patch 的是各自模块命名空间里的这个名字,互不影响)。
_singleton: ComfyClient | None = None


def get_comfy_client() -> ComfyClient:
    global _singleton
    if _singleton is None:
        _singleton = ComfyClient()
    return _singleton


def reset_comfy_client() -> None:
    """测试/重配置用:丢弃缓存的单例,下次 get_comfy_client() 重新构造。"""
    global _singleton
    _singleton = None
