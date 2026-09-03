"""动态选项来源:某个入参的合法值域取决于**另一个入参**的当前值。

目前唯一的来源是 `comfy_styles` —— ComfyUI-Easy-Use 的 `easy stylesSelector`:
`styles`(风格包)决定 `select_styles`(具体风格)的清单。注册模板时冻结下来的
静态 enum 只是**默认包**的那一份,用户在 Playground 里切了包之后,合法值就换了一
整批,静态白名单会把新包里的风格全判成非法(422)。

这里做两件事:
1. `style_values(pack)` —— 打 sidecar 拿某个包的合法值,带**进程内 TTL 缓存**
   (默认 10 分钟),免得每次预测都去敲一次 ComfyUI;
2. `resolve_dynamic_enums(schema, payload)` —— 扫 input_schema 里带
   `x-options-source` 的字段,解析出各自的包名并取回值列表,交给
   `validate_service_input(..., dynamic_enums=...)`。

**sidecar 不可达时一律返回「没有动态清单」**(并 warning),让校验退回静态 enum ——
拿不到清单不该让一次预测直接 500 / 全盘拒绝。
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from src.services.comfy.client import ComfyError, get_comfy_client

logger = logging.getLogger(__name__)

# 风格清单几乎不变(装/删风格包才会动),10 分钟足够新鲜,又能把一串连续预测
# 的 sidecar 往返压成一次。
STYLE_CACHE_TTL_S = 600.0

# pack → (取回时刻(monotonic), 合法值列表)。进程内、无上限——包的数量是几十级别。
_style_cache: dict[str, tuple[float, list[str]]] = {}


def reset_style_options_cache() -> None:
    """测试/重配置用:清空 TTL 缓存。"""
    _style_cache.clear()


async def style_values(pack: str, *, client: Any = None) -> list[str] | None:
    """某风格包里的合法取值(`name` 列表)。拿不到 → `None`(调用方退回静态 enum)。"""
    now = time.monotonic()
    hit = _style_cache.get(pack)
    if hit is not None and now - hit[0] < STYLE_CACHE_TTL_S:
        return hit[1]
    c = client if client is not None else get_comfy_client()
    try:
        items = await c.styles(pack)
    except (ComfyError, httpx.HTTPError, ValueError, OSError) as e:
        # 退回静态 enum 校验:sidecar 抖一下不该让预测挂掉。
        logger.warning("读取风格包 %r 的清单失败,本次退回静态 enum 校验:%s", pack, e)
        return None
    values = [
        i["name"] for i in (items or [])
        if isinstance(i, dict) and isinstance(i.get("name"), str) and i["name"]
    ]
    _style_cache[pack] = (now, values)
    return values


def _pack_for(props: dict, spec: dict, payload: dict) -> str | None:
    """字段的 `x-options-depends-on` 指向的那个参数的**实际值**;没传就用它的 default。"""
    dep = spec.get("x-options-depends-on")
    if not isinstance(dep, str) or not dep:
        return None
    v = payload.get(dep)
    if v is None:
        v = (props.get(dep) or {}).get("default")
    return v if isinstance(v, str) and v else None


async def resolve_dynamic_enums(input_schema: dict, payload: Any) -> dict[str, list]:
    """input_schema 里声明了动态来源的字段 → `{key: 允许值列表}`。

    为什么在**调用点预取**、而不是把 `validate_service_input` 改成 async:那个函数是
    一段纯同步的手写校验,被 4 个测试文件和 `/v1/services/{name}/predictions` 之外的
    地方直接调用;改成 async 会把 await 传染给所有调用方(以及 schema 端点这种压根
    不需要 sidecar 的路径)。这里只在**真有字段声明了依赖**时多一次 await,校验函数
    保持同步、可单测、零 I/O。
    """
    props = (input_schema or {}).get("properties") or {}
    if not isinstance(payload, dict):
        payload = {}
    out: dict[str, list] = {}
    for key, spec in props.items():
        if not isinstance(spec, dict) or spec.get("x-options-source") != "comfy_styles":
            continue
        pack = _pack_for(props, spec, payload)
        if pack is None:
            continue
        values = await style_values(pack)
        if values:
            out[key] = values
    return out
