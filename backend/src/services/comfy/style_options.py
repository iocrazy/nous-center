"""动态选项来源:某个入参的合法值域取决于**另一个入参**的当前值。

目前唯一的来源是 `comfy_styles` —— ComfyUI-Easy-Use 的 `easy stylesSelector`:
`styles`(风格包)决定 `select_styles`(具体风格)的清单。注册模板时冻结下来的
静态 enum 只是**默认包**的那一份,用户在 Playground 里切了包之后,合法值就换了一
整批,静态白名单会把新包里的风格全判成非法。

这里做两件事:
1. `style_values(pack)` —— 打 sidecar 拿某个包的合法值,带**进程内 TTL 缓存**;
2. `resolve_dynamic_enums(schema, payload)` —— 扫 input_schema 里带
   `x-options-source` 的字段,解析出各自的包名并取回值列表,交给
   `validate_service_input(..., dynamic_enums=...)`。

三个必须分清的结果(混一起就是 bug):
  · **取不到**(sidecar 挂了/超时)→ `None` → 调用方退回静态 enum,只 warning,
    绝不让一次预测 500;
  · **取到了但是空**(包存在但没加载出风格)→ `[]` → **按空白名单执行**,任何值都拒。
    退回默认包的静态 enum 才是危险的:那正是本模块要防的"拿 A 包的清单校验 B 包";
  · **取到了** → 值列表。

缓存三条约束:
  · 成功 10 分钟(风格清单只有装/删风格包才会动);
  · 失败与空结果只缓存 30 秒(**负缓存**)—— sidecar 卡死时不必每个预测都去等满超时,
    但也不能把一次抖动记上十分钟;
  · 条目数有上限并按插入序驱逐 —— 包名来自请求,不设上限就是个能被喂大的进程内字典。
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from src.services.comfy.client import ComfyError, get_comfy_client

logger = logging.getLogger(__name__)

# 成功结果的新鲜度。风格清单几乎不变,10 分钟足够,又能把一串连续预测压成一次往返。
STYLE_CACHE_TTL_S = 600.0
# 失败 / 空结果的新鲜度(负缓存)。sidecar 活着但卡死时,不必每个预测都等满超时;
# 30 秒也短到 sidecar 一恢复就能自愈。
STYLE_CACHE_FAILURE_TTL_S = 30.0
# 缓存条目上限(按插入序驱逐最旧的)。
STYLE_CACHE_MAX_ENTRIES = 256
# 这是**预校验**用的取数,不是用户在等的渲染:超时要短。sidecar 正常时这条是毫秒级,
# 卡住时也只拖 5 秒就退回静态 enum(路由层给前端列清单的那条仍用默认 15 秒)。
STYLE_FETCH_TIMEOUT_S = 5.0

# pack → (取回时刻(monotonic), 合法值列表 | None)。None = 那次取失败(负缓存)。
_style_cache: dict[str, tuple[float, list[str] | None]] = {}


def reset_style_options_cache() -> None:
    """测试/重配置用:清空 TTL 缓存。"""
    _style_cache.clear()


def _cache_put(pack: str, values: list[str] | None) -> None:
    """写缓存并维持上限。dict 保插入序,驱逐 = 弹最前面那个。

    先 pop 再写:刷新一个已有 key 时把它挪到队尾,免得"刚取回的热条目"因为当初插入
    得早而被优先驱逐。
    """
    _style_cache.pop(pack, None)
    while len(_style_cache) >= STYLE_CACHE_MAX_ENTRIES:
        _style_cache.pop(next(iter(_style_cache)))
    _style_cache[pack] = (time.monotonic(), values)


def _cache_get(pack: str) -> tuple[bool, list[str] | None]:
    """→ `(命中?, 值)`。命中的 `None` 是"上次取失败"的负缓存,不是"没缓存"。"""
    hit = _style_cache.get(pack)
    if hit is None:
        return False, None
    ts, values = hit
    # 失败与空结果只活 30 秒,成功的活 10 分钟。
    ttl = STYLE_CACHE_TTL_S if values else STYLE_CACHE_FAILURE_TTL_S
    if time.monotonic() - ts >= ttl:
        return False, None
    return True, values


async def style_values(pack: str, *, client: Any = None) -> list[str] | None:
    """某风格包里的合法取值(`name` 列表)。

    `None` = 取不到(调用方退回静态 enum);`[]` = 取到了但这个包是空的(调用方按空
    白名单执行)。两者语义不同,别合并。
    """
    hit, cached = _cache_get(pack)
    if hit:
        return cached
    c = client if client is not None else get_comfy_client()
    try:
        items = await c.styles(pack, timeout=STYLE_FETCH_TIMEOUT_S)
    except (ComfyError, httpx.HTTPError, ValueError, OSError) as e:
        # 退回静态 enum 校验:sidecar 抖一下不该让预测挂掉。负缓存 30 秒,免得
        # sidecar 卡死时每个预测都去等满超时。
        logger.warning("读取风格包 %r 的清单失败,本次退回静态 enum 校验:%s", pack, e)
        _cache_put(pack, None)
        return None
    values = [
        i["name"] for i in (items or [])
        if isinstance(i, dict) and isinstance(i.get("name"), str) and i["name"]
    ]
    _cache_put(pack, values)
    return values


def _pack_for(props: dict, spec: dict, payload: dict) -> str | None:
    """字段的 `x-options-depends-on` 指向的那个参数的**实际值**;没传就用它的 default。

    只有**受信任**的值才会被拿去打 sidecar:依赖参数自己声明了 enum 就必须落在里面,
    没声明 enum 就只认 schema 里的 default。原因是本函数跑在
    `validate_service_input` **之前** —— 不设这道闸,任何持 key 的调用方随便 POST 一个
    包名就能让后端往 sidecar 发一次请求、并在进程内缓存里占一格。落不进白名单的值
    直接返回 None:它本来就会被随后的静态 enum 校验拒掉,不必先替它跑一趟网络。
    """
    dep = spec.get("x-options-depends-on")
    if not isinstance(dep, str) or not dep:
        return None
    dep_spec = props.get(dep) or {}
    default = dep_spec.get("default")
    v = payload.get(dep)
    if v is None:
        v = default
    if not isinstance(v, str) or not v:
        return None
    allowed = dep_spec.get("enum")
    if isinstance(allowed, list):
        return v if v in allowed else None
    return v if v == default else None


async def resolve_dynamic_enums(input_schema: dict, payload: Any) -> dict[str, list]:
    """input_schema 里声明了动态来源的字段 → `{key: 允许值列表}`。

    为什么在**调用点预取**、而不是把 `validate_service_input` 改成 async:那个函数是
    一段纯同步的手写校验,被 4 个测试文件和 `/v1/services/{name}/predictions` 之外的
    地方直接调用;改成 async 会把 await 传染给所有调用方(以及 schema 端点这种压根
    不需要 sidecar 的路径)。这里只在**真有字段声明了依赖**时多一次 await,校验函数
    保持同步、可单测、零 I/O。

    取不到清单的字段**不进结果**(调用方自然退回静态 enum);取到空清单的字段以 `[]`
    进结果(空白名单,任何值都拒)。
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
        if values is not None:
            out[key] = values
    return out
