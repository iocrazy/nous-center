"""选项依赖:一个入参的选项清单由**另一个入参的当前值**决定。

动机(krea2 实机):`style_pack` 切成别的风格包后,`styles` 的缩略图网格纹丝不动 ——
注册时冻结的 275 项静态 enum 是 `fooocus_styles` 那一份,新包里的风格既渲不出来、
传上去也必被静态白名单判成 422。这里测的是通用机制(mapping → schema → 运行期校验),
不是 krea2 专用逻辑。

sidecar 一律 mock —— 测试不碰真 ComfyUI。
"""
import pytest

from src.services.comfy import style_options
from src.services.service_schema import validate_service_input

WF = {"1243": {"class_type": "easy stylesSelector",
               "inputs": {"styles": "fooocus_styles", "select_styles": ""}}}

# 注册时冻结的静态清单(= 默认包 fooocus_styles 的那一份)。
STATIC = [{"value": "Fooocus Enhance", "label": "Fooocus-优化增强", "image": "https://x/a.jpg"},
          {"value": "sai-anime", "label": "SAI-动漫", "image": "https://x/b.jpg"}]


async def _mk(client, name):
    r = await client.post("/api/v1/comfy-templates", json={"name": name, "workflow": WF})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _mapping(**overrides):
    styles = {
        "key": "styles", "label": "风格(可多选,随风格包切换)", "type": "string",
        "comfy_node_id": "1243", "comfy_input": "select_styles", "required": False,
        "multiple": True, "options": STATIC,
        "options_depends_on": "style_pack", "options_source": "comfy_styles",
    }
    styles.update(overrides)
    return {"exposed_params": [
        {"key": "style_pack", "label": "风格包", "type": "string",
         "comfy_node_id": "1243", "comfy_input": "styles", "required": False,
         "default": "fooocus_styles",
         "options": ["fooocus_styles", "krea2_397styles-anime_动漫"]},
        styles,
    ]}


class FakeClient:
    """只实现 styles();记录调用次数,用来断言 TTL 缓存真的挡住了第二次往返。"""

    def __init__(self, packs=None, boom=False):
        self.packs = packs or {
            "fooocus_styles": ["Fooocus Enhance", "sai-anime"],
            "krea2_397styles-anime_动漫": ["krea-cel", "krea-ink"],
        }
        self.boom = boom
        self.calls: list[str] = []

    async def styles(self, pack: str):
        self.calls.append(pack)
        if self.boom:
            from src.services.comfy.client import ComfyError
            raise ComfyError("sidecar 掉线")
        return [{"name": n, "name_cn": n, "thumbnail": f"https://x/{n}.jpg"}
                for n in self.packs.get(pack, [])]


@pytest.fixture(autouse=True)
def _clear_style_cache():
    """TTL 缓存是模块级的 —— 用例之间必须清,否则顺序一变断言就飘。"""
    style_options.reset_style_options_cache()
    yield
    style_options.reset_style_options_cache()


# 全仓约定:pydantic 请求体校验失败经 main.py 的 RequestValidationError handler 统一
# 翻成 **400 + code=validation_error**(OpenAI 风格错误信封),不是裸 FastAPI 的 422。
BAD_REQUEST = 400


def _assert_rejected(r, needle: str = ""):
    assert r.status_code == BAD_REQUEST, r.text
    assert r.json()["error"]["code"] == "validation_error", r.text
    if needle:
        assert needle in r.text


# ---------- mapping 层 ----------


@pytest.mark.asyncio
async def test_mapping_round_trip(client):
    """新字段能存能取:PUT mapping → GET 模板详情原样吐回。"""
    tid = await _mk(client, "dep-roundtrip")
    r = await client.put(f"/api/v1/comfy-templates/{tid}/mapping", json=_mapping())
    assert r.status_code == 200, r.text

    detail = (await client.get(f"/api/v1/comfy-templates/{tid}")).json()
    styles = next(p for p in detail["exposed_params"] if p["key"] == "styles")
    assert styles["options_depends_on"] == "style_pack"
    assert styles["options_source"] == "comfy_styles"
    assert styles["multiple"] is True
    # 静态 options 仍在(默认包的兜底),没被依赖声明顶掉
    assert styles["options"][0]["value"] == "Fooocus Enhance"
    # 不声明依赖的字段:两个键都是 None,老映射零变化
    pack = next(p for p in detail["exposed_params"] if p["key"] == "style_pack")
    assert pack["options_depends_on"] is None and pack["options_source"] is None


@pytest.mark.asyncio
async def test_depends_on_unknown_key_rejected(client):
    """指向不存在的 key → 拒绝(否则运行期静默退回静态 enum,症状查不出原因)。"""
    tid = await _mk(client, "dep-unknown")
    r = await client.put(f"/api/v1/comfy-templates/{tid}/mapping",
                         json=_mapping(options_depends_on="no_such_key"))
    _assert_rejected(r, "no_such_key")


@pytest.mark.asyncio
async def test_depends_on_self_rejected(client):
    tid = await _mk(client, "dep-self")
    r = await client.put(f"/api/v1/comfy-templates/{tid}/mapping",
                         json=_mapping(options_depends_on="styles"))
    _assert_rejected(r)


@pytest.mark.asyncio
async def test_depends_on_without_source_rejected(client):
    """只写 depends_on 不写 source:运行期不知道去哪儿取清单 → 发布时就拒。"""
    tid = await _mk(client, "dep-nosource")
    r = await client.put(f"/api/v1/comfy-templates/{tid}/mapping",
                         json=_mapping(options_source=None))
    _assert_rejected(r)


@pytest.mark.asyncio
async def test_unknown_source_rejected(client):
    """options_source 是枚举(目前只有 comfy_styles),乱写的来源直接拒。"""
    tid = await _mk(client, "dep-badsource")
    r = await client.put(f"/api/v1/comfy-templates/{tid}/mapping",
                         json=_mapping(options_source="from_mars"))
    _assert_rejected(r)


# ---------- schema 层 ----------


@pytest.mark.asyncio
async def test_schema_exposes_dependency_keys(client):
    """schema 输出 x-options-depends-on / x-options-source,静态 enum + meta 照旧。"""
    tid = await _mk(client, "dep-schema")
    assert (await client.put(f"/api/v1/comfy-templates/{tid}/mapping",
                             json=_mapping())).status_code == 200

    props = (await client.get("/v1/services/dep-schema/schema")).json()["input_schema"]["properties"]
    prop = props["styles"]
    assert prop["x-options-depends-on"] == "style_pack"
    assert prop["x-options-source"] == "comfy_styles"
    # 兜底/离线展示的静态清单没被删
    assert prop["enum"] == ["Fooocus Enhance", "sai-anime"]
    assert prop["x-option-meta"][0]["image"] == "https://x/a.jpg"
    assert prop["x-multiple"] is True
    # 依赖参数自己不带这两个键
    assert "x-options-depends-on" not in props["style_pack"]


# ---------- 运行期校验 ----------


def _schema():
    return {"type": "object", "properties": {
        "style_pack": {"type": "string", "default": "fooocus_styles",
                       "enum": ["fooocus_styles", "krea2_397styles-anime_动漫"]},
        "styles": {"type": "string", "enum": ["Fooocus Enhance", "sai-anime"],
                   "x-multiple": True, "x-options-depends-on": "style_pack",
                   "x-options-source": "comfy_styles"}}}


@pytest.mark.asyncio
async def test_validation_allows_values_from_switched_pack(monkeypatch):
    """切到别的包后,**新包**里的风格放行 —— 静态 enum 里根本没有这些值。"""
    fake = FakeClient()
    monkeypatch.setattr(style_options, "get_comfy_client", lambda: fake)

    payload = {"style_pack": "krea2_397styles-anime_动漫", "styles": "krea-cel,krea-ink"}
    dyn = await style_options.resolve_dynamic_enums(_schema(), payload)
    assert dyn == {"styles": ["krea-cel", "krea-ink"]}
    assert validate_service_input(_schema(), payload, dynamic_enums=dyn) == []


@pytest.mark.asyncio
async def test_validation_rejects_values_not_in_switched_pack(monkeypatch):
    """切包后,旧包(静态 enum)里的值不再合法 —— 动态清单**覆盖**静态 enum。"""
    fake = FakeClient()
    monkeypatch.setattr(style_options, "get_comfy_client", lambda: fake)

    payload = {"style_pack": "krea2_397styles-anime_动漫", "styles": "sai-anime"}
    dyn = await style_options.resolve_dynamic_enums(_schema(), payload)
    errs = validate_service_input(_schema(), payload, dynamic_enums=dyn)
    assert len(errs) == 1 and "sai-anime" in errs[0], errs


@pytest.mark.asyncio
async def test_pack_falls_back_to_default_when_not_submitted(monkeypatch):
    """没传依赖参数 → 用它 schema 里的 default 当包名。"""
    fake = FakeClient()
    monkeypatch.setattr(style_options, "get_comfy_client", lambda: fake)

    dyn = await style_options.resolve_dynamic_enums(_schema(), {"styles": "sai-anime"})
    assert fake.calls == ["fooocus_styles"]
    assert dyn == {"styles": ["Fooocus Enhance", "sai-anime"]}


@pytest.mark.asyncio
async def test_sidecar_down_falls_back_to_static_enum(monkeypatch, caplog):
    """sidecar 不可达:不抛错,退回静态 enum 校验 + 一条 warning(预测不该 500)。"""
    fake = FakeClient(boom=True)
    monkeypatch.setattr(style_options, "get_comfy_client", lambda: fake)

    payload = {"style_pack": "krea2_397styles-anime_动漫", "styles": "sai-anime"}
    with caplog.at_level("WARNING"):
        dyn = await style_options.resolve_dynamic_enums(_schema(), payload)
    assert dyn == {}
    assert any("风格包" in r.message or "风格包" in r.getMessage() for r in caplog.records)
    # 退回静态 enum:静态清单里有的值照旧放行
    assert validate_service_input(_schema(), payload, dynamic_enums=dyn) == []


@pytest.mark.asyncio
async def test_ttl_cache_avoids_second_sidecar_call(monkeypatch):
    """同一个包第二次解析走进程内缓存,不再打 sidecar。"""
    fake = FakeClient()
    monkeypatch.setattr(style_options, "get_comfy_client", lambda: fake)

    payload = {"style_pack": "fooocus_styles", "styles": "sai-anime"}
    await style_options.resolve_dynamic_enums(_schema(), payload)
    await style_options.resolve_dynamic_enums(_schema(), payload)
    assert fake.calls == ["fooocus_styles"], "第二次必须命中缓存"

    # 换个包 = 换个缓存键,该打还得打
    await style_options.resolve_dynamic_enums(
        _schema(), {"style_pack": "krea2_397styles-anime_动漫", "styles": "krea-cel"})
    assert fake.calls == ["fooocus_styles", "krea2_397styles-anime_动漫"]

    # 过期后重新取
    style_options.reset_style_options_cache()
    await style_options.resolve_dynamic_enums(_schema(), payload)
    assert fake.calls.count("fooocus_styles") == 2


@pytest.mark.asyncio
async def test_no_dependency_declared_means_no_sidecar_call(monkeypatch):
    """回归:没有字段声明依赖时,预测路径一次 sidecar 都不打。"""
    fake = FakeClient()
    monkeypatch.setattr(style_options, "get_comfy_client", lambda: fake)

    plain = {"type": "object", "properties": {
        "pick": {"type": "string", "enum": ["a", "b"]}}}
    assert await style_options.resolve_dynamic_enums(plain, {"pick": "a"}) == {}
    assert fake.calls == []


@pytest.mark.asyncio
async def test_prediction_endpoint_uses_dynamic_pack(client, monkeypatch):
    """端到端接线:预测入口按**提交的包**校验 —— 静态 enum 里有、新包里没有的值被拒。

    故意断言"拒绝"这一侧:通过的那一侧会真的排一个渲染任务去敲 sidecar(本机可能真
    跑着 ComfyUI),测试不该碰它。放行逻辑由上面的单元用例覆盖。
    """
    tid = await _mk(client, "dep-predict")
    assert (await client.put(f"/api/v1/comfy-templates/{tid}/mapping",
                             json=_mapping())).status_code == 200

    fake = FakeClient()
    monkeypatch.setattr(style_options, "get_comfy_client", lambda: fake)

    r = await client.post(
        "/v1/services/dep-predict/predictions",
        headers={"Prefer": "respond-async"},
        json={"input": {"style_pack": "krea2_397styles-anime_动漫",
                        "styles": "sai-anime"}})
    assert r.status_code == 422, r.text
    assert "sai-anime" in r.text
    assert fake.calls == ["krea2_397styles-anime_动漫"], "校验用的是提交的包,不是默认包"
