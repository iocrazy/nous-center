# Image 细粒度图收敛 — PR-3(动态多 CLIP)Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development / executing-plans。checkbox 跟踪。

**Goal:** Load CLIP 支持用户**自己增删 CLIP 条目**(每条 = 文件 + 精度)+ 一个 `type`(架构)选择器。单编码器(flux2/qwen)端到端走通(已验);**多编码器执行 gated**(磁盘只有单编码器模型,无法真验——按 feedback_verify_real_model 不发跑不起来的路径)。实现用户早先明确要的「现在就要双、多 clip,自己添加删除」。

**Tech:** 前端 React/TS(新 clip_stack widget)+ 后端 node.yaml/executor + runner gated 文案。CI = tsc+vite+vitest 前端 + 后端 pytest。

**Branch:** `feat/image-granular-convergence-pr3`。

**Spec:** `2026-05-21-image-granular-convergence-design.md` §3.2 / §4.3 / §5.2。

**前置:** PR-1(#121)+ PR-2(#122)merged。Load CLIP 现为单 `file`+`weight_dtype`(componentRole clip)。

---

## 关键设计决策
1. **数据形态**:`data.clips = [{file, weight_dtype}, ...]` + `data.type`(架构)。bundle:`{_type:flux2_clip, type, encoders:[{kind:clip, file, dtype}, ...]}`(PR-1 已是此形,只是 encoders 现在可多条)。
2. **四态头**:Load CLIP 从节点级 `componentRole`(单文件 ComponentStatusHeader)改为 **clip_stack 每行一个状态点**(多文件,节点级单 file 头不适配)。移除 node.yaml 的 `componentRole`。
3. **多编码器 gated**:runner `_build_request` 见 `len(encoders)>1` → 抛清晰 gated 错误(替换 PR-1 的 "PR-3 才支持" 占位文案)。单编码器(任意 type)继续走通。**不**写未经真模型验证的 merge_conditionings 数学(spec §9 future 点亮)。
4. **back-compat**:exec_load_clip + 前端 widget 兜底旧单 `file` 格式(无 `clips` 时包成 `[{file, weight_dtype}]`)——保 PR-1/PR-2 期存的单 CLIP workflow 不破。

---

## File Structure
| 文件 | 动作 | 职责 |
|---|---|---|
| `frontend/src/models/nodeRegistry.ts` | Modify | `WidgetType` 加 `'clip_stack'` |
| `frontend/src/components/nodes/DeclarativeNode.tsx` | Modify | `ClipStackWidget`(行=component_select+dtype+状态点+删,底部加按钮)+ WidgetRenderer case |
| `backend/nodes/flux2-components/node.yaml` | Modify | Load CLIP:`clips`(clip_stack)+`type`(select),移除 componentRole |
| `backend/nodes/flux2-components/executor.py` | Modify | `exec_load_clip` 从 `clips` 列表产 encoders(兜底旧 `file`) |
| `backend/src/runner/runner_process.py` | Modify | `_build_request` 多编码器 gated 文案 |
| `backend/tests/` + `frontend/src/**/*.test.*` | Create/Modify | 见各 Task |

---

## Task 1: 后端 — Load CLIP clips+type bundle(多 encoder)

**Files:** node.yaml / executor.py / test_flux2_components_loaders.py(更新 clip 用例)

- [ ] **Step 1: 失败测试**(更新现有 `test_load_clip_single_encoder` + 加多条 + back-compat)
```python
@pytest.mark.asyncio
async def test_load_clip_multi_encoder():
    out = await EX["flux2_load_clip"]({"type": "flux1", "clips": [
        {"file": "/m/clipL.safe", "weight_dtype": "bfloat16"},
        {"file": "/m/t5.safe", "weight_dtype": "fp8_e4m3"},
    ]}, {})
    assert out["clip"] == {"_type": "flux2_clip", "type": "flux1", "encoders": [
        {"kind": "clip", "file": "/m/clipL.safe", "dtype": "bfloat16"},
        {"kind": "clip", "file": "/m/t5.safe", "dtype": "fp8_e4m3"},
    ]}

@pytest.mark.asyncio
async def test_load_clip_single_via_clips():
    out = await EX["flux2_load_clip"]({"clips": [{"file": "/m/c.safe", "weight_dtype": "default"}]}, {})
    assert out["clip"]["type"] == "flux2"
    assert out["clip"]["encoders"] == [{"kind": "clip", "file": "/m/c.safe", "dtype": "default"}]

@pytest.mark.asyncio
async def test_load_clip_legacy_single_file_fallback():
    # PR-1/PR-2 期存的单 file 格式仍可解析
    out = await EX["flux2_load_clip"]({"file": "/m/c.safe", "weight_dtype": "bfloat16"}, {})
    assert out["clip"]["encoders"] == [{"kind": "clip", "file": "/m/c.safe", "dtype": "bfloat16"}]
```

- [ ] **Step 2: 跑确认失败**
- [ ] **Step 3: 实现 `exec_load_clip`**
```python
async def exec_load_clip(data: dict, inputs: dict) -> dict:
    clips = data.get("clips")
    if not clips and data.get("file"):  # back-compat:PR-1/PR-2 单 file
        clips = [{"file": data["file"], "weight_dtype": data.get("weight_dtype")}]
    encoders = [
        {"kind": "clip", "file": c["file"], "dtype": c.get("weight_dtype") or _DEFAULT_DTYPE}
        for c in (clips or []) if c.get("file")
    ]
    return {"clip": {"_type": "flux2_clip", "type": data.get("type") or "flux2", "encoders": encoders}}
```
- [ ] **Step 4: node.yaml Load CLIP**
```yaml
  flux2_load_clip:
    label: "Load CLIP"
    ...   # 移除 componentRole
    widgets:
      - { name: clips, label: "CLIP", widget: clip_stack }
      - { name: type,  label: "架构", widget: select, options: [flux2, flux1, sdxl, sd3, qwen], default: flux2 }
```
- [ ] **Step 5: 跑通 + 回归** `pytest tests/test_flux2_components_loaders.py tests/test_flux2_components_sampling.py`
- [ ] **Step 6: Commit** `feat(image): PR-3 — Load CLIP clips+type bundle(多 encoder + 旧 file 兜底)`

---

## Task 2: runner _build_request — 多编码器 gated 文案

**Files:** runner_process.py / test_runner_build_request_granular.py

- [ ] **Step 1: 失败测试**(更新 `test_granular_multi_encoder_not_yet` 断言新文案 + 单编码器仍通)
```python
def test_granular_multi_encoder_gated():
    inp = _granular_inputs()
    inp["latent"]["conditioning"]["clip"]["encoders"].append({"kind":"clip","file":"/m/c2.safe","dtype":"default"})
    with pytest.raises(ValueError, match="多编码器架构.*未就绪|执行未就绪"):
        _build_request(_node(inp))
```
- [ ] **Step 2-3: 实现** —把 PR-1 的占位文案换成正式 gated:
```python
    if len(encoders) != 1:
        raise ValueError(
            f"多编码器架构 '{cond_d['clip'].get('type','?')}'({len(encoders)} 个 encoder)"
            f"执行未就绪 —— 需对应多编码器模型 backend(见 spec 2026-05-21 §9);"
            f"当前可用 flux2/qwen 单编码器")
```
- [ ] **Step 4-5: 跑通 + 回归** `pytest tests/test_runner_build_request_granular.py`
- [ ] **Step 6: Commit** `feat(image): PR-3 — runner 多编码器 CLIP 执行 gated 清晰报错`

---

## Task 3: 前端 — clip_stack widget

**Files:** nodeRegistry.ts / DeclarativeNode.tsx / 新 test

- [ ] **Step 1: 失败测试**(`ClipStackWidget.test.tsx`:渲染初始行、点「添加」加行、点删除减行、改 file/dtype 回调)。
- [ ] **Step 2: 跑确认失败**
- [ ] **Step 3: 实现**
  - `WidgetType` 加 `'clip_stack'`。
  - `ClipStackWidget`(类比 LoraStackWidget):`value: {file,weight_dtype}[]`;每行 `<ComponentSelectWidget role="clip">` + dtype `<NodeSelect>`(default/bfloat16/fp8_e4m3)+ 小状态点(`useComponentState(componentStateKey({file,device:'auto',dtype}))`)+ 删除;底部「+ 添加 CLIP」。空值兜底:`Array.isArray(value)?value:(data.file?[{file,weight_dtype}]:[])` —— 但 widget 只拿到 value,旧 file 兜底放节点数据迁移(见 Step 3b)。
  - WidgetRenderer 加 `case 'clip_stack'`。
  - **3b 旧格式迁移**:DeclarativeNode 读 widget 值时,clip_stack 若 `data.clips` 空但 `data.file` 有 → 用 `[{file:data.file, weight_dtype:data.weight_dtype}]`(或在 ClipStackWidget 内兜底)。
- [ ] **Step 4: 跑通 + `tsc -b`**
- [ ] **Step 5: Commit** `feat(image): PR-3 — clip_stack widget(增删 CLIP 行 + 每行状态点)`

---

## Task 4: 预检 + 真机验证

- [ ] **Step 1:** 后端 `pytest -q` + `ruff check src tests`;前端 `tsc -b && vitest run && npm run build`。
- [ ] **Step 2: 真机(vite/本地 backend)** —Load CLIP:见 clip_stack 一行(file+精度)+「添加 CLIP」;点添加 → 两行;**单条 flux2(Qwen3)端到端出图**(真模型);**两条配置保存正常 + Run 报清晰 gated 错误**(不崩、不静默错图)。截图留证。
- [ ] **Step 3:** 开 PR → CI 绿 → auto-merge。

---

## 收尾
- 过渡期 Family B 仍在(PR-4 删)。多编码器执行点亮见 spec §9(有 flux.1/sdxl 多编码器模型时)。
