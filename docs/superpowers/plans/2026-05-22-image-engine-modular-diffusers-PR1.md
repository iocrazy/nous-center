# PR-1 — Modular Diffusers 引擎骨架(并存灰度)Plan

> REQUIRED SUB-SKILL: executing-plans。
> Spec: `docs/superpowers/specs/2026-05-22-image-engine-modular-diffusers-design.md`(plan-eng-review 通过,D1-D6 + 自查风险)。

**Goal**:在 runner 里加一条 `ModularPipeline` 出图路径,**与现有 ImageSampler 并存**(flag 选择),不删旧。固化 spike 的基线出图;落实 review 决议 D2/D5/D6 + 自查 P1。本 PR **不**碰量化桥接(PR-2)、LoRA(PR-3)、删旧(PR-4)。

**Branch**:`feat/image-modular-engine-pr1`。**前置**:收敛 5 PR + 本 spec 已 merged/push。

**约束(来自 review)**:
- **D2**:`diffusers.modular*` 的 import **只允许在 `image_modular.py` 一个文件**。新路径实现现有 image adapter 的 `.infer()` 接口(与 TTS/LLM adapter 一致),不新建抽象层。
- **D5**:钉死 diffusers 版本;CI 加 wiring 测(mock,不需 GPU);CLAUDE.md 加「改图像引擎/升 diffusers 前必跑 smoke」门。
- **D6**:standalone A/B 受控对比(同模型/seed/步数/分辨率),量纯采样 + 总耗时,查清 6.4s vs 27s 来源。
- **自查 P1**:验 ComponentsManager × 现有 ModelManager 在长活 runner 共存(连续多请求换模型不泄漏/不 OOM)。

---

## Task 1:`image_modular.py` 引擎骨架(adapter 接口 + diffusers.modular 隔离)

**Files**:`backend/src/services/inference/image_modular.py`(新)+ `backend/tests/test_image_modular_wiring.py`(新)

- [ ] **Step 1 失败测试(CI wiring,mock torch/diffusers)**:`ModularImageBackend` 实现 `.infer(ImageRequest)`。mock `ModularPipeline`/`ComponentsManager`,断言:
  - `from_pretrained(repo, components_manager=...)` 被调;
  - `load_components(torch_dtype=...)` 按 per-component dtype 调;
  - `pipe(prompt, num_inference_steps, height, width, generator(seed))` 参数从 ImageRequest 正确映射;
  - 返回 `InferenceResult(media_type="image/png", ...)`。
```python
def test_modular_backend_maps_request_to_pipe(monkeypatch):
    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [<fake PIL>]
    # monkeypatch ModularPipeline.from_pretrained → fake; ComponentsManager → fake
    be = ModularImageBackend(repo=..., components_manager=fake_cm)
    res = be.infer(ImageRequest(prompt="x", steps=7, width=512, height=512, seed=42))
    assert res.media_type == "image/png"
    fake_pipe.assert_called_once()
    kw = fake_pipe.call_args.kwargs
    assert kw["num_inference_steps"] == 7 and kw["width"] == 512
```
- [ ] **Step 2 跑确认失败**
- [ ] **Step 3 实现**:`ModularImageBackend(InferenceAdapter)` —— `diffusers.modular*` import **只在本文件**;`__init__` 持 `ComponentsManager`;`infer()` 建/取 `ModularPipeline`(Flux2Klein blocks)→ `load_components` → `pipe(...)` → PNG bytes + UsageMeter。lazy import diffusers(模块顶层不 import,避免 conftest mock torch 时 collection 崩)。
- [ ] **Step 4 跑通**:`pytest tests/test_image_modular_wiring.py -q`(CI 可跑,无 GPU)。
- [ ] **Step 5 ruff + commit** `feat(image): ModularImageBackend 骨架 + wiring 测(diffusers.modular 隔离一文件)`

## Task 2:引擎选择器(并存灰度,不删旧)

**Files**:`backend/src/services/model_manager.py`(或 runner `_node_executor`)+ `backend/tests/test_image_engine_selector.py`

- [ ] **Step 1 失败测试**:flag(env `NOUS_IMAGE_ENGINE=modular|legacy`,默认 `legacy`)选择 backend。断言 `modular`→`ModularImageBackend`、默认/`legacy`→现有 `DiffusersImageBackend`。
- [ ] **Step 2 跑确认失败**
- [ ] **Step 3 实现**:在 image adapter 取用处加选择器;两路并存,默认仍走 legacy(灰度安全)。
- [ ] **Step 4 跑通 + 回归**(选择器 + 现有 image 测试不破)
- [ ] **Step 5 ruff + commit** `feat(image): 引擎选择器 NOUS_IMAGE_ENGINE(modular/legacy 并存,默认 legacy)`

## Task 3:diffusers 版本钉死 + CLAUDE.md smoke 门(D5)

**Files**:`backend/pyproject.toml`(或 requirements)+ `CLAUDE.md`

- [ ] **Step 1**:把 diffusers 钉到当前精确安装源(`0.38.0.dev0` 是 git 快照 → 钉到具体 commit/URL,不能只写 `0.38.0.dev0`;在注释写明「Modular Diffusers experimental,升级前必跑 image A/B smoke」)。先确认当前安装来源(`uv pip show diffusers` 的 location / git ref)。
- [ ] **Step 2**:CLAUDE.md 加一节「## 图像引擎」:改 `image_modular.py`/`image_sampler.py` 或升 diffusers 前,**必须**跑 `tests/manual/smoke_image_ab.py`(Task 4)并确认 SSIM ≥ 阈值。
- [ ] **Step 3 commit** `chore(image): 钉 diffusers 版本 + CLAUDE.md 图像引擎 smoke 门(D5)`

## Task 4:standalone A/B 受控对比 smoke(D6 — 查清 6.4s vs 27s)

**Files**:`backend/tests/manual/smoke_image_ab.py`(新,真模型/GPU,非 CI)

- [ ] **Step 1**:同 REPO/seed=42/steps=20/1024²,**两引擎各跑**(legacy ImageSampler vs ModularImageBackend),分别计时:**纯采样循环**(denoise loop)+ **总耗时**(含 encode/decode)。存两图。
- [ ] **Step 2**:计算两图 SSIM(`skimage.metrics.structural_similarity`),打印。阈值 ≥ 0.97(同架构同 seed 应高度一致;低于则查差异)。
- [ ] **Step 3 跑真模型**(cuda:1 Pro 6000):记录 ① SSIM ② 两引擎纯采样 vs 总耗时 → **定位 6.4s vs 27s 来源**(真优化 / 口径不一 / ImageSampler 可去开销)。把结论写回 spec §1。
- [ ] **Step 4 commit** `test(image): A/B 受控 smoke(SSIM + 分段计时,查清 Modular vs ImageSampler 性能)`

## Task 5:ComponentsManager 长活生命周期验证(自查 P1)

**Files**:`backend/tests/manual/smoke_modular_lifecycle.py`(新,真模型/GPU,非 CI)

- [ ] **Step 1**:模拟长活 runner —— 同一个 `ComponentsManager`,**连续 N 次** infer,中间**切换模型/dtype**(如 bf16 → 另一组件配置),每次 `nvidia-smi` 记显存。
- [ ] **Step 2**:断言显存**不单调上涨**(无泄漏)、不 OOM;ComponentsManager 的缓存/eviction 跨请求行为符合预期(同模型复用、换模型释放)。
- [ ] **Step 3 跑真模型**:记录显存曲线 + 结论(共存是否安全;若有泄漏/双记账,记入 spec 风险,可能影响 PR-4 切换前提)。
- [ ] **Step 4 commit** `test(image): ComponentsManager 长活 runner 生命周期 smoke(连续多请求换模型不泄漏)`

## Task 6:PR + CI + 真机

- [ ] 后端全套 `pytest -q` + `ruff`;前端不涉及。
- [ ] PR → CI 绿(wiring 测在 CI 跑;真模型 smoke 本地跑过附结论)→ **确认全 pass(逐项,吸取 #126 教训)** → auto-merge。
- [ ] 默认仍 legacy,灰度安全;Modular 路径靠 `NOUS_IMAGE_ENGINE=modular` 开。
- [ ] PR 描述附:A/B SSIM + 性能结论、生命周期结论、3 自查风险中 P1×2 的验证结果。
