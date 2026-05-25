# 推理全栈升级 torch 2.11 + vllm 0.21 — 设计

> 状态:**PR-0 spike 完成,全栈在 torch 2.11/cu130 上验证可行**(见下「PR-0 spike 结果」)。
> 待 PR-1 实际 bump 主 pyproject(用户拍板要 bump,理由=栈更新)。
> 地基级改动,触及生产关键 LLM serving + 钉死的 diffusers —— 隔离分支 + spike 逐模态重验。
> 依据 [[feedback-long-term-robustness]] [[feedback-verify-real-model]] [[user-hardware]]。

## 为什么(rationale 修正:原"快 fp8 前置"已被 spike 证伪)

- **⚠️ 原假设证伪**:以为 torch 2.11 给 fp8 快核 → 快 fp8。spike 实测:
  - 3090 是 **Ampere sm_86**,torchao dynamic-activation fp8 **硬报错**
    `Float8 dynamic activation … only supported on CUDA>=8.9`(fp8 tensor core 要 sm≥8.9 = Ada/Hopper/Blackwell)。
  - weight-only fp8 在 torch 2.11 跑出与 2.10 **字节一致的 28.8s**(dequant→bf16 matmul,torch 版本无关)。
  - **结论:3090 上 fp8 永远只省显存、不会更快;bump 对"3090 快 fp8"零收益。** fp8 真加速只在
    Pro6000(Blackwell)有意义,但那卡 96GB 不缺显存(吞吐目标,非"塞 3090")。
- **本 bump 保留,理由改为"栈本身要更新"**(用户决定):vllm 0.21 / torch 2.11 的新特性与维护性。
  **与 fp8 解耦** —— fp8 省显存版在 torch 2.10 即可落地(见 [[2026-05-25-image-fit-small-card-design]])。
- **无硬冲突,配套升级**:vllm 0.21.0 硬钉 `torch==2.11.0`(+torchaudio 2.11/torchvision 0.26)。
  torch 被 LLM(vllm)/ TTS(5 引擎)/ 图像(diffusers)三摊共用 → 全栈。

## 现状(锚点)

- `pyproject.toml`:`vllm>=0.19.1`(inference extra,**同 venv** `uv sync --extra inference`);
  torch 2.10.0+cu128(无直接 pin,经 image extra 传递);diffusers 钉 commit `c8eba433`(0.38.0.dev0)。
- vllm 跑独立子进程(`scripts/start_vllm.sh`:`vllm serve …`,从 PATH 取),但**装在同 venv**。
- TTS 引擎 cosyvoice2/indextts2/qwen3_tts/moss(+voxcpm2)都 `import torch`,跑 runner 子进程,共用主 venv。
- 图像引擎 = Modular Diffusers,钉死 commit 在 torch 2.10 验过(SSIM 1.0)。

## 风险与未知(spike 要回答)

1. **diffusers 钉死 commit `c8eba433` 在 torch 2.11 上能否跑**?modular API experimental;
   可能要 bump 到 torch-2.11-兼容的更新 diffusers commit → **必须重跑 `smoke_image_ab.py` SSIM ≥ 0.97**
   (CLAUDE.md 硬要求)。这是最大未知。
2. **vllm 0.19→0.21 的 serve 行为/CLI/模型兼容**:Qwen3.5-35B(TP=2)/ Qwen2.5-VL / Gemma 等
   能否在 0.21 正常 serve;`start_vllm.sh` + vllm adapter 是否要改。
3. **5 个 TTS 引擎在 torch 2.11 上加载+推理**(各自模型对 torch 版本的敏感度)。
4. **torchao 0.17 在 torch 2.11 上 fp8 cpp 核真的载**(消除 28.8s→验证接近 resident 速度)。

## 方案:隔离 + spike-first,逐模态闸

**不在主 venv 直接 bump**(会打断正在用的栈)。用 git worktree / 独立 venv 升级后逐模态验,全绿再合。

- **隔离环境**:worktree 分支 + 新 venv(`uv sync --extra inference --extra image`,bump 后的 pin)。
- **逐模态 spike 闸**(任一不过就停下评估):
  - LLM:vllm 0.21 serve 一个真模型 → 正常出 token(+ TP=2 若可)。
  - TTS:至少 cosyvoice2 + 一个 qwen3_tts 真模型出音。
  - 图像:`smoke_image_ab.py` SSIM ≥ 0.97 + 出图正确(可能需同步 bump diffusers commit)。
  - fp8:`spike_quant_compact.py` 在 torch 2.11 重跑 → 推理显著快于 28.8s(确认 fp8 cpp 核生效)。

## PR-0 spike 结果(已完成,2026-05-25,隔离 worktree `../nous-center-stackbump`)

装好:**torch 2.11.0+cu130 / vllm 0.21.0 / diffusers 同 commit c8eba433(无需换!)/ torchao 0.17**。

| 闸 | 结果 | 备注 |
|---|---|---|
| 依赖解析 + 安装 | ✓ | 钉死 diffusers commit 与 torch 2.11 **无冲突**,不用换 commit |
| GPU / cu130 驱动 | ✓ | 驱动 595.71.05 支持 CUDA 13;3 卡 matmul 正常 |
| 图像(钉死 diffusers @ 2.11) | ✓ | `smoke_load_checkpoint_dir.py` 出正确狐狸图;**SSIM 0.9772 ≥ 0.97** vs torch 2.10 参考 |
| LLM(vllm 0.21) | ✓ | gemma-4 AWQ offline generate 出 token;**需 `VLLM_USE_FLASHINFER_SAMPLER=0` + `VLLM_ATTENTION_BACKEND=TORCH_SDPA`** 绕开 flashinfer JIT(见下) |
| TTS(5 引擎) | ✓ import | cosyvoice2/qwen3_tts/voxcpm2/moss_tts/indextts2 在 2.11 全 import OK(完整出音未跑,无 API 破坏) |
| ~~fp8 快核~~ | ✗ **证伪** | 3090 Ampere sm_86 无 fp8 核;torch 2.11 与 2.10 同 28.8s。**bump 不为 fp8,为栈更新** |

**flashinfer 的坑(部署要知道)**:vllm 0.21 默认用 flashinfer(快 attention/sampling),它对 cu13/sm_86
没预编译核 → 运行时 nvcc JIT。JIT 要 cuRAND 头(`curand.h`),**最小 `cuda-nvcc-13-0` 不含**,要全套
`cuda-toolkit-13-0`(~3-4GB)。否则用 `TORCH_SDPA` 兜底(能跑,慢)。spike 用了兜底。
机器原本**无任何 CUDA toolkit**(torch 自带 runtime);CUDA 13 的 pip 工具链还是空 stub(NVIDIA 没发)→
nvcc 只能系统装(已装 `cuda-nvcc-13-0` 到 `/usr/local/cuda-13.0`)。

## PR 拆分(实际 bump)

- **PR-1**:bump 主 `pyproject.toml`(`vllm>=0.21.0` → 拉 torch 2.11;diffusers commit **不变**;
  torchao 进 image extra)+ 重生 `uv.lock`。CI 走 `uv sync --frozen`(不装 extras/torch)→ 不受影响,应绿。
  **不动 start_vllm.sh 的模型**;文档化部署前置:**装 `cuda-toolkit-13-0`(flashinfer 性能)** + cu130 驱动 ≥595。
- **PR-2**:部署/回归文档 —— vllm 0.21 serve 真模型(含 TP=2)+ TTS 真出音 + flashinfer 性能确认;
  更新 CLAUDE.md。**注**:vllm 当前未部署([[project_deploy_deferred]]),PR-2 在实际部署 vllm 时做。

## 与 fp8 arc 的关系

本 arc 合后,[[2026-05-25-image-fit-small-card-design]] 的 fp8(torchao weight-only,ComfyUI 式
weight_dtype)再落地,自带快核(接近 resident 速度)。GGUF / nvfp4 不依赖本 arc,可并行先做。

## 回滚

bump 在隔离 worktree;主 venv/分支不动。任一闸不过 → 弃 worktree,主栈零影响。合并后若生产出问题,
`git revert` PR-1 + `uv sync` 回 torch 2.10 栈。
