# 推理全栈升级 torch 2.11 + vllm 0.21 — 设计

> 状态:**PR-0 spike 进行中**(隔离 worktree `../nous-center-stackbump` / 分支 spike/stack-bump-torch211;
> 装好 torch 2.11+cu130 / vllm 0.21 / diffusers 同 commit / torchao,GPU 已验)。
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

## PR 拆分(plan 细化,spike 闸全绿后)

- **PR-0 spike**:隔离环境装 torch 2.11 + vllm 0.21(+ 可能新 diffusers commit),逐模态验(上面 4 闸)。
  产出:确定的 pin 组合 + 各模态验证结论 + diffusers commit 是否要换 + SSIM 数。
- **PR-1**:bump `pyproject.toml`(vllm 0.21 / torch 2.11 间接 / diffusers commit / torchaudio·torchvision),
  改 `start_vllm.sh` + vllm adapter(若 0.21 有 CLI/API 变)+ uv.lock。CI 绿(CI 不跑真模型)。
- **PR-2**:真模型回归套(LLM/TTS/图像 smoke + SSIM)记录到 docs;更新 CLAUDE.md「图像引擎」的 commit。

## 与 fp8 arc 的关系

本 arc 合后,[[2026-05-25-image-fit-small-card-design]] 的 fp8(torchao weight-only,ComfyUI 式
weight_dtype)再落地,自带快核(接近 resident 速度)。GGUF / nvfp4 不依赖本 arc,可并行先做。

## 回滚

bump 在隔离 worktree;主 venv/分支不动。任一闸不过 → 弃 worktree,主栈零影响。合并后若生产出问题,
`git revert` PR-1 + `uv sync` 回 torch 2.10 栈。
