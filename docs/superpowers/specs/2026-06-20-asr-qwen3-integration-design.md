# ASR 模态接入 — Qwen3-ASR-1.7B 走 vLLM(镜像 embedding 整合)

- Date: 2026-06-20
- Status: 设计 + **PR-0 spike ✅ 已真机验通**(2026-06-20)
- Trigger: 用户要给 nous-center 加语音识别(ASR)。

## PR-0 Spike 结果(2026-06-20,真机验通)

`vllm serve Qwen3-ASR-1.7B`(钉死的 0.22)→ `POST /v1/audio/transcriptions` 真音频转写成功:
中文样本 `assets/voices/default_zh_female.wav` → `{"text":"希望你以后能够做得比我还好哟。"}`。

**核心假设全部成立**:vLLM 0.22 原生认 `Qwen3ASRForConditionalGeneration`、`/v1/audio/
transcriptions` OpenAI 兼容端点可用、中文转写正确。**无需 bump vllm**。

**关键集成约束(PR-1 必须照做,踩了一圈才通)**:
1. **`VLLM_USE_FLASHINFER_SAMPLER=0` 必设** —— 本机 CUDA 编译链对 flashinfer JIT 是坏的
   (`/usr/local/cuda` 缺 `curand.h`;混 venv `nvidia/cu13/include` 又版本冲突
   `__cudaLaunch not declared`)。生成式模型在 3090(sm_86)走 flashinfer 采样会 JIT 编译→崩;
   关掉用原生 PyTorch 采样即过。生产 LLM 没事是因为 Blackwell 用预编译 flashinfer、不触发该编译。
2. **`ninja` 要在 PATH**(在 `.venv/bin`;生产 systemd PATH 已含 linuxbrew,但 runner 起 vLLM 时要确保)。
3. **音频要 PyAV 可解码格式**:vLLM `multimodal/media/audio.py:load_audio` 用 PyAV;测试那个
   **IEEE-Float 24kHz WAV 被拒**("Invalid or unsupported audio file"),`ffmpeg -ar 16000 -ac 1
   -c:a pcm_s16le` 转标准 PCM 后即过。→ 端点要么 ffmpeg 归一化输入,要么文档声明支持格式。
4. 模型在 `LOCAL_MODELS_PATH/speech/Qwen3-ASR-1.7B`(当前 nvme2/NTFS;`weight_utils` 警告
   NTFS 不能 auto-prefetch,不影响功能)。
5. profiling 显存:`--max-num-seqs` + `--limit-mm-per-prompt '{"audio":N}'` 压低音频项数避 OOM;
   `--enforce-eager` 起得快(跳 torch.compile/cudagraph)。

对比 MiMo-V2.5-ASR(8B,中文方言强,但只有
  transformers 路径)与 Qwen3-ASR(vLLM day-0,52 语种,轻量)后,用户拍 **Qwen3-ASR-1.7B**。
  目标:像 embedding/LLM 那样做成**直接引擎 + OpenAI 式端点**,不走画布工作流。

## 1. 背景 / 现状

- nous-center 现**无 ASR 模态**。models.yaml 只有 `llm / tts / embedding / image`(+ SeedVR2 超分)。
- 最近先例 = **embedding 接入(#516,2026-06-12)**:`type: embedding` + `vllm_runner: pooling`
  → 同一个 vLLM `openai.api_server` 子进程,暴露 `/v1/embeddings`。ASR 镜像这套。
- vLLM 已钉 `>=0.22.0`(torch 2.11/Blackwell,2026-06-03 bump),晚于 Qwen3-ASR 的 1 月 day-0
  → **大概率已支持**,但其 ASR 的 `--runner`/`--task` 形态 + transcription 端点必须 spike 真验。

### 现状坐标

| 关注点 | 坐标 |
|---|---|
| vLLM 启动命令构建 | `src/services/inference/llm_vllm.py:284` `python -m vllm.entrypoints.openai.api_server …`;L290 `cmd += ["--runner", self._vllm_runner]` |
| embedding 端点 | `src/api/routes/openai_compat.py:487` `POST /v1/embeddings`(代理到 vLLM 子进程) |
| 既有 audio 路由 | `src/api/routes/audio.py` `/api/v1/audio`(仅 upload/get 文件,**无转写**) |
| 模型类目同步点 | model_scanner(depth 硬编码 2 处)+ 前端 4 处(类目/tab/检测)—— embedding 当初的坑 |
| 模型根 | `LOCAL_MODELS_PATH`(= `MODELS_ROOT/nous`,2026-06-19 收口);ASR 模型落 `audio/asr/Qwen3-ASR-1.7B` |
| vllm 看门狗 | `src/services/vllm_watchdog.py`:端口存在 ⟺ vLLM 后端(LLM/embedding);ASR 同样有端口,纳入 |

## 2. 决策(用户已拍)

- 模型:**Qwen3-ASR-1.7B**(非 MiMo;非 0.6B)。Apache-2.0。
- 形态:**直接引擎 + `/v1/audio/transcriptions`**(OpenAI 兼容),**不走工作流**。
- 引擎地基建一次,后续别的 ASR 模型(0.6B / MiMo)可增量接。

## 3. 目标设计

### 3.1 总体:复刻 embedding 的 vLLM 直引擎路径

```
models.yaml: type=asr, vllm_runner=<transcription 形态>  ← PR-0 spike 定
   ↓ ModelManager 起 vLLM 子进程(openai.api_server + --runner …)
vLLM 子进程暴露 /v1/audio/transcriptions
   ↓ 代理
nous-center POST /v1/audio/transcriptions(multipart 音频 → 转发 → 文本)
```

### 3.2 PR 拆分

**PR-0 — Spike(真机验核心假设,先于一切)**
- 下载 Qwen3-ASR-1.7B 到 `LOCAL_MODELS_PATH/audio/asr/Qwen3-ASR-1.7B`。
- standalone 起 `vllm serve`(钉死的 0.22),确认:暴露 `/v1/audio/transcriptions`、真音频转写正确、
  确定 `--runner`/`--task` 具体值 + 是否需额外依赖(ffmpeg/librosa/音频解码)。
- **不通过则止步**——决定是否要 bump vllm(撞大工程),回报用户再定。

**PR-1 — 模型注册 + 引擎接线**
- models.yaml 加 `type: asr` 条目(vllm_runner=spike 结论)。
- `llm_vllm.py`:已有 `--runner` 注入;补 ASR 任务的命令细节(若 spike 发现需 `--task transcription`
  或音频依赖)。
- model_type 类目接入:model_scanner depth 两处 + metadata service `asr` 归类。

**PR-2 — `/v1/audio/transcriptions` 端点**
- `openai_compat.py` 加 OpenAI 兼容端点:multipart `file` + `model` + `language?` →
  代理到对应 vLLM ASR 子进程(复用 embedding 的 source_type=model 选址逻辑)。
- multipart 音频处理(复用 `audio.py` 上传/`audio_io.py`);鉴权/计量走既有 M:N key + 配额。

**PR-3 — 前端 + 服务化**
- 前端加 ASR 类目/tab(对齐 embedding 当初的 4 处同步);测试页(传音频 → 出文本)。
- service detector + ServiceCategory 加 ASR(复发模式:新枚举须同步 tab + detector,#282/#467 教训)。
- 可发布成外部服务(像四个图像服务那样,经 `/v1/audio/transcriptions` 对外)。

## 4. 验收

- [ ] PR-0:真音频经 vLLM Qwen3-ASR-1.7B 转写正确(中英各一,WER 合理);确定运行形态。
- [ ] `/v1/audio/transcriptions` OpenAI 兼容:`curl -F file=@a.wav -F model=qwen3_asr` 返回文本。
- [ ] 模型可在 models.yaml 注册 + 网页常驻/启停(像 LLM)。
- [ ] vllm 看门狗/monitor 纳入 ASR 子进程,不误杀。
- [ ] ruff + 单测(端点/类目);真机 e2e(建 key → 调端点 → 计量入库)。

## 5. 非目标 / 风险

- 不走画布工作流(ASR 是音频进文本出的直接推理)。
- 不接 MiMo(本 arc 只 Qwen3-ASR;地基可复用,MiMo 8B transformers 路径另立)。
- **风险**:vllm 0.22 若对 Qwen3-ASR transcription 支持不全 → 可能要 bump vllm(大工程,撞
  torch/Blackwell 全栈),PR-0 spike 是闸门。音频解码依赖(ffmpeg)可能要补。
- 流式 ASR(realtime)不在本 arc;先离线文件转写。

## 6. 前置

- 下载 Qwen3-ASR-1.7B(Apache-2.0,HF `Qwen/Qwen3-ASR-1.7B` 或 ModelScope)到模型根。
- 确认 vLLM 0.22 的音频/transcription 依赖(ffmpeg 等)在 inference extra 内或补装。
