# ASR 换代 — MOSS-Transcribe-Diarize 走 SGLang Omni 微服务,取代 Qwen3-ASR

- Date: 2026-07-20
- Status: 设计(用户已拍 serving=SGLang、常驻 systemd、完全取代 Qwen3-ASR)
  + **PR-0 spike ✅ PASS(2026-07-20 真机)**——详见 `infra/moss-asr/SPIKE.md`。
  要点:attention 必须 `torch_native`(sm_86 上 flashinfer 要 JIT、triton 数值坏);
  CUDA 工具链 pip 补装(nvcc 13.3 系,配 lib64/libcudart 软链);配置走 `--config` YAML
  的 `server_args_overrides`;verbose_json 的 speaker 在 `segments[].text` 的 `[Sxx]` 前缀
  (无独立字段,PR-2 正则抠);mem_fraction 0.5(实测 13.4GB);微服务只吃 16k WAV
  (mp3 靠 backend ffmpeg 归一化,torchcodec 缺 libav 不修)。7.7min 播客 31.5s / 53 段 / S01-S02 正确。
- Trigger: 用户装了 `MOSS-Transcribe-Diarize-0.9B`(ModelScope),问能否替代 Qwen3-ASR。

## 0. 能力验证(2026-07-20 真机已验,transformers 路径)

smoke `tests/manual/smoke_moss_asr_diarize.py`(prod venv 零安装,GPU2 3090):

- 转写正确(与 Qwen3-ASR 同样本一字不差);加载 1.3s / **1.82GB 显存**;单句 ~1s。
- **时间戳内建**:输出规范 `[start][Sxx]text[end]`(秒),段级,单调、末端≈时长。
- **说话人分离内建**:合成 [A][B][A] 正确分 S01/S02;**真 7.7 分钟双人播客** 53 段,
  S01=主持人/S02=嘉宾 全程归属正确,短插话不串。
- **热词**:prompt 尾追加 `热词提示:…`;steer 同音用字生效(做得→做的),缺席人名不幻觉。
- 模型 = 自定义架构 `MossTranscribeDiarizeForConditionalGeneration`(trust_remote_code),
  **prod vLLM 0.22 registry 不认** → 不能走既有 vLLM 直引擎路径。
- 踩坑(已记 memory):prod transformers 5.6.0.dev0 对 >100k 词表 tokenizer 误触
  `_patch_mistral_regex` 崩;transformers 路径需 monkeypatch。**SGLang 路径不受此影响**
  (自带独立 venv),但 spike 若用 transformers 兜底则要带上。

## 1. 决策(用户已拍)

| 轴 | 决策 |
|---|---|
| serving | **SGLang Omni**(MOSS 官方推荐;原生 `/v1/audio/transcriptions` + verbose_json diar 段) |
| 进程形态 | **常驻 systemd unit**(仿 aligner:独立 venv/端口/unit,web UI 不管启停;1.9GB 常驻无压力) |
| 与 Qwen3-ASR 关系 | **完全取代**:退役 Qwen3-ASR vLLM 引擎 + `nous-aligner` 对齐器微服务(MOSS 内建时间戳) |
| 升级耦合 | MOSS/SGLang 在独立 venv,**不动 backend pyproject 的 vllm 0.22/torch 2.11 钉**;两条升级轨互不牵扯 |

## 2. 拓扑

```
上传音频 → nous backend POST /v1/audio/transcriptions   (既有端点,改造代理目标)
              │ ffmpeg 归一化 16k/mono (复用 _ffmpeg_to_wav16k)
              │ auth / grant / quota (复用现有 M:N key + 秒计费)
              ▼ HTTP (NOUS_MOSS_ASR_URL, 默认 http://127.0.0.1:8003)
   nous-moss-asr.service — infra/moss-asr/ 独立 venv + systemd —
     sgl-omni serve <MODELS_ROOT>/nous/speech/MOSS-Transcribe-Diarize
       /v1/audio/transcriptions (response_format=verbose_json → diar 段)
       GPU: CUDA_VISIBLE_DEVICES=<3090 UUID> (绝不 Pro 6000 — GSP 固件崩卡,同 aligner unit 做法)
       显存: --mem-fraction-static 压小 (README 示例 0.80 是抢卡,常驻共享 3090 不可;spike 定具体值)
```

新增 `infra/moss-asr/`:`setup.sh`(uv 独立 venv 装 sglang-omni)、`README.md`、
`infra/systemd/nous-moss-asr.service`。镜像 `infra/aligner/` 全套范式。

## 3. 端点与输出(按既有模式定)

- `POST /v1/audio/transcriptions` 入参不变:`file, model, language?, context?, timestamps?`。
  - `context` → 映射为 MOSS 热词:默认提示 + `热词提示:{context}`(README 配方,已真机验)。
  - `timestamps=true` → 响应带分段(MOSS 免费自带;不再调对齐器)。
- 响应保持 OpenAI 兼容、`text` 仍首字段:
  ```json
  {"text": "...", "language": "...", "usage": {"type": "duration", "seconds": N},
   "segments": [{"start": 0.28, "end": 3.24, "speaker": "S01", "text": "..."}]}
  ```
  - `segments` 为增量字段(diar 段,秒;前端格式化 mm:ss)。旧 `words`(字级)随对齐器退役
    删除——MOSS 是段级不是字级,不伪造兼容。language:MOSS 是否回语种由 spike 确认,
    不回则 null(字段保留)。
- 计量口径不变:自算音频秒数扣配额(`_wav16k_seconds`),`record_llm_usage` 照旧。
- 降级:MOSS 服务不可达 → 503(**不是**静默降级——它已是唯一 ASR 主路)。

## 4. PR 拆分(逐一实施;Opus 子代理执行,主循环把关)

- **PR-0 — spike(闸门,先于一切)**:`infra/moss-asr/` 建独立 venv 装 sglang-omni,
  `sgl-omni serve` 本地起 MOSS,curl 真音频验:① cu130 装得起跑得起;② verbose_json 真回
  说话人+时间戳分段;③ 记录响应 JSON 具体结构 + 定 `--mem-fraction-static`/显存实测。
  **不通则止步回报**(退路 = transformers 微服务,今日已验通的路径)。
- **PR-1 — 微服务化**:setup.sh + systemd unit(UUID 钉 3090)+ README;healthz 探活。
- **PR-2 — 端点切换**:`openai_compat.py` 转写端点代理目标切 MOSS(vLLM chat 路径 →
  MOSS transcriptions 路径),解析 verbose_json → `segments`,context→热词,503 语义。
  单测(mock MOSS 响应)。
- **PR-3 — 退役旧栈**:models.yaml 删 Qwen3-ASR 条目、`_asr_chat_transcribe`/
  `_asr_align_timestamps`/aligner 调用删除、`infra/aligner/` + `nous-aligner.service` 退役
  (unit disable;目录留一个 release 周期再删)、前端 Playground 时间戳/words UI 改 segments。
- **PR-4 — 前端增强**:Playground 显示说话人分段时间轴(mm:ss 格式化,hms 换算)。

每 PR:ruff + 单测过;PR-2 起真机 e2e(建 key → curl 端点 → 计量入库)。

## 5. 验收

- [ ] PR-0:cu130 上 sgl-omni 起 MOSS,verbose_json 回真分段(播客样本,S01/S02 正确)。
- [ ] `/v1/audio/transcriptions` 带 `timestamps=true` 回 `segments[]`(speaker/start/end/text)。
- [ ] `context` 热词经 MOSS 生效(同 smoke 判据:steer 得/的)。
- [ ] Qwen3-ASR 引擎 + aligner 微服务退役,无残留调用;`nous-moss-asr.service` 开机自启。
- [ ] 计量:秒数口径与切换前一致;key/grant/quota 全链路真机 e2e 过。

## 6. 非目标 / 风险

- 不做流式 ASR;不做 UI 启停 MOSS(常驻)。
- 不动 backend 的 vllm/torch/diffusers 钉;SGLang 升级独立在 infra/moss-asr venv 内。
- **风险**:sglang-omni 在本机 cu130 的安装/兼容未验(PR-0 闸门);其 verbose_json 段结构
  未见实物(PR-0 记录);常驻显存占用需实测定值。90 分钟长音频的 max_new_tokens 上限
  需在 PR-2 按时长动态给。
