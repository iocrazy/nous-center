# ASR 增强 — context 偏置 + 语种识别 + 时间戳

- Date: 2026-06-21
- Status: 设计
- Trigger: 用户:Qwen3-ASR 现在只返回纯文本,问「不是有时间戳吗?还能返回什么?」。调研结论 =
  开源 Qwen3-ASR-1.7B 原生支持**语种识别(LID)** + **context 偏置**(system 提示注入热词/领域词);
  **时间戳**要配单独的 `Qwen3-ForcedAligner-0.6B`(强制对齐,词/字级)+ `qwen-asr` 工具链,
  vLLM 的 OpenAI 转写端点做不到。用户:两个都要。

## 0. 调研事实(真机已验)

- 转写端点 `/v1/audio/transcriptions`:只回 `{text, usage.seconds}`;`verbose_json`/`srt`/`vtt`
  **被拒**(`Currently do not support verbose_json for Qwen3-ASR`)→ 拿不到时间戳。
- chat 路径 `/v1/chat/completions`(audio_url + 可选 system):原始输出稳定为
  **`language {LANG}<asr_text>{TEXT}`**(实测多次一致)→ 同时给**语种** + 文本;system 消息 =
  **context 偏置**槽(chat_template 确认有 system_text 槽)。
- 时间戳:`Qwen3-ForcedAligner-0.6B`(已下,1.8G,arch 同 qwen3_asr)`align(audio, text, language)`
  → 每词/字 `.text/.start_time/.end_time`。**两步走**:先 ASR 出文本,再对齐器吃 (音频+文本) 出时间。
  只能经 `qwen-asr` 工具包,不能经 vLLM OpenAI 端点。

## 1. 分两条 arc

### Arc A — context 偏置 + 语种识别(不加模型,改 serving 路径)

后端 `audio_transcriptions`(`openai_compat.py`)把转发从转写端点 → **chat/completions**:
- 入参加可选 `context`(form 字段)→ 作为 `system` 消息注入(热词/领域/人名偏置)。
- 音频归一化后 base64 成 `data:audio/wav;base64,...` 放进 `audio_url`。
- 解析返回:`language (.+?)<asr_text>(.*)` → `{text, language}`;无 `<asr_text>` 标记则
  整体当 text、language=None(防御式)。
- **metering**:chat 回的是 token usage 不是音频秒数 → 自算时长(归一化 wav = 16k/mono/s16le,
  `seconds = len(pcm_bytes) / (16000*2)`),沿用 `record_llm_usage` 的 audio 秒计费,口径不变。
- 响应体保持 OpenAI 兼容:`{text, language, usage:{type:"duration",seconds}}`(text 仍是首字段,
  外部纯文本客户端不受影响;language 是增量字段)。

前端 Playground(`ServiceDetail.tsx` isAsr 分支):
- 加「领域提示 / 热词」可选输入框 → 随 multipart 带 `context`。
- 输出区显示**检测到的语种** badge + 文本。

### Arc B — 时间戳(加 ForcedAligner + qwen-asr,spike 先行)

**PR-0 spike ✅ 已完成(2026-06-21 真机)**,独立 venv 装 `qwen-asr`,GPU0 空闲 3090 实测:

| 测法 | 加载 | 显存 | 推理 | 时间戳 |
|---|---|---|---|---|
| 方案① 全套(ASR 1.7B + 对齐器 0.6B 一把抓) | 2.1s | 5.5GB | 8.5s | 14 字级,正确 |
| **方案② 独立对齐器**(喂现成文本 align) | 0.9s | **1.72GB** | **0.85s** | 14 字级,正确(`希`[0.320→0.480]…末端3.04s 对齐) |

**决策:采方案②。** 独立 `Qwen3ForcedAligner.from_pretrained()` 只 1.72GB / 0.85s,喂
(音频, vLLM 已出文本, 语言)→ 字/词级时间戳。复用现有 vLLM ASR 出文本,**不加第二份 ASR、
不动快文本主路**,时间戳纯增量后处理。

集成方式(方案②):
- 注册 `qwen3_forced_aligner`(0.6B)为新模型/引擎类目(对齐器,非 vLLM)。
- backend 需能调 `qwen-asr` 的 `Qwen3ForcedAligner`:在 prod venv 装 `qwen-asr`(spike 用独立
  venv 装通;需验证与现有 vllm/transformers 版本不冲突 —— B-1 先验)。冲突则退化为独立 aligner
  子进程/runner(自带 venv,IPC 喂 audio+text 拿 timestamps)。
- 转写端点加 `timestamps=true`:后端先 vLLM 出文本(含 LID),再对齐器 align → 返回
  `{text, language, words:[{text,start,end}]}`。对齐器按需加载、可驱逐(小,1.7GB)。

时间戳走 chat/转写端点都接不了 → 时间戳模式必然是 nous 自己的一条 API 形态(例如转写端点加
`timestamps=true` → 后端触发 ASR + 对齐两步,返回 `{text, language, words:[{text,start,end}]}`)。
前端:Playground 加「时间戳」开关 + 分段时间轴展示。

## 2. PR 拆分(每个独立分支/PR,走 CI)

- **A-1**(后端):转写端点切 chat 路径 + context + LID + 自算时长 metering。
- **A-2**(前端):Playground context 框 + 语种显示。
- **B-0**(spike):qwen-asr + ForcedAligner 真机验证 + 集成方式决策(产出更新本 spec)。
- **B-1+**(实施):按 spike 结论拆(模型注册 / 对齐器加载 / 转写端点 timestamps 模式 / 前端时间轴)。

## 3. 验收

- [ ] A:Playground 传音频得到 文本 + 语种;填 context(人名/术语)后该词识别更准(真机对比)。
- [ ] A:外部 API 纯文本客户端不受影响(text 仍首字段,200 OK)。
- [ ] A:用量记账秒数与切换前一致(自算时长 == 音频真实时长 ±0.1s)。
- [ ] B-0:真机出正确词/字时间戳(start/end 单调、末 end ≈ 音频时长),显存/延迟量化记录。
- [ ] B:时间戳模式端到端(音频 → 文本 + 词级时间戳),Playground 时间轴可视;ruff+tsc+test 绿。
