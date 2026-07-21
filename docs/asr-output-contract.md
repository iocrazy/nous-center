# nous ASR 对外输出契约 v1

`POST /v1/audio/transcriptions` 的响应契约。**与底层引擎无关**:任何 ASR 引擎
(Qwen3-ASR 已退役、MOSS-Transcribe-Diarize 现役、未来任何模型)的私有输出格式一律在
后端归一化到本契约,客户端永不因换引擎而改代码。2026-07 Qwen3→MOSS 换代已实证:
`text/language/usage` 零变动。

## 响应结构

```json
{
  "text": "纯文本转写全文",
  "language": "zh" ,
  "usage": {"type": "duration", "seconds": 8},
  "segments": [
    {"start": 0.26, "end": 3.24, "speaker": "S01", "text": "该段纯文本"}
  ]
}
```

## 字段承诺(客户端可依赖的不变量)

| 字段 | 承诺 |
|---|---|
| `text` | **永远存在、永远第一字段、永远纯文本**(无时间戳/说话人 markup)。只读 `text` 的最简客户端在任何引擎下都成立。 |
| `language` | 永远存在。取值 = **引擎提供的语种,或引擎不提供时由后端文本字符集检测**(`_detect_language`,zh/ja/ko/ru/en,ISO 639-1 码);引擎与检测均无法判定才为 `null`(verbose_json 格式下为 `"und"`),**不会缺字段**。 |
| `usage` | 永远 `{"type":"duration","seconds":int≥1}`,秒 = 音频真实时长(计费口径),与引擎无关(后端自算)。 |
| `segments` | 仅当请求 `timestamps=true` 时出现。每段:`start`/`end` 秒(float,单调不减)、`speaker`(`"S01"` 式标签,引擎无说话人分离能力时为 `null`,**不会缺字段**)、`text` 纯文本。 |

## 入参契约(同样引擎无关)

`file`(multipart,任意常见音频格式,后端 ffmpeg 归一化)、`model`(服务名)、
`timestamps`(bool,要分段)、`context`(热词/领域偏置,引擎不支持时静默忽略)、
`language`(OpenAI 兼容保留位,当前不消费)、`response_format`(**已消费**,见下节)。

## 演进规则

1. **只加不改**:新能力 = 新增字段/新增可选入参;既有字段的名字、类型、语义、
   nullability 永不变。破坏性变更 = 新版本路径,不复用 v1。
2. **能力降级显式化**:引擎缺某能力 → 对应字段置 `null`(language/speaker)或不出现
   (segments),绝不改变结构或塞私有格式进 `text`。
3. **新引擎接入清单**:实现一个归一化函数(参照 `openai_compat.py` 的
   `_asr_moss_transcribe`,返回 `(text, language, segments)` 三元组)+ 对照本契约过一遍
   字段承诺 + `tests/test_asr_transcription.py` 加该引擎的 mock 用例。引擎私有协议
   (如 Qwen3 的 `<asr_text>` 标记、MOSS 的 `[Sxx]` 前缀)**只允许存在于归一化函数内部**。
4. 严格 OpenAI-Whisper `verbose_json` 兼容:经 `response_format=verbose_json`
   走**独立出参分支**映射(我们的 segments → Whisper 段形状),**不动本契约默认输出**。
   **已实现**(2026-07-21,PR-7),见下节。

## `response_format` 出参格式(§4 演进钩子的实现)

`response_format` 是入参,选 `POST /v1/audio/transcriptions` 的出参形状。默认输出(上文
「响应结构」)一字不动;`verbose_json` 是**另一条分支**,供 OpenAI-Whisper 生态 SDK
(openai-python 等)直连。未知取值 → 400。

| `response_format` | 出参 | 分段 | 用途 |
|---|---|---|---|
| 缺省 / `json` | 本契约默认:`{text, language, usage}`(+ `timestamps=true` 时 `segments`) | 由 `timestamps` 门控 | nous 客户端;段带 `speaker`(平台增强) |
| `text` | 纯文本 body(`text/plain`),无 JSON 包裹、无 `language`/`usage`/`segments` | 无 | OpenAI `text` 语义,只要转写全文 |
| `verbose_json` | OpenAI-Whisper 段形状(见下) | **隐含**(始终带 `segments`,无需另传 `timestamps`) | OpenAI SDK 直连、Whisper 生态工具 |

`verbose_json` 形状:

```json
{
  "task": "transcribe",
  "language": "zh",
  "duration": 8.0,
  "text": "纯文本全文",
  "segments": [
    {"id": 0, "seek": 0, "start": 0.26, "end": 3.24, "text": "段纯文本",
     "tokens": [], "temperature": 0.0, "avg_logprob": 0.0,
     "compression_ratio": 1.0, "no_speech_prob": 0.0, "speaker": "S01"}
  ]
}
```

- `language`:引擎或后端检测得出;均无 → `"und"`(OpenAI 语义:未定语种;不为 null)。
- `duration`:float 秒 = 后端自算的 `audio_seconds`(与 `usage.seconds` 同一口径;归一化
  wav 算出,不依赖引擎透传时长)。
- `segments[].{seek,tokens,temperature,avg_logprob,compression_ratio,no_speech_prob}`:
  **中性占位** —— MOSS 不产这些 token 级/逐段置信度指标,给 OpenAI SDK 惯例默认值(字段
  在、值中性),SDK 不因缺字段报错。
- `segments[].speaker`:nous **附加字段**(OpenAI schema 无此键),SDK 忽略未知字段;无
  说话人归属的段为 `null`。

## 历史

- 2026-06-21:v1 雏形(Qwen3-ASR:text/language/usage + words 字级时间戳)。
- 2026-07-20:`words`(字级,对齐器)→ `segments`(段级+speaker,MOSS 内建)。
  这是唯一一次破坏性变更,随对齐器退役发生;此后按演进规则冻结。
- 2026-07-21:本契约文档化,spec `2026-07-20-moss-asr-sglang-serving-design.md` §3 为实现出处。
- 2026-07-21(PR-7):§4 演进钩子落地 —— `response_format` 开始消费(`json`/`text`/`verbose_json`,
  未知值 400);新增 OpenAI-Whisper `verbose_json` 出参分支(独立映射,默认输出不变);
  `language` 语义扩展为「引擎提供或后端文本字符集检测,均无才 null / und」。纯新增,默认契约冻结不动。
