# legacy 1:1 子系统全删 计划 spec(服务层 API PR-5 展开)

状态:计划(2026-06-03)。用户拍:**全删 legacy 1:1 子系统**(继 [[project_service_api_layer]] 的统一
prediction 契约 PR-1~4 之后)。查实这不是 spec #319 PR-5 那行「删 legacy key」—— 是 **auth + files/responses
数据作用域 + instance/preset/synthesize 子系统**的多 PR 大重构。本 spec 把它有序拆开 + 给 files/responses
重作用域设计。**改 auth 核心 blast radius 大、改作用域会丢文件归属,每 PR 必真机验 + 用户在环。**

## 0. 前置事实(2026-06-03 查 DB + 代码)
- **0 个 legacy 1:1 key**(`instance_api_keys.instance_id` 全 NULL),**0 个 preset 服务**(source_type 只 model+workflow),
  12 个全 M:N key → **整个 legacy 1:1 子系统空转、无消费者**。clean cut 安全(无真东西可破)。
- mediahub 等上游只在规划文档提 API,无实际调用。

## 1. legacy 1:1 子系统清单(要删/迁的)
| 类别 | 文件/符号 | 现状 | 目标 |
|---|---|---|---|
| 路由-instance CRUD | `routes/instances.py`(`/api/v1/instances` 增删改查) | legacy 1:1 instance(= preset 实例) | 删(M:N 用 `services.py`) |
| 路由-1:1 key CRUD | `routes/instance_keys.py`(`/instances/{id}/keys`) | legacy 1:1 key | 删(M:N 用 `api_keys.py` + grants) |
| 路由-TTS 合成 | `routes/instance_service.py`(`/synthesize`;`/run` 已删 PR-2) | legacy preset TTS 同步 | 删 |
| auth-legacy | `deps_auth.verify_bearer_token`(files/audio/responses 用) | 只认 legacy key(M:N 实际 403) | 迁 → `verify_bearer_token_any` 后删 |
| auth-legacy | `deps_auth.verify_instance_key`(只 /synthesize 用) | legacy 1:1 URL auth | 删 |
| auth-混合 | `verify_bearer_token_any` 里的 legacy 分支(`key.instance_id` 非空) | 死分支(0 legacy key) | 删 → M:N only |
| 数据作用域 | `files.py` / `responses.py` 按 `instance_id` 切(`FileRow.instance_id`) | legacy 1:1 绑定 | **重作用域 → 按 `api_key_id`** |
| preset 模型 | `VoicePreset` + `/api/v1/voices` + `source_type=preset` | legacy TTS preset | 评估删(TTS 该走 workflow) |

## 2. files/responses 重作用域设计(关键,数据模型改)
现状:`FileRow.instance_id` 把上传文件绑到一个 instance(已发布服务)。M:N 一个 key 可访问多服务,**没有单一
instance** → 作用域失配。
**目标**:文件/responses 作用域**绑到调用方 API key**(谁传的谁拥有),不绑服务。
- `FileRow.instance_id` → `FileRow.api_key_id`(谁上传);dedup 键 `(api_key_id, sha256)`。
- `responses` 同理按 `api_key_id` 切。
- auth 改 `verify_bearer_token_any` → 拿 `key`(M:N 不解析 instance);作用域用 `key.id`。
- 幂等 ALTER 加 `api_key_id` 列;无 legacy 数据无需迁移(空转)。

## 3. 有序 PR 拆分(低耦合先、auth 核心最后)
- **PR-5a** `/v1/audio/speech`(openai_compat:427)→ M:N:改 `verify_bearer_token_any`(handler 用 engine
  非 instance,最干净)。真机验:带 M:N key 调 speech 出音。
- **PR-5b** files/responses **重作用域**:`FileRow.api_key_id`(+responses)+ auth 改 `verify_bearer_token_any` +
  幂等 ALTER。**最该谨慎**(丢文件归属)。真机验:M:N key 传文件→引用→预测取到。
- **PR-5c** 删 `verify_bearer_token`(5a+5b 后无用)+ `verify_bearer_token_any` 简化成 **M:N only**(删 legacy 分支)。
  真机验:所有 compat(openai/ollama/anthropic)+ predictions 带 M:N key 都没坏。
- **PR-5d** 删 legacy instance/preset/synthesize 子系统:`instance_service.py`(/synthesize)+ `verify_instance_key` +
  评估删 `instances.py`/`instance_keys.py`(确认 M:N services/keys 全覆盖)+ **删 `test_instance_keys.py`**(整个测
  legacy preset/synthesize/1:1)+ 修 `test_event_types`(去 instance_service)`test_prediction_service_pr2`(去读 instance_service)。
- **PR-5e**(评估)preset/VoicePreset/`/voices` 删 or 留:TTS 该走 workflow 服务;若用户不再要 preset 概念则删。

## 4. 每 PR 验证铁律
- CI 绿(无 legacy 引用残留;改 auth 后 compat 测全过)。
- **真机**(ADMIN_PASSWORD 设的真后端,记得 `source .env` 否则不 spawn runner):受影响的对外端点带 **M:N key**
  实测没坏(speech 出音 / 文件传引用取 / chat completions / predictions)。
- 用户在环:auth/作用域改动合前用户确认真机过。

## 5. 风险
- auth 核心(PR-5c)blast radius = 全对外 API;必 compat 测 + 真机三协议验。
- 作用域改(PR-5b)= 数据模型;改错丢文件归属。空转期改最安全(无真数据)。
- `instances.py`/`instance_keys.py` 删前必确认没被 M:N 路径或前端复用(前端「实例」概念可能还引)。
参见 [[project_service_api_layer]]、[[project_positioning]]、[[feedback_long_term_robustness]]、[[feedback_verify_real_model]]。
