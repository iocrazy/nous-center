# 服务层 / API 设计 spec —— 统一类型化「预测」契约

状态:设计(2026-06-03)。用户要求「好好思考服务端/api 这块,对照开源」→ 先读源(ComfyUI 本机 /
Replicate Cog / BentoML)再出 spec。落地分多 PR,**大改,改对外契约 + 执行路径,必真机验**。
前置:[[project_output_delivery_service_layer]](输出交付归服务层 —— 本 spec 把它落实)、
[[project_positioning]](nous-center 是推理 infra,服务层是对外 API 边界)、[[feedback_read_comfyui_source]]。

## 0. 现状定性(explorer 实锤,file:line)

核心抽象其实已成形:`ServiceInstance`(`models/service_instance.py:18`)= 发布的工作流/模型快照 +
`exposed_inputs/outputs` I/O 元数据;M:N API Key(`InstanceApiKey` + `ApiKeyGrant` + `ResourcePack`
配额 + `AlertRule`,`models/api_gateway.py`);三套兼容层(OpenAI/Anthropic/Ollama compat)。骨架对。

**三个结构性裂缝:**
1. **工作流无法从兼容层调** —— `openai_compat.py:191` 对 `source_type=workflow` 直接 `501`,只有
   `source_type=model`(vLLM 代理)能走。工作流只能走 `instance_service.py:104 /v1/instances/{id}/run`
   (老式 1:1 key、异步 202+task_id)。**M:N key 调不了工作流**(没有匹配端点)。
2. **两套调用范式没收敛**:模型=同步代理(OpenAI 形),工作流=异步任务(ExecutionTask + task_id)。
   没有统一的「跑这个服务」入口;auth/quota 校验在 legacy 与 M:N 路径分散重复(M:N 跳过了 status /
   rate-limit 检查,各 compat 各补,`ollama_compat.py:74` 是后补的)。
3. **I/O 无类型透传** —— `exposed_inputs/outputs`(`models/schemas.py:393 ExposedParam`)只是元数据
   (node_id + 字段名),发布时不生成 per-service schema,执行时不校验,响应是无类型 object。
   **这正是 [[project_output_delivery_service_layer]] 那个坑的根:输出(图 URL/TTL/格式)没有服务层契约。**

## 1. 参考源精读结论(读源,不是凭印象)

### ComfyUI(本机 /home/heygo/sites/ComfyUI,server.py)—— 工作流调用契约
- `POST /prompt`{prompt 图, client_id} → **同步返回 `prompt_id`**,执行全异步。发布前 `execution.py:1098`
  校验图(class_type 存在 / 连线类型兼容 / 至少一个 OUTPUT_NODE),错误 `node_errors` 按 node_id。
- `ws /ws?clientId=` 推:`executing` / `progress_state`{nodes:{id:{value,max,state}}} / `executed`
  {node, output:{images:[{filename,subfolder,type}]}} / `execution_error` / `execution_success`。
- `GET /history/{prompt_id}` → {outputs:{node_id:{images:[{filename,subfolder,type}]}}, status}。
- `GET /view?filename=&subfolder=&type=` 取产物文件。
- `GET /object_info` → 每个 node 的 input(required/optional + 类型)/ output 类型 —— **节点即 schema 源**。
- **教训**:工作流的进度/取产物契约直接对齐它(我们工作流就是 comfy 形,还能让上游 comfy 生态接);
  我们 node.yaml 的 widget 类型 = comfy object_info,是 per-service schema 的现成来源。

### Replicate Cog(v0.9.4,python/cog/{predictor,server/http,schema}.py)—— 类型化 I/O + 异步 + 交付【主轴】
- **声明即 schema**:`predict(self, prompt: str = Input(description=..., ge=, le=, choices=))→Path`,
  `Input()` 就是 `pydantic.Field` 换默认位;`get_input_type` 用 inspect.signature → `create_model("Input")`
  → **自动出 `/openapi.json`**(`components.schemas.Input/Output`),`x-order` 保声明顺序,`choices`→enum。
- **一个端点,头切执行模式**(最该抄):`POST /predictions`
  - 默认同步:阻塞返结果 `200` + 终态。
  - `Prefer: respond-async` → `202` + `status:starting`,结果走 webhook。
  - `Prefer: wait=N` → 阻塞至多 N 秒,超时返未完成对象让客户端转轮询(Replicate 网关层做,非 Cog OSS)。
  - `Accept: text/event-stream` → SSE(start/output/log/metric/completed)。
- **Prediction 资源对象**(同步响应 / webhook payload / 轮询 都用它):
  `{id, status(starting|processing|succeeded|failed|canceled), input, output, error, logs, metrics{predict_time}, created_at, started_at, completed_at}`。
- **webhook** = 完整 Prediction 对象 + `webhook_events_filter:[start,output,logs,completed]`,高频事件 500ms 节流。
- **文件产物**:默认 inline data-URI;设了 `output_file_prefix`/`--upload-url` 则 PUT 上传换成 URL;
  **异步必须上传模式**(webhook 塞不下大文件)。**这就是输出交付该落的地方**。
- `GET /health-check`(STARTING/READY/BUSY/SETUP_FAILED…)。

### BentoML 1.3(@bentoml.service / @bentoml.api / @bentoml.task)—— 印证异步任务形 + runner 范式
- I/O 纯靠方法类型注解(+ pydantic / `pathlib.Path` / `PIL.Image`)→ 自动 OpenAPI + Swagger。
- **`@bentoml.task` 长任务**:自动出 `/submit`(POST,返 task id)`/status` `/get` `/cancel` `/retry`,
  配 `external_queue:True` 把排不上的请求缓冲不拒绝。**= 我们工作流的 submit→status→get,验证了形。**
- runner 范式:1.2 起删了独立 Runner,模型运行时 = 一个独立 `@service`,靠 `bentoml.depends()` RPC。
  **对比**:我们的 `RunnerSupervisor` 集中管 GPU 子进程,边界比 BentoML 的 depends RPC 更强 —— 这块**保留**,
  本 spec 不动 runner 层,只动其上的 API/服务契约。

## 2. 目标设计 —— 统一「预测(prediction)」契约

**一句话**:把分裂的调用层收敛成一个**类型化、可同步可异步的「跑服务」契约**,model/workflow/app 同一形;
OpenAI/Ollama/Anthropic 退化成上面薄薄的协议 shim(只为 LLM/chat 客户端);输出交付落在这层。

### 2.1 类型化 per-service schema(对齐 Cog `Input/Output` + ComfyUI object_info)
- 发布时(`workflow_publish.py`)从 `exposed_inputs/outputs` + 各 node 的 node.yaml widget 定义,**生成
  per-service JSON-Schema**(input schema + output schema),存进 `ServiceInstance`(新字段 `io_schema`)。
- 暴露 `GET /v1/services/{name}/openapi.json`(或 `/schema`)—— 机器可发现、可生成客户端/表单。
- 调用时按 input schema **校验 + 强类型**(补当前执行期不校验的洞,explorer 观察点)。
- 复用现有 node.yaml widget 类型(select/slider/string/seed…)→ schema 的 type/enum/min/max。

### 2.2 一个调用端点,头切模式(对齐 Cog `Prefer`)
- `POST /v1/services/{name}/predictions`(body = input 对象,按 2.1 schema):
  - 默认**同步**:阻塞返终态 Prediction(LLM/快图,秒级)。
  - `Prefer: respond-async` → `202` + `status:processing` + prediction id,结果走 webhook / 轮询。
  - `Prefer: wait=N` → 阻塞至多 N 秒,超时返未完成对象转轮询(网关层实现)。
  - `Accept: text/event-stream` → SSE 进度流(对齐 ComfyUI ws 事件)。
- model/workflow/app 三种 source_type **全走这一个端点**,内部按 source_type 分派(model→vLLM 同步;
  workflow→executor 异步;app→…)。**修掉 openai_compat.py:191 的 501**。

### 2.3 Prediction 资源(对齐 Cog PredictionResponse + ComfyUI history,泛化现有 ExecutionTask)
- `GET /v1/predictions/{id}` 轮询;`POST /v1/predictions/{id}/cancel`。
- 对象:`{id, service, status(queued|processing|succeeded|failed|canceled), input, output, error,
  logs, metrics{predict_time,...}, created_at, started_at, completed_at}`。
- 把现有 `ExecutionTask`(`instance_service.py` 用的)**泛化成 Prediction**(统一 model/workflow 都产出它)。

### 2.4 进度流(对齐 ComfyUI ws + Cog SSE)
- 每个 prediction 一条 SSE/ws:`processing` / `progress`(节点级,复用工作流 UI 已有的 node 进度事件,
  见 [[project_workflow_ui_bugs]] 的进度通道)/ `output` / `completed` / `error`。
- webhook(对齐 Cog):`webhook` URL + `webhook_events_filter`,payload = 完整 Prediction 对象,高频 500ms 节流。

### 2.5 输出交付契约(落实 [[project_output_delivery_service_layer]],对齐 Cog 文件产物)
- prediction.output 里的图像/文件:**小 → inline data-URI;大 / 异步 → 签名 URL + TTL**。
- 复用现有 `image_output_storage`(已有签名 URL + TTL + reap orphans)—— 这就是交付实现,**从 node widget
  上移到服务层输出契约**(节点只产 latent/图,持久化/签名/TTL/格式由服务层按 prediction 决定)。
- 撤掉节点里的「URL 有效期」widget(用户早质疑过那不该是节点的事)。

### 2.6 兼容层退化成 shim
- OpenAI `/v1/chat/completions`、Ollama、Anthropic = 翻译层:解析各自协议 → 调 2.2 的统一路径 → 翻译回。
- auth/quota **收敛到一处**(统一 `resolve_target_service` + 配额消费),compat 不再各自校验(修 explorer 指出的分散漏)。

## 3. 接入点(map 到现状)

| 现状 | 改成 |
|---|---|
| `ServiceInstance.exposed_inputs/outputs`(元数据) | + `io_schema`(发布生成的 JSON-Schema),`/services/{name}/openapi.json` |
| `ExecutionTask` | 泛化 `Prediction`(model/workflow 统一产出) |
| `/v1/instances/{id}/run`(202+task_id,1:1) | `/v1/services/{name}/predictions`(同步/异步头切,M:N) |
| `openai_compat.py:191` workflow→501 | 走统一 prediction 路径(source_type 分派) |
| 各 compat 分散 auth/quota | 收敛到 `resolve_target_service` + 统一配额消费 |
| `image_output_storage` 签名 URL | prediction.output 的交付实现(服务层契约) |
| 节点「URL 有效期」widget | 删,移到服务层输出契约 |
| `RunnerSupervisor` / runner 层 | **不动**(边界比 BentoML depends 更强,保留) |

## 4. PR 拆分(用户拍板:**不要向后兼容,一步到位** —— 每 PR 直接替换/删 legacy,不并存过渡层)

单 admin、开发期、无需保护的外部调用方 → clean cut。每 PR 仍是独立逻辑改动(走 CI/review),但**用新的
替掉旧的、顺手删 legacy**,不留 deprecated 并存。

- **PR-1 类型化 schema**:发布生成 per-service JSON-Schema + `GET /services/{name}/openapi.json` + 调用期校验。
  (这块本就纯加,作目标态的基座。)
- **PR-2 统一 Prediction + 端点 + 删旧调用路径**:`POST /services/{name}/predictions`(Prefer 头切)+
  `GET /predictions/{id}` + cancel;泛化 ExecutionTask→Prediction;**修 workflow 501**;
  **同时删** `/v1/instances/{id}/run`、`/v1/instances/{id}/synthesize` 老端点(功能并入统一路径)。
- **PR-3 进度流 + webhook**:SSE/ws per prediction(复用 node 进度事件)+ webhook(Cog 形)。
- **PR-4 输出交付契约**:prediction.output 图→data-URI / 签名 URL+TTL(复用 image_output_storage);
  **删节点「URL 有效期」widget**(交付上移服务层)。
- **PR-5 compat shim + auth/quota 收敛 + 删 legacy key 路径**:OpenAI/Ollama/Anthropic 翻译到统一路径;
  **删 legacy 1:1 `InstanceApiKey.instance_id` 路径 + `verify_instance_key`**,auth 只剩 M:N 一条;配额校验收一处。

## 5. 风险 / 坑
- **大改对外契约 + 执行路径** —— 必须真机端到端验(发布→拿 schema→同步调→异步调+轮询→进度→取图)。
- **不要向后兼容(用户拍板)**:直接删 legacy(`/instances/run`·`synthesize`·1:1 key·`verify_instance_key`·
  节点 URL widget)。**唯一前置检查**:确认 mediahub 等上游([[project_positioning]] 提的上层应用)**没在调**
  现有 OpenAI-compat / `/instances/run` —— 若在调,clean cut 时同步改上游(不留兼容)。开发期大概率没真消费者。
- M:N auth/quota 收敛别开新洞(explorer 指出现状分散校验,收敛是机会也是风险点)。
- 输出交付的签名 URL/TTL 安全(path traversal / 越权),复用 image_output_storage 已有防护。
- DB:`ExecutionTask`→`Prediction` 泛化 + 删 legacy key 字段,是 schema 迁移(开发期可直接重建,无需保数据迁移脚本)。
- 不动 runner 层(GPU 子进程边界保留),本 spec 只动 API/服务契约层。

参见 [[project_output_delivery_service_layer]]、[[project_unified_model_mgmt_gap]]、[[project_positioning]]、
[[feedback_read_comfyui_source]]、[[feedback_long_term_robustness]]、[[feedback_push_before_impl]]。
