# ComfyUI 通用工作流桥 — 设计 spec

日期:2026-08-10。状态:设计已获用户批准(会话内逐节确认 + UI 设计稿 v5)。
UI 设计稿(artifact):https://claude.ai/code/artifact/abadb607-277b-4488-8466-bf0885fda480

## 1. 背景与定位

- 目标:在 nous-engine 内建立**通用 ComfyUI 工作流桥**——任意 ComfyUI 工作流
  (API 格式 JSON)可导入为 nous 服务,经 prediction API 对外提供。**不是**只接
  MiniMax H3;H3 的三个官方模板(T2V/I2V/R2V)是首批内容与验收用例。
- 分工:**Infinite-Canvas(IC)是生成 UI**(其 MiniMax H3 节点/Comfy 节点),经
  prediction API 调 nous-engine;**nous-engine 是算力服务层**(sidecar 管理、模板
  注册、任务执行、鉴权计量);**ComfyUI 是推理引擎**(独立 sidecar 进程)。
- 与 `comfyui-replicate` skill 的关系:那条 SOP 是把 ComfyUI 工作流**复刻**成 nous
  原生画布工作流;本桥是**不复刻直接跑**。互补,桥优先。
- 首版 provider 只做本地 ComfyUI sidecar;服务层按 provider 抽象设计,二期插
  MiniMax 官方 API(2K 精修 H3-Regenerate-2K 未开源,只有 API 有)/RunningHub
  云 ComfyUI 时接口零改。

## 2. 关键背景事实(设计依据)

- MiniMax H3:33B 全模态视频+立体声生成,两 checkpoint(FL2VA/Ref2VA),文本编码
  器用 Qwen3-VL-32B 隐状态。BF16 全量 123.6GB,单卡 96G(RTX Pro 6000)放不下。
- ComfyUI 2026-08-03 day-0 支持(Comfy-Org/ComfyUI #15224):4 新节点 + T2V/I2V/R2V
  模板;Comfy-Org repack 权重(bf16/INT8/剪枝 INT8=42.5G)。**本地跑 INT8 档**
  (~60-70G,96G 单卡可行),质量梯队:API 全量+2K 精修 > bf16 > INT8 > 剪枝 INT8 > INT4。
- IC 同类实现已验证的模式(参照 `Infinite-Canvas/main.py`):workflow JSON 模板 +
  `.config.json` 字段映射(node/input/type/default/min-max/options/bind_prompt/
  random_enabled)、`POST /prompt` → 轮询 `/history/{id}` → `/view` 下载、产物按扩展
  名分拣 image/video/audio/text、preview/调试节点过滤、校验错误翻译、下载 socket 超时。

## 3. 架构总览

```
IC 节点 ──POST /v1/services/{tpl}/predictions (Prefer: respond-async)──▶ nous-engine
                                                                      │
  轮询 GET /v1/predictions/{id} / webhook ◀── ExecutionTask(PG) ◀── run_workflow_task
                                                                      │
                                              comfyui_workflow 节点(InvokableNode)
                                                                      │
                                              ComfyUIProvider(external_providers 家族)
                                                                      │
                    upload media → patch JSON → POST /prompt → poll /history → 下载产物
                                                                      │
                                       ComfyUI sidecar(systemd nous-engine-comfyui, :8188)
```

组件:
- **ComfyUI sidecar**:独立安装(≥0.30.0),systemd 单元 `nous-engine-comfyui`,纳入
  `enginectl`。`CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES` 绑 Pro 6000。
  **前置条件:GSP 固件 bug 缓解脚本必须先上机**(满载崩卡拖黑 3090,见 memory)。
- **ComfyUIProvider**:进 `src/services/external_providers/` 家族,实现
  `ExternalGenRequest/Result` 契约(契约本质"驱动外部程序→产出本地文件",HTTP
  sidecar 适配;借此给家族命名/文档松绑,不再限于 CLI)。经 governor 排队。
- **`comfyui_workflow` 后端节点**(`src/services/nodes/`,InvokableNode):data=
  模板服务引用 + exposed 参数值;invoke 内走 provider,可跑数小时。
- **`video_output` 输出节点** + 视频/音频产物存储(仿 `image_output_storage.py`)。
- **模板 = 服务**:`service_instances.source_type` 新增第四类 `comfy_template`
  (现有 preset|workflow|model)。workflow JSON 快照 + exposed_params 即服务 source。

## 4. 模板注册与字段映射

- 数据:`ComfyTemplate` 持有 ComfyUI **API 格式** workflow JSON + 
  `exposed_params: [{name, type(text|textarea|int|float|enum|media), node_id,
  input_name, default, min, max, step, options, required, random(seed 类)}]`。
  与 IC `.config.json` 同形;与 nous 现有 `ExposedParam`/`build_service_io_schema`
  同一套 schema 语言,prediction input schema 由此生成,复用 node_id 漂移守护测试思路。
- 注册途径:
  1. `POST /api/v1/comfy-templates`(name + workflow JSON → 创建 comfy_template 服务);
     PUT/DELETE 同族;字段映射随服务配置保存。
  2. 目录扫描 `backend/comfy_templates/*.json`(含同 schema 的映射),启动 upsert,方便 git 管理。
- 节点 schema 读取:代理 sidecar `GET /object_info`,取节点类的输入定义(类型/枚举
  选项/min-max),值推断兜底。重新上传 JSON 时校验 node_id 映射,失效项高亮。

## 5. 执行数据流

1. 客户端(IC/试用面板)`POST /v1/services/{tpl}/predictions`,`Prefer: respond-async`
   → 202 + ExecutionTask 落库(同步模式维持现有 600s 封顶,长视频必走异步)。
2. `run_workflow_task` 执行单节点工作流 → `comfyui_workflow.invoke`:
   - governor 串行化(sidecar 显存独占);提交前 `gpu_free_probe` 查空闲显存,不足则等待;
   - media 参数上传 sidecar `/upload/image`;
   - 按 exposed_params patch workflow JSON;seed 未指定 → 随机生成并**回写到结果**;
   - `POST /prompt` → 每 2s 轮询 `GET /history/{prompt_id}`,总超时 `NOUS_COMFY_TIMEOUT`
     (默认 14400s=4h,env 可调;超时仅停止等待,ComfyUI 侧任务与 history 不受影响);
   - 产物分拣(仿 IC):扩展名→kind(image/video/audio/text/file);有正式输出时丢弃
     PreviewImage/对比节点产物,抑制调试文本节点;`/view` 下载,socket 超时 120s;
   - 产物落 nous 存储,返回 outputs(签名 URL + kind + 元信息)。
3. 终态 → ExecutionTask completed/failed → webhook(若配)/客户端轮询取回。
4. 取消:`POST /v1/predictions/{id}/cancel` → 转发 sidecar `/interrupt` + 队列删除。
5. 所有 sidecar HTTP 一律 `httpx.AsyncClient(trust_env=False)`(防 mihomo ALL_PROXY 劫持)。

## 6. 存储

仿 `image_output_storage.py` 增视频/音频产物存储:磁盘目录 + DB 记录 + 现有 files
路由签名 URL。视频抽首帧写入 ExecutionTask 现有 `output_thumbnails` 字段(画廊复用,
无 schema 变更)。保留/清理策略与 image 现状一致。

## 7. 错误处理

- ComfyUI 校验失败 JSON → 错误翻译层转人话(仿 IC `comfy_prompt_error_message`),
  进 prediction error 字段;
- sidecar 不可达:服务 health 标 degraded,新 prediction 快速失败;
- sidecar 中途崩溃/重启(history 丢 prompt_id):task failed + 原因;
- 轮询超时:task failed(注明 ComfyUI 侧可能仍在跑);
- OOM 等运行时错误:透传 ComfyUI 报错;governor 串行 + gpu_free_probe 预检为主要预防。

## 8. UI(四表面,详见 artifact v5;全部为现有表面扩展,零新页面)

1. **总入口两层**:设置 overlay 新增「ComfyUI 桥」Section(sidecar 地址/探活/队列/
   版本/超时);`/services` 列表加「ComfyUI 桥」筛选分类 + 行尾「桥」徽标 +
   「新建服务▾」下拉加「导入 ComfyUI 工作流」。
2. **ServiceDetail·编辑 = 读图选节点字段配置**:React Flow 渲染**通用节点卡**(类名
   + class_type + #id + 已暴露数徽标,边取 JSON;不认识自定义节点类型也能画)。点节点
   弹输入配置(暴露开关/显示名/min·max·step/随机);类型/枚举/范围来自 `/object_info`。
   下方字段汇总表与图双视图同编一份 exposed_params。sidecar 状态行 + 上传替换区
   (node_id 失效高亮)。保存即更新服务 schema。
3. **Playground·运行**:`SchemaDrivenForm`/`SchemaDrivenOutput` 现成(已支持
   image/video 字段与 video 渲染)。**唯一新增:异步运行态**——视频服务走
   respond-async,输出区显示 202→排队位次→渲染耗时→完成内嵌播放 + 取消按钮 +
   等价 API 调用预览。图像服务仍同步,不受影响。
4. **历史画廊(HistoryOverlay)+ 灯箱**:视频卡(首帧缩略 + ▶ 角标 + 时长 + 服务名);
   灯箱按扩展名分流,mp4 渲染 `<video controls>`;LightboxMeta 元信息面板零改动,
   video 产物加下载。**TaskPanel 零改动白送**(视频任务=普通 ExecutionTask;可选
   增强:排队 pill 加位次)。

## 9. IC 侧 adapter 契约(IC 仓库实现,本 spec 只锁接口)

- 发现:`GET /api/v1/services`(筛 comfy_template)+ 服务 io schema。
- 提交:`POST /v1/services/{name}/predictions`,bearer=instance key,
  `Prefer: respond-async`,`input` 按 exposed_params;media 值传 data URI 或先传文件端点。
- 等待:轮询 `GET /v1/predictions/{id}`(或 webhook);取消 `POST .../cancel`。
- 结果:`output` 内产物签名 URL 列表(kind 标注),IC 下载落本地素材。

## 10. 测试

- 单测(CI,mock ComfyUI):参数 patch 与 node_id 漂移守护、产物分拣与 preview 过滤、
  错误翻译、模板注册 API、schema 生成、`/object_info` 解析。
- 真机 smoke(非 CI):`tests/manual/smoke_comfy_h3.py`——注册 H3 模板→生成短视频→
  校验 mp4 存在且含音轨(结构性校验;不做 SSIM golden,量化档位不保证复现)。
- 前端:vitest 覆盖字段配置状态、异步运行态轮询、画廊/灯箱 video 分支。

## 11. 部署/运维

- `infra/systemd/` 增 `nous-engine-comfyui` 单元,`enginectl` 纳管。
- H3 权重:Comfy-Org repack INT8 档入 ComfyUI models 目录(磁盘预算 ~60-70G/档)。
- 前置:GSP 缓解脚本上机;`CUDA_DEVICE_ORDER=PCI_BUS_ID`。
- ComfyUI 版本 ≥0.30.0(H3 day-0);sidecar 升级属运维操作,桥只依赖其 HTTP API
  (/prompt,/history,/view,/upload/image,/queue,/interrupt,/object_info)。

## 12. 已知限制与二期

- **限制**:sidecar 是 `gpu_allocator` 管不到的独立进程,视频任务期间与常驻 LLM 引擎
  共享 Pro 6000 显存——靠 gpu_free_probe 预检 + ComfyUI 自身 offload 缓解,不做硬隔离。
- **限制**:本地开源权重无 2K 精修;量化档位画质相对官方 API 有折损(待实测)。
- **二期**:MiniMax 官方 API provider(同家族插入)、RunningHub 云 ComfyUI provider、
  细粒度渲染进度(ComfyUI WS → node-progress)、`comfyui_workflow` 节点画布前端渲染。
