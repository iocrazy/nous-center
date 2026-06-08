# 创作台四功能发布为外部 API 服务(P4)

**Status**: Draft
**Author**: heygo(设计 by Claude)
**Date**: 2026-06-08
**上游**: [2026-06-07-native-image-features-replication-design.md](2026-06-07-native-image-features-replication-design.md) 的 P4

## 0. 目标

把创作台(`/studio`)已真机验通的四功能(文生图 Z-Image / 图片编辑 Flux2 / 细节增强 SeedVR2 /
角度控制 Qwen-Edit),各发布为一个可被**外部 API 自动调用**的服务,统一经
`POST /v1/images/generations`(火山式统一端点 + `model` 选服务)调用出图。

## 1. 决策(用户 2026-06-08 拍板)

- **显存**:先保障每个功能**独立跑通**(角度控制 Qwen-Edit 54GB 与常驻 vLLM 42GB 在 96GB 装不下 →
  用时手动让位 / 按需独占卡即可)。fp8 量化或 VRAM 守卫跨模态驱逐这类**共存优化**留作后续独立 arc,
  不阻塞 P4。
- **端点形态**:**扩展现有 `/v1/images/generations`**(不新增 `/v1/images/edits`)。依据火山方舟
  Seedream/SeedEdit 调研:火山是「**单一 images/generations 端点 + `image` 字段(URL 或 base64
  data URI,可数组多图)**」,靠 `model` + 有无 `image` 区分文生图/图生图/编辑,**不是** OpenAI 那种
  分离的 `/v1/images/edits` + multipart 文件上传。这与本项目既有「火山式统一端点」哲学一致(#340)。
- **发布入口**:创作台每个 tab 加「发布为服务」按钮(用户先前已拍)。

## 2. 现状勘查(已读码确认)

- **发布契约** `workflow_publish.py` `POST /api/v1/workflows/{id}/publish` 已就位:冻结 snapshot +
  校验每个 `exposed.node_id` 存在 + 出图字段白名单(`_IMAGE_OUTPUT_FIELDS`)+ `category=image` 自动
  探测(靠 snapshot 里有 `flux2_vae_decode` 节点)+ 翻 `wf.status="published"`。
- **执行核心** `run_published_workflow._merge_inputs` 已支持把 caller `{key: value}` 注入任意 exposed
  节点,`NODE_PRIMARY_SLOT` 已含 `image_input→image`。**工作流执行层的图注入管线(#370)是通的**。
- **卡点 = 外部端点只收 prompt**:`/v1/images/generations` 现逻辑 `_pick_prompt_input_key` 选一个文本
  exposed input 把 `body.prompt` 注进去。4 功能里 3 个(编辑/增强/角度)需要 `image` 输入,**增强
  完全无 prompt**(会触发 `no_prompt_input` 报错)。
- **`_extract_image_urls` echo bug(外部路径复发 #372)**:它扫 `result["outputs"]` 里**所有**节点
  的 `image_url`。`image_input` executor 返回 `{"image_url": <上传图签URL>}` —— 含 image_input 的
  工作流(编辑/增强/角度)外部调用会把**输入图**也捞进结果,可能当输出返回。
- **输出终端**:`image_output` 是 **sink 节点(无输出)**,真正 emit `image_url` 的是 `flux2_vae_decode`
  (dec)/ `seedvr2_upscale`(up)。故 `exposed_outputs` 指向 **dec/up**,不是 image_output。

## 3. 四服务 exposed schema(钉死)

节点 id 来自创作台 `buildXxxWorkflow`(Studio.tsx);field 名经 `_merge_inputs`(slot=`input_name`)
+ executor 读取核对一致。

| 服务 | exposed_inputs | exposed_outputs | category |
|---|---|---|---|
| 文生图 Z-Image | `prompt→{node:enc, slot:text}` | `{node:dec, field:image_url}` | auto(有 dec) |
| 图片编辑 Flux2 | `image→{img, image}` + `prompt→{enc, text}` | `{dec, image_url}` | auto |
| 细节增强 SeedVR2 | `image→{img, image}` + `resolution→{up, resolution}` | `{up, image_url}` | **显式 image**(无 dec) |
| 角度控制 Qwen-Edit | `image→{img, image}` + `prompt→{enc, text}` | `{dec, image_url}` | auto |

- 每个 input:`{node_id, key, input_name, type, required}`;`key` 是外部 body 字段名(prompt/image/
  resolution),`input_name` 是节点 data slot。
- SeedVR2 工作流没有 `flux2_vae_decode` → publish 的 `_detect_category` 落空 → 前端 publish 时显式传
  `category="image"`。

## 4. 实施(PR 拆分)

### PR-1(后端):`/v1/images/generations` 扩展 + extract 修复 + 测试

1. **请求模型** `ImageGenerationRequest`:
   - `prompt` 改 `str | None`(增强无 prompt);
   - 加 `image: str | list[str] | None`(URL 或 base64 data URI,单串/数组);
   - `model_config = ConfigDict(extra="allow")` 收 `resolution` 等额外参数。
2. **通用参数合并**(替换「只塞 prompt」):
   - `exposed_keys = {p.key for p in svc.exposed_inputs}`;
   - body 里任意字段(含 extra)名命中 `exposed_keys` → 注入 `inputs[key]`;
   - OpenAI 兼容兜底:若服务有文本 input 且 body 给了 `prompt` 但 key 不字面叫 `prompt`,经
     `_pick_prompt_input_key` 注入;
   - `inputs` 为空才报错(不再强制要文本 input)。
3. **`_extract_image_urls(result, snapshot, exposed_outputs)`**:
   - 优先按 `exposed_outputs` 的 node_id 取其 output 的 `image_url`;
   - 兜底扫全部 output 但**跳过 `image_input` 类型节点**(用 snapshot 的 node_id→type 映射);
   - 修 echo bug,且对老服务(#340)向后兼容。
4. **图字段格式**:本轮先吃 **base64 data URI**(image_input executor 已支持 `data:`;curl 可发);
   URL 输入(需下载 + SSRF 校验)留 follow-up,不阻塞独立跑通。
5. 单测:扩 `tests/test_images_generations.py`(image 注入、增强无 prompt、extract 跳 image_input
   echo);WorkflowExecutor.execute 仍 mock(CI 无 GPU)。

**PR-1 独立可验**:不依赖前端 —— 手动 `POST /api/v1/workflows` 存图 + `POST .../publish` 建服务
+ 建 key/grant + curl `/v1/images/generations` 真机出图(每功能一条),验「独立跑通」。

### PR-2(前端):创作台「发布为服务」按钮

1. 每个 tab 加按钮 → 弹窗(服务名 `^[a-z][a-z0-9-]{1,62}$` + label);
2. `POST /api/v1/workflows`(存当前 build 的 {name,nodes,edges},非 auto_generated)→
   `POST /api/v1/workflows/{id}/publish`(带 §3 的 exposed_inputs/outputs,SeedVR2 传 category=image);
3. 成功 toast + 显示调用示例(model=服务名);tsc/build 本地过。

### PR-3:真机端到端验证

建 key + grant 四服务 → curl 四条(文生图纯 prompt / 编辑+增强+角度带 base64 图)→ 确认出图正确。
角度控制验时手动让 vLLM 让位(`POST /engines/{name}/unload?force=true`,完后 load 回)。

## 5. 不做(本轮)

- Qwen-Edit fp8 / VRAM 守卫跨模态驱逐(共存优化,后续 arc)。
- image URL 远程下载输入(先 base64);多图 `image2..image10` 火山式多参考(plumbing 已支持逗号分隔)。
- `b64_json` 响应(先 url)。
