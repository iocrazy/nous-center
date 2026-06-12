# 出图拖到输入(迭代范式)— design

2026-06-12。对齐 Infinite-Canvas「把生成图拖回当输入」,让用户把 image_output 画廊里的某张
生成图直接作为另一个采样器的输入图(img2img / 参考编辑 / SeedVR2 超分),**无需重新上传** → 迭代。

## 背景:IC 怎么做(已读 static/js/canvas.js)

- 生成图缩略图可拖拽,`dataTransfer` 带 `application/x-canvas-output-image` = 图 url。
- 落到 `image` 节点 → `handleImageNodeDropEvent` → `setImageNodeFromOutput(nodeId, url)` 把该节点
  url 设成生成图。生成图于是成为下游 generator 的输入 → 迭代环。
- 另有 OUTPUT 节点右键「转为输入组 / 复制为新的输入组」(`convertOutputNodeToInputGroup`):把输出图
  整体变成 `image` 节点组,重连下游。

## nous 现状 + 约束

- `image_input`(image-io 声明式节点,可放置)产 `image` 端口喂采样器;widget `image`(`image_upload`)。
- **硬约束**:`exec_image_input`(backend/nodes/image-io/executor.py:18-20)**强制 `data.image` 为
  base64 data URI**(`startswith("data:")`),给签名 URL 直接 `RuntimeError`。
- 生成图是**签名 URL**(`/files/images/<date>/<uuid>.<ext>?token=...`),不是 base64 → 现在塞不进 image_input。
- React Flow 拖拽落点 + 自动连线有可测性限制(合成事件 drop 能、连接拖拽不能 → 靠单测,见
  [[reference_reactflow_devtools_verify]])。

## 方案:两条互补路径,都落到「image_input 节点持有生成图 url」

### PR-1 后端:image_input 接受已存图 URL(透传)

`exec_image_input` 放开 `src` 来源:
- `data:` 开头(base64 data URI)→ 现逻辑(解码 + 落盘签 URL)。**零回归**。
- 本站图 URL(`/files/images/...` 或绝对同源)→ **不重新落盘**:`resolve_path` 校验文件在盘,
  重签延 TTL(防生成图签名 1h 过期),返回 `{image_url, ...}`。引用一张已生成图,不必重传。
- 其余(空 / 非法)→ 现 `RuntimeError`。

签名 URL 解析复用 `image_output_storage`(`verify_token`/`resolve_path`/`sign_existing_image`);
runner 端 `_resolve_input_image_path` 本就把签名 URL → 本地路径,下游链路无需改。

### PR-2 前端:画廊图 → image_input

- **拖拽源**:image_output 画廊缩略图(#492 网格)设 `draggable`,`dragstart` 写 `dataTransfer`
  (自定义 mime `application/x-nous-image-url` + `text/uri-list` 兜底)= 图 url。
- **drop 落点**:`image_input` 节点(DeclarativeNode 的 `image_upload` widget)接 drop → 设
  `data.image = url`(对齐 IC `setImageNodeFromOutput`)。
- **「转为输入」按钮(MVP 主路,更稳)**:画廊每图 hover 一个「转为输入」按钮 → 在该输出节点附近
  `addNode` 一个 `image_input` 节点(`data.image = url`),用户再连到采样器 `image` 端口。
  React Flow 自动连线可测性差 → 不自动连,只 spawn 预填节点;拖拽落点作增强。

## 边界 / 决策

- **TTL 过期**:生成图签名 URL TTL 1h;迭代时可能已过期 → image_input 透传时 `resolve_path` + 重签延 TTL
  (与 run-history arc「画廊旧图重签」follow-up 同源,见 [[project_run_history_artifacts]])。
- **不做 IC 的「组」**:nous typed DAG 没有自由画布的 group 容器;批量「转为输入」= spawn 多个
  image_input 节点(每图一个),不做 group 包装。
- **安全**:只接受**本站** image storage URL(token 校验 + resolve_path 限定在 outputs root),
  不接受任意外链(防 SSRF / 任意读盘)。

## 测试

- 后端:`exec_image_input` 接 base64(原,零回归)/ 接本站签名 URL(新,透传 + 重签)/ 接外链或非法(报错)。
- 前端:画廊缩略图可拖(dragstart 设 dataTransfer)/ image_input drop 设 `data.image` / 「转为输入」spawn
  预填 image_input 节点。

## PR 拆分

- **PR-1 后端**:`exec_image_input` 接受已存图 URL(透传 + 重签)+ 单测。
- **PR-2 前端**:画廊拖拽源 + image_input drop target + 「转为输入」按钮 + 单测。

关联 [[project_lightbox_image_node_enhancements]] [[project_native_image_features]] [[project_run_history_artifacts]]
[[reference_reactflow_devtools_verify]]
