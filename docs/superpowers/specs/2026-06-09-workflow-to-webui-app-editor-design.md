# 工作流 → WebUI 应用编辑器(复刻 Infinite-Canvas)

- 日期: 2026-06-09
- 状态: 设计
- 参考: `/media/heygo/Program/projects-code/github-repos/Infinite-Canvas`(`static/js/api-settings.js`)
- 相关: [service-api-layer](./2026-06-03-service-api-layer-design.md)、
  [studio-publish-image-services](./2026-06-08-studio-publish-image-services-design.md)

## 1. 目标

把任意 nous 工作流(节点图)一键转成一个「填表即跑」的 WebUI 应用 ——
忠实复刻 Infinite-Canvas 的「测试画布」体验,但用 **nous 自己的节点**(非
ComfyUI)和 React + React Flow 实现:

- 右侧:只读节点图,渲染 nous 节点卡片;
- 节点上:逐 **widget** 勾选,被勾选的参数暴露成表单字段;
- 左侧:从勾选项实时生成的富表单(slider / select / 文本 / 图片上传 / 开关 /
  随机种子);
- 底部/顶部:运行,把表单值回填进工作流并执行,展示结果。

落点三处(用户指定):

1. **发布为服务弹窗**(`PublishDialog`)—— 发布时逐参数暴露并配置。
2. **服务详情页**(`ServiceDetail`)—— 新增「应用编辑」tab,发布后还能改配置。
3. **Playground tab** —— 即生成的运行表单(已存在,补富控件)。

## 2. 现状(关键发现:后端基本已就位)

| 能力 | 现状 | 缺口 |
|---|---|---|
| 暴露参数数据结构 | `ExposedParam{node_id, key, input_name, label, type, required, default, constraints}` 已含富字段(`schemas.py:396`) | 无 |
| 运行回填 | `_merge_inputs` 按 `key`→读、`input_name`(=node.data 字段名)→写(`workflow_service_runner.py:56`) | 无 —— 逐 widget 天然可行 |
| 运行表单 | `SchemaDrivenForm` 已按 `type`+`constraints.enum` 渲染 string/多行/number/boolean/file/select(`SchemaDrivenForm.tsx`) | 缺 slider / 图片预览 / 随机种子 |
| 发布产出 | `PublishDialog` 只做**节点级**勾选,硬编码 `type:'string'` + `defaultSlotForNode`,从不填富字段 | 需升级为逐 widget + 富元数据 |
| 发布后改配置 | `ServicePatch` 只允许改 `status`(`services.py:102`) | 需支持 PATCH `exposed_inputs/outputs` |
| 节点 widget 元数据 | `DECLARATIVE_NODES` 每个 widget 自带 `widget` 类型 + `min/max/step/precision/options/default`(`nodeRegistry.ts`) | 无 —— **比 Infinite-Canvas 强**:它靠字段名猜类型,我们直接拿精确类型 |

结论:核心契约和执行路径已具备,本特性主要是**前端编辑/表单 UX** + **一处后端
PATCH 扩展**。

## 3. nous 的关键优势:widget 类型不用猜

Infinite-Canvas 用 `rhWorkflowFieldKind` 靠字段名/值正则猜 IMAGE/NUMBER/BOOLEAN。
nous 的 widget 是强类型的,勾选某节点的某 widget 时直接查 `DECLARATIVE_NODES`
得到精确控件。映射表(widget → ExposedParam):

| WidgetType | ExposedParam.type | constraints | 备注 |
|---|---|---|---|
| `textarea` | `string` | `{}`(默认多行) | rows 带入 constraints.rows |
| `input` | `string` | `{format:'single_line'}` | 单行 |
| `slider` | `number`(precision=0 → `integer`) | `{min,max,step}` | 富控件渲染成滑杆 |
| `checkbox` | `boolean` | `{}` | 开关 |
| `select` / `model_select` / `component_select` / `lora_select` / `agent_select` / `seedvr2_model_select` | `string` | `{enum:[...], enum_labels?}` | 静态 options 直接带入;动态(model/component)运行时拉取,见 §6 |
| `image_upload` | `image` | `{}` | 图片上传+预览 |
| `lora_stack` / `clip_stack` | (首期不暴露) | — | 复合控件,暴露体验差,首期跳过,标注 TODO |

- `input_name` = widget 的 `name`(就是 node.data 字段名)。
- `key` = 调用方字段名,默认 = `input_name`,冲突时加 `节点短id_` 前缀。
- `label` = widget 的 `label`,可在编辑器里覆写。
- `default` = 节点当前 `data[name]` 值,回退 widget `default`。

## 4. 架构

一个**可复用核心组件** `WorkflowAppEditor`,被弹窗和服务页 tab 共用:

```
WorkflowAppEditor
├── 左:实时表单预览(复用增强后的 SchemaDrivenForm,readOnly 预览态)
└── 右:只读节点图(React Flow)
        节点 = AppEditorNode(复刻 nous 节点样式 + 每行 widget 带 checkbox)
        勾选 widget → 累加/移除 ExposedParam
```

数据流:`(snapshot/workflow, 已有 exposed) → 内部 ExposedParam[] 草稿 → onChange`。
组件不关心持久化;调用方决定是 publish 还是 patch。

### 节点图渲染

- 复用工作流的 `nodes`(含 position)+ `edges` 喂给 React Flow,`nodesDraggable=
  false`、`nodesConnectable=false`、`elementsSelectable=false`,只保留 pan/zoom。
- 新节点组件 `AppEditorNode`:头部沿用 nous 节点卡片样式(类目色条 + 标题 +
  badge,见 `BaseNode`/`DeclarativeNode`),body 列出该节点 type 在
  `DECLARATIVE_NODES` 里的 widgets,每行左侧一个 checkbox。勾中的行高亮 +
  节点整体加 `has-exposed` 视觉标记(对齐 Infinite-Canvas 的绿色/高亮节点)。
- 非声明式节点(text_input/image_input/image_output 等纯 I/O 节点)按其
  primary slot 暴露成单字段(沿用 `NODE_PRIMARY_SLOT` 的语义)。

### 输入 vs 输出

- 输入:勾选 widget / 入口节点 → `exposed_inputs`。
- 输出:勾选出口节点(image_output / text_output / save_* / preview_*)→
  `exposed_outputs`。沿用 publish 现有 output 校验(image envelope 白名单)。

## 5. PR 切分(每 lane 独立分支 + PR,走 CI/CD)

> 依赖:PR-1 与 PR-5 可独立先行;PR-2 是 PR-3/PR-4 的基础。

### PR-0(本 spec,docs)
push 本设计文档,开 docs PR。

### PR-1 后端:ServicePatch 支持改 exposed
- `ServicePatch` 增 `exposed_inputs / exposed_outputs / label`(可选)。
- PATCH 时若带 exposed:复用 `workflow_publish._node_ids` 对 `svc.workflow_snapshot`
  校验每个 `node_id` 存在(否则 422);复用 image envelope 白名单校验 outputs。
- 写入后 `invalidate("services")`;`updated_at` 刷新。**不**改 snapshot_hash
  (schema 映射变更不动快照本体)。
- 测试:patch exposed 成功 / 引用不存在 node_id → 422 / 缓存失效。
- 前端 `usePatchService` 增 `exposed_inputs/outputs/label` 可选入参。

### PR-2 共享组件:WorkflowAppEditor
- `frontend/src/components/workflow/AppEditorNode.tsx`、`WorkflowAppEditor.tsx`。
- widget → ExposedParam 派生工具 `deriveExposedParam(nodeType, widget, node)`。
- 只读 React Flow 画布 + 逐 widget checkbox + 左侧实时表单预览。
- 单测:派生映射正确(slider→number+min/max、select→enum、input_name=widget.name);
  勾选/取消增删 ExposedParam;key 冲突加前缀。
- (React Flow 框选/连线类交互本特性不需要;勾选是普通 click,可单测 +
  chrome-devtools 验,见用户 memory `reference_reactflow_devtools_verify`。)

### PR-3 发布弹窗升级
- `PublishDialog` 第 1/2 步换成 `WorkflowAppEditor`(或在节点行内展开 widget 勾选),
  产出富 `ExposedParam`。第 3 步命名发布不变。
- 删除 `defaultSlotForNode` 硬编码 `type:'string'` 的旧路径(改由派生)。
- 兼容:无 widget 定义的节点回退 primary-slot 单字段。

### PR-4 服务详情页「应用编辑」tab
- `ServiceDetail.TABS` 增 `{id:'app-editor', label:'应用编辑', icon: SlidersHorizontal}`。
- tab 内挂 `WorkflowAppEditor`,源 = `svc.workflow_snapshot` + 现有 `exposed_inputs/
  outputs`;保存按钮走 PR-1 的 PATCH;保存后 invalidate `['service', id]`。
- 顶部「保存配置 / 删除」对齐 Image4。

### PR-5 Playground 富控件对齐
- `SchemaDrivenForm` 增:
  - `slider`:`type=number/integer` 且 `constraints.{min,max}` 齐 → range 滑杆 +
    数字框联动。
  - `select`:已支持 `constraints.enum`,补 `enum_labels` 显示友好名。
  - `image`:文件选择 → 本地预览缩略图 + 上传(走现有图片上传端点,产出 URL/
    base64,对齐 `image_input` 节点期望)。
  - 随机种子:字段名含 seed 或 `constraints.random` → 数字框旁「🎲」按钮(用
    lucide `Dices`,不用 emoji)。
  - `boolean`:开关样式。
- 单测覆盖各控件渲染 + 值回传。

## 6. 待决 / 风险

- **动态 select(model_select/component_select)**:options 运行时从
  `/v1/engines`、组件库拉取。首期:编辑器里把当前可选值快照进 `constraints.enum`,
  Playground 直接用;后续可改成运行时拉取(标 TODO,不阻塞)。
- **image_upload 在 Playground 的产物格式**:需与 `image_input` 节点消费的格式
  一致(base64 data URI 还是签名 URL)—— PR-5 落地前用真机 smoke 验一次(用户
  memory `feedback_verify_real_model`)。
- **lora_stack / clip_stack** 复合控件首期不暴露,标 TODO。
- 输出交付(图持久化/签名 URL/TTL)归服务层输出契约,见
  `project_output_delivery_service_layer`,本特性不动这块。

## 7. 真机验证

按用户习惯,UI 改动用 chrome-devtools 真机点测(勾选 widget→表单实时更新→运行
出结果),逐个抓 build 抓不到的口径 bug;图片/采样类用真模型 smoke 验一次。
