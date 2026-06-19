# API Key 访问矩阵 — 对外出口控制台(2026-06-19)

## 问题

`/api-keys` 是单管理员对外 API 的**访问控制中心**(用户原话:「统一对外出口都在 /api-keys 管理」)。但当前是**以 key 为中心**的列表:一行一个 key,里面挂服务名 chip(只有名字,无类型/背后/用量)。

痛点:
- 看不出每个授权服务是什么类型(llm/embedding/image/app)、背后绑哪个模型/工作流。
- 「对外出口」的主体其实是**服务**(服务才是对外暴露的东西,key 只是凭证),key-centric 视图本末倒置。
- 想知道「哪些 key 能调某服务」要逐个 key 翻。

调用面已就位(#560):`/v1/models` 按 key 返回其 active-grant 服务(服务名 + 类目),`model` 字段选服务。本设计只升级**管理面**,不动调用面。

## 设计:服务 × Key 访问矩阵

把 `/api-keys` 升级成以**服务为行、key 为列、格子=授权态**的矩阵控制台。一页看清「暴露了哪些服务、各被哪些 key 可调、用得如何」,授权 = 点格子。

```
                          │ master- │ embed-  │ ideogram4- │ ...
  服务(行,按类目分组)     │ all-svc │ public  │ external   │
─────────────────────────┼─────────┼─────────┼────────────┼──
 LLM
  qwen3-6-35b   →fp8 35B  │   ●     │   ○     │    ○       │
  qwen3-5-api   →int4     │   ●     │   ○     │    ○       │
 EMBEDDING
  qwen3-embedding-8b      │   ●     │   ●     │    ○       │
 IMAGE
  ideogram4    →wf 3234.. │   ●     │   ○     │    ●       │
  ...
```

- **行 = 服务**:服务名 · 类目徽章(llm/embedding/image/app)· 背后(model 名 或 workflow)· 状态 · 今日调用数。按类目分组。
- **列 = key**:label + prefix(+ 今日总调用)。
- **格 = 授权态**:● active / ◐ paused / ○ none。**点击切换**(none→grant、active→revoke)。
- 调用面对照:某 key 那一**列**被点亮的服务 = 这把 key `/v1/models` 看到 / 能 `model=` 调的清单。矩阵就是这条「一链多模型」清单的编辑器。
- 规模:服务(~13)×key(~8)一屏可见;key 多 → 横向滚动 + 服务列 sticky。
- key 的明文/复制/启用禁用/删除等仍可达(行内或点 key 表头进详情)。保留旧「列表视图」开关零回归。

## 后端契约(PR-B)

新增 `GET /api/v1/keys/matrix`(admin),一次拉齐矩阵所需:

```json
{
  "services": [
    {"id": "323...", "name": "qwen3-6-35b", "category": "llm",
     "source_type": "model", "backing": "qwen3_6_35b_a3b_fp8",
     "status": "active", "today_calls": 71}
  ],
  "keys": [
    {"id": "329...", "label": "master-all-services", "key_prefix": "sk-mast-55",
     "is_active": true, "today_calls": 137}
  ],
  "grants": [
    {"id": "...", "key_id": "329...", "service_id": "323...", "status": "active"}
  ]
}
```

- `backing` = source_type=model→source_name(模型 id);workflow→`wf:{workflow_id}`(前端可再解析工作流名)。
- `today_calls`:服务级/ key 级从 `llm_usage`+`tts_usage` 当日聚合(复用 usage 聚合;无则 0)。
- 格子点击复用既有端点:grant = `POST /api/v1/keys/{key_id}/grants` `{instance_id: service_id}`;revoke = `DELETE /api/v1/grants/{grant_id}`(矩阵里带 grant_id)。**不新增写端点**。

## PR 拆分

- **PR-A(本 doc)**:spec,先 push。
- **PR-B(后端)**:`GET /api/v1/keys/matrix` 聚合端点 + 单测(scoped、today_calls、backing 解析)。
- **PR-C(前端)**:`/api-keys` 矩阵视图(行=服务分组、列=key、格点击 toggle 调既有 grant 增删 + 失效缓存刷新)+ 「矩阵/列表」视图开关(列表保留)。

## 不做 / 边界

- 不动调用面(`/v1/chat`·`/v1/models` 等,#560 已成)。
- 服务「定义」(绑哪个模型/工作流、暴露参数)仍在**服务页**管;矩阵只读展示 backing,不在此编辑服务本身(访问控制 vs 资源定义分离,像 IAM)。
- quota/计费维度暂不进矩阵(用量页已有),v1 只显示 today_calls 作为「活跃度」线索。
