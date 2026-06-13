# 每模型显存预算可配(网页配,百分比/绝对值/auto)

日期:2026-06-13 · 状态:spec(用户拍板:灵活设置 + 有推荐值 + 网页配百分比或绝对值)
来源:embedding 上线后,Pro 6000 多模型共卡,vLLM `gpu_memory_utilization` 写死在 models.yaml,
embedding 占了 19GB(实际只需 ~10GB),且百分比对多租户不好平衡。

## 问题

1. vLLM 显存预算旋钮 `gpu_memory_utilization` = **整卡百分比**,管总预算(权重+激活+KV)。
2. 现状:embedding 在 models.yaml `params` 写死 0.2/0.3;LLM 走 adapter auto 公式。改要动 git 跟踪的 yaml。
3. 百分比对「一张卡塞多个模型」难平衡(改一个要重算其他)。用户要:**每模型可配、有推荐值、网页上填百分比或绝对值**。

## 设计

### 数据模型 — runtime overlay 新增 `vram_budget`
存 `configs/runtime_overrides.json`(gitignore,即时持久不被 git 冲;复用现有 overlay 机制,
`_OVERRIDABLE_KEYS` 加 `vram_budget`)。形:
```json
{ "qwen3_embedding_4b": { "vram_budget": { "mode": "absolute", "value": 11 } } }
```
- `mode`: `"auto"`(adapter 公式,默认)/ `"percent"`(整卡比例 0–1)/ `"absolute"`(GB)。
- `value`: percent 时是 0–1 小数;absolute 时是 GB 数;auto 时忽略。

### 解析(adapter 加载时)→ gpu_memory_utilization
统一收敛到 vLLM 的 `--gpu-memory-utilization`(单一旋钮,够用;`kv-cache-memory-bytes` 留后续细化):
- `auto` → 现有 auto 公式(零回归)。
- `percent` → 直接用 value。
- `absolute` → `value_gb / card_total_gb`(加载时按目标卡真实总显存算;Pro6000=96G → 11G≈0.115)。
- 显式 overlay 优先级 > models.yaml params 的 gpu_memory_utilization > auto 公式。

### 推荐值计算(UI 显示「推荐」+ auto 落地）
按模型 footprint(models.yaml `vram_mb` = 估算权重占用)+ type:
- embedding/tts(几乎不用 KV):`权重GB × 1.25`(权重+激活+小批 KV)。
- llm:`权重GB + 6`(权重 + 一档实用 KV);vl 同。
- image:不走 vLLM,本旋钮 N/A(image runner 另一套,不在本 spec)。
- 返回 `{recommended_gb, recommended_percent, card_total_gb}` 供 UI 双单位展示。

## 实施(2 PR)

### PR-1 后端
1. `config.py`:`_OVERRIDABLE_KEYS` 加 `vram_budget`;`set_runtime_override` 接结构化 value;
   `_apply_runtime_overrides` 把 `vram_budget` 叠进 cfg。
2. `llm_vllm.py`:加载时解析 `vram_budget`(auto/percent/absolute → utilization),优先级如上。
3. 推荐值函数 `recommend_vram_budget(model_id)`(纯计算,CI 安全)。
4. API:`PATCH /api/v1/engines/{name}/vram-budget`(body: {mode, value})写 overlay;
   `GET /api/v1/engines/{name}/vram-budget`(返当前设置 + 推荐值 + 卡总显存)。
5. 改后需重载模型生效(vLLM 子进程重启)—— 响应提示。
6. 测试:解析三模式 → utilization 映射、推荐值口径、overlay 读写、API wiring(CI 安全,不起真 vLLM)。

### PR-2 前端
1. ModelsOverlay:LLM/embedding/tts(vLLM 类)卡片右键或配置入口加「显存预算」。
2. 控件:mode 切换(auto / 百分比 / 绝对GB)+ value 输入 + 「推荐 X GB(Y%)」提示 +
   「需重新加载生效」标注。
3. `engines.ts` 加 hook;真机验:设绝对值 11G → 重载 → nvidia-smi 占用对得上。

## 不做(本轮)
- `kv-cache-memory-bytes` 绝对 KV(更细,vLLM 0.22 支持,留 footprint 不准时再上)。
- image runner 的显存预算(另一套机制,不在 vLLM 旋钮范围)。
- 自动「多模型不超卡」的全局编排(本 spec 只给单模型旋钮 + 推荐值;编排是后续 arc)。
