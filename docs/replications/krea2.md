# krea2 复刻 manifest

来源: `Krea2-全能总控-11模式.json`(ComfyUI UI 导出) | 日期: 2026-08-30(2026-08-31 更新)
方式: **ComfyUI 桥(模板即服务)**,非原生复刻 —— 与 `minimax-h3-r2v` 同一条路。

## 为什么走桥而不是原生复刻

工作流 214 节点 / 18 分组 / 11 种玩法,依赖 `Krea2EditGroundedEncode`、
`SeedVR2*`、`llama_cpp_*`、`LazySwitchKJ`、`easy compare` 等大量第三方节点包。
原生复刻要在 nous 侧补齐这些节点能力,工作量与收益不成比例;且并发瓶颈
(`comfy_bridge._SEM(1)`)与原生 image runner 串行队列相同,复刻不会更快。

## UI → API 格式转换

桥吃 API 格式(`{node_id: {class_type, inputs}}`),原始文件是 UI 格式。
**不需要手点 ComfyUI 的「导出(API)」** —— comfy-cli 自带客户端转换:

```python
from comfy_cli.workflow_to_api import convert_ui_to_api
api = convert_ui_to_api(ui_workflow, object_info)   # object_info 取自 GET /object_info
```

结果: 214 节点 → **136 节点**(2026-08-31 用户改工作流后重传:212 → **135 节点**)。被解析消除的是 GetNode×46 / SetNode×19
(KJNodes 前端虚拟节点,链路直连) + Note×8 / MarkdownNote×2,以及 3 个
`mode=4`(bypass)的 LoRA 节点。

`comfy validate --workflow <api.json> --input <object_info.json>` → **valid: true, 0 errors**。
6 条 warning 全是 SeedVR2 节点包的 `COMFY_MATCHTYPE_V3` 通配类型,静态校验器
解析不了,误报。

## 模式路由(11 种玩法如何切)

单个 `INTConstant#3101`(value=1..11)→ `KREA_MODE` 总线 → 10 个
`easy compare(a==b)` → 13 个 `LazySwitchKJ`。惰性求值:未选中的分支不执行。

三层:
1. **提示词模板**(组 07,仅影响 1/2/3):`#4425` mode==3 ? 四视图模板 : `#4422` mode==2 ? 巨物模板 : 默认模板
2. **模式4 资源四切换**(组 08):mode==4 时 MODEL/POSITIVE/NEGATIVE/LATENT 整体换成 Kontext 编辑链
3. **输出七级级联**(组 17):`#3156`(11) → `#3155`(10) → … → `#3150`(5) → else = `KREA_OUTPUT_1_4`

玩法编号: 1 普通/风格参考 · 2 巨物美学 · 3 四视图 · 4 漫画转真人 · 5 包臀裙人物 ·
6 单图编辑 · 7 双图编辑 · 8 局部重绘 · 9 图像扩展 · 10 SeedVR2 2K · 11 SeedVR2 4K

## 创建的资源(卸载按从下往上顺序删)

- 服务 + 模板: `krea2` / template_id `352457419630055424`
  → `DELETE /api/v1/comfy-templates/352457419630055424`(级联删 ServiceInstance)
  (删服务那侧现在也会连带删模板,两边对称 —— #681)
- API key: 未铸(验证走 ADMIN_TOKEN 的 admin-session 回退路径)
- 模型文件: **共享资产,勿删** —— 全部由 ComfyUI sidecar 自己的
  `extra_model_paths.yaml` 管理(`/media/heygo/program/models/nous/media/`),
  nous 侧没有引用

## exposed 契约

`POST /v1/services/krea2/predictions`(注意前缀 `/v1/`,不是 `/api/v1/`)
带 `Prefer: respond-async` → 202 → 轮询 `GET /v1/predictions/{id}`。

| key | 类型 | 节点 | 必填 | 说明 |
|---|---|---|---|---|
| `mode` | integer(enum 1-11) | `3101.value` | 否(默认 1) | 玩法选择 |
| `prompt` | string | `64.value` | 否 | 提示词 |
| `style_pack` | string(enum 48) | `1243.styles` | 否(默认 fooocus_styles) | 风格包 |
| `styles` | string(enum 275,**多选**) | `1243.select_styles` | 否 | 风格,逗号串;带缩略图 |
| `main_image` | file | `227.image` | 否 | 主图,玩法 4-11 实际必填(含遮罩) |
| `ref_image` | file | `2021.image` | 否 | 参考图,玩法 4/7 |
| `seed` | integer | `75.seed` | 否(random) | 随机种子 |
| `edit_lora_strength` | number 0-2 | `2017.strength_model` | 否(默认 1.0) | 编辑 LoRA 强度,玩法 5-9 |

**风格选择器**:`styles` 的 275 个选项各带中文名 + 缩略图,数据来自 sidecar 的
`/easyuse/prompt/styles?name=<包>`,经 nous 的 `GET /api/v1/comfy/styles?pack=<包>`
代理(#678)。Playground 渲成缩略图网格 + 搜索 + 多选(#679)。
`select_styles` 在 workflow 快照里**没有序列化**(原工作流没选任何风格),桥在运行时
补写这个 key —— 所以重传工作流时 stale 检测会把 `styles` 报成 stale,那是误报。

**多选的传输形态是逗号串**,不是数组(对齐 ComfyUI-Easy-Use 的
`prompt.py:196 select_styles.split(',')`)。名字必须与风格包内**完全一致**,
不认识的名字被静默跳过(`prompt.py:205 continue`)—— 例如 `SAI Anime` 是错的,
正确写法是 `sai-anime`。

**已知契约缺陷**: 桥的 `exposed_params` 是平表,表达不了「选了玩法 10 就必须给图、
prompt 无意义」这类条件必填。补救是把适用玩法写进 `label`,schema 能带出去。

产物结构(注意是嵌套的,不是顶层 `items`):
```
output.outputs.bridge.items[0].url    → /files/images/<date>/<hash>.png?token=…&expires=…
```

## 真机验证

| 日期 | mode | 参数 | predict_time | 语义核对 |
|---|---|---|---|---|
| 08-30 | 1 普通 | seed 12345 | 26.1s | ✓ 电影感森林实拍狐狸 |
| 08-30 | 3 四视图 | 同上 | 32.8s | ✓ 正面/侧面×2/背面,白底 |
| 08-30 | 1 普通 | `styles=sai-anime` | 28.7s | ✓ 明确的动漫/吉卜力风 |
| 08-31 | 1 普通 | 重传两图版后冒烟 | 26.5s | ✓ 出图正常 |
| 08-31 | 1 普通 | `styles=sai-anime,Fooocus Cinematic`(多选) | 44.9s | ✓ 动漫风 + 电影感广角/丁达尔光,与单 sai-anime 的近景构图明显不同 |

mode 1 vs mode 3 差异显著 → 模式路由端到端生效。
加/不加 `sai-anime` 差异显著 → 风格选择器生效。
单选 vs 多选差异显著 → 多选叠加生效(#683 修好校验之后才通;在那之前多选值一律 422)。

⚠️ **玩法 1-4 固定种子也不可复现** —— 那条路要过 `llama_cpp_instruct_adv` 做提示词
扩写,LLM 采样本身不确定。别拿"同 seed 出图不同"当 bug。

## 运维注意

- sidecar `nous-engine-comfyui` 绑 `CUDA_VISIBLE_DEVICES=1`(Pro 6000 96G);
  工作流里 `*MultiGPU` 加载器写的 `cuda:0` 在该掩码下正好指向这张卡。
- 玩法 10/11 是 SeedVR2 4K 放大,耗时长;`comfy_bridge._SEM(1)` 全局串行,
  一个 4K 任务在跑时 `krea2` 和 `minimax-h3-r2v` 的所有请求都排队。
- 玩法 1-4 会加载 `llama_cpp` 的 4B GGUF 做提示词扩写,与 Krea2 主模型抢同一张卡显存。

## 相关 PR

  #678  combo 选项支持每项元数据(label/image)+ `GET /api/v1/comfy/styles` 代理
  #679  Playground 缩略图选择器(搜索 + 多选)
  #681  桥模板/服务删除对称 —— 孤儿模板不再删不掉
  #683  多选枚举字段的校验逐项比对(否则多选值永远 422)
