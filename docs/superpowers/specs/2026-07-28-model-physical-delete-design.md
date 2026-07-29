# 引擎库模型物理删除 —— 设计

日期:2026-07-28
状态:设计已确认,待实现

## 背景

引擎库(UI `/models`,组件 `ModelsOverlay.tsx`)右键菜单里有一个「删除」项,写死
`disabled: true` 的占位(`ModelsOverlay.tsx:311` 附近),后端没有任何删除端点。
模型下线只能手工 `rm -rf` 模型目录 + 手工删 `configs/models.d/<id>.yaml` + 手工清
DB 里的元数据/运行时覆盖行,漏一步就留下幽灵条目或幽灵配置。

本设计补上这条路径:引擎库里右键即可**物理删除**磁盘文件,并同步清理注册表。

## 目标 / 非目标

**目标**

- 引擎库 5 类条目全部可删:registry 整模型、自动发现整模型、SeedVR2 超分单文件、
  单文件组件(diffusion_models / clip / vae)、LoRA。
- 真删(`shutil.rmtree` / `os.unlink`),磁盘立即释放。不做回收站。
- 注册表同步清理:`configs/models.d/<id>.yaml`、DB `model_metadata`、DB
  `model_runtime_override`、ModelRegistry 热重载、各层扫描缓存。
- 删除后返回**残留源码引用报告**,告诉用户仓库里还有哪些源文件提到这个模型。

**非目标**

- 不做回收站 / 撤销。用户明确选择直接 rm -rf。
- 不自动修改源码,也不派 agent 改。残留引用只报不删 —— 线上服务不该改自己的仓库。
  交接靠对话框的「复制清理 prompt」按钮,由用户粘给 coding agent。
- 不改写 legacy `configs/models.yaml` 的 YAML 结构(该文件目前 `models: []`,
  是空锚点)。若将来它真有条目,命中的条目进残留引用报告,由人工处理。
- 不动引用该模型的服务实例本身(删模型后服务会指向空模型,由用户自行处置)。

## 决策记录

| 决策 | 选择 | 理由 |
|---|---|---|
| 删除范围 | 全部 5 类条目 | 图像 tab 下 38 个条目里大部分是单文件组件/LoRA,只删整模型等于删不掉大头 |
| 删除语义 | 直接 rm -rf | 用户要求「物理删除」;回收站同盘 rename 不释放磁盘,且 `.trash` 会被扫描器扫到需额外排除 |
| 模型正在显存里 | 硬拒,要求先卸载 | 避免 vLLM 子进程 / image runner 还在 mmap 权重文件时删目录导致的半死状态 |
| 被服务引用 | 阻止,`force=true` 可越过 | 预检列出引用它的服务,用户知情后可强删 |
| 源码残留 | 删除后出 grep 报告,只报不删 | 源码改动是人的活;运行时端点自删代码不可接受 |
| 残留交给谁清 | 前端「复制清理 prompt」按钮,人工粘给 coding agent | 后端零额外状态、不落待办文件;生产服务不自己派 agent 改仓库 |
| 端点形状 | 两步(预检 + 执行) | 确认框需要在删之前显示「将释放 34.9GB / 要删哪个 yaml / 哪些服务引用」;单端点 409 重试拿不到这些 |

## 架构

新建 `backend/src/services/model_deleter.py`,承担目标解析、路径安全、预检、执行、
注册表清理、残留扫描全部逻辑。`backend/src/api/routes/engines.py` 已经 950 行,
只加两个薄路由,不再往里堆逻辑。

```
POST /api/v1/engines/delete/preflight   body {name}            → DeletePreflight
POST /api/v1/engines/delete             body {name, force}     → DeleteResult
```

两个端点都挂 `dependencies=[Depends(require_admin)]`,与其余写操作一致。

**为什么用 POST 而不是 `DELETE /engines/{name}`**:组件条目的 `name` 形如
`component:diffusion_models:/media/.../x.safetensors`,含 `/`,做不了 path 参数。
现有 `/component/unload`、`/seedvr2/unload` 已是同样的 body 风格。

## 条目 → 物理目标解析

`resolve_target(name, engines_snapshot) -> Target`

| kind | name 形态 | 物理目标 |
|---|---|---|
| `model` | engine key(如 `qwen3_6_35b_a3b_fp8`) | `LOCAL_MODELS_PATH/<cfg.local_path>` 整目录 |
| `upscale` | `seedvr2:<filename>` | `LOCAL_MODELS_PATH/image/SEEDVR2/<filename>` 单文件 |
| `component` / `lora` | `component:<role>:<abs_path>` | 该 `abs_path` 单文件 |

`Target` 携带:`path`、`is_dir`、`kind`、`engine_key`(仅 `model` 有)、`local_path`。

未知 name → 404。

## 路径安全闸门

删任何东西之前,`assert_safe_target(target)` 必须全过,否则 400:

1. `path.resolve()` 后必须落在允许根之内。允许根 = `LOCAL_MODELS_PATH.resolve()`,
   若 `LORA_PATHS` 指向别的根(默认 `<MODELS_ROOT>/comfyui/models/loras`,在
   `LOCAL_MODELS_PATH` 之外),该根一并加入白名单。
2. 拒绝允许根本身;拒绝相对深度 < 2 的目录 —— 不许删 `image/`、`llm/`、`speech/`
   这类类型目录。
3. 目标是符号链接 → 只 `unlink` 链接本身,**绝不**顺着链接 `rmtree` 到别处。
4. `name` 中含 `..` 段 → 直接 400,不进解析。

## 预检

`POST /api/v1/engines/delete/preflight`,返回:

```json
{
  "name": "qwen3_6_35b_a3b_fp8",
  "target_path": "/media/heygo/program/models/nous/llm/Qwen3.6-35B-A3B-FP8",
  "is_dir": true,
  "size_bytes": 37500000000,
  "blockers": {
    "loaded": {"status": "loaded", "gpu": 1},
    "services": [{"id": "...", "name": "qwen36-fp8-a1b2"}]
  },
  "registry_cleanup": {
    "models_d_yaml": "configs/models.d/qwen3_6_35b_a3b_fp8.yaml",
    "model_metadata": true,
    "runtime_overrides": 2
  },
  "code_refs": [{"file": "src/api/routes/services.py", "line": 377, "text": "..."}]
}
```

- `blockers.loaded` 非空 = **硬拒**,`force` 也不放行。判定按 kind 分:
  `kind=model` 用 engines.py 现有的 `_is_engine_loaded` / `_get_loaded_gpus`;
  `upscale` / `component` / `lora` 用 `engine_catalog` 已有的
  `_loaded_index` / `_component_loaded_index` 结果(即条目 `status`)。
  `loaded` 与 `loading` 都算已加载。
- `blockers.services` = `service_instances` 中 `source_type='model'` 且
  `source_name=<engine_key>` 的行。非空 = **软拒**,`force=true` 放行。
  仅 `kind=model` 有此检查(单文件组件/LoRA 不会被服务实例按 key 引用)。
- `size_bytes` 在删除前算(目录走一次 `rglob` 求和),供确认框显示「将释放 X GB」。
- `code_refs` 预检时也算一份,让确认框提前显示残留引用。

## 执行

`POST /api/v1/engines/delete` body `{name, force}`:

1. **服务端重跑预检**。不信任前端传来的除 `force` 以外的任何东西。
   - 硬 blocker(loaded)在 → 409。
   - 软 blocker(services)在且 `force` 非真 → 409,附带引用列表。
2. **删磁盘**。目录走 `shutil.rmtree(path, onerror=收集)`,单文件走 `os.unlink`。
   `onerror` 收集失败项而不是抛异常 —— 保证第 3 步注册表清理仍会执行,并把
   「磁盘删了一半」如实报给用户,不静默。
3. **注册表清理**(仅 `kind=model` 需要全套;组件/LoRA/SeedVR2 只需第 4 步缓存):
   - `configs/models.d/<engine_key>.yaml` 存在则删除。
   - `DELETE FROM model_metadata WHERE engine_key = <key>`
   - `DELETE FROM model_runtime_override WHERE model_id = <key>`
   - `mgr._registry.reload()` 热重载模型定义。
4. **缓存失效**(全部条目类型都要):
   - `model_scanner.invalidate_scan_cache()`
   - `model_metadata_service.invalidate_local_scan_cache()`
   - `lora_scanner.invalidate_cache()`
   - `component_scanner.invalidate_component_cache()`
   - `response_cache.invalidate("engines")`。服务实例本身不动,故不需要
     `invalidate("services")`。
5. **残留引用扫描** → 结果并入响应返回。

   实现取舍(2026-07-29):扫描实际跑在第 1 步**之前**(复用服务端重跑的那次预检),
   避免为一次删除跑两遍 `git grep`。代价是 `models.d/<key>.yaml` 会把自己扫进来,而
   它在第 3 步就被删了 —— 响应构造时用 `drop_refs_to()` 把指向该文件的条目剔掉,
   否则报告会指使用户去清一个已不存在的文件。删除模型目录不影响源码引用,所以其余
   结果与「删后再扫」等价。

响应:

```json
{
  "deleted": true,
  "target_path": "...",
  "freed_bytes": 37500000000,
  "disk_errors": [],
  "registry_cleaned": {"models_d_yaml": true, "model_metadata": true, "runtime_overrides": 2},
  "code_refs": [...]
}
```

`disk_errors` 非空时 `deleted` 仍为 `true`(部分删除),前端以警告样式展示未删项。

## 残留引用扫描

`scan_code_refs(terms) -> list[CodeRef]`

- 优先 `git grep -n -F -- <term>` 在仓库根执行:自动跳过 `.gitignore` 覆盖的
  `.venv` / `node_modules` / `frontend/dist` / 日志,速度也最好。
- 非 git 环境(或 `git grep` 返回非 0/1 退出码)fallback 到 `os.walk` + 后缀白名单
  (`.py .ts .tsx .yaml .yml .md .sh .json`),显式跳过 `.venv`、`node_modules`、
  `dist`、`__pycache__`、`.git`。
- 搜索词 = engine key + `local_path` + 目录名/文件名(基名)。去重;丢弃长度 < 4 的
  词避免噪音。
- 结果上限 200 条,超出时截断并在响应里标 `truncated: true`。
- 扫描失败(git 不可用、超时)不影响删除成败,返回空列表 + 一条 `scan_error` 说明。

### 谁来清理这些残留

**没有任何自动 agent。**后端是生产服务(systemd `nous-engine-backend`),不允许由
一个 admin HTTP 端点触发去改自己的仓库源码 —— 无论直接改还是派 agent 改。

交接方式:对话框在残留列表旁给一个「复制清理 prompt」按钮,把结果拼成一段现成的
提示词放进剪贴板,用户粘给 Claude Code(或任意 coding agent)执行。形如:

```
已从 nous-engine 物理删除模型 `qwen3_6_35b_a3b_fp8`
(目录 /media/heygo/program/models/nous/llm/Qwen3.6-35B-A3B-FP8,已释放 34.9GB)。
注册表已清理:configs/models.d/qwen3_6_35b_a3b_fp8.yaml、model_metadata、
model_runtime_override。

以下源文件仍引用它,请逐个判断并清理(不确定的先报给我,不要盲删):
- src/api/routes/services.py:377  ...
- configs/image_arch/flux2/xxx.yaml:12  ...
```

不落盘、不产生待办文件,后端零额外状态。

## 前端

- `ModelsOverlay.tsx`:「删除」菜单项去掉 `disabled: true`,`onClick` 打开对话框,
  保留 `danger: true` 红色样式。菜单项对所有 kind 可用。
- 新建 `frontend/src/components/models/DeleteModelDialog.tsx`:
  - 打开即调预检,加载态给骨架。
  - 展示:目标路径、将释放空间、注册表清理清单、引用它的服务、代码残留引用
    (可折叠列表)。
  - `blockers.loaded` 非空 → 只显示「该模型正在显存中,请先卸载」+ 关闭按钮,
    不给删除按钮。
  - 确认方式:**输入框内键入模型 `display_name` 完全一致**才激活「永久删除」按钮。
  - `blockers.services` 非空 → 额外一个必勾复选框「我知道这些服务将指向不存在的
    模型」,勾选后映射为请求里的 `force: true`。
  - 删除成功 → toast 报释放空间;`disk_errors` 非空时改为 warning toast。
  - 残留引用列表旁「复制清理 prompt」按钮(见上节),`navigator.clipboard.writeText`;
    `code_refs` 为空时不渲染该按钮。剪贴板 API 不可用(非 HTTPS/无权限)时降级为
    展开一个只读 `<textarea>` 让用户自行全选复制 —— 生产是明文 HTTP 内网访问
    (见 CLAUDE.md 与 memory「别加 HSTS」),`navigator.clipboard` 在非安全上下文
    下确实会缺失,这个降级不是可选项。
- `frontend/src/api/engines.ts`:新增 `useDeletePreflight`(query,按 name 缓存)
  与 `useDeleteEngine`(mutation)。成功后 invalidate `engines` 与 `services` 查询。

## 错误处理

| 情况 | 响应 |
|---|---|
| 未知条目 name | 404 |
| name 含 `..` / 解析后越界 / 深度不足 | 400 |
| 模型已加载或加载中 | 409(`force` 无效) |
| 有服务引用且未 `force` | 409 + 引用列表 |
| rmtree 部分失败 | 200 + `disk_errors` 非空 |
| 注册表清理某步失败 | 200 + `registry_cleaned` 里该项为 false + 日志 error |

## 测试

`backend/tests/test_model_delete.py` —— 全程 `monkeypatch` `LOCAL_MODELS_PATH` 到
`tmp_path` 造假模型树,**绝不碰真模型盘**:

- 5 类 kind 各自删对目标(整模型删目录、SeedVR2/组件/LoRA 删单文件)
- 路径越界:`../` 逃逸、绝对路径注入、软链指向根外 → 400 且真实目标未动
- 深度闸门:试图删 `image/` 类型目录 → 400
- 已加载 → 409,且 `force=true` 依然 409
- 有服务引用 → 409;`force=true` → 200 且服务实例未被删
- 注册表清理:`models.d/<key>.yaml` 确实被删、`model_metadata` 行被删、
  `model_runtime_override` 行被删
- 各缓存失效函数被调用(spy)
- rmtree 部分失败 → `disk_errors` 如实返回,注册表清理仍执行
- `scan_code_refs` 在非 git 目录下 fallback 生效

`frontend/src/components/models/DeleteModelDialog.test.tsx`:

- 名字输入不匹配时删除按钮禁用,完全匹配后启用
- `blockers.loaded` 非空时不渲染删除按钮
- `blockers.services` 非空时未勾复选框则按钮禁用
- `code_refs` 非空 → 渲染「复制清理 prompt」按钮,点击后写入剪贴板的文本包含模型名
  与全部残留条目;`navigator.clipboard` 缺失时改为渲染只读 textarea 降级

## 实现顺序

1. `model_deleter.py` 目标解析 + 路径安全 + 单测(先红后绿)
2. 预检逻辑(size / blockers / registry_cleanup)+ 单测
3. `scan_code_refs` + 单测
4. 执行逻辑(删盘 + 注册表清理 + 缓存失效)+ 单测
5. `engines.py` 两个路由 + 路由级测试
6. `api/engines.ts` hooks
7. `DeleteModelDialog.tsx` + 前端测试
8. `ModelsOverlay.tsx` 接线,去掉 disabled 占位
