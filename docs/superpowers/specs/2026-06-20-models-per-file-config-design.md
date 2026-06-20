# 模型配置一模型一文件 — `configs/models.d/<id>.yaml`

- Date: 2026-06-20
- Status: 设计
- Trigger: 用户:模型定义都挤在单个 `configs/models.yaml`(17 模型/150 行),增删改不便(diff 大、
  并行加模型冲突、滚动找)。改成 **一模型一文件**(ComfyUI 模块化感):增=丢文件、删=rm、改=只动一个。

## 1. 现状

- `configs/models.yaml`:单文件,`models:` 下 17 个模型 dict 的 list。
- **两个独立读取器**(都要改):
  - `src/config.py` `load_model_configs(path)` — 主进程 API(monitor/instances/engines/metadata/scanner)。
  - `src/services/inference/registry.py` `ModelRegistry._load` — 直 `open()` 读,ModelManager/runner 子进程用。
- 运行时覆盖(resident/gpu/vram_budget)已在 `runtime_overrides.json` overlay(两读取器都套);本设计只动**静态定义**的存放方式。

## 2. 设计

### 2.1 抽共享 helper(单一来源)

`config.py` 新增 `collect_model_entries(yaml_path: Path) -> list[dict]`:
1. glob `<yaml_path 同目录>/models.d/*.yaml`,每个文件 = 一个模型 dict(或 `{models:[...]}` 容错);
2. 再读 `yaml_path` 自身的 `models:` list(向后兼容 / 共存);
3. 按 id 去重合并(models.d 优先),返回 list。

两读取器都改用它:
- `load_model_configs`:用 `collect_model_entries(resolved)` 拿 list,后续 per-entry 构建逻辑不变。
- `ModelRegistry._load`:用 `collect_model_entries(Path(config_path))` 替代 `data.get("models",[])`。

### 2.2 迁移

- 17 个条目 → `configs/models.d/<id>.yaml`(每文件一个模型 dict,顶层即 adapter/id/paths/type/...)。
- `configs/models.yaml` → 留 `models: []` + 注释指向 models.d/(保留文件:NOUS_MODELS_YAML 默认路径、
  显式传 path 的测试、collect 的同目录锚点都还在)。

## 3. 验收

- [ ] `load_model_configs()` 与 `ModelRegistry(path).specs` 都返回全部 17 模型(与迁移前 id 集合一致)。
- [ ] 改 `MODELS_ROOT` 无关;改某模型只动 models.d/<id>.yaml,diff 仅该文件。
- [ ] backend 启动模型扫描自检数不变;ruff + test_config(load_model_configs 断言 cosyvoice2 等)绿。
- [ ] 真机:加载任一模型照常(经合并后的 registry)。

## 4. 非目标 / 风险

- 不改运行时 overlay、不改 image 自动扫描(image 模型不在 models.yaml)。
- 不改 per-entry 字段语义。
- 风险:两读取器口径必须一致 → 共享 helper 强制单一来源,避免分叉。
