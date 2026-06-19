# 模型/库路径收口 — `.env` 单一根 + comfyui 式相对结构

- Date: 2026-06-19
- Status: 设计
- Trigger: 用户要把散在 `config.py` 的多个 nvme2 绝对路径(`LOCAL_MODELS_PATH`/`LORA_PATHS`/
  `NAS_*`/`COSYVOICE_REPO_PATH`/`INDEXTTS_REPO_PATH`)收成 **`.env` 一个总根** + 单独配置文件
  里的相对结构(ComfyUI `base_path` 模式)。动机:重格 nvme2 / 搬盘时只改一处,不用满 config 改路径。

## 1. 现状

`backend/src/config.py` 里各自独立的绝对路径设定(live `.env` 多数被覆盖到 `/media/heygo/Program/...`):

| 设定 | 当前值 | 消费文件数 |
|---|---|---|
| `LOCAL_MODELS_PATH` | `/media/heygo/Program/models/nous` | 11(`Path(LOCAL_MODELS_PATH)/model_path` 到处用) |
| `LORA_PATHS` | `/media/heygo/Program/models/comfyui/models/loras`(**另一棵子树**) | 2 |
| `NAS_MODELS_PATH` | `/media/heygo/Program/models/nous`(= LOCAL,名存实亡) | 4 |
| `NAS_OUTPUTS_PATH` | `/media/heygo/Program/models/nous/outputs` | 4 |
| `COSYVOICE_REPO_PATH` | `/media/heygo/Program/projects-code/github-repos/CosyVoice`(代码库) | 2 |
| `INDEXTTS_REPO_PATH` | `/media/heygo/Program/projects-code/github-repos/index-tts`(代码库) | 2 |

- 所有消费点都用 `settings.XXX`(无 `os.environ` 直读)→ 改设定**派生方式**即可,消费代码零改。
- `configs/model_paths.yaml` 已存在,是 **role→glob**(相对 `LOCAL_MODELS_PATH`,给 component_scanner),
  **本设计不动它**(它的 base 仍是 `LOCAL_MODELS_PATH`,只是后者改为派生)。
- `model_config = {"extra": "ignore"}` → 去掉旧字段后 `.env` 残留键不报错。

## 2. 用户决策(2026-06-19 已拍)

1. 相对结构 → **单独 yaml**(comfyui 式)。
2. `NAS_*` → **并进 `MODELS_ROOT`**(已名存实亡,当本地子路径派生)。
3. TTS 代码库(`COSYVOICE/INDEXTTS_REPO_PATH`)→ **也收**(走第二个根 `REPOS_ROOT`,因它们在 `projects-code/github-repos/` 另一棵树)。

## 3. 设计

### 3.1 `.env` — 两个根(模型 + 代码库)

```
MODELS_ROOT=/media/heygo/Program/models
REPOS_ROOT=/media/heygo/Program/projects-code/github-repos
```

删掉 live `.env` 与 `.env.example` 里的 `LOCAL_MODELS_PATH`/`LORA_PATHS`/`NAS_MODELS_PATH`/
`NAS_OUTPUTS_PATH`(`COSYVOICE/INDEXTTS_REPO_PATH` 本就只在 config.py 默认)。

### 3.2 新建 `backend/configs/model_roots.yaml` — 相对子根

```yaml
models:                          # 相对 MODELS_ROOT
  local:   nous                  # → LOCAL_MODELS_PATH / NAS_MODELS_PATH
  loras:   comfyui/models/loras  # → LORA_PATHS
  outputs: nous/outputs          # → NAS_OUTPUTS_PATH
repos:                           # 相对 REPOS_ROOT
  cosyvoice: CosyVoice           # → COSYVOICE_REPO_PATH
  indextts:  index-tts           # → INDEXTTS_REPO_PATH
```

### 3.3 `config.py` — 派生

- 新增字段 `MODELS_ROOT` / `REPOS_ROOT`(env-settable,有默认)。
- 删除上述 6 个字段,改为 `@property` 从根 + `model_roots.yaml` 派生:
  - `LOCAL_MODELS_PATH = MODELS_ROOT / models.local`
  - `NAS_MODELS_PATH   = LOCAL_MODELS_PATH`(并进)
  - `NAS_OUTPUTS_PATH  = MODELS_ROOT / models.outputs`
  - `LORA_PATHS        = MODELS_ROOT / models.loras`
  - `COSYVOICE_REPO_PATH = REPOS_ROOT / repos.cosyvoice`
  - `INDEXTTS_REPO_PATH  = REPOS_ROOT / repos.indextts`
- `model_roots.yaml` 加载 **fail-soft**:文件缺失/坏 → 用 config.py 内置默认相对值(与 yaml 同),
  绝不让缺配置文件崩启动。

## 4. 验收

- [ ] `settings.LOCAL_MODELS_PATH` 等 6 个派生值 == 当前绝对值(逐一比对,bit 一致)。
- [ ] 改 `MODELS_ROOT` 一行 → 6 个里属于它的全部跟着变(单元验证)。
- [ ] backend 启动模型扫描自检数不变(diffusion_models/clip/vae/loras/checkpoint 计数同迁移前)。
- [ ] ruff + 相关单测(config / model_scanner / 启动)绿。
- [ ] 真机:`git pull` + 改 live `.env` 为两个根 + 重启,模型照常加载、出图 e2e 通。

## 5. 非目标

- 不动 `model_paths.yaml`(role→glob 层)。不改任何消费代码。
- 不引入第三个根;DB 连接(`DATABASE_URL`)/数据目录(pg cluster)与本设计无关。
- 不做 NAS 真远程存储(NAS_* 已并本地)。

## 6. 风险

- `config.py` 是中枢;`get_settings()` 走 `@lru_cache` + `settings.yaml` 覆盖。`@property` 不入
  pydantic 字段,不影响序列化/yaml 覆盖(yaml 只覆盖 `MODELS_ROOT`/`REPOS_ROOT` 这种真字段)。
- 单一 PR(config 改 + yaml 新增 + .env.example),真机验证后合。
