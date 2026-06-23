# 格式化重建 SOP — 从裸机到全栈在跑

重装系统(或换盘格式化)后,把 nous-center 带回「全栈在跑」的标准流程。
核心工具是 **`infra/bootstrap.sh`** 编排器(设计见
`docs/superpowers/specs/2026-06-23-fresh-format-bootstrap-design.md`)。

## ⚠️ 关键前提:全量 bootstrap 要在 **nous-prod 检出**里跑

systemd unit(`infra/systemd/*.service`)把检出路径**写死成 `…/nous-prod`**
(`WorkingDirectory`/`ExecStart`)。所以不管你在哪跑 bootstrap,systemd 永远 serve
`nous-prod`。**provision 的检出必须 == 被 serve 的检出**,否则装出的服务起不来
(services 段现在会校验:`nous-prod` 无 venv 直接拒装并提示)。

> 结论:把生产检出 clone/worktree 成 `…/nous-prod`,**在它里面**跑全量 bootstrap。
> dev 检出(`nous-center`)是可选的、事后再加(`git worktree`)。

## TL;DR

```bash
ROOT=/media/heygo/Program/projects-code/_playground

# 0. 装系统级前提(bootstrap 不代装,见下「OS 层」)
# 1. clone 成生产检出 nous-prod(systemd 指向它)
git clone git@github.com:iocrazy/nous-center.git "$ROOT/nous-prod"
cd "$ROOT/nous-prod"
touch .nous-production            # deploy.sh 凭它放行

# 2. 体检:一屏看清缺什么(只读,零改动)
./infra/bootstrap.sh --check

# 3. 全量一键(需 root;幂等,已就位即跳过)—— 在 nous-prod 里跑
sudo ./infra/bootstrap.sh
#    若 .env 是新建的 → 脚本会停下让你填 DATABASE_URL 密码等,填完重跑
#    若要从备份恢复 DB:sudo ./infra/bootstrap.sh --restore /path/to/nous_center.dump

# 4.(可选)事后加 dev 检出,各用各的 worktree,不抢生产
git -C "$ROOT/nous-prod" worktree add "$ROOT/nous-center" master
```

跑完 `--check` 应全 OK,公网 `https://api.iocrazy.com/healthz` 返回 200。

## OS 层(bootstrap 不代装,只 `--check` 报缺)

格式化后先就位这些,bootstrap 才能接手:

- **NVIDIA 驱动 + CUDA**:`nvidia-smi` 能列出 3 张卡。注意 PRO 6000 Blackwell GSP
  固件 bug —— boot 不要常驻大模型,显示器**别接 GPU0 以外**的逻辑见
  `hardware.yaml` 注释与 GSP 缓解留档(#604)。
- **大盘挂载**:`/media/heygo/Program`(模型权重 + DB 备份落此盘)。
- **基础 CLI**:`git curl openssl`,以及 `uv`(Python)、`node`/`npm`(前端)。
- **systemd**(发行版自带)。

## 机器特定 secret(bootstrap 不伪造,必须人工带入)

这些无法 commit、无法脚本生成,bootstrap 检测到缺会标「需人工」并指源:

- **`backend/.env` 的 `DATABASE_URL` 密码** 等机器特定值 —— 从 `.env.example` 起后人工填。
  admin secret(`ADMIN_PASSWORD`/`ADMIN_SESSION_SECRET`)bootstrap 会用
  `gen-admin-secrets.sh` 自动补缺。
- **cloudflared 隧道凭证**(`~/.cloudflared/cert.pem` + `<tunnel>.json` + `config.yml`)
  —— 从备份盘复制,或 `cloudflared tunnel login` 重新授权。
- **DB 数据** —— 靠最近一次 `nous-dbbackup` 的 dump,`--restore <dump>` 恢复;不传则
  建空库,backend 首启 `create_all` 自建 schema(单管理员、无 alembic)。
- **模型权重** —— 在大盘、不入 git。格式化若保留大盘则无需重下。

## bootstrap 阶段一览

`sudo ./infra/bootstrap.sh` 依次跑(每段幂等,已就位即跳过):

| 阶段 | 做什么 | 权限 |
|---|---|---|
| preflight | OS/盘/驱动/CLI 体检(只读) | 任意 |
| secrets   | `.env`(模板+admin secret)+ 检测 cloudflared 凭证 | root |
| db        | apt 装 pg17(自动加 PGDG 源)+ 建 role/库 + 可选 `--restore` | root |
| deps      | 后端 `uv sync --extra inference` + aligner venv | 真实用户(非 root) |
| build     | 前端 `npm ci` + `npm run build` | 真实用户 |
| checkout  | 派生专用 prod worktree + 标记 + `.env` symlink | 真实用户 |
| services  | `install.sh`(单元+nousctl+sudoers+enable)+ healthz 自检 | root |

单独重跑某段:`sudo ./infra/bootstrap.sh --stage db`。

## 生产/dev 检出分离(可选但推荐)

systemd 单元指向**专用 prod 检出** `…/nous-prod`(detached,deploy 独占),dev 用
`nous-center`,互不抢检出。详见 [PROD_CHECKOUT.md](PROD_CHECKOUT.md)。

要点:**venv/dist 是 per-checkout 的 gitignore 物**。所以 prod 检出建好后,要在
**它内部**再跑一遍 deps/build(即 `cd …/nous-prod && sudo ./infra/bootstrap.sh`),
否则 systemd 起 backend 时缺 vllm → 常驻 0/N。

## 验收

```bash
./infra/bootstrap.sh --check                       # 应全 OK
systemctl is-active nous-backend nous-cloudflared nous-status nous-aligner
curl -s --noproxy '*' https://api.iocrazy.com/healthz   # {"status":"ok"}
nousctl status                                     # 全栈一览
```
