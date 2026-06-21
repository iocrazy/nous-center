# 专用生产检出(nous-prod)

生产与开发**检出分离**(2026-06-21)。背景:systemd `nous-backend` 等 unit 的
`WorkingDirectory`/`ExecStart` 写死指向某个 git 检出,后端从那里 `import src.*` 并
serve `frontend/dist`。如果生产和 dev 共用同一个检出,多个 dev session 切分支 / 改
working tree 就会和发版互相抢检出(部署到半成品代码、或被迫等别人交还检出)。

解法:一个**专用生产检出** `…/_playground/nous-prod`,systemd 只指它;所有 dev
session 各用自己的 worktree,谁都不碰生产。

## 布局

| 路径 | 角色 |
|---|---|
| `…/_playground/nous-center` | dev 主检出(可被任意 session 切分支) |
| `…/_playground/nous-prod`   | **生产专用**,detached、只读、deploy 脚本独占 |

systemd unit(`infra/systemd/*.service`)里的绝对路径都指向 `nous-prod`。

## 真相源(不随检出走的东西)

- **secret**:`nous-prod/backend/.env` → symlink 到 `nous-center/backend/.env`(单一来源)。
- **resident/gpu/vram 运行时覆盖**:在 **Postgres**(`runtime_override_store`,启动 hydrate),
  不是文件 → prod 用同一个 DB(`DATABASE_URL`),状态天然保留。
- `node_modules` / `.venv` / `dist`:gitignore,每个检出各自一份(deploy 的 `reset --hard`
  不动 untracked/ignored 文件)。

## 发版

只在 `nous-prod` 跑:

```bash
cd …/_playground/nous-prod && ./infra/deploy.sh
```

`deploy.sh` 凭 `nous-prod/.nous-production` 标记放行(dev 检出没这个标记 → 直接拒绝,
防止在 dev 检出误 `reset --hard` 卷走未提交改动)。脚本 `git fetch + reset --hard
origin/master` → 前端 build(dist 时间戳校验)→ `sudo systemctl restart nous-backend`
→ 自检 /healthz(本机 + 公网)。

## 一次性搭建(已执行,留档以备重建 / Phase2 迁盘)

```bash
ROOT=/media/heygo/Program/projects-code/_playground
# 1. 建专用生产检出(worktree,detached 在 origin/master)
cd "$ROOT/nous-center"
git worktree add --detach "$ROOT/nous-prod" origin/master
# 2. 标记 + 单一来源 .env
touch "$ROOT/nous-prod/.nous-production"
ln -s "$ROOT/nous-center/backend/.env" "$ROOT/nous-prod/backend/.env"
# 3. 前端依赖 + 首次 build
cd "$ROOT/nous-prod/frontend" && npm ci && npm run build
# 4. 安装重指后的 unit + 重启
cd "$ROOT/nous-prod" && sudo ./infra/systemd/install.sh
sudo systemctl daemon-reload
sudo systemctl restart nous-backend nous-status nous-healthprobe
```
