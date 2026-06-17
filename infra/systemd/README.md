# systemd 部署

把 `backend (uvicorn)` 和 `cloudflared tunnel` 跑成开机自启的 systemd 服务，
取代 `nohup ... & disown`。机器重启 / shell 退出后两个服务仍在跑。

## 一次性安装

```bash
sudo ./infra/systemd/install.sh
```

## 验证

```bash
systemctl status nous-backend nous-cloudflared
journalctl -u nous-backend -f         # 实时日志
```

## 修改后重启

改了 `backend/.env` 或 service 文件 → 重新加载 + 重启：

```bash
sudo systemctl daemon-reload          # 仅在改了 .service 文件后才需要
sudo systemctl restart nous-backend
```

## 卸载

```bash
sudo ./infra/systemd/install.sh uninstall
```

## 设计选择

- **没把 `vite dev` 服务化** — vite dev server 仅给本机开发用，生产路径走
  backend 直 serve `frontend/dist`（PR #32），不需要常驻。
- **`nous-cloudflared` `Requires=nous-backend` + `PartOf=nous-backend`** — Requires
  让 backend 死了就把 tunnel 也停掉（避免 cloudflare 把空 origin 挂域名上扔 502）。
  但 Requires 只单向传播「停」：`restart nous-backend` 会停掉隧道，backend 回来后隧道
  **不会**自动跟着起 → 每次发版都静默掐断公网（2026-06-16 踩到的 530）。`PartOf` 补上
  这个：隧道**跟随** backend 的 restart/stop，重启后端隧道自动回来。
- **`MemoryHigh=88G` + `MemoryMax=96G`** on backend — vLLM 之类有 OOM 史，给 host
  留余地。早期是 8G，但 V1' Lane A 之后 wikeeyang fp8mixed 的 dequant 中间态会冲到
  ~18G，8G 上限会让加载在没有 journal 输出的情况下被 SIGKILL。后来 64G —— 但
  Ideogram-4 bf16 整装整模型(54G,from_pretrained + stream pin)峰值 ~76G，64G 会在
  加载中途把整个后端 SIGKILL（2026-06-13 活机 e2e 坐实，连带 vLLM 被一起带走）。
  主机是 125G RAM，96G 能装下该峰值且仍留 ~29G 给 OS + cloudflared + vLLM host 侧。
  单文件 fp8 路(~18G)不受影响。再大的模型/并发多模型可继续上调，留 host ~25G 余地即可。
- **`StartLimitIntervalSec=0` + `RestartSec=15`** on backend — systemd 默认「10s 内
  重启 5 次就彻底放弃」，启动期 OOM/瞬时崩溃环最易触发,触发后服务永久下线、无人知晓。
  单管理员推理机宁可一直重试也不要静默死 → 关掉重启上限,RestartSec=15 拉慢重试防刷屏。
- **`OOMScoreAdjust=-500`** on backend — host 全局 OOM 时让内核优先杀别的(桌面/临时
  工具),保住推理后端(本机核心服务)。注意只影响 host 级「选谁杀」;后端自己超
  `MemoryMax=96G` 时仍由 cgroup OOM 在本 cgroup 内处理,与此无关。
- **`--protocol http2`** for cloudflared — 国内某些 ISP 屏蔽 UDP/7844 (QUIC)，
  http2 是已知能 work 的回落。
- **`nous-healthprobe.timer`(每 2 分钟)** — 本地健康巡检(`infra/monitoring/
  nous-healthprobe.sh`):探后端本机存活(`/healthz`)、后端自报健康(`/health` 的
  database / load_failures)、**公网隧道存活**(`<public>/health` 非 530/000)。结果进
  journal(`journalctl -u nous-healthprobe`)。硬故障(后端连不上 / DB 挂 / 隧道 down)
  退出非 0 → unit 标 failed,将来接告警只需给探针 service 加 `OnFailure=<alert>.service`。
  **为何要它**:systemd 的 `active` 会骗人 —— 2026-06-16 cloudflared 进程 `active` 但
  edge 连接掉到 0、公网 530,只有真正打一发 HTTP 才看得出来。**不报裸 status==degraded**
  (Lane-K llm runner supervisor 常驻 running:false → 恒 degraded,但 vLLM 独立 spawn、
  LLM 服务正常 → 报它纯噪声)。`NOUS_PUBLIC_URL` / `NOUS_LOCAL_URL` / `NOUS_PROBE_TIMEOUT`
  可覆盖。
- **隧道自愈(探针内置)** — cloudflared 在烂网络上会进「半开僵尸」:进程 `active`、
  edge 连接全死且**它自己不重连**(2026-06-17 卡死 2.5h,systemd 全程 active、journal 静默
  2.5h、`tunnel info` 报 0 connection)。systemd `active` 检测不到,只有真打 HTTP 过 edge
  才看得出。探针探到「**后端本机健康 但 公网持续 530/502/000**」连续 `NOUS_AUTOHEAL_THRESHOLD`
  (默认 2,×2min)次 → `sudo systemctl restart nous-cloudflared` 自愈,reset streak(自带
  ~4min 冷却防 restart 风暴)。**只在「后端活、唯独隧道死」时动手**——后端本身挂了重启隧道
  没用,不碰。授权靠 `infra/security/nous-healthprobe.sudoers`(install.sh 装 `/etc/sudoers.d/
  nous-healthprobe` 0440 + visudo 校验):只给 heygo 无密码 `systemctl restart nous-cloudflared`
  这一条。`NOUS_TUNNEL_AUTOHEAL=0` 关自愈退回只巡检+日志。没装 sudoers 时探针记 `[HEAL-FAIL]`
  不崩。
- **`nous-status`(独立公开状态监控)** — `infra/monitoring/status_service.py`,**纯 stdlib
  / 独立进程 / 独立端口(127.0.0.1:8001)/ 独立 unit**,刻意不 `Requires`/`PartOf`
  nous-backend —— **后端进程挂了它还活着、显示「后端 API:中断」**(对齐 status.claude.ai
  是独立平台,不是系统模块)。用系统 `/usr/bin/python3` 跑(不碰 backend venv/torch)。自己
  跑 `nvidia-smi` + 读 `/proc` 拿硬件(每卡显存/利用/温度、CPU%、内存、负载、uptime),
  `urllib` 探 `<backend>/health` 拿组件在线/离线。公开无登录,只露硬件概况 + 组件在线/离线,
  不露模型路径/密钥/内部错误。`/`(HTML 自动刷新 15s)、`/api.json`、`/healthz`。
  与 SPA 内 admin 状态页(#547,`/status` 详细版)并存:一个对外独立监控、一个登录后详查。

### 公网暴露 nous-status(cloudflared,需 cloudflared auth)

走独立子域,避免和 admin SPA 的 `/status` 路径冲突。编辑 `~/.cloudflared/config.yml`,在
catch-all(`http_status:404`)**之前**加一条 ingress:

```yaml
ingress:
  - hostname: api.iocrazy.com
    service: http://localhost:8000
  - hostname: status.iocrazy.com      # 新增:独立监控
    service: http://localhost:8001
  - service: http_status:404
```

再建 DNS 路由 + 重启隧道:

```bash
cloudflared tunnel route dns nous-center status.iocrazy.com
sudo systemctl restart nous-cloudflared
```

之后 `https://status.iocrazy.com` 公开可访问(无需登录)。

## 日志

systemd 走 journald，不再写 `/tmp/backend.log`。日志自动轮转，磁盘可控。

```bash
journalctl -u nous-backend --since '1 hour ago'
journalctl -u nous-cloudflared -p err               # 仅 ERROR
journalctl -u nous-backend --vacuum-time=7d         # 仅保留 7 天
```
