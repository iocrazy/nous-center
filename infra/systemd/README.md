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

## 日志

systemd 走 journald，不再写 `/tmp/backend.log`。日志自动轮转，磁盘可控。

```bash
journalctl -u nous-backend --since '1 hour ago'
journalctl -u nous-cloudflared -p err               # 仅 ERROR
journalctl -u nous-backend --vacuum-time=7d         # 仅保留 7 天
```
