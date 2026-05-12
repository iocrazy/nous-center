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
- **`nous-cloudflared` `Requires=nous-backend`** — backend 死了就把 tunnel 也
  停掉，避免 cloudflare 把空 origin 挂在域名上扔 502 给用户。
- **`MemoryMax=64G`** on backend — vLLM 之类有 OOM 史，给 host 留余地。早期是
  8G，但 V1' Lane A 之后 wikeeyang fp8mixed 的 dequant 中间态会冲到 ~18G，
  8G 上限会让加载在没有 journal 输出的情况下被 SIGKILL。64G 在 96G 主机上
  留 ~30G 给 OS + cloudflared。如果你计划同时加载多个 image 模型，提高到
  `MemoryHigh=80G` + `MemoryMax=88G` 更稳。
- **`--protocol http2`** for cloudflared — 国内某些 ISP 屏蔽 UDP/7844 (QUIC)，
  http2 是已知能 work 的回落。

## 日志

systemd 走 journald，不再写 `/tmp/backend.log`。日志自动轮转，磁盘可控。

```bash
journalctl -u nous-backend --since '1 hour ago'
journalctl -u nous-cloudflared -p err               # 仅 ERROR
journalctl -u nous-backend --vacuum-time=7d         # 仅保留 7 天
```
