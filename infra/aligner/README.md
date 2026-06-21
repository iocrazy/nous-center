# nous-aligner — ForcedAligner 时间戳微服务

ASR 时间戳(词/字级 start/end)的独立微服务。spec
`docs/superpowers/specs/2026-06-21-asr-context-lid-timestamps-design.md`(Arc B)。

## 为什么独立进程 / 独立 venv

时间戳要 `Qwen3-ForcedAligner-0.6B` + `qwen-asr` 工具包,而 **`qwen-asr` 钉死
`transformers==4.57.6`**,和 backend 的 `transformers 5.6-dev` / vllm 0.22 冲突(装一起会
降级 transformers 砸坏 vllm/图像)。所以对齐器跟 `nous-status` 一样:独立进程、独立 venv、
独立端口、独立 systemd unit,和 backend 完全隔离。backend 仅在 `timestamps=true` 时 HTTP
调它;它挂了/没开,纯文本 ASR 主路不受影响(降级)。

## 接口

```
GET  /healthz                         → 200 {"status":"ok"} / 503 loading
POST /align {audio_b64, text, language} → {"words":[{"text","start","end"}], "n":N}
```
`audio_b64` = 16k/mono/s16le WAV 的 base64(backend 归一化后传)。

## 搭建(一次性 / 重建 prod 检出时)

```bash
# 1. 下模型(若没下)
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B \
  --local_dir $MODELS_ROOT/nous/speech/Qwen3-ForcedAligner-0.6B
# 2. 建独立 venv(qwen-asr,~5GB)
./infra/aligner/setup.sh
# 3. 装 + 起 systemd unit
sudo cp infra/systemd/nous-aligner.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now nous-aligner
# 4. 自检
curl -s http://127.0.0.1:8002/healthz
```

`.venv` 每检出一份(gitignore);GPU 默认 `cuda:0`(env `NOUS_ALIGNER_DEVICE` 可改),端口 8002。
