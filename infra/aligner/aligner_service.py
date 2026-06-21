#!/usr/bin/env python3
"""nous-aligner — 独立的 ForcedAligner 时间戳微服务(spec asr-context-lid-timestamps,Arc B)。

为什么独立进程/独立 venv:`qwen-asr` 钉死 `transformers==4.57.6`,与 backend 的
transformers 5.6-dev / vllm 0.22 / torch 2.11 冲突(装一起会降级 transformers 砸坏
vllm/图像)。所以对齐器跟 nous-status 一样,独立进程 + 独立 venv + 独立端口 + 独立 unit,
和 backend 完全隔离。backend 仅在 timestamps=true 时 HTTP 调它;它挂了/没开,纯文本主路不受影响。

接口:
  GET  /healthz                       → 200 ok / 503 (模型未就绪)
  POST /align {audio_b64, text, lang} → {"words":[{"text","start","end"}], "n":N}

音频:audio_b64 = 16k/mono/s16le WAV 的 base64(backend 归一化后传来)。文本 = 已转写文本。
对齐器吃 (音频, 文本, 语言) 做强制对齐,输出每词/字 start/end 秒。

环境变量:
  NOUS_ALIGNER_MODEL   对齐器模型目录(默认 MODELS_ROOT/nous/speech/Qwen3-ForcedAligner-0.6B)
  NOUS_ALIGNER_DEVICE  cuda:N(默认 cuda:0;PCI_BUS_ID 序)
  NOUS_ALIGNER_PORT    监听端口(默认 8002)
  MODELS_ROOT          模型根(默认 /media/heygo/Program/models)
"""
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # 必须在 import torch 前(cuda:N 对齐 nvidia-smi)

import base64
import json
import logging
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s aligner %(levelname)s %(message)s")
log = logging.getLogger("aligner")

MODELS_ROOT = os.environ.get("MODELS_ROOT", "/media/heygo/Program/models")
MODEL_DIR = os.environ.get(
    "NOUS_ALIGNER_MODEL", f"{MODELS_ROOT}/nous/speech/Qwen3-ForcedAligner-0.6B"
)
DEVICE = os.environ.get("NOUS_ALIGNER_DEVICE", "cuda:0")
PORT = int(os.environ.get("NOUS_ALIGNER_PORT", "8002"))

_aligner = None
_lock = threading.Lock()  # align 不保证线程安全;串行化(单管理员 infra 够用)


def _load():
    global _aligner
    import torch
    from qwen_asr import Qwen3ForcedAligner

    log.info("loading ForcedAligner from %s on %s ...", MODEL_DIR, DEVICE)
    _aligner = Qwen3ForcedAligner.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map=DEVICE
    )
    log.info("ForcedAligner ready")


def _align(audio_wav: bytes, text: str, language: str | None):
    # qwen-asr align 接受 path/URL/base64/(ndarray,sr);写临时 wav 文件最稳。
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tf.write(audio_wav)
        path = tf.name
    try:
        with _lock:
            results = _aligner.align(audio=path, text=text, language=language or "Chinese")
        segs = results[0] if results else []
        return [
            {"text": s.text, "start": round(float(s.start_time), 3), "end": round(float(s.end_time), 3)}
            for s in segs
        ]
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # 静音默认访问日志(用我们的 logger)
        pass

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            self._send(200 if _aligner is not None else 503,
                       {"status": "ok" if _aligner is not None else "loading"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/align":
            self._send(404, {"error": "not found"})
            return
        if _aligner is None:
            self._send(503, {"error": "aligner not ready"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
            audio_b64 = payload.get("audio_b64")
            text = payload.get("text")
            language = payload.get("language")
            if not audio_b64 or not text:
                self._send(400, {"error": "audio_b64 and text required"})
                return
            words = _align(base64.b64decode(audio_b64), text, language)
            self._send(200, {"words": words, "n": len(words)})
        except Exception as e:  # noqa: BLE001
            log.exception("align failed")
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    _load()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    log.info("nous-aligner listening on 127.0.0.1:%d", PORT)
    srv.serve_forever()


if __name__ == "__main__":
    main()
