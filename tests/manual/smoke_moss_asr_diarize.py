#!/usr/bin/env python
"""Standalone smoke: MOSS-Transcribe-Diarize 0.9B — transcription + diarization + timestamps.

真机验证(非 CI):确认本地已安装的 MOSS-Transcribe-Diarize 能加载、转写、出时间戳、
并在多说话人音频上产出 [S01]/[S02] 说话人标注。走 transformers 原生路径(trust_remote_code),
**不碰 vLLM**——MOSS 是自定义架构 MossTranscribeDiarizeForConditionalGeneration,prod 钉死的
vLLM 0.22 registry 不认它;本 smoke 只用 transformers 直跑,零安装。

用法(用 prod venv 的 python 直接跑,只 import、不装任何包):
    CUDA_DEVICE_ORDER=PCI_BUS_ID SMOKE_DEVICE=cuda:2 \
        backend/.venv/bin/python tests/manual/smoke_moss_asr_diarize.py

    # 有真多人录音时(才是真正的 diarization 质量判据):
    ... tests/manual/smoke_moss_asr_diarize.py --audio /path/to/meeting.wav

关键约束(踩过的坑):
  - **import torch 前必须 CUDA_DEVICE_ORDER=PCI_BUS_ID**,否则 torch FASTEST_FIRST 把
    Pro 6000 排到 cuda:0、3090 错位 → 选错卡 / OOM。脚本顶部已 setdefault,命令再显式带一遍最稳。
  - 默认 SMOKE_DEVICE=cuda:2 = 空闲的第二块 3090(PCI 序);别用 cuda:1(Pro 6000,勿扰生产)。
  - 只有单说话人样本 assets/voices/default_zh_female.wav → 脚本用 ffmpeg 变调合成一段
    [A][B][A] 双说话人音频来 exercise diarization 头;真质量判据仍需喂真多人录音。
"""

from __future__ import annotations

# —— 必须在 import torch 之前 ——
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

REPO = Path(__file__).resolve().parents[2]  # tests/manual/<file> → repo root
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", REPO.parent / "models"))
DEFAULT_MODEL = MODELS_ROOT / "nous" / "speech" / "MOSS-Transcribe-Diarize"
DEFAULT_AUDIO = REPO / "assets" / "voices" / "default_zh_female.wav"
SCRATCH = Path(
    os.environ.get(
        "SMOKE_SCRATCH",
        "/tmp/claude-1000/-media-heygo-program-projects-code-repos-nous-center/scratch-moss",
    )
)

# README 记录的默认提示:带时间戳 + 说话人编号的转写
DEFAULT_PROMPT = (
    "请将音频转写为文本,每一段需以起始时间戳和说话人编号([S01]、[S02]、[S03]…)开头,"
    "正文为对应的语音内容,并在段末标注结束时间戳,以清晰标明该段语音范围。"
)

# 热词提示后缀(README 记录:默认提示后追加「热词提示:词1, 词2, ...」)
HOTWORD_SUFFIX = "热词提示:{words}"

# 输出规范:[start_time][Sxx]text[end_time] 反复拼接
SEGMENT_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]\[(S\d+)\](.*?)\[(\d+(?:\.\d+)?)\]", re.DOTALL)


def build_prompt(hotwords: list[str] | None = None) -> str:
    if hotwords:
        return DEFAULT_PROMPT + HOTWORD_SUFFIX.format(words=", ".join(hotwords))
    return DEFAULT_PROMPT


def plain_text(raw: str) -> str:
    """剥掉时间戳/说话人标注,只留正文,便于对比热词是否改变了转写用字。"""
    return "".join(seg[2].strip() for seg in SEGMENT_RE.findall(raw)) or raw.strip()


def hms(seconds: str | float) -> str:
    """秒 → mm:ss(<1h)或 h:mm:ss(≥1h),对齐前端 transcript 的 [mm:ss] 展示。"""
    total = int(round(float(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def log(msg: str) -> None:
    print(msg, flush=True)


def _neutralize_mistral_regex_patch() -> None:
    """绕开 prod transformers 5.6.0.dev0 的 bug:大词表(Qwen2 151k > 100000)会误触
    `_patch_mistral_regex`,而该 Mistral 专用分支 crash 在 `tokenizer.backend_tokenizer`
    (raw tokenizers.Tokenizer 无此属性)。Qwen2 不需要该修正 → 直接原样返回 tokenizer。
    """
    try:
        from transformers import tokenization_utils_tokenizers as tut
    except Exception:
        return
    backend = getattr(tut, "TokenizersBackend", None)
    if backend is not None and hasattr(backend, "_patch_mistral_regex"):
        backend._patch_mistral_regex = lambda self, tokenizer, *a, **k: tokenizer


def load_audio_16k_mono(path: Path) -> np.ndarray:
    wav, _ = librosa.load(str(path), sr=16000, mono=True)
    return np.asarray(wav, dtype=np.float32)


def synth_two_speaker(src: Path, out: Path) -> Path:
    """把单说话人样本变调造出「第二个人」,拼成 [A][gap][B][gap][A]。

    纯 smoke 用途:同一段话的原声 + 降调声 + 原声,让 diarization 头有两种音色可分。
    真质量判据仍需真多人录音。
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    # 单条 concat 滤镜命令:A + gap + B(变调) + gap + A,全部归一化到 16k/mono/s16。
    # 用 concat *滤镜*(非 demuxer)避免各片段编码/采样格式不一致导致的坏包。
    # [b] = asetrate 降采样率→降调 + atempo 补回时长,得到听感不同的「B」。
    filt = (
        "[0:a]aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono[a];"
        "[0:a]asetrate=16000*0.82,aresample=16000,atempo=1.2195,"
        "aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono[b];"
        "anullsrc=r=16000:cl=mono,atrim=0:0.35,"
        "aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono[g1];"
        "anullsrc=r=16000:cl=mono,atrim=0:0.35,"
        "aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono[g2];"
        "[a][g1][b][g2][a]concat=n=5:v=0:a=1[out]"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-filter_complex", filt, "-map", "[out]",
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out)],
        check=True,
    )
    return out


def transcribe(model, processor, device, dtype, audio: np.ndarray, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": "placeholder"},  # 占位符,渲染 <|audio_pad|>;实际波形经 processor(audio=) 传
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, audio=audio)

    moved = {}
    for k, v in inputs.items():
        if not torch.is_tensor(v):
            moved[k] = v
            continue
        moved[k] = v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)

    prompt_len = moved["input_ids"].shape[1]
    with torch.inference_mode():
        out_ids = model.generate(**moved, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = out_ids[0, prompt_len:]
    decoded = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()


def report(title: str, raw: str) -> dict:
    log(f"\n===== {title} =====")
    log("---- 原始输出 ----")
    log(raw or "(空)")
    segs = SEGMENT_RE.findall(raw)
    speakers = sorted({s[1] for s in segs})
    log("---- 解析后的分段 ----")
    if not segs:
        log("(未解析出 [start][Sxx]text[end] 分段——检查输出格式)")
    for start, spk, seg_text, end in segs:
        log(f"  [{hms(start)} → {hms(end)}] {spk}: {seg_text.strip()}")
    log(f"---- 小结: {len(segs)} 段, 说话人 {speakers or '无'} ----")
    return {"segments": len(segs), "speakers": speakers, "chars": len(raw)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--audio", type=Path, default=None, help="真多人录音路径;缺省用内置单人样本 + 合成双人")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--hotwords", type=lambda s: [w.strip() for w in s.split(",") if w.strip()],
                    default=None, help="逗号分隔的热词;缺省跑内置 steering/robustness 两个探针")
    ap.add_argument("--skip-synth", action="store_true", help="不合成双人音频")
    args = ap.parse_args()

    device_str = os.environ.get("SMOKE_DEVICE", "cuda:2")
    if not args.model.exists():
        log(f"[FATAL] 模型不存在: {args.model}")
        return 2
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        log("[WARN] CUDA_DEVICE_ORDER != PCI_BUS_ID,设备编号可能错位")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    log(f"[env] torch={torch.__version__} device={device} dtype={dtype}")
    if device.type == "cuda":
        log(f"[env] GPU = {torch.cuda.get_device_name(device)}")

    t0 = time.time()
    log(f"[load] {args.model} ...")
    model = (
        AutoModelForCausalLM.from_pretrained(str(args.model), trust_remote_code=True, dtype="auto")
        .to(dtype=dtype)
        .to(device)
        .eval()
    )
    _neutralize_mistral_regex_patch()
    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)
    log(f"[load] done in {time.time() - t0:.1f}s")
    if device.type == "cuda":
        log(f"[vram] {torch.cuda.memory_allocated(device) / 1e9:.2f} GB allocated")

    results = {}

    # 1) 单人真样本 —— 验加载/转写/时间戳
    single_src = args.audio if args.audio else DEFAULT_AUDIO
    if not single_src.exists():
        log(f"[FATAL] 音频不存在: {single_src}")
        return 2
    single_audio = load_audio_16k_mono(single_src)
    t = time.time()
    raw = transcribe(model, processor, device, dtype, single_audio,
                     build_prompt(), args.max_new_tokens)
    log(f"[infer] 单人 {single_src.name} 用时 {time.time() - t:.1f}s")
    results["single"] = report(f"单说话人: {single_src.name}", raw)
    baseline_text = plain_text(raw)

    # 1b) 热词 / context 偏置 —— 验 promptable 槽是否生效
    #  probe A(steering): 该样本存在 得/的 同音歧义,喂热词看能否 steer 用字。
    #  probe B(robustness): 喂音频中不存在的人名,看是否被无脑塞进转写(好模型不该幻觉)。
    #  kind: steering=期望改变用字; robustness=期望不被硬塞; custom=用户自定义
    #  内置探针是针对短样本标定的,仅在没传 --audio 时跑;传了 --audio 只跑用户显式热词
    if args.hotwords:
        hotword_probes = [("自定义", "custom", args.hotwords)]
    elif args.audio is None:
        hotword_probes = [("得/的 同音", "steering", ["做的"]),
                          ("缺席人名", "robustness", ["李华", "王芳"])]
    else:
        hotword_probes = []
    log("\n===== 热词 / context 偏置 =====")
    log(f"[baseline 无热词] 正文: {baseline_text}")
    hw_summary = []
    for tag, kind, words in hotword_probes:
        t = time.time()
        raw_hw = transcribe(model, processor, device, dtype, single_audio,
                            build_prompt(words), args.max_new_tokens)
        txt = plain_text(raw_hw)
        changed = txt != baseline_text
        injected = any(w in txt for w in words)
        note = {"steering": "← steer 生效" if changed else "未 steer",
                "robustness": "⚠️硬塞了缺席词" if injected else "✓未幻觉",
                "custom": "← 变了" if changed else "= 同 baseline"}[kind]
        log(f"[热词={words}] ({tag}/{kind}, {time.time() - t:.1f}s) 正文: {txt}  {note}")
        hw_summary.append({"tag": tag, "kind": kind, "changed": changed, "injected": injected})
    results["hotwords"] = hw_summary

    # 2) 合成双人 —— 验 diarization 头(仅在没提供真多人录音时)
    if args.audio is None and not args.skip_synth:
        try:
            synth = synth_two_speaker(DEFAULT_AUDIO, SCRATCH / "two_speaker.wav")
            t = time.time()
            raw2 = transcribe(model, processor, device, dtype, load_audio_16k_mono(synth),
                              build_prompt(), args.max_new_tokens)
            log(f"[infer] 合成双人 用时 {time.time() - t:.1f}s")
            results["synthetic_2spk"] = report("合成双说话人 [A][B][A](变调造 B)", raw2)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            log(f"[WARN] 合成双人音频失败(ffmpeg?): {e}")

    # —— 判据 ——
    log("\n===== 判据 =====")
    s = results.get("single", {})
    ok_load = s.get("chars", 0) > 0
    ok_ts = s.get("segments", 0) > 0
    log(f"  [{'✅' if ok_load else '❌'}] 模型加载 + 出文本")
    log(f"  [{'✅' if ok_ts else '❌'}] 单人出时间戳分段 ({s.get('segments', 0)} 段)")
    hw = results.get("hotwords") or []
    if hw:
        robustness = [h for h in hw if h["kind"] == "robustness"]
        steering = [h for h in hw if h["kind"] == "steering"]
        no_hallucinate = all(not h["injected"] for h in robustness)
        steered = any(h["changed"] for h in steering)
        log(f"  [✅] 热词 prompt 槽可用 (跑了 {len(hw)} 个探针)")
        if robustness:
            log(f"  [{'✅' if no_hallucinate else '⚠️ '}] 不把缺席热词硬塞进转写 (robustness)")
        if steering:
            log(f"  [{'✅' if steered else 'ℹ️ '}] 热词能 steer 同音用字 (弱样本;真判据需带领域术语的音频)")
    two = results.get("synthetic_2spk")
    if two is not None:
        multi = len(two.get("speakers", [])) >= 2
        log(f"  [{'✅' if multi else '⚠️ '}] 合成双人识别出 ≥2 说话人 (识别到 {two.get('speakers')})"
            + ("" if multi else " — 合成音频是弱判据,真结论请喂真多人录音"))
    log("\n真 diarization 质量判据:--audio 喂一段真多人录音再看 cpCER / 说话人切分是否对齐。")
    return 0 if (ok_load and ok_ts) else 1


if __name__ == "__main__":
    sys.exit(main())
