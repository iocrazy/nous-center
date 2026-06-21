"""ASR 转写端点的纯逻辑单测(spec asr-context-lid):

chat 路径输出解析(语种 + 文本)+ 音频时长计量(自算)。端到端转写需真模型(见
tests/manual / 真机验证),这里只锁不依赖 GPU/vLLM 的纯函数。
"""
import io
import wave

from src.api.routes.openai_compat import _ASR_OUT_RE, _wav16k_seconds


def _parse(content: str):
    m = _ASR_OUT_RE.match(content)
    return (m.group(2).strip(), m.group(1).strip()) if m else (content.strip(), None)


def test_asr_output_parse_language_and_text():
    # Qwen3-ASR chat 原始输出:`language {LANG}<asr_text>{TEXT}`
    assert _parse("language Chinese<asr_text>希望你以后能够做得比我还好哟。") == (
        "希望你以后能够做得比我还好哟。", "Chinese",
    )
    assert _parse("language English<asr_text>Hello world.") == ("Hello world.", "English")


def test_asr_output_parse_fallback_no_marker():
    # 无 <asr_text> 标记(格式异常/未来变更)→ 整体当文本、language=None,不崩
    assert _parse("just plain text") == ("just plain text", None)
    assert _parse("") == ("", None)


def _make_wav16k(seconds: float) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # s16le
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return buf.getvalue()


def test_wav16k_seconds():
    assert _wav16k_seconds(_make_wav16k(4.0)) == 4
    assert _wav16k_seconds(_make_wav16k(0.3)) == 1  # 至少 1 秒计费
    # 损坏/非 wav 字节 → 退化估算不崩(至少 1)
    assert _wav16k_seconds(b"not a wav") >= 1
