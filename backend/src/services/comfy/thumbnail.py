"""First-frame thumbnail extraction for video outputs (gallery preview).

Shells out to `ffmpeg` rather than pulling in a Python video decoding dep —
ffmpeg is the de-facto standard already assumed available on boxes that run
ComfyUI-adjacent media pipelines, but it's optional here: a box without it
just gets no thumbnail (caller falls back to a generic video icon), never a
hard failure of the whole workflow run.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


async def extract_first_frame(video_path: Path) -> bytes | None:
    """Return the first frame of `video_path` as PNG bytes, or None.

    Never raises: missing ffmpeg binary (FileNotFoundError) or a non-zero
    ffmpeg exit code (corrupt/unreadable input, unsupported codec, etc.)
    both degrade to None so gallery rendering never breaks on a bad video.
    """
    with tempfile.TemporaryDirectory(prefix="nous-thumb-") as tmp_dir:
        out_path = Path(tmp_dir) / "frame.png"
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", str(video_path),
                "-frames:v", "1", "-f", "image2", str(out_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            returncode = await proc.wait()
        except FileNotFoundError:
            logger.warning("extract_first_frame: ffmpeg not found on PATH")
            return None

        if returncode != 0:
            logger.warning(
                "extract_first_frame: ffmpeg exited %d for %s", returncode, video_path,
            )
            return None

        try:
            return out_path.read_bytes()
        except OSError:
            return None
