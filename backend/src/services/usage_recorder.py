from sqlalchemy.ext.asyncio import AsyncSession

from src.models.tts_usage import TTSUsage


async def record_tts_usage(
    session: AsyncSession,
    engine: str,
    characters: int,
    duration_ms: int | None = None,
    rtf: float | None = None,
    cached: bool = False,
) -> None:
    """Record a TTS synthesis event for usage statistics."""
    usage = TTSUsage(
        engine=engine,
        characters=characters,
        duration_ms=duration_ms,
        rtf=rtf,
        cached=cached,
    )
    session.add(usage)
    await session.commit()
