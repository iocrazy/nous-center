from sqlalchemy import select

from src.models.tts_usage import TTSUsage
from src.services.usage_recorder import record_tts_usage


async def test_record_tts_usage(db_session):
    await record_tts_usage(
        session=db_session,
        engine="cosyvoice2",
        characters=10,
        duration_ms=2000,
        rtf=0.5,
        cached=False,
    )
    result = await db_session.execute(select(TTSUsage))
    row = result.scalar_one()
    assert row.engine == "cosyvoice2"
    assert row.characters == 10
