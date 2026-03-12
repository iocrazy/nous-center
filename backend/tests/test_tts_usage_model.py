from sqlalchemy import select

from src.models.tts_usage import TTSUsage


async def test_create_tts_usage(db_session):
    usage = TTSUsage(engine="cosyvoice2", characters=42, duration_ms=3200, rtf=0.8, cached=False)
    db_session.add(usage)
    await db_session.commit()

    result = await db_session.execute(select(TTSUsage))
    row = result.scalar_one()
    assert row.engine == "cosyvoice2"
    assert row.characters == 42
    assert row.duration_ms == 3200
    assert row.id > 0  # snowflake ID
