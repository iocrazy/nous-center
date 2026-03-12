from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase

from src.config import get_settings


class Base(DeclarativeBase):
    pass


def create_engine():
    settings = get_settings()
    return create_async_engine(settings.DATABASE_URL)


def create_session_factory(engine=None):
    if engine is None:
        engine = create_engine()
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


_session_factory = None


async def get_async_session():
    """FastAPI dependency for async DB sessions."""
    global _session_factory
    if _session_factory is None:
        _session_factory = create_session_factory()
    async with _session_factory() as session:
        yield session
