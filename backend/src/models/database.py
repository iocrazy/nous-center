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


def get_session_factory():
    """进程内**共享**的 async session factory(memoized)。

    round4 #1:别再到处裸调 `create_session_factory()` —— 那每次都 `create_async_engine`
    新建一个 engine + 连接池且从不 dispose。usage_service(每次推理)、计费、workflow_runner
    等热/后台路径都这么调 → 连接池泄漏 + 无池化收益。这里 memoize 一个工厂复用同一 engine。
    需要独立 engine 的(测试 fixture)仍可显式 create_session_factory(engine=...)。
    """
    global _session_factory
    if _session_factory is None:
        _session_factory = create_session_factory()
    return _session_factory


async def get_async_session():
    """FastAPI dependency for async DB sessions。"""
    async with get_session_factory()() as session:
        yield session
