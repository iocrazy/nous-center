"""Structured log tables — moved from the standalone SQLite ``log_db`` into the
main PostgreSQL database (spec 2026-06-10). One DB to manage.

Four log kinds mirror the original ``log_db.py`` SQLite schema 1:1. The
``timestamp`` column stays a CST string (``"%Y-%m-%d %H:%M:%S"``) on purpose:
the frontend LogsOverlay parses that exact format and the ``since`` filter does
a fixed-width lexicographic compare — keeping it a string ports the read
contract verbatim. Ordering is by ``id`` (monotonic autoincrement), not the
timestamp, so a string column loses nothing here.
"""
from sqlalchemy import BigInteger, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base

# BIGINT doesn't autoincrement on SQLite (only INTEGER PRIMARY KEY does), but the
# test suite runs on SQLite. with_variant → BIGINT on PG (prod), INTEGER on SQLite
# (tests) so the id autoincrements on both.
_AutoId = BigInteger().with_variant(Integer, "sqlite")


class RequestLog(Base):
    __tablename__ = "request_logs"

    id: Mapped[int] = mapped_column(_AutoId, primary_key=True, autoincrement=True)
    timestamp: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    method: Mapped[str] = mapped_column(String(10), nullable=False)
    path: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    ip: Mapped[str | None] = mapped_column(String(45), default="")
    user_agent: Mapped[str | None] = mapped_column(String(500), default="")


class AppLog(Base):
    __tablename__ = "app_logs"

    id: Mapped[int] = mapped_column(_AutoId, primary_key=True, autoincrement=True)
    timestamp: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    level: Mapped[str] = mapped_column(String(10), nullable=False)
    module: Mapped[str | None] = mapped_column(String(100), default="")
    message: Mapped[str | None] = mapped_column(Text, default="")
    location: Mapped[str | None] = mapped_column(String(200), default="")


class FrontendLog(Base):
    __tablename__ = "frontend_logs"

    id: Mapped[int] = mapped_column(_AutoId, primary_key=True, autoincrement=True)
    timestamp: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(30), nullable=False)
    message: Mapped[str | None] = mapped_column(Text, default="")
    page: Mapped[str | None] = mapped_column(String(500), default="")
    stack: Mapped[str | None] = mapped_column(Text, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(_AutoId, primary_key=True, autoincrement=True)
    timestamp: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    path: Mapped[str | None] = mapped_column(String(500), default="")
    method: Mapped[str | None] = mapped_column(String(10), default="")
    ip: Mapped[str | None] = mapped_column(String(45), default="")
    detail: Mapped[str | None] = mapped_column(Text, default="")
