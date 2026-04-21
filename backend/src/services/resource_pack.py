"""ResourcePack quota service — atomic consume + pack selection.

Why the atomic UPDATE (critical gap #1 from the 2026-04-21 plan review):

  Two concurrent requests each calling consume(100) on a pack with 150 left
  must NOT both succeed. A naive read-then-write loses the race:

     req A: read used=0 total=150
     req B: read used=0 total=150
     req A: write used=100 (ok)
     req B: write used=100 (WRONG: pack now over-consumed)

  The fix is a single SQL UPDATE with the budget check in the WHERE clause.
  If the row doesn't match (insufficient units, expired, deleted), the
  UPDATE affects 0 rows and the caller knows it lost the race. No locks,
  no application-level retry loops, no races.

The selection policy: consume from the pack expiring soonest. Users with
multiple packs get FIFO; packs without expiry sort last. consume() does
NOT span packs: if you want to charge 100 units and no single pack has
>=100 free, it fails. Splitting across packs adds code + tests for no
real user benefit at internal-beta scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.api_gateway import ResourcePack


def _naive_utc(dt: datetime | None) -> datetime | None:
    """SQLite strips tzinfo on storage, so DB reads come back naive. Unify
    application-side datetimes to naive UTC before any comparison so the
    code runs identically on both PG and SQLite."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class QuotaExhausted(Exception):
    """No pack with enough remaining units."""


@dataclass
class ConsumeResult:
    pack_id: int
    remaining_units: int  # total - used on the chosen pack, AFTER consume


async def consume(
    session: AsyncSession,
    *,
    grant_id: int,
    units: int,
    now: datetime | None = None,
) -> ConsumeResult:
    """Atomically charge `units` against the best-fit pack on `grant_id`.

    Raises QuotaExhausted if no single pack has >= `units` free, or if the
    only eligible packs are expired.
    """
    if units <= 0:
        # Consuming 0/negative is a no-op — surface loudly, don't silently
        # accept garbage from the caller.
        raise ValueError(f"units must be positive, got {units}")

    now = _naive_utc(now or datetime.now(timezone.utc))

    # Pick packs in FIFO-by-expiry order. null expiry is last (NULLS LAST).
    # We select IDs only; the atomic UPDATE below is what actually charges.
    from sqlalchemy import nulls_last
    stmt = (
        select(ResourcePack.id)
        .where(
            ResourcePack.grant_id == grant_id,
            (ResourcePack.expires_at.is_(None)) | (ResourcePack.expires_at > now),
            ResourcePack.total_units - ResourcePack.used_units >= units,
        )
        .order_by(nulls_last(ResourcePack.expires_at.asc()))
    )
    candidates = (await session.execute(stmt)).scalars().all()

    for pack_id in candidates:
        # Atomic compare-and-swap: the WHERE clause is the budget check.
        # If a racing consumer won the pack since we read the candidate
        # list, the UPDATE affects 0 rows and we try the next candidate.
        upd = (
            update(ResourcePack)
            .where(
                ResourcePack.id == pack_id,
                ResourcePack.total_units - ResourcePack.used_units >= units,
                (ResourcePack.expires_at.is_(None)) | (ResourcePack.expires_at > now),
            )
            .values(used_units=ResourcePack.used_units + units)
        )
        result = await session.execute(upd)
        if result.rowcount == 1:
            # Win. Read back the post-state for the return value.
            await session.commit()
            row = await session.get(ResourcePack, pack_id)
            return ConsumeResult(
                pack_id=pack_id,
                remaining_units=row.total_units - row.used_units,
            )
        # Race loser — keep trying.

    await session.rollback()
    raise QuotaExhausted(f"grant {grant_id}: no pack has {units} units free")


async def peek_remaining(
    session: AsyncSession, *, grant_id: int, now: datetime | None = None
) -> int:
    """Sum of remaining units across all non-expired packs for a grant.

    Read-only; use for dashboards and pre-flight "would this request fit".
    Note that peek + consume is not atomic — consume is the source of truth.
    """
    now = _naive_utc(now or datetime.now(timezone.utc))
    stmt = select(
        (ResourcePack.total_units - ResourcePack.used_units)
    ).where(
        ResourcePack.grant_id == grant_id,
        (ResourcePack.expires_at.is_(None)) | (ResourcePack.expires_at > now),
    )
    rows = (await session.execute(stmt)).scalars().all()
    return sum(max(0, r) for r in rows)
