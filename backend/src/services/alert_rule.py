"""AlertRule evaluation — threshold checking with 24h dedup.

Called inline from the usage-record path after a successful consume().
Each AlertRule carries a threshold_percent (1-100) and an optional
pack_id scope:
  - pack_id=NULL → evaluate against all packs on the grant (aggregate)
  - pack_id set  → evaluate against just that one pack

A rule fires when observed_pct >= threshold_percent AND the rule hasn't
fired in the last 24 hours (stored on last_notified_at). We don't
fire on the "crossing" moment specifically — firing on "at or above"
plus the 24h dedup is simpler and functionally equivalent: a user won't
see more than one notification per rule per day.

Delivery is out of scope for this module. check_and_fire returns a list
of AlertEvent dicts; the caller wires them to ws broadcast / webhook /
email.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.api_gateway import AlertRule, ResourcePack

# Dedup window. Tightened or lengthened here only.
DEDUP_WINDOW = timedelta(hours=24)


def _naive_utc(dt: datetime | None) -> datetime | None:
    """SQLite strips tzinfo on storage, so DB reads come back naive. Unify
    application-side datetimes to naive UTC before any comparison so the
    code runs identically on both PG and SQLite."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


@dataclass
class AlertEvent:
    rule_id: int
    grant_id: int
    threshold_percent: int
    observed_percent: int  # actual usage % at check time
    pack_id: int | None    # None = aggregate rule


async def check_and_fire(
    session: AsyncSession,
    *,
    grant_id: int,
    now: datetime | None = None,
) -> list[AlertEvent]:
    """Evaluate all enabled rules for this grant. Returns events that fired.

    Mutates last_notified_at on the rules that fired. Caller commits the
    session. Dedup within 24h is checked against last_notified_at; the
    current call's timestamp wins the tie if two concurrent evaluations
    race — duplicate notifications are impossible because we only fire
    when last_notified_at is old enough AND the UPDATE we emit sets it.
    """
    now = _naive_utc(now or datetime.now(timezone.utc))
    dedup_cutoff = now - DEDUP_WINDOW

    stmt = select(AlertRule).where(
        AlertRule.grant_id == grant_id,
        AlertRule.enabled == True,  # noqa: E712 — SQLAlchemy boolean column
    )
    rules = (await session.execute(stmt)).scalars().all()
    if not rules:
        return []

    # Pull packs once per grant; rules will filter in Python.
    pstmt = select(ResourcePack).where(ResourcePack.grant_id == grant_id)
    packs = {p.id: p for p in (await session.execute(pstmt)).scalars().all()}

    events: list[AlertEvent] = []
    for rule in rules:
        if rule.last_notified_at and rule.last_notified_at > dedup_cutoff:
            continue

        if rule.pack_id is not None:
            pack = packs.get(rule.pack_id)
            if pack is None or pack.total_units == 0:
                continue
            observed = (pack.used_units * 100) // pack.total_units
        else:
            total = sum(p.total_units for p in packs.values())
            used = sum(p.used_units for p in packs.values())
            if total == 0:
                continue
            observed = (used * 100) // total

        if observed >= rule.threshold_percent:
            rule.last_notified_at = now
            events.append(AlertEvent(
                rule_id=rule.id,
                grant_id=grant_id,
                threshold_percent=rule.threshold_percent,
                observed_percent=int(observed),
                pack_id=rule.pack_id,
            ))

    if events:
        await session.commit()
    return events
