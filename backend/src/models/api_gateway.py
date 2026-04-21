"""API gateway tables: ApiKeyGrant + ResourcePack + AlertRule.

  ApiKeyGrant (M:N api_key <-> service_instance)
    │  one api key can be granted access to many service instances;
    │  the grant is also the scoping unit for quota + alerts.
    │
    ├── ResourcePack (1:N)
    │    character/token/call packs purchased against a grant.
    │    consume() is atomic compare-and-swap: UPDATE ... WHERE used + n <= total.
    │
    └── AlertRule (1:N)
         threshold-based notifications (% of a pack used).
         24h debounce via last_notified_at.

Snowflake IDs everywhere (see utils/snowflake.py). BigInteger on PG,
INTEGER on SQLite (Wave 1 pattern, see models/memory.py for the reason).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, ForeignKey, Index, Integer, String,
    UniqueConstraint,
)

from src.models.database import Base
from src.utils.snowflake import snowflake_id


class ApiKeyGrant(Base):
    """M:N binding between InstanceApiKey and ServiceInstance.

    When a grant exists, the api key may invoke /v1/chat/completions (and the
    other protocol routes) targeting this particular instance via the `model`
    field. Status = active | paused | retired:
      - active:  request goes through, quota consumed
      - paused:  402 service_paused, quota untouched
      - retired: 410 service_retired, resource packs are settled/cleared
    """
    __tablename__ = "api_key_grants"

    id = Column(BigInteger, primary_key=True, default=snowflake_id)
    api_key_id = Column(
        BigInteger,
        ForeignKey("instance_api_keys.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    instance_id = Column(
        BigInteger,
        ForeignKey("service_instances.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status = Column(String(20), nullable=False, default="active")
    # When the customer opened this grant. null = seeded by the system
    # (e.g., the auto-activation on key creation decision D4).
    activated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    paused_at = Column(DateTime(timezone=True), nullable=True)
    retired_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("api_key_id", "instance_id", name="uq_grant_key_instance"),
        Index("ix_grant_active", "api_key_id", "status"),
    )


class ResourcePack(Base):
    """Purchased quota packs.

    total_units and used_units are in the meter_dim of the grant's
    ServiceInstance (tokens / chars / duration_ms / calls).

    consume() MUST be atomic: use a single SQL UPDATE with a WHERE clause
    enforcing used + n <= total, and RETURNING (or check affected row
    count) so racing callers cannot over-consume. See
    services/resource_pack.py for the helper.
    """
    __tablename__ = "resource_packs"

    id = Column(BigInteger, primary_key=True, default=snowflake_id)
    grant_id = Column(
        BigInteger,
        ForeignKey("api_key_grants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Display name shown on the catalog page (e.g., "10万字包").
    name = Column(String(100), nullable=False)
    total_units = Column(BigInteger, nullable=False)
    used_units = Column(BigInteger, nullable=False, default=0)
    # null = never expires. Packs whose expires_at has passed are skipped
    # by consume() (treated as 0 remaining).
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    purchased_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    # "free_trial" for the default D4 grant; "purchased" for real packs.
    # Reserved for future billing integration.
    source = Column(String(20), nullable=False, default="purchased")

    __table_args__ = (
        Index("ix_pack_grant_active", "grant_id", "expires_at"),
    )


class AlertRule(Base):
    """Percent-of-pack threshold alerts.

    threshold_percent: int 0-100. When used_units / total_units crosses this
    line (either direction — we only fire going up), the system notifies the
    api key owner via in-app toast + optional webhook (Phase 2+).

    last_notified_at + 24h dedup prevents notification storms.
    """
    __tablename__ = "alert_rules"

    id = Column(BigInteger, primary_key=True, default=snowflake_id)
    grant_id = Column(
        BigInteger,
        ForeignKey("api_key_grants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    threshold_percent = Column(Integer, nullable=False)
    # Optional: scope rule to a single pack. Null = across all packs for
    # the grant (useful for "total usage crossing X%").
    pack_id = Column(
        BigInteger,
        ForeignKey("resource_packs.id", ondelete="CASCADE"),
        nullable=True,
    )
    enabled = Column(Boolean, nullable=False, default=True)
    last_notified_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_alert_grant_enabled", "grant_id", "enabled"),
    )
