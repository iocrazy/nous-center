from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Index,
    Integer,
    JSON,
    String,
)
from sqlalchemy.orm import deferred

from src.models.database import Base
from src.utils.snowflake import snowflake_id


class ServiceInstance(Base):
    """A callable service. v3 unified concept: service = instance = app.

    A service always has a workflow behind it (trivial 1-3 node workflow for
    quick-provisioned services, full DAG for workflow-published ones). The
    `workflow_snapshot` is the frozen execution graph at publish time;
    re-publishing the source workflow produces a new ServiceInstance row
    (versioning is external; `version` is the row's own counter).
    """
    __tablename__ = "service_instances"

    id = Column(BigInteger, primary_key=True, default=snowflake_id)
    source_type = Column(String(20), nullable=False, default="preset")  # preset | workflow | model
    source_id = Column(BigInteger, nullable=True, index=True)
    source_name = Column(String(128), nullable=True)
    # v3 contract: name is the public identifier and the routing key
    # (`/v1/apps/{name}` and `?model={name}`). Must match `^[a-z][a-z0-9-]{1,62}$`,
    # enforced by Pydantic at the API boundary and a CHECK in PG.
    name = Column(String(100), nullable=False, unique=True)
    type = Column(String(20), default="tts", nullable=False)
    status = Column(String(20), default="active", nullable=False)
    category = Column(String(20), nullable=True)   # llm | tts | vl | app
    meter_dim = Column(String(20), nullable=True)  # tokens | chars | duration | calls
    endpoint_path = Column(String(200), nullable=True)
    params_override = Column(JSON, default=dict)
    rate_limit_rpm = Column(Integer, nullable=True)
    rate_limit_tpm = Column(Integer, nullable=True)

    # ---- v3 publish contract --------------------------------------
    # FK to the source workflow (auto-generated trivial workflow for
    # quick-provisioned services, user-authored DAG for published ones).
    # Nullable for legacy preset/model rows pre-migration.
    workflow_id = Column(BigInteger, nullable=True)
    # Frozen ComfyUI-style api JSON. Big — use deferred() so list/lookup
    # queries don't pay for it. Dispatch path MUST .options(undefer(...)).
    workflow_snapshot = deferred(Column(JSON, nullable=False, default=dict))
    # External-facing input/output schemas (list of {key, label, node_id,
    # input_name, type, ...}). Also deferred for the same reason.
    exposed_inputs = deferred(Column(JSON, nullable=False, default=list))
    exposed_outputs = deferred(Column(JSON, nullable=False, default=list))
    # SHA-256 of workflow_snapshot — non-unique index for dedup hints.
    snapshot_hash = Column(String(80), nullable=True)
    snapshot_schema_version = Column(Integer, nullable=False, default=1)
    version = Column(Integer, nullable=False, default=1)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_service_instances_source", "source_type", "source_id"),
        Index("idx_service_snapshot_hash", "snapshot_hash"),
    )
