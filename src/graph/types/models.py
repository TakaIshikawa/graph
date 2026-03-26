from __future__ import annotations

from datetime import datetime, timezone
from pydantic import BaseModel, Field

from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class KnowledgeUnit(BaseModel):
    id: str = ""
    source_project: SourceProject
    source_id: str
    source_entity_type: str
    title: str
    content: str
    content_type: ContentType = ContentType.INSIGHT
    metadata: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    confidence: float | None = None
    utility_score: float | None = None
    embedding: list[float] | None = Field(default=None, exclude=True)
    created_at: datetime = Field(default_factory=_utcnow)
    ingested_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class KnowledgeEdge(BaseModel):
    id: str = ""
    from_unit_id: str
    to_unit_id: str
    relation: EdgeRelation
    weight: float = 1.0
    source: EdgeSource = EdgeSource.INFERRED
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


class SyncState(BaseModel):
    source_project: str
    source_entity_type: str
    last_sync_at: datetime
    last_source_id: str | None = None
    items_synced: int = 0
