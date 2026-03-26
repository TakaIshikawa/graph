"""Base adapter interface for source project ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod

from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


class IngestResult:
    """Container for what an adapter produces."""

    def __init__(self) -> None:
        self.units: list[KnowledgeUnit] = []
        self.edges: list[KnowledgeEdge] = []


class SourceAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter name matching SourceProject enum value."""

    @property
    @abstractmethod
    def entity_types(self) -> list[str]:
        """Entity types this adapter can ingest."""

    @abstractmethod
    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        """Ingest knowledge from source.

        If since is provided, only ingest items created/updated after last_sync_at.
        If entity_types is provided, only ingest those types.
        """
