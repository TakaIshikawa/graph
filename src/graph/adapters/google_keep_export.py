"""Adapter for Google Keep export - wraps existing Google Keep adapter."""

from __future__ import annotations

from graph.adapters.base import IngestResult, SourceAdapter
from graph.adapters.google_keep import GoogleKeepAdapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


class GoogleKeepExportAdapter(SourceAdapter):
    """Wrapper around GoogleKeepAdapter for Google Keep exports."""

    @property
    def name(self) -> str:
        return "google_keep_export"

    @property
    def entity_types(self) -> list[str]:
        return ["note"]

    def __init__(self, path: str = "") -> None:
        self.path = path
        self._google_keep = GoogleKeepAdapter(path=path)

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        """Ingest using GoogleKeepAdapter and remap source_project."""
        result = self._google_keep.ingest(since=since, entity_types=entity_types)

        # Remap source_project to GOOGLE_KEEP_EXPORT
        for unit in result.units:
            unit.source_project = SourceProject.GOOGLE_KEEP_EXPORT
            unit.source_id = unit.source_id.replace("google_keep:", "google_keep_export:", 1)

        return result
