"""Adapter for Evernote export ENEX files - wraps existing ENEX adapter."""

from __future__ import annotations

from graph.adapters.base import IngestResult, SourceAdapter
from graph.adapters.enex import EnexAdapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


class EvernoteExportAdapter(SourceAdapter):
    """Wrapper around EnexAdapter for Evernote exports."""

    @property
    def name(self) -> str:
        return "evernote_export"

    @property
    def entity_types(self) -> list[str]:
        return ["note"]

    def __init__(self, path: str = "") -> None:
        self.path = path
        self._enex = EnexAdapter(path=path)

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        """Ingest using EnexAdapter and remap source_project."""
        result = self._enex.ingest(since=since, entity_types=entity_types)

        # Remap source_project to EVERNOTE_EXPORT
        for unit in result.units:
            unit.source_project = SourceProject.EVERNOTE_EXPORT
            unit.source_id = unit.source_id.replace("enex:", "evernote_export:", 1)

        return result
