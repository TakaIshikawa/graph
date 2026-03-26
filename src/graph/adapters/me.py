"""Adapter for the 'me' project (YAML portfolio metadata)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class MeAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "me"

    @property
    def entity_types(self) -> list[str]:
        return ["project_metadata"]

    def __init__(self, config_path: str = "") -> None:
        self.config_path = config_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        path = Path(self.config_path)
        if not path.exists():
            return result

        with open(path) as f:
            data = yaml.safe_load(f)

        projects = data.get("projects", [])
        for proj in projects:
            if not proj.get("enabled", True):
                continue

            tags = proj.get("metadata", {}).get("tags", [])
            description = proj.get("description", "")

            last_updated = (
                proj.get("updateRules", {}).get("lastUpdated")
                or datetime.now(timezone.utc).isoformat()
            )

            unit = KnowledgeUnit(
                source_project=SourceProject.ME,
                source_id=proj["id"],
                source_entity_type="project_metadata",
                title=proj["name"],
                content=description,
                content_type=ContentType.METADATA,
                metadata={
                    "repo_path": proj.get("repoPath", ""),
                    "url": proj.get("metadata", {}).get("url", ""),
                },
                tags=tags,
                created_at=last_updated,
            )
            result.units.append(unit)

        return result
