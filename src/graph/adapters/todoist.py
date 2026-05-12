"""Adapter for Todoist CSV exports with tasks, projects, and priorities."""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState

# Todoist CSV columns
_COL_TYPE = "TYPE"
_COL_CONTENT = "CONTENT"
_COL_PRIORITY = "PRIORITY"
_COL_INDENT = "INDENT"
_COL_AUTHOR = "AUTHOR"
_COL_RESPONSIBLE = "RESPONSIBLE"
_COL_DATE = "DATE"
_COL_DATE_LANG = "DATE_LANG"
_COL_TIMEZONE = "TIMEZONE"

# Priority mapping: Todoist uses 1=highest, 4=lowest
_PRIORITY_TAGS = {
    "1": "p1",
    "2": "p2",
    "3": "p3",
    "4": "p4",
}


class TodoistAdapter(SourceAdapter):
    """Import Todoist CSV exports preserving tasks, projects, and priorities."""

    @property
    def name(self) -> str:
        return "todoist"

    @property
    def entity_types(self) -> list[str]:
        return ["task", "project"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if not self.path:
            return result

        csv_path = Path(self.path).expanduser()
        if not csv_path.exists():
            return result

        allowed_types = set(entity_types) if entity_types else None

        try:
            text = csv_path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            return result

        latest_by_indent: dict[int, dict[str, object]] = {}
        reader = csv.DictReader(io.StringIO(text))
        for row_number, row in enumerate(reader, start=2):
            content = (row.get(_COL_CONTENT) or "").strip()
            if not content:
                continue

            row_type = (row.get(_COL_TYPE) or "").strip().lower()
            indent_raw = row.get(_COL_INDENT, "").strip()
            indent = int(indent_raw) if indent_raw.isdigit() else 1

            # Determine entity type: TYPE=task or indent>1 → task; otherwise project
            if row_type == "task" or indent > 1:
                entity_type = "task"
            elif row_type == "project":
                entity_type = "project"
            else:
                entity_type = "task"

            # Due date
            priority_raw = (row.get(_COL_PRIORITY) or "").strip()
            due_date = (row.get(_COL_DATE) or "").strip() or None

            # Deterministic source ID
            id_input = f"{content}|{due_date or ''}"
            digest = hashlib.sha1(id_input.encode("utf-8")).hexdigest()[:16]
            source_id = f"todoist:{entity_type}:{digest}"

            parent = self._nearest_parent(latest_by_indent, indent)
            emitted = not allowed_types or entity_type in allowed_types

            if emitted:
                # Priority tag
                tags: list[str] = []
                if priority_raw in _PRIORITY_TAGS:
                    tags.append(_PRIORITY_TAGS[priority_raw])

                author = (row.get(_COL_AUTHOR) or "").strip() or None
                responsible = (row.get(_COL_RESPONSIBLE) or "").strip() or None
                date_lang = (row.get(_COL_DATE_LANG) or "").strip() or None
                tz = (row.get(_COL_TIMEZONE) or "").strip() or None

                metadata: dict = {
                    "indent": indent,
                    "source_row_number": row_number,
                }
                if parent is not None:
                    metadata["parent_source_id"] = parent["source_id"]
                    metadata["parent_title"] = parent["title"]
                if priority_raw:
                    metadata["priority"] = int(priority_raw) if priority_raw.isdigit() else priority_raw
                if due_date:
                    metadata["due_date"] = due_date
                if author:
                    metadata["author"] = author
                if responsible:
                    metadata["responsible"] = responsible
                if date_lang:
                    metadata["date_lang"] = date_lang
                if tz:
                    metadata["timezone"] = tz

                unit = KnowledgeUnit(
                    source_project=SourceProject.TODOIST,
                    source_id=source_id,
                    source_entity_type=entity_type,
                    title=content,
                    content=content,
                    content_type=ContentType.ARTIFACT,
                    metadata=metadata,
                    tags=sorted(tags),
                )
                result.units.append(unit)

                if parent is not None and parent.get("emitted"):
                    result.edges.append(
                        KnowledgeEdge(
                            id=self._edge_id(str(parent["source_id"]), source_id),
                            from_unit_id=str(parent["source_id"]),
                            to_unit_id=source_id,
                            relation=EdgeRelation.CONTAINS,
                            source=EdgeSource.SOURCE,
                            metadata={
                                "source_project": SourceProject.TODOIST.value,
                                "relation_type": "todoist_hierarchy",
                                "parent_title": parent["title"],
                                "child_title": content,
                                "child_indent": indent,
                                "source_row_number": row_number,
                            },
                        )
                    )

            latest_by_indent[indent] = {
                "source_id": source_id,
                "title": content,
                "entity_type": entity_type,
                "emitted": emitted,
            }
            for stale_indent in [level for level in latest_by_indent if level > indent]:
                del latest_by_indent[stale_indent]

        result.units.sort(key=lambda u: (u.source_entity_type, u.source_id))
        result.edges.sort(key=lambda e: e.id)
        return result

    def _nearest_parent(self, latest_by_indent: dict[int, dict[str, object]], indent: int) -> dict[str, object] | None:
        for parent_indent in sorted((level for level in latest_by_indent if level < indent), reverse=True):
            return latest_by_indent[parent_indent]
        return None

    def _edge_id(self, parent_source_id: str, child_source_id: str) -> str:
        digest = hashlib.sha1(f"{parent_source_id}|{child_source_id}|contains".encode("utf-8")).hexdigest()[:16]
        return f"todoist:contains:{digest}"
