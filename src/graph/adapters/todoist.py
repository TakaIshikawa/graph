"""Adapter for Todoist CSV exports with tasks, projects, and priorities."""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState

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

        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
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

            if allowed_types and entity_type not in allowed_types:
                continue

            # Priority tag
            priority_raw = (row.get(_COL_PRIORITY) or "").strip()
            tags: list[str] = []
            if priority_raw in _PRIORITY_TAGS:
                tags.append(_PRIORITY_TAGS[priority_raw])

            # Due date
            due_date = (row.get(_COL_DATE) or "").strip() or None
            author = (row.get(_COL_AUTHOR) or "").strip() or None
            responsible = (row.get(_COL_RESPONSIBLE) or "").strip() or None
            date_lang = (row.get(_COL_DATE_LANG) or "").strip() or None
            tz = (row.get(_COL_TIMEZONE) or "").strip() or None

            # Deterministic source ID
            id_input = f"{content}|{due_date or ''}"
            digest = hashlib.sha1(id_input.encode("utf-8")).hexdigest()[:16]
            source_id = f"todoist:{entity_type}:{digest}"

            metadata: dict = {
                "indent": indent,
            }
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

        result.units.sort(key=lambda u: (u.source_entity_type, u.source_id))
        return result
