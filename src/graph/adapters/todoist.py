"""Adapter for Todoist CSV exports with tasks, projects, and priorities."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from datetime import datetime, timezone
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
_COL_LABELS = "LABELS"
_COL_SECTION = "SECTION"

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
        return ["task", "project", "person"]

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

        allowed_types = set(entity_types) if entity_types else {"task", "project"}

        try:
            text = csv_path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            return result

        latest_by_indent: dict[int, dict[str, object]] = {}
        person_units: dict[str, KnowledgeUnit] = {}
        person_edges: dict[str, KnowledgeEdge] = {}
        reader = csv.DictReader(io.StringIO(text))
        for row_number, row in enumerate(reader, start=2):
            content = self._row_value(row, _COL_CONTENT)
            if not content:
                continue

            row_type = self._row_value(row, _COL_TYPE).lower()
            indent_raw = self._row_value(row, _COL_INDENT)
            indent = int(indent_raw) if indent_raw.isdigit() else 1

            # Determine entity type: TYPE=task or indent>1 → task; otherwise project
            if row_type == "task" or indent > 1:
                entity_type = "task"
            elif row_type == "project":
                entity_type = "project"
            else:
                entity_type = "task"

            # Due date
            priority_raw = self._row_value(row, _COL_PRIORITY)
            due_date = self._row_value(row, _COL_DATE) or None

            # Deterministic source ID
            id_input = f"{content}|{due_date or ''}"
            digest = hashlib.sha1(id_input.encode("utf-8")).hexdigest()[:16]
            source_id = f"todoist:{entity_type}:{digest}"

            parent = self._nearest_parent(latest_by_indent, indent)
            emitted = entity_type in allowed_types
            person_emitted = "person" in allowed_types
            author = self._row_value(row, _COL_AUTHOR) or None
            responsible = self._row_value(row, _COL_RESPONSIBLE) or None

            if emitted:
                # Priority tag
                tags: list[str] = []
                if priority_raw in _PRIORITY_TAGS:
                    tags.append(_PRIORITY_TAGS[priority_raw])
                labels = self._labels(row)
                for label in labels:
                    normalized = self._label_tag(label)
                    if normalized and normalized not in tags:
                        tags.append(normalized)

                date_lang = self._row_value(row, _COL_DATE_LANG) or None
                tz = self._row_value(row, _COL_TIMEZONE) or None
                section = self._row_value(row, _COL_SECTION) or None
                recurring = self._parse_bool(self._first_row_value(row, "recurring", "Recurring", "is_recurring"))
                recurrence = self._first_row_value(row, "recurrence", "Recurrence", "due_recurring", "Due Recurring") or None
                due_timezone = self._first_row_value(row, "due_timezone", "Due Timezone", "Due Time Zone") or None
                reminder_at = self._normalize_datetime(self._first_row_value(row, "reminder_at", "Reminder At", "Reminder", "Reminder Date"))
                reminder_timezone = self._first_row_value(row, "reminder_timezone", "Reminder Timezone", "Reminder Time Zone") or None
                project_name = self._first_row_value(row, "project_name", "Project Name", "Project") or None
                section_name = self._first_row_value(row, "section_name", "Section Name") or None

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
                if recurring is not None:
                    metadata["recurring"] = recurring
                if recurrence:
                    metadata["recurrence"] = recurrence
                if due_timezone:
                    metadata["due_timezone"] = due_timezone
                if reminder_at:
                    metadata["reminder_at"] = reminder_at
                if reminder_timezone:
                    metadata["reminder_timezone"] = reminder_timezone
                if labels:
                    metadata["labels"] = labels
                if project_name:
                    metadata["project_name"] = project_name
                if section:
                    metadata["section"] = section
                if section_name:
                    metadata["section_name"] = section_name

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

            if person_emitted:
                for field_name, value in (("author", author), ("responsible", responsible)):
                    if not value:
                        continue
                    person_unit = self._person_unit(value)
                    if person_unit is None:
                        continue
                    person_units.setdefault(person_unit.source_id, person_unit)
                    if emitted:
                        edge = self._person_edge(source_id, person_unit.source_id, entity_type, content, field_name, row_number)
                        person_edges.setdefault(edge.id, edge)

            latest_by_indent[indent] = {
                "source_id": source_id,
                "title": content,
                "entity_type": entity_type,
                "emitted": emitted,
            }
            for stale_indent in [level for level in latest_by_indent if level > indent]:
                del latest_by_indent[stale_indent]

        result.units.extend(person_units.values())
        result.edges.extend(person_edges.values())
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

    def _person_unit(self, value: str) -> KnowledgeUnit | None:
        normalized = self._normalize_person(value)
        if not normalized:
            return None
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
        email_match = re.search(r"[\w.!#$%&'*+/=?^`{|}~-]+@[\w.-]+\.[A-Za-z]{2,}", value)
        email = email_match.group(0).casefold() if email_match else ""
        title = value.strip()
        return KnowledgeUnit(
            source_project=SourceProject.TODOIST,
            source_id=f"todoist:person:{digest}",
            source_entity_type="person",
            title=title,
            content=title,
            content_type=ContentType.METADATA,
            metadata={
                "name": title,
                "email": email or None,
                "normalized": normalized,
                "normalized_name": normalized,
                "source": "todoist_collaborator",
            },
            tags=["person", "todoist"],
        )

    def _person_edge(
        self,
        source_id: str,
        person_source_id: str,
        entity_type: str,
        title: str,
        field_name: str,
        row_number: int,
    ) -> KnowledgeEdge:
        digest = hashlib.sha1(f"{source_id}|{person_source_id}|{field_name}|references".encode("utf-8")).hexdigest()[:16]
        return KnowledgeEdge(
            id=f"todoist:references:{digest}",
            from_unit_id=source_id,
            to_unit_id=person_source_id,
            relation=EdgeRelation.REFERENCES,
            source=EdgeSource.SOURCE,
            metadata={
                "source_project": SourceProject.TODOIST.value,
                "relation_type": f"todoist_{field_name}",
                "from_entity_type": entity_type,
                "to_entity_type": "person",
                "field": field_name,
                "role": field_name,
                "title": title,
                "source_row_number": row_number,
            },
        )

    def _normalize_person(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().casefold())

    def _row_value(self, row: dict[str, str], column: str) -> str:
        lowered = {str(key).casefold(): value for key, value in row.items()}
        value = row.get(column)
        if value is None:
            value = lowered.get(column.casefold())
        return str(value or "").strip()

    def _first_row_value(self, row: dict[str, str], *columns: str) -> str:
        normalized = {self._normalize_key(str(key)): value for key, value in row.items()}
        for column in columns:
            value = row.get(column)
            if value is None:
                value = normalized.get(self._normalize_key(column))
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _normalize_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    def _parse_bool(self, value: str) -> bool | None:
        text = value.strip().casefold()
        if not text:
            return None
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
        return None

    def _normalize_datetime(self, value: str) -> str:
        parsed = self._parse_datetime(value)
        return parsed.isoformat() if parsed else value

    def _parse_datetime(self, value: str) -> datetime | None:
        text = value.strip()
        if not text:
            return None
        for candidate in (text, text.replace("Z", "+00:00")):
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                continue
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _labels(self, row: dict[str, str]) -> list[str]:
        raw = self._row_value(row, _COL_LABELS)
        labels: list[str] = []
        for item in re.split(r"[,;|]", raw):
            label = item.strip()
            if label and label not in labels:
                labels.append(label)
        return labels

    def _label_tag(self, label: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", label.casefold()).strip("-")
        return normalized
