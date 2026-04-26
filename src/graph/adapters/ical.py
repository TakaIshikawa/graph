"""Adapter for local iCalendar files."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class ICalAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "ical"

    @property
    def entity_types(self) -> list[str]:
        return ["calendar_event"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "calendar_event" not in entity_types:
            return result

        root = Path(self.path).expanduser()
        if not root.exists():
            return result

        files = self._iter_files(root)
        sync_at = self._sync_datetime(since) if since else None
        for file_path in files:
            source_path = self._source_path(root, file_path)
            try:
                events = self._parse_events(file_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue

            for index, event in enumerate(events):
                try:
                    unit = self._event_to_unit(event, source_path, index)
                except (KeyError, ValueError):
                    continue
                if sync_at is not None and not self._changed_since(unit.metadata, sync_at):
                    continue
                result.units.append(unit)

        return result

    def _iter_files(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() == ".ics" else []
        if root.is_dir():
            return sorted(path for path in root.rglob("*.ics") if path.is_file())
        return []

    def _source_path(self, root: Path, path: Path) -> str:
        if root.is_file():
            return path.name
        return path.relative_to(root).as_posix()

    def _parse_events(self, text: str) -> list[dict[str, list[dict[str, object]]]]:
        lines = self._unfold_lines(text)
        events: list[dict[str, list[dict[str, object]]]] = []
        current: dict[str, list[dict[str, object]]] | None = None

        for line in lines:
            upper = line.upper()
            if upper == "BEGIN:VEVENT":
                current = {}
                continue
            if upper == "END:VEVENT":
                if current is not None:
                    events.append(current)
                current = None
                continue
            if current is None:
                continue

            parsed = self._parse_property(line)
            if parsed is None:
                continue
            name, params, value = parsed
            current.setdefault(name, []).append({"params": params, "value": value})

        return events

    def _unfold_lines(self, text: str) -> list[str]:
        lines: list[str] = []
        for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            if raw_line.startswith((" ", "\t")) and lines:
                lines[-1] += raw_line[1:]
            elif raw_line:
                lines.append(raw_line)
        return lines

    def _parse_property(self, line: str) -> tuple[str, dict[str, str], str] | None:
        if ":" not in line:
            return None
        key, value = line.split(":", 1)
        parts = key.split(";")
        name = parts[0].upper()
        params: dict[str, str] = {}
        for part in parts[1:]:
            if "=" not in part:
                continue
            param_name, param_value = part.split("=", 1)
            params[param_name.upper()] = param_value.strip('"')
        return name, params, value

    def _event_to_unit(
        self,
        event: dict[str, list[dict[str, object]]],
        source_path: str,
        index: int,
    ) -> KnowledgeUnit:
        uid = self._text(event, "UID")
        if not uid:
            raise ValueError("VEVENT missing UID")

        start = self._datetime_text(event, "DTSTART")
        end = self._datetime_text(event, "DTEND")
        created = self._datetime_text(event, "CREATED")
        updated = self._datetime_text(event, "LAST-MODIFIED") or self._datetime_text(event, "DTSTAMP")
        title = self._text(event, "SUMMARY") or "Untitled calendar event"
        description = self._text(event, "DESCRIPTION")
        location = self._text(event, "LOCATION")
        organizer = self._participant(event, "ORGANIZER")
        attendees = [self._format_participant(item) for item in event.get("ATTENDEE", [])]
        attendees = [attendee for attendee in attendees if attendee]

        metadata = {
            "uid": uid,
            "start": start,
            "end": end,
            "location": location,
            "organizer": organizer,
            "attendees": attendees,
            "source_path": source_path,
        }
        if created:
            metadata["created"] = created
        if updated:
            metadata["updated"] = updated

        event_time = start or created or updated
        created_at = self._parse_datetime_value(event_time) if event_time else None

        return KnowledgeUnit(
            source_project=SourceProject.ME,
            source_id=f"{source_path}#{uid or index}",
            source_entity_type="calendar_event",
            title=title,
            content=self._content(description, start, end, location, organizer, attendees),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=self._categories(event),
            created_at=created_at or datetime.now(timezone.utc),
        )

    def _text(self, event: dict[str, list[dict[str, object]]], name: str) -> str:
        values = event.get(name, [])
        if not values:
            return ""
        return self._unescape_text(str(values[0]["value"])).strip()

    def _datetime_text(self, event: dict[str, list[dict[str, object]]], name: str) -> str:
        values = event.get(name, [])
        if not values:
            return ""
        value = str(values[0]["value"]).strip()
        return self._parse_datetime_value(value).isoformat()

    def _participant(self, event: dict[str, list[dict[str, object]]], name: str) -> str:
        values = event.get(name, [])
        if not values:
            return ""
        return self._format_participant(values[0])

    def _format_participant(self, item: dict[str, object]) -> str:
        params = item.get("params", {})
        value = self._unescape_text(str(item.get("value", ""))).strip()
        if value.lower().startswith("mailto:"):
            value = value[7:]
        if isinstance(params, dict):
            common_name = str(params.get("CN", "")).strip()
            if common_name and value:
                return f"{common_name} <{value}>"
        return value

    def _categories(self, event: dict[str, list[dict[str, object]]]) -> list[str]:
        tags: list[str] = []
        for item in event.get("CATEGORIES", []):
            for category in self._split_escaped_commas(str(item["value"])):
                tag = self._unescape_text(category).strip()
                if tag and tag not in tags:
                    tags.append(tag)
        return tags

    def _content(
        self,
        description: str,
        start: str,
        end: str,
        location: str,
        organizer: str,
        attendees: list[str],
    ) -> str:
        lines: list[str] = []
        if description:
            lines.append(description)
        details = [
            ("Start", start),
            ("End", end),
            ("Location", location),
            ("Organizer", organizer),
            ("Attendees", ", ".join(attendees)),
        ]
        lines.extend(f"{label}: {value}" for label, value in details if value)
        return "\n".join(lines)

    def _changed_since(self, metadata: dict, sync_at: datetime) -> bool:
        timestamps = [
            self._parse_datetime_value(str(metadata[key]))
            for key in ("updated", "created", "start")
            if metadata.get(key)
        ]
        return any(timestamp > sync_at for timestamp in timestamps)

    def _parse_datetime_value(self, value: str) -> datetime:
        value = value.strip()
        if not value:
            raise ValueError("empty datetime")
        if len(value) == 8 and value.isdigit():
            return datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc)
        if value.endswith("Z"):
            return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        if "T" in value and "-" not in value:
            return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        if isinstance(since.last_sync_at, datetime):
            sync_at = since.last_sync_at
        else:
            sync_at = datetime.fromisoformat(str(since.last_sync_at))
        if sync_at.tzinfo is None:
            return sync_at.replace(tzinfo=timezone.utc)
        return sync_at.astimezone(timezone.utc)

    def _unescape_text(self, text: str) -> str:
        output: list[str] = []
        index = 0
        while index < len(text):
            char = text[index]
            if char == "\\" and index + 1 < len(text):
                next_char = text[index + 1]
                if next_char in {"n", "N"}:
                    output.append("\n")
                else:
                    output.append(next_char)
                index += 2
                continue
            output.append(char)
            index += 1
        return "".join(output)

    def _split_escaped_commas(self, text: str) -> list[str]:
        parts: list[str] = []
        current: list[str] = []
        escaped = False
        for char in text:
            if escaped:
                current.append("\\" + char)
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == ",":
                parts.append("".join(current))
                current = []
            else:
                current.append(char)
        if escaped:
            current.append("\\")
        parts.append("".join(current))
        return parts
