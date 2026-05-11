"""Adapter for local iCalendar files."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class ICalAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "ical"

    @property
    def entity_types(self) -> list[str]:
        return ["calendar_event", "calendar_task"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        requested = set(entity_types or self.entity_types)
        if not requested.intersection(self.entity_types):
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
                    units = self._component_to_units(event, source_path, index)
                except (KeyError, ValueError):
                    continue
                for unit in units:
                    if unit.source_entity_type not in requested:
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
            if upper in {"BEGIN:VEVENT", "BEGIN:VTODO"}:
                current = {"__TYPE__": [{"params": {}, "value": upper.removeprefix("BEGIN:")}]}
                continue
            if upper in {"END:VEVENT", "END:VTODO"}:
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

    def _component_to_units(
        self,
        event: dict[str, list[dict[str, object]]],
        source_path: str,
        index: int,
    ) -> list[KnowledgeUnit]:
        unit = self._event_to_unit(event, source_path, index)
        rrule = self._text(event, "RRULE")
        if not rrule or unit.source_entity_type != "calendar_event":
            return [unit]

        return self._expand_rrule(unit, rrule)

    def _event_to_unit(
        self,
        event: dict[str, list[dict[str, object]]],
        source_path: str,
        index: int,
    ) -> KnowledgeUnit:
        component_type = self._text(event, "__TYPE__") or "VEVENT"
        entity_type = "calendar_task" if component_type == "VTODO" else "calendar_event"
        uid = self._text(event, "UID")
        if not uid:
            raise ValueError(f"{component_type} missing UID")

        start = self._datetime_text(event, "DTSTART")
        end = self._datetime_text(event, "DTEND")
        due = self._datetime_text(event, "DUE")
        completed = self._datetime_text(event, "COMPLETED")
        created = self._datetime_text(event, "CREATED")
        updated = self._datetime_text(event, "LAST-MODIFIED") or self._datetime_text(event, "DTSTAMP")
        title = self._text(event, "SUMMARY") or ("Untitled calendar task" if entity_type == "calendar_task" else "Untitled calendar event")
        description = self._text(event, "DESCRIPTION")
        location = self._text(event, "LOCATION")
        organizer = self._participant(event, "ORGANIZER")
        attendees = [self._format_participant(item) for item in event.get("ATTENDEE", [])]
        attendees = [attendee for attendee in attendees if attendee]
        categories = self._categories(event)

        metadata = {
            "uid": uid,
            "component": component_type,
            "start": start,
            "end": end,
            "due": due,
            "completed": completed,
            "location": location,
            "organizer": organizer,
            "attendees": attendees,
            "categories": categories,
            "rrule": self._text(event, "RRULE"),
            "source_path": source_path,
        }
        if created:
            metadata["created"] = created
        if updated:
            metadata["updated"] = updated

        event_time = start or due or completed or created or updated
        created_at = self._parse_datetime_value(event_time) if event_time else None

        return KnowledgeUnit(
            source_project=SourceProject.CALENDAR,
            source_id=f"{source_path}#{uid or index}",
            source_entity_type=entity_type,
            title=title,
            content=self._content(description, start, end or due, location, organizer, attendees),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=categories,
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
        item = values[0]
        value = str(item["value"]).strip()
        params = item.get("params", {})
        tzid = str(params.get("TZID", "")) if isinstance(params, dict) else ""
        return self._parse_datetime_value(value, tzid=tzid).isoformat()

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
            for key in ("updated", "created", "start", "due", "completed")
            if metadata.get(key)
        ]
        return any(timestamp > sync_at for timestamp in timestamps)

    def _expand_rrule(self, unit: KnowledgeUnit, rrule: str) -> list[KnowledgeUnit]:
        start_text = str(unit.metadata.get("start") or "")
        if not start_text:
            return [unit]
        start = self._parse_datetime_value(start_text)
        end = self._parse_datetime_value(str(unit.metadata["end"])) if unit.metadata.get("end") else None
        rule = self._rrule_parts(rrule)
        freq = rule.get("FREQ", "").upper()
        if freq not in {"DAILY", "WEEKLY", "MONTHLY", "YEARLY"}:
            return [unit]

        count = self._parse_count(rule.get("COUNT"))
        until = self._parse_datetime_value(rule["UNTIL"]) if rule.get("UNTIL") else None
        limit = min(count or 100, 100)
        interval = self._parse_count(rule.get("INTERVAL")) or 1
        duration = end - start if end else None

        units: list[KnowledgeUnit] = []
        current = start
        for occurrence in range(limit):
            if until and current > until:
                break
            copy = unit.model_copy(deep=True)
            copy.source_id = f"{unit.source_id}#{occurrence + 1}"
            copy.created_at = current
            copy.updated_at = current
            copy.metadata["recurrence_index"] = occurrence + 1
            copy.metadata["start"] = current.isoformat()
            if duration:
                copy.metadata["end"] = (current + duration).isoformat()
            units.append(copy)
            current = self._add_interval(current, freq, interval)
        return units or [unit]

    def _rrule_parts(self, rrule: str) -> dict[str, str]:
        parts: dict[str, str] = {}
        for item in rrule.split(";"):
            if "=" in item:
                key, value = item.split("=", 1)
                parts[key.strip().upper()] = value.strip()
        return parts

    def _parse_count(self, value: str | None) -> int | None:
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def _add_interval(self, value: datetime, freq: str, interval: int) -> datetime:
        if freq == "DAILY":
            return value + timedelta(days=interval)
        if freq == "WEEKLY":
            return value + timedelta(weeks=interval)
        if freq == "MONTHLY":
            return self._add_months(value, interval)
        return self._add_months(value, interval * 12)

    def _add_months(self, value: datetime, months: int) -> datetime:
        month = value.month - 1 + months
        year = value.year + month // 12
        month = month % 12 + 1
        days_in_month = [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        return value.replace(year=year, month=month, day=min(value.day, days_in_month[month - 1]))

    def _parse_datetime_value(self, value: str, tzid: str = "") -> datetime:
        value = value.strip()
        if not value:
            raise ValueError("empty datetime")

        # All-day date (YYYYMMDD)
        if len(value) == 8 and value.isdigit():
            return datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc)

        # UTC timestamp with Z suffix
        if value.endswith("Z"):
            return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)

        # Basic format with TZID parameter (e.g., DTSTART;TZID=America/New_York:20230101T120000)
        if tzid and "T" in value and "-" not in value:
            naive_dt = datetime.strptime(value, "%Y%m%dT%H%M%S")
            try:
                tz = ZoneInfo(tzid)
                localized_dt = naive_dt.replace(tzinfo=tz)
                return localized_dt.astimezone(timezone.utc)
            except ZoneInfoNotFoundError:
                # Fall back to UTC if timezone is not recognized
                return naive_dt.replace(tzinfo=timezone.utc)

        # Basic format without TZID (naive timestamp)
        if "T" in value and "-" not in value:
            return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)

        # ISO format timestamps
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
