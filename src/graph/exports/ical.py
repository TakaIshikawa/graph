"""iCalendar (RFC 5545) export adapter for knowledge unit collections."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import datetime, timezone

from graph.types.models import KnowledgeUnit

# RFC 5545 line length limit (75 octets excluding CRLF)
_MAX_LINE_OCTETS = 75

# RFC 5545 line folding continuation
_CRLF = "\r\n"


def export_units_to_ical(
    units: Iterable[KnowledgeUnit],
    *,
    calendar_name: str = "Knowledge Units Export",
    prodid: str = "-//GraphKnowledge//Export//EN",
) -> str:
    """
    Export knowledge units as an iCalendar (.ics) document (RFC 5545).

    Each unit becomes a VEVENT with:
      - DTSTART: unit.created_at
      - SUMMARY: unit.title
      - DESCRIPTION: unit.content
      - LOCATION: unit.metadata.get("location")
      - CATEGORIES: unit.tags
      - UID: deterministic from unit.id

    Args:
        units: Iterable of KnowledgeUnit instances to export
        calendar_name: X-WR-CALNAME value
        prodid: PRODID value identifying the producer

    Returns:
        RFC 5545 compliant iCalendar string
    """
    units_list = list(units)

    lines: list[str] = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        f"PRODID:{prodid}",
        f"X-WR-CALNAME:{_escape_text(calendar_name)}",
    ]

    for unit in units_list:
        lines.extend(_render_vevent(unit))

    lines.append("END:VCALENDAR")

    # Join with CRLF per RFC 5545 and fold long lines
    folded = _CRLF.join(_fold_line(line) for line in lines)
    # Trailing CRLF
    return folded + _CRLF


def _render_vevent(unit: KnowledgeUnit) -> list[str]:
    """Render a single knowledge unit as a VEVENT component."""
    lines: list[str] = ["BEGIN:VEVENT"]

    # UID - deterministic from unit id
    uid = _make_uid(unit.id)
    lines.append(f"UID:{uid}")

    # Timestamps
    dtstart = _format_datetime(unit.created_at)
    lines.append(f"DTSTART:{dtstart}")

    # DTSTAMP is required by RFC 5545 - use ingested_at or now
    dtstamp = _format_datetime(unit.ingested_at)
    lines.append(f"DTSTAMP:{dtstamp}")

    # SUMMARY from title
    if unit.title:
        lines.append(f"SUMMARY:{_escape_text(unit.title)}")

    # DESCRIPTION from content
    if unit.content:
        lines.append(f"DESCRIPTION:{_escape_text(unit.content)}")

    # LOCATION from metadata
    location = unit.metadata.get("location") if unit.metadata else None
    if location:
        lines.append(f"LOCATION:{_escape_text(str(location))}")

    # CATEGORIES from tags
    if unit.tags:
        categories = ",".join(_escape_text(tag) for tag in sorted(unit.tags))
        lines.append(f"CATEGORIES:{categories}")

    # LAST-MODIFIED from updated_at
    if unit.updated_at:
        lines.append(f"LAST-MODIFIED:{_format_datetime(unit.updated_at)}")

    lines.append("END:VEVENT")
    return lines


def _make_uid(unit_id: str) -> str:
    """Generate a deterministic UID from unit ID."""
    if unit_id:
        return f"{unit_id}@graphknowledge"
    # Fallback for units without IDs
    return f"{uuid.uuid4()}@graphknowledge"


def _format_datetime(dt: datetime | None) -> str:
    """Format a datetime as RFC 5545 DATE-TIME (UTC form with Z suffix)."""
    if dt is None:
        dt = datetime.now(timezone.utc)

    # Convert to UTC if timezone-aware
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y%m%dT%H%M%SZ")

    # Naive datetime - treat as UTC
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _escape_text(text: str) -> str:
    """
    Escape text per RFC 5545 section 3.3.11.

    Backslash, semicolons, commas, and newlines must be escaped.
    """
    # Order matters: escape backslash first
    text = text.replace("\\", "\\\\")
    text = text.replace(";", "\\;")
    text = text.replace(",", "\\,")
    # Newlines become literal \n
    text = text.replace("\r\n", "\\n")
    text = text.replace("\r", "\\n")
    text = text.replace("\n", "\\n")
    return text


def _fold_line(line: str) -> str:
    """
    Fold a content line per RFC 5545 section 3.1.

    Lines longer than 75 octets (bytes in UTF-8) are folded by inserting
    a CRLF followed by a single whitespace character.
    """
    encoded = line.encode("utf-8")
    if len(encoded) <= _MAX_LINE_OCTETS:
        return line

    parts: list[str] = []
    # First line gets full 75 octets
    first_chunk = _safe_utf8_slice(line, 0, _MAX_LINE_OCTETS)
    parts.append(first_chunk)
    remaining = line[len(first_chunk) :]

    # Continuation lines: space + up to 74 octets of content
    while remaining:
        chunk = _safe_utf8_slice(remaining, 0, _MAX_LINE_OCTETS - 1)
        parts.append(" " + chunk)
        remaining = remaining[len(chunk) :]

    return _CRLF.join(parts)


def _safe_utf8_slice(text: str, start_char: int, max_bytes: int) -> str:
    """
    Slice text from start_char position, ensuring we don't exceed max_bytes
    in UTF-8 and don't split multi-byte characters.
    """
    substr = text[start_char:]
    # Try progressively shorter slices until we fit
    for end in range(len(substr), 0, -1):
        candidate = substr[:end]
        if len(candidate.encode("utf-8")) <= max_bytes:
            return candidate
    return ""
