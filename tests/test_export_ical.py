"""Tests for iCalendar export adapter."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

from graph.exports import export_units_to_ical, get_exporter, list_exporters
from graph.exports.registry import export_units
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, 30, tzinfo=timezone.utc)
UPDATED_TIME = datetime(2026, 5, 2, 8, 30, 45, tzinfo=timezone.utc)
INGESTED_TIME = datetime(2026, 5, 1, 11, 0, 0, tzinfo=timezone.utc)


def _unit(
    unit_id: str,
    title: str,
    content: str = "",
    *,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
    ingested_at: datetime | None = None,
) -> KnowledgeUnit:
    """Create a test unit with sensible defaults."""
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=content or f"{title} content",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=tags or [],
        created_at=created_at or UNIT_TIME,
        updated_at=updated_at or UPDATED_TIME,
        ingested_at=ingested_at or INGESTED_TIME,
    )


# --- Basic event generation ---


def test_export_generates_valid_icalendar_structure():
    """Test that export generates valid iCalendar structure."""
    units = [_unit("unit-1", "Test Unit")]
    ical = export_units_to_ical(units)

    assert ical.startswith("BEGIN:VCALENDAR\r\n")
    assert "END:VCALENDAR\r\n" in ical
    assert "VERSION:2.0\r\n" in ical
    assert "BEGIN:VEVENT\r\n" in ical
    assert "END:VEVENT\r\n" in ical


def test_export_single_event_has_required_properties():
    """Test that a single event contains all required RFC 5545 properties."""
    unit = _unit("unit-1", "Test Unit")
    ical = export_units_to_ical([unit])

    assert "UID:unit-1@graphknowledge\r\n" in ical
    assert "DTSTART:20260501T101530Z\r\n" in ical
    assert "DTSTAMP:20260501T110000Z\r\n" in ical
    assert "SUMMARY:Test Unit\r\n" in ical


def test_export_includes_prodid():
    """Test that PRODID is included."""
    ical = export_units_to_ical([_unit("u1", "T")])
    assert "PRODID:" in ical


def test_export_custom_prodid():
    """Test custom PRODID value."""
    ical = export_units_to_ical(
        [_unit("u1", "T")], prodid="-//Custom//Product//EN"
    )
    assert "PRODID:-//Custom//Product//EN\r\n" in ical


def test_export_custom_calendar_name():
    """Test custom calendar name."""
    ical = export_units_to_ical(
        [_unit("u1", "T")], calendar_name="My Calendar"
    )
    assert "X-WR-CALNAME:My Calendar\r\n" in ical


# --- Metadata field mapping ---


def test_export_maps_content_to_description():
    """Test that unit content maps to DESCRIPTION."""
    unit = _unit("unit-1", "Title", "This is the content")
    ical = export_units_to_ical([unit])

    assert "DESCRIPTION:This is the content\r\n" in ical


def test_export_maps_location_from_metadata():
    """Test that metadata location maps to LOCATION."""
    unit = _unit("unit-1", "Title", metadata={"location": "Tokyo, Japan"})
    ical = export_units_to_ical([unit])

    assert "LOCATION:Tokyo\\, Japan\r\n" in ical


def test_export_maps_tags_to_categories():
    """Test that tags map to CATEGORIES."""
    unit = _unit("unit-1", "Title", tags=["ai", "research", "python"])
    ical = export_units_to_ical([unit])

    # Tags are sorted
    assert "CATEGORIES:ai,python,research\r\n" in ical


def test_export_maps_updated_at_to_last_modified():
    """Test that updated_at maps to LAST-MODIFIED."""
    unit = _unit("unit-1", "Title", updated_at=UPDATED_TIME)
    ical = export_units_to_ical([unit])

    assert "LAST-MODIFIED:20260502T083045Z\r\n" in ical


def test_export_omits_description_when_content_empty():
    """Test that DESCRIPTION is omitted when content is empty."""
    unit = KnowledgeUnit(
        id="unit-1",
        source_project=SourceProject.MAX,
        source_id="source-1",
        source_entity_type="note",
        title="Title",
        content="",
        content_type=ContentType.INSIGHT,
        created_at=UNIT_TIME,
        ingested_at=INGESTED_TIME,
        updated_at=UPDATED_TIME,
    )
    ical = export_units_to_ical([unit])

    assert "DESCRIPTION:" not in ical


def test_export_omits_location_when_not_in_metadata():
    """Test that LOCATION is omitted when not in metadata."""
    unit = _unit("unit-1", "Title", metadata={"other": "value"})
    ical = export_units_to_ical([unit])

    assert "LOCATION:" not in ical


def test_export_omits_categories_when_no_tags():
    """Test that CATEGORIES is omitted when there are no tags."""
    unit = _unit("unit-1", "Title", tags=[])
    ical = export_units_to_ical([unit])

    assert "CATEGORIES:" not in ical


# --- Timezone handling ---


def test_export_handles_utc_timestamps():
    """Test UTC timestamp formatting."""
    dt = datetime(2026, 3, 15, 14, 30, 0, tzinfo=timezone.utc)
    unit = _unit("unit-1", "Title", created_at=dt)
    ical = export_units_to_ical([unit])

    assert "DTSTART:20260315T143000Z\r\n" in ical


def test_export_converts_non_utc_timezone_to_utc():
    """Test that non-UTC timezones are converted to UTC."""
    jst = timezone(timedelta(hours=9))
    dt = datetime(2026, 3, 15, 23, 30, 0, tzinfo=jst)  # 14:30 UTC
    unit = _unit("unit-1", "Title", created_at=dt)
    ical = export_units_to_ical([unit])

    assert "DTSTART:20260315T143000Z\r\n" in ical


def test_export_handles_naive_datetime_as_utc():
    """Test that naive datetimes are treated as UTC."""
    dt = datetime(2026, 6, 1, 12, 0, 0)
    unit = _unit("unit-1", "Title", created_at=dt)
    ical = export_units_to_ical([unit])

    assert "DTSTART:20260601T120000Z\r\n" in ical


# --- Multiple units export ---


def test_export_multiple_units():
    """Test exporting multiple units generates multiple VEVENTs."""
    units = [
        _unit("unit-1", "First"),
        _unit("unit-2", "Second"),
        _unit("unit-3", "Third"),
    ]
    ical = export_units_to_ical(units)

    assert ical.count("BEGIN:VEVENT") == 3
    assert ical.count("END:VEVENT") == 3
    assert "UID:unit-1@graphknowledge" in ical
    assert "UID:unit-2@graphknowledge" in ical
    assert "UID:unit-3@graphknowledge" in ical


def test_export_multiple_units_all_within_single_calendar():
    """Test that all VEVENTs are within a single VCALENDAR."""
    units = [_unit(f"unit-{i}", f"Unit {i}") for i in range(5)]
    ical = export_units_to_ical(units)

    assert ical.count("BEGIN:VCALENDAR") == 1
    assert ical.count("END:VCALENDAR") == 1


# --- Units without timestamps ---


def test_export_unit_without_explicit_timestamps_uses_defaults():
    """Test that units without explicit timestamps still generate valid events."""
    unit = KnowledgeUnit(
        id="unit-no-ts",
        source_project=SourceProject.MAX,
        source_id="source-no-ts",
        source_entity_type="note",
        title="No Timestamps",
        content="Content",
        content_type=ContentType.INSIGHT,
    )
    ical = export_units_to_ical([unit])

    # Should still have DTSTART (from default created_at)
    assert "DTSTART:" in ical
    assert "DTSTAMP:" in ical


# --- Empty graph export ---


def test_export_empty_units():
    """Test exporting empty iterable produces valid but event-less calendar."""
    ical = export_units_to_ical([])

    assert "BEGIN:VCALENDAR\r\n" in ical
    assert "END:VCALENDAR\r\n" in ical
    assert "VERSION:2.0\r\n" in ical
    assert "BEGIN:VEVENT" not in ical


# --- Special characters in text fields ---


def test_export_escapes_commas_in_text():
    """Test that commas are escaped per RFC 5545."""
    unit = _unit("unit-1", "Hello, World", "First, second, third")
    ical = export_units_to_ical([unit])

    assert "SUMMARY:Hello\\, World\r\n" in ical
    assert "DESCRIPTION:First\\, second\\, third\r\n" in ical


def test_export_escapes_semicolons_in_text():
    """Test that semicolons are escaped per RFC 5545."""
    unit = _unit("unit-1", "A;B;C", "x;y;z")
    ical = export_units_to_ical([unit])

    assert "SUMMARY:A\\;B\\;C\r\n" in ical
    assert "DESCRIPTION:x\\;y\\;z\r\n" in ical


def test_export_escapes_backslashes_in_text():
    """Test that backslashes are escaped per RFC 5545."""
    unit = _unit("unit-1", "path\\to\\file", "C:\\Users\\test")
    ical = export_units_to_ical([unit])

    assert "SUMMARY:path\\\\to\\\\file\r\n" in ical
    assert "DESCRIPTION:C:\\\\Users\\\\test\r\n" in ical


def test_export_escapes_newlines_in_text():
    """Test that newlines are escaped to literal \\n per RFC 5545."""
    unit = _unit("unit-1", "Title", "Line 1\nLine 2\nLine 3")
    ical = export_units_to_ical([unit])

    assert "DESCRIPTION:Line 1\\nLine 2\\nLine 3\r\n" in ical


def test_export_escapes_crlf_in_text():
    """Test that CRLF sequences are escaped to single \\n."""
    unit = _unit("unit-1", "Title", "Line 1\r\nLine 2\r\nLine 3")
    ical = export_units_to_ical([unit])

    assert "DESCRIPTION:Line 1\\nLine 2\\nLine 3\r\n" in ical


def test_export_handles_unicode_in_all_fields():
    """Test that Unicode characters are preserved."""
    unit = _unit(
        "unit-unicode",
        "日本語タイトル",
        "内容: 你好世界 🌍",
        tags=["タグ", "标签"],
        metadata={"location": "東京都渋谷区"},
    )
    ical = export_units_to_ical([unit])

    assert "SUMMARY:日本語タイトル" in ical
    assert "DESCRIPTION:内容: 你好世界 🌍" in ical
    assert "LOCATION:東京都渋谷区" in ical


def test_export_handles_special_chars_in_tags():
    """Test escaping of special characters in CATEGORIES (tags)."""
    unit = _unit("unit-1", "Title", tags=["tag,with,commas", "tag;semi"])
    ical = export_units_to_ical([unit])

    # Commas and semicolons in individual tag values should be escaped
    assert "CATEGORIES:tag\\,with\\,commas,tag\\;semi\r\n" in ical


# --- Maximum field length handling (line folding) ---


def test_export_folds_long_lines():
    """Test that lines exceeding 75 octets are folded per RFC 5545."""
    long_title = "A" * 200
    unit = _unit("unit-1", long_title)
    ical = export_units_to_ical([unit])

    # The SUMMARY line should be folded (continuation lines start with space)
    lines = ical.split("\r\n")
    # Find SUMMARY line and check folding
    summary_found = False
    for i, line in enumerate(lines):
        if line.startswith("SUMMARY:"):
            summary_found = True
            # Check that it's followed by continuation lines
            assert len(line.encode("utf-8")) <= 75
            # Next line should be a continuation (starts with space)
            if i + 1 < len(lines):
                assert lines[i + 1].startswith(" ")
            break
    assert summary_found


def test_export_folds_long_description():
    """Test that long DESCRIPTION is properly folded."""
    long_content = "B" * 300
    unit = _unit("unit-1", "Title", long_content)
    ical = export_units_to_ical([unit])

    # Unfold and verify content is intact
    unfolded = _unfold_ical(ical)
    assert f"DESCRIPTION:{long_content}" in unfolded


def test_export_folds_respects_utf8_boundaries():
    """Test that line folding doesn't split multi-byte UTF-8 characters."""
    # Each Japanese character is 3 bytes in UTF-8
    # 25 characters = 75 bytes, which is at the boundary
    content = "あ" * 50  # 150 bytes
    unit = _unit("unit-1", "Title", content)
    ical = export_units_to_ical([unit])

    # Should be valid UTF-8 on all lines
    for line in ical.split("\r\n"):
        line.encode("utf-8")  # Should not raise

    # Unfold and verify content
    unfolded = _unfold_ical(ical)
    assert f"DESCRIPTION:{content}" in unfolded


# --- RFC 5545 compliance ---


def test_export_uses_crlf_line_endings():
    """Test that output uses CRLF line endings per RFC 5545."""
    ical = export_units_to_ical([_unit("u1", "T")])

    # Split by \r\n should yield more parts than splitting by just \n
    # (all line breaks should be CRLF)
    assert "\r\n" in ical
    # Remove all CRLF then check no bare LF remains
    stripped = ical.replace("\r\n", "")
    assert "\n" not in stripped
    assert "\r" not in stripped


def test_export_ends_with_crlf():
    """Test that output ends with CRLF."""
    ical = export_units_to_ical([_unit("u1", "T")])
    assert ical.endswith("\r\n")


def test_export_vcalendar_contains_version():
    """Test VERSION property is present and correct."""
    ical = export_units_to_ical([_unit("u1", "T")])
    assert "VERSION:2.0\r\n" in ical


def test_export_vevent_uid_is_unique_per_unit():
    """Test that UIDs are unique across units."""
    units = [_unit(f"unit-{i}", f"Title {i}") for i in range(10)]
    ical = export_units_to_ical(units)

    uids = []
    for line in _unfold_ical(ical).split("\r\n"):
        if line.startswith("UID:"):
            uids.append(line)
    assert len(uids) == 10
    assert len(set(uids)) == 10


def test_export_dtstart_format_is_utc():
    """Test DTSTART uses UTC format (ending with Z)."""
    ical = export_units_to_ical([_unit("u1", "T")])

    unfolded = _unfold_ical(ical)
    for line in unfolded.split("\r\n"):
        if line.startswith("DTSTART:"):
            value = line.split(":", 1)[1]
            assert value.endswith("Z")
            # Format: YYYYMMDDTHHMMSSz
            assert len(value) == 16


# --- Registry integration ---


def test_ical_exporter_registered_in_registry():
    """Test that iCal exporter is registered in the export registry."""
    exporters = list_exporters()
    assert "ical" in exporters


def test_get_ical_exporter_from_registry():
    """Test retrieving iCal exporter from registry."""
    exporter = get_exporter("ical")
    assert exporter is export_units_to_ical


def test_ical_exporter_works_through_registry():
    """Test that iCal export works when called through registry."""
    units = [_unit("unit-1", "Registry Test")]
    ical = export_units("ical", units, calendar_name="Registry Cal")

    assert "BEGIN:VCALENDAR" in ical
    assert "X-WR-CALNAME:Registry Cal" in ical
    assert "SUMMARY:Registry Test" in ical


# --- Edge cases ---


def test_export_unit_with_empty_title():
    """Test unit with empty title omits SUMMARY."""
    unit = _unit("unit-1", "")
    ical = export_units_to_ical([unit])

    assert "SUMMARY:" not in ical


def test_export_unit_with_empty_id_generates_uid():
    """Test unit with empty id still gets a UID."""
    unit = KnowledgeUnit(
        id="",
        source_project=SourceProject.MAX,
        source_id="source-1",
        source_entity_type="note",
        title="No ID",
        content="Content",
        content_type=ContentType.INSIGHT,
        created_at=UNIT_TIME,
        ingested_at=INGESTED_TIME,
        updated_at=UPDATED_TIME,
    )
    ical = export_units_to_ical([unit])

    assert "UID:" in ical
    # Should end with @graphknowledge
    unfolded = _unfold_ical(ical)
    for line in unfolded.split("\r\n"):
        if line.startswith("UID:"):
            assert line.endswith("@graphknowledge")


def test_export_large_collection():
    """Test export of a large collection."""
    units = [_unit(f"unit-{i}", f"Title {i}", f"Content {i}") for i in range(100)]
    ical = export_units_to_ical(units)

    assert ical.count("BEGIN:VEVENT") == 100
    assert ical.count("END:VEVENT") == 100


def test_export_accepts_generator_input():
    """Test that export works with generator input (not just lists)."""
    def gen():
        for i in range(3):
            yield _unit(f"unit-{i}", f"Title {i}")

    ical = export_units_to_ical(gen())
    assert ical.count("BEGIN:VEVENT") == 3


# --- Helper functions ---


def _unfold_ical(ical: str) -> str:
    """Unfold iCalendar content lines (reverse of folding)."""
    # Per RFC 5545: folded lines have CRLF followed by a single whitespace
    return ical.replace("\r\n ", "").replace("\r\n\t", "")
