from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.ical import ICalAdapter
from graph.types.models import SyncState


def test_ical_parses_event_with_tzid_america_new_york(tmp_path):
    """TZID-bearing timestamps should be converted to timezone-aware UTC."""
    cal = tmp_path / "calendar.ics"
    cal.write_text(
        """BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Test//Test//EN
BEGIN:VEVENT
UID:event-with-tzid@example.com
DTSTART;TZID=America/New_York:20260503T120000
DTEND;TZID=America/New_York:20260503T130000
SUMMARY:Meeting in New York
DESCRIPTION:Team sync
LOCATION:NYC Office
CREATED:20260501T100000Z
LAST-MODIFIED:20260502T150000Z
END:VEVENT
END:VCALENDAR
""",
        encoding="utf-8",
    )

    result = ICalAdapter(path=str(cal)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_entity_type == "calendar_event"
    assert unit.title == "Meeting in New York"
    assert unit.content_type == "artifact"

    # DTSTART 12:00 PM in America/New_York = 16:00 UTC (EDT, UTC-4)
    # DTEND 13:00 PM in America/New_York = 17:00 UTC (EDT, UTC-4)
    assert unit.metadata["start"] == "2026-05-03T16:00:00+00:00"
    assert unit.metadata["end"] == "2026-05-03T17:00:00+00:00"
    assert unit.metadata["created"] == "2026-05-01T10:00:00+00:00"
    assert unit.metadata["updated"] == "2026-05-02T15:00:00+00:00"

    # created_at should use start time converted to UTC
    assert unit.created_at == datetime(2026, 5, 3, 16, 0, tzinfo=timezone.utc)


def test_ical_parses_event_with_tzid_utc(tmp_path):
    """TZID=UTC should work correctly."""
    cal = tmp_path / "calendar.ics"
    cal.write_text(
        """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:event-utc@example.com
DTSTART;TZID=UTC:20260503T120000
SUMMARY:UTC Event
END:VEVENT
END:VCALENDAR
""",
        encoding="utf-8",
    )

    result = ICalAdapter(path=str(cal)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["start"] == "2026-05-03T12:00:00+00:00"


def test_ical_parses_all_day_date_event(tmp_path):
    """All-day dates (YYYYMMDD) should remain UTC-normalized."""
    cal = tmp_path / "calendar.ics"
    cal.write_text(
        """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:all-day@example.com
DTSTART:20260503
DTEND:20260504
SUMMARY:All Day Event
END:VEVENT
END:VCALENDAR
""",
        encoding="utf-8",
    )

    result = ICalAdapter(path=str(cal)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["start"] == "2026-05-03T00:00:00+00:00"
    assert unit.metadata["end"] == "2026-05-04T00:00:00+00:00"


def test_ical_parses_utc_timestamp_with_z_suffix(tmp_path):
    """UTC timestamps with Z suffix should parse correctly."""
    cal = tmp_path / "calendar.ics"
    cal.write_text(
        """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:utc-z@example.com
DTSTART:20260503T120000Z
DTEND:20260503T130000Z
SUMMARY:UTC Z Event
END:VEVENT
END:VCALENDAR
""",
        encoding="utf-8",
    )

    result = ICalAdapter(path=str(cal)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["start"] == "2026-05-03T12:00:00+00:00"
    assert unit.metadata["end"] == "2026-05-03T13:00:00+00:00"


def test_ical_parses_naive_timestamp_defaults_to_utc(tmp_path):
    """Naive timestamps without TZID should default to UTC."""
    cal = tmp_path / "calendar.ics"
    cal.write_text(
        """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:naive@example.com
DTSTART:20260503T120000
SUMMARY:Naive Event
END:VEVENT
END:VCALENDAR
""",
        encoding="utf-8",
    )

    result = ICalAdapter(path=str(cal)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["start"] == "2026-05-03T12:00:00+00:00"


def test_ical_handles_invalid_tzid_gracefully(tmp_path):
    """Invalid TZID should fall back to UTC and not abort the import."""
    cal = tmp_path / "calendar.ics"
    cal.write_text(
        """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:invalid-tz@example.com
DTSTART;TZID=Invalid/Timezone:20260503T120000
SUMMARY:Invalid TZ Event
END:VEVENT
END:VCALENDAR
""",
        encoding="utf-8",
    )

    result = ICalAdapter(path=str(cal)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Should fall back to UTC
    assert unit.metadata["start"] == "2026-05-03T12:00:00+00:00"


def test_ical_handles_malformed_tzid_parameter(tmp_path):
    """Malformed TZID parameter should fall back gracefully."""
    cal = tmp_path / "calendar.ics"
    cal.write_text(
        """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:malformed@example.com
DTSTART;TZID=:20260503T120000
SUMMARY:Malformed TZID Event
END:VEVENT
END:VCALENDAR
""",
        encoding="utf-8",
    )

    result = ICalAdapter(path=str(cal)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    # Empty TZID should fall back to naive UTC
    assert unit.metadata["start"] == "2026-05-03T12:00:00+00:00"


def test_ical_preserves_multiple_timezone_events(tmp_path):
    """Multiple events with different timezones should be converted correctly."""
    cal = tmp_path / "calendar.ics"
    cal.write_text(
        """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:event-ny@example.com
DTSTART;TZID=America/New_York:20260503T100000
SUMMARY:NY Morning
END:VEVENT
BEGIN:VEVENT
UID:event-la@example.com
DTSTART;TZID=America/Los_Angeles:20260503T100000
SUMMARY:LA Morning
END:VEVENT
BEGIN:VEVENT
UID:event-london@example.com
DTSTART;TZID=Europe/London:20260503T100000
SUMMARY:London Morning
END:VEVENT
END:VCALENDAR
""",
        encoding="utf-8",
    )

    result = ICalAdapter(path=str(cal)).ingest()

    assert len(result.units) == 3

    # NY 10:00 AM EDT (UTC-4) = 14:00 UTC
    ny_unit = next(u for u in result.units if u.title == "NY Morning")
    assert ny_unit.metadata["start"] == "2026-05-03T14:00:00+00:00"

    # LA 10:00 AM PDT (UTC-7) = 17:00 UTC
    la_unit = next(u for u in result.units if u.title == "LA Morning")
    assert la_unit.metadata["start"] == "2026-05-03T17:00:00+00:00"

    # London 10:00 AM BST (UTC+1) = 09:00 UTC
    london_unit = next(u for u in result.units if u.title == "London Morning")
    assert london_unit.metadata["start"] == "2026-05-03T09:00:00+00:00"


def test_ical_since_filter_with_tzid_timestamps(tmp_path):
    """since filter should work correctly with TZID-converted timestamps."""
    cal = tmp_path / "calendar.ics"
    cal.write_text(
        """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:old-event@example.com
DTSTART;TZID=America/New_York:20260501T120000
LAST-MODIFIED:20260501T160000Z
SUMMARY:Old Event
END:VEVENT
BEGIN:VEVENT
UID:new-event@example.com
DTSTART;TZID=America/New_York:20260503T120000
LAST-MODIFIED:20260503T160000Z
SUMMARY:New Event
END:VEVENT
END:VCALENDAR
""",
        encoding="utf-8",
    )

    result = ICalAdapter(path=str(cal)).ingest(
        since=SyncState(
            source_project="ical",
            source_entity_type="calendar_event",
            last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
        )
    )

    assert len(result.units) == 1
    assert result.units[0].title == "New Event"


def test_ical_mixed_timestamp_formats_in_single_calendar(tmp_path):
    """Calendar with mixed timestamp formats should parse all correctly."""
    cal = tmp_path / "calendar.ics"
    cal.write_text(
        """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:tzid-event@example.com
DTSTART;TZID=America/New_York:20260503T120000
SUMMARY:TZID Event
END:VEVENT
BEGIN:VEVENT
UID:utc-event@example.com
DTSTART:20260503T120000Z
SUMMARY:UTC Event
END:VEVENT
BEGIN:VEVENT
UID:all-day-event@example.com
DTSTART:20260503
SUMMARY:All Day Event
END:VEVENT
BEGIN:VEVENT
UID:naive-event@example.com
DTSTART:20260503T120000
SUMMARY:Naive Event
END:VEVENT
END:VCALENDAR
""",
        encoding="utf-8",
    )

    result = ICalAdapter(path=str(cal)).ingest()

    assert len(result.units) == 4
    assert {u.title for u in result.units} == {
        "TZID Event",
        "UTC Event",
        "All Day Event",
        "Naive Event",
    }


def test_ical_event_with_attendees_and_tzid(tmp_path):
    """Event with attendees and TZID should parse correctly."""
    cal = tmp_path / "calendar.ics"
    cal.write_text(
        """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:meeting@example.com
DTSTART;TZID=America/New_York:20260503T140000
DTEND;TZID=America/New_York:20260503T150000
SUMMARY:Team Meeting
ORGANIZER;CN=Alice:mailto:alice@example.com
ATTENDEE;CN=Bob:mailto:bob@example.com
ATTENDEE;CN=Carol:mailto:carol@example.com
END:VEVENT
END:VCALENDAR
""",
        encoding="utf-8",
    )

    result = ICalAdapter(path=str(cal)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Team Meeting"
    # 14:00 EDT = 18:00 UTC
    assert unit.metadata["start"] == "2026-05-03T18:00:00+00:00"
    assert unit.metadata["organizer"] == "Alice <alice@example.com>"
    assert unit.metadata["attendees"] == [
        "Bob <bob@example.com>",
        "Carol <carol@example.com>",
    ]
