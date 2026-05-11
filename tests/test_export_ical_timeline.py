from __future__ import annotations

from datetime import date, datetime, timezone

from graph.export import export_units_to_ical_timeline
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    title: str,
    *,
    content: str = "Body",
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=["beta", "alpha"],
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def test_export_units_to_ical_timeline_exports_all_day_event():
    text = export_units_to_ical_timeline([unit("unit-a", "Alpha", metadata={"date": date(2026, 5, 3)})])

    assert text.startswith("BEGIN:VCALENDAR\r\n")
    assert "BEGIN:VEVENT\r\n" in text
    assert "UID:unit-a@graph.local\r\n" in text
    assert "DTSTART;VALUE=DATE:20260503\r\n" in text
    assert "SUMMARY:Alpha\r\n" in text
    assert text.endswith("END:VCALENDAR\r\n")


def test_export_units_to_ical_timeline_exports_datetime_event():
    text = export_units_to_ical_timeline(
        [unit("unit-a", "Alpha", metadata={"scheduled_at": "2026-05-03T10:30:00+09:00"})]
    )

    assert "DTSTART:20260503T013000Z\r\n" in text


def test_export_units_to_ical_timeline_escapes_text_and_uses_deterministic_uid():
    text = export_units_to_ical_timeline(
        [
            unit(
                "unit-a",
                "Title, semi; slash\\ line\nnext",
                content="Line 1\nLine, 2; slash\\",
                metadata={"date": "2026-05-03"},
            )
        ]
    )

    assert "UID:unit-a@graph.local\r\n" in text
    assert "SUMMARY:Title\\, semi\\; slash\\\\ line\\nnext\r\n" in text
    assert "DESCRIPTION:Line 1\\nLine\\, 2\\; slash\\\\\\nTags: alpha\\, beta\r\n" in text
    assert "DTSTAMP:19700101T000000Z\r\n" in text


def test_export_units_to_ical_timeline_sorts_events_deterministically():
    units = [
        unit("unit-b", "Beta", metadata={"date": "2026-05-04"}),
        unit("unit-a", "Alpha", metadata={"date": "2026-05-03"}),
    ]

    first = export_units_to_ical_timeline(units)
    second = export_units_to_ical_timeline(reversed(units))

    assert first == second
    assert first.index("SUMMARY:Alpha") < first.index("SUMMARY:Beta")


def test_export_units_to_ical_timeline_skips_items_without_usable_dates():
    text = export_units_to_ical_timeline(
        [
            unit("unit-a", "Alpha", metadata={"date": "not a date"}),
            unit("unit-b", "Beta", metadata={}),
        ]
    )

    assert "BEGIN:VEVENT" not in text
