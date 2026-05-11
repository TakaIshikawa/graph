from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.export import export_units_to_timelinejs
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

CREATED_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)
INGESTED_TIME = datetime(2026, 5, 2, 8, 30, tzinfo=timezone.utc)
UPDATED_TIME = datetime(2026, 5, 3, 9, 45, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    title: str | None = None,
    *,
    content: str = "A compact research note.",
    metadata: dict | None = None,
    tags: list[str] | None = None,
    source_project: SourceProject | str = SourceProject.CSV,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title or f"Title {unit_id}",
        content=content,
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=tags or ["timeline", "export"],
        created_at=CREATED_TIME,
        ingested_at=INGESTED_TIME,
        updated_at=UPDATED_TIME,
    )


def events(text: str) -> list[dict]:
    return json.loads(text)["events"]


def test_export_units_to_timelinejs_emits_timeline_events_with_metadata_dates():
    data = json.loads(
        export_units_to_timelinejs(
            [
                unit(
                    "unit-a",
                    "Launch note",
                    content="The project launched.\n\nDetails follow.",
                    metadata={
                        "start_date": "2026-04-01T12:30:00+09:00",
                        "end_date": "2026-04-02",
                        "image_url": "https://example.test/launch.png",
                    },
                    tags=["release", "alpha", "release"],
                    source_project=SourceProject.PINBOARD,
                )
            ]
        )
    )

    assert data == {
        "events": [
            {
                "end_date": {"day": 2, "month": 4, "year": 2026},
                "group": "pinboard",
                "media": {"url": "https://example.test/launch.png"},
                "start_date": {"day": 1, "hour": 3, "minute": 30, "month": 4, "year": 2026},
                "tags": ["alpha", "release", "release"],
                "text": {
                    "headline": "Launch note",
                    "text": "The project launched. Details follow.",
                },
            }
        ]
    }


def test_export_units_to_timelinejs_sorts_events_by_date_then_unit_identity():
    text_a = export_units_to_timelinejs(
        [
            unit("unit-b", "Beta", metadata={"date": "2026-02-01"}),
            unit("unit-c", "Charlie", metadata={"date": "2026-01-01"}),
            unit("unit-a", "Alpha", metadata={"date": "2026-02-01"}),
        ]
    )
    text_b = export_units_to_timelinejs(
        [
            unit("unit-a", "Alpha", metadata={"date": "2026-02-01"}),
            unit("unit-c", "Charlie", metadata={"date": "2026-01-01"}),
            unit("unit-b", "Beta", metadata={"date": "2026-02-01"}),
        ]
    )

    assert text_a == text_b
    assert [event["text"]["headline"] for event in events(text_a)] == ["Charlie", "Alpha", "Beta"]


def test_export_units_to_timelinejs_falls_back_to_unit_timestamps():
    event = events(export_units_to_timelinejs(unit("unit-a", metadata={"date": "not a date"})))[0]

    assert event["start_date"] == {
        "year": 2026,
        "month": 5,
        "day": 1,
        "hour": 10,
        "minute": 15,
    }


def test_export_units_to_timelinejs_detects_common_date_and_url_metadata_keys():
    exported = events(
        export_units_to_timelinejs(
            [
                unit("published", metadata={"published_at": "2026-03-04", "source_url": "https://example.test/p"}),
                unit("event", metadata={"event_start": "2026-03-03", "media": {"url": "https://example.test/e.png"}}),
                unit("completed", metadata={"completed_at": "2026-03-05"}),
            ]
        )
    )

    assert [event["text"]["headline"] for event in exported] == [
        "Title event",
        "Title published",
        "Title completed",
    ]
    assert exported[0]["media"] == {"url": "https://example.test/e.png"}
    assert exported[1]["media"] == {"url": "https://example.test/p"}


def test_export_units_to_timelinejs_writes_file_and_returns_stats(tmp_path):
    path = tmp_path / "nested" / "timeline.json"

    stats = export_units_to_timelinejs([unit("unit-a"), unit("unit-b")], path)
    written = path.read_text(encoding="utf-8")

    assert json.loads(written)["events"][0]["text"]["headline"] == "Title unit-a"
    assert stats == {
        "path": str(path),
        "event_count": 2,
        "skipped_count": 0,
        "bytes_written": path.stat().st_size,
    }
