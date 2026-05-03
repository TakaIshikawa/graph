from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_timeline_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_timeline_csv_is_deterministic_for_same_payload():
    timeline = {
        "buckets": [
            {
                "bucket": "2026-02",
                "start": "2026-02-01",
                "sources": {"max": 1},
                "total": 1,
            },
            {
                "bucket": "2026-01",
                "start": "2026-01-01",
                "sources": {"readwise": 1, "max": 1},
                "total": 2,
            },
        ],
        "stats": {"bucket": "month"},
    }

    first = export_source_timeline_csv(timeline)
    second = export_source_timeline_csv(timeline)

    assert first == second
    assert _rows(first) == [
        {
            "bucket_start": "2026-01-01",
            "bucket_end": "2026-01-31",
            "bucket_label": "2026-01",
            "source_project": "max",
            "source_entity_type": "unknown",
            "unit_count": "1",
            "top_titles": "",
        },
        {
            "bucket_start": "2026-01-01",
            "bucket_end": "2026-01-31",
            "bucket_label": "2026-01",
            "source_project": "readwise",
            "source_entity_type": "unknown",
            "unit_count": "1",
            "top_titles": "",
        },
        {
            "bucket_start": "2026-02-01",
            "bucket_end": "2026-02-28",
            "bucket_label": "2026-02",
            "source_project": "max",
            "source_entity_type": "unknown",
            "unit_count": "1",
            "top_titles": "",
        },
    ]


def test_source_timeline_csv_uses_stable_fallbacks_for_missing_fields():
    text = export_source_timeline_csv(
        {
            "buckets": [
                {
                    "bucket": "2026-01-01",
                    "start": "2026-01-01",
                    "sources": {None: 1},
                    "total": 1,
                }
            ],
            "events": [
                {
                    "date": "2026-01-01",
                    "source": None,
                    "title": None,
                }
            ],
            "stats": {"bucket": "day"},
        }
    )

    assert _rows(text) == [
        {
            "bucket_start": "2026-01-01",
            "bucket_end": "2026-01-01",
            "bucket_label": "2026-01-01",
            "source_project": "unknown",
            "source_entity_type": "unknown",
            "unit_count": "1",
            "top_titles": "Untitled",
        }
    ]


def test_source_timeline_csv_quotes_commas_quotes_and_newlines():
    text = export_source_timeline_csv(
        {
            "buckets": [
                {
                    "bucket": "2026-W01",
                    "start": "2025-12-29",
                    "sources": {"feed, alpha": 2},
                    "total": 2,
                }
            ],
            "events": [
                {
                    "date": "2026-01-01",
                    "source_project": "feed, alpha",
                    "source_entity_type": 'note "raw"',
                    "title": 'First, "quoted"\nTitle',
                },
                {
                    "date": "2026-01-02",
                    "source_project": "feed, alpha",
                    "source_entity_type": 'note "raw"',
                    "title": "Second title",
                },
            ],
            "stats": {"bucket": "week"},
        }
    )

    assert '"feed, alpha"' in text
    assert '"note ""raw"""' in text
    assert '"First, ""quoted"" Title; Second title"' in text
    assert _rows(text)[0] == {
        "bucket_start": "2025-12-29",
        "bucket_end": "2026-01-04",
        "bucket_label": "2026-W01",
        "source_project": "feed, alpha",
        "source_entity_type": 'note "raw"',
        "unit_count": "2",
        "top_titles": 'First, "quoted" Title; Second title',
    }


def test_source_timeline_csv_path_mode_writes_file_and_returns_rows_written(tmp_path):
    path = tmp_path / "reports" / "timeline.csv"
    timeline = {
        "buckets": [
            {
                "bucket": "2026",
                "start": "2026-01-01",
                "sources": {"max": 3},
                "total": 3,
            }
        ],
        "stats": {"bucket": "year"},
    }

    stats = export_source_timeline_csv(timeline, path)

    assert stats == {"path": str(path), "rows_written": 1}
    assert _rows(path.read_text(encoding="utf-8"))[0]["bucket_end"] == "2026-12-31"


def test_source_timeline_csv_is_importable_from_graph_export():
    from graph.export import export_source_timeline_csv as imported

    assert imported is export_source_timeline_csv
