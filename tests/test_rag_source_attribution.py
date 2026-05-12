from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

from graph.rag import summarize_source_attribution


@dataclass
class ResultStub:
    id: str
    source_project: str
    title: str
    metadata: dict


def test_summarize_source_attribution_groups_by_source_project_with_unknown_fallback():
    rows = summarize_source_attribution(
        [
            {
                "id": "a",
                "source_project": "alpha",
                "title": "Alpha note",
                "metadata": {"author": "Ada"},
            },
            {
                "id": "b",
                "source_project": "alpha",
                "title": "Beta note",
                "metadata": {"author": "Ben", "url": "https://example.test"},
            },
            {"id": "c", "title": "Unknown note", "metadata": {"author": "Cy"}},
        ]
    )

    assert rows == [
        {
            "source_project": "alpha",
            "count": 2,
            "result_ids": ["a", "b"],
            "title_samples": ["Alpha note", "Beta note"],
            "metadata_key_coverage": {"author": 2, "url": 1},
            "latest_updated_at": None,
        },
        {
            "source_project": "unknown",
            "count": 1,
            "result_ids": ["c"],
            "title_samples": ["Unknown note"],
            "metadata_key_coverage": {"author": 1},
            "latest_updated_at": None,
        },
    ]


def test_summarize_source_attribution_supports_objects_nested_units_and_metadata_dates():
    rows = summarize_source_attribution(
        [
            ResultStub(
                id="object-1",
                source_project="objects",
                title="Object result",
                metadata={"updated_at": "2026-05-10T12:00:00Z", "kind": "memo"},
            ),
            {
                "unit": {
                    "id": "nested-1",
                    "source_project": "nested",
                    "title": "Nested result",
                    "metadata": {"published_at": "2026-04-01", "kind": "brief"},
                }
            },
            (
                {
                    "id": "tuple-1",
                    "source_project": "nested",
                    "title": "Tuple result",
                    "updated_at": datetime(2026, 5, 12, 9, 0, tzinfo=timezone.utc),
                    "metadata": {"kind": "brief", "score_note": "high"},
                },
                0.9,
            ),
        ]
    )

    assert rows == [
        {
            "source_project": "nested",
            "count": 2,
            "result_ids": ["nested-1", "tuple-1"],
            "title_samples": ["Nested result", "Tuple result"],
            "metadata_key_coverage": {"kind": 2, "published_at": 1, "score_note": 1},
            "latest_updated_at": "2026-05-12T09:00:00+00:00",
        },
        {
            "source_project": "objects",
            "count": 1,
            "result_ids": ["object-1"],
            "title_samples": ["Object result"],
            "metadata_key_coverage": {"kind": 1, "updated_at": 1},
            "latest_updated_at": "2026-05-10T12:00:00+00:00",
        },
    ]


def test_summarize_source_attribution_metadata_key_coverage_is_deterministic():
    first = summarize_source_attribution(
        [
            {"id": "b", "source_project": "alpha", "metadata": {"z": 1, "a": 1}},
            {"id": "a", "source_project": "alpha", "metadata": {"a": 2, "empty": ""}},
        ]
    )
    second = summarize_source_attribution(
        [
            {"id": "a", "source_project": "alpha", "metadata": {"a": 2, "empty": ""}},
            {"id": "b", "source_project": "alpha", "metadata": {"z": 1, "a": 1}},
        ]
    )

    assert first == second
    assert first[0]["metadata_key_coverage"] == {"a": 2, "z": 1}


def test_summarize_source_attribution_parses_top_level_and_metadata_date_values():
    rows = summarize_source_attribution(
        [
            {"id": "date", "source_project": "alpha", "updated_at": date(2026, 5, 1)},
            {
                "id": "metadata",
                "source_project": "alpha",
                "metadata": {"published_at": "2026-05-03"},
            },
            {"id": "invalid", "source_project": "alpha", "updated_at": "not-a-date"},
        ]
    )

    assert rows[0]["latest_updated_at"] == "2026-05-03T00:00:00+00:00"


def test_summarize_source_attribution_caps_title_samples_and_is_importable():
    rows = summarize_source_attribution(
        [
            {"id": str(index), "source_project": "alpha", "title": f"Title {index}"}
            for index in range(5)
        ]
    )

    assert rows[0]["title_samples"] == ["Title 0", "Title 1", "Title 2"]
    assert callable(summarize_source_attribution)
