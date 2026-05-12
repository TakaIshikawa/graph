from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from graph.rag.result_attribution_audit import audit_result_attribution
from graph.types.models import KnowledgeUnit


def test_audit_result_attribution_reports_default_field_coverage_and_sources():
    payload = audit_result_attribution(
        [
            {
                "id": "a",
                "source_project": "notes",
                "title": "Roadmap",
                "updated_at": "2026-05-10T10:00:00Z",
                "url": "https://example.test/a",
            },
            {
                "id": "b",
                "metadata": {"source_project": "web", "title": "Launch", "source_url": "https://example.test/b"},
            },
            {"title": "Untitled source"},
        ],
        limit=2,
    )

    assert payload["totals"] == {
        "result_count": 3,
        "required_fields": ["source", "title", "timestamp", "url", "stable_id"],
        "complete_result_count": 1,
    }
    assert payload["field_coverage"] == {
        "source": {"present": 2, "missing": 1},
        "title": {"present": 3, "missing": 0},
        "timestamp": {"present": 1, "missing": 2},
        "url": {"present": 2, "missing": 1},
        "stable_id": {"present": 2, "missing": 1},
    }
    assert payload["source_distribution"] == [
        {"source": "notes", "count": 1},
        {"source": "unknown", "count": 1},
        {"source": "web", "count": 1},
    ]
    assert payload["results"][1]["missing_fields"] == ["timestamp"]
    assert [row["result_id"] for row in payload["representative_rows"]] == ["a", "b"]


def test_audit_result_attribution_supports_objects_nested_units_and_scored_tuples():
    unit = KnowledgeUnit(
        id="unit-1",
        source_project="readwise",
        source_id="source-1",
        source_entity_type="highlight",
        title="Highlight",
        content="Text",
        metadata={"author": "Ada", "canonical_url": "https://book.test"},
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    wrapper = SimpleNamespace(id="wrapper", unit=unit)

    payload = audit_result_attribution(
        [
            wrapper,
            (
                {
                    "source_id": "tuple-source",
                    "source": "archive",
                    "name": "Archived page",
                    "date": "2026-02-03",
                    "link": "archive.test/page",
                },
                0.9,
            ),
        ],
        required_fields=["stable_id", "source", "author", "url", "timestamp"],
    )

    assert payload["field_coverage"]["author"] == {"present": 1, "missing": 1}
    assert payload["results"][0]["values"]["author"] == "Ada"
    assert payload["results"][0]["values"]["timestamp"] == "2026-01-02T00:00:00+00:00"
    assert payload["results"][1]["result_id"] == "tuple-source"
    assert payload["results"][1]["missing_fields"] == ["author"]


def test_audit_result_attribution_deduplicates_custom_fields():
    payload = audit_result_attribution(
        [{"id": "a", "title": "A"}],
        required_fields=["TITLE", "title", "stable_id"],
    )

    assert payload["totals"]["required_fields"] == ["title", "stable_id"]
    assert payload["field_coverage"] == {
        "title": {"present": 1, "missing": 0},
        "stable_id": {"present": 1, "missing": 0},
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": 0},
        {"limit": True},
        {"required_fields": []},
        {"required_fields": [""]},
        {"required_fields": ["unknown"]},
    ],
)
def test_audit_result_attribution_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        audit_result_attribution([], **kwargs)
