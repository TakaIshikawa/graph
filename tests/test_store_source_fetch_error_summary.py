from __future__ import annotations

from graph.store.source_fetch_error_summary import source_fetch_error_summary
from graph.types.models import KnowledgeUnit


def _unit(source_id: str, metadata: dict, *, updated_at: str = "2026-05-01T00:00:00+00:00"):
    return KnowledgeUnit(
        id=f"unit-{source_id}",
        source_project="web",
        source_id=source_id,
        source_entity_type="page",
        title=source_id,
        content="",
        metadata=metadata,
        updated_at=updated_at,
    )


def test_source_fetch_error_summary_groups_repeated_errors_by_source_host_status_and_kind():
    units = [
        _unit(
            "a",
            {
                "url": "https://Example.com/a",
                "fetch_errors": [
                    {"status_code": 404, "error_kind": "http", "seen_at": "2026-05-01T01:00:00+00:00"},
                    {"status": 410, "kind": "http", "seen_at": "2026-05-02T01:00:00+00:00"},
                ],
            },
        ),
        _unit("a", {"source_url": "https://example.com/b", "status_code": 500, "error_kind": "timeout"}),
    ]

    assert source_fetch_error_summary(units) == [
        {
            "source_id": "a",
            "source_project": "web",
            "host": "example.com",
            "status_class": "4xx",
            "error_kind": "http",
            "count": 2,
            "last_seen_at": "2026-05-02T01:00:00+00:00",
        },
        {
            "source_id": "a",
            "source_project": "web",
            "host": "example.com",
            "status_class": "5xx",
            "error_kind": "timeout",
            "count": 1,
            "last_seen_at": "2026-05-01T00:00:00+00:00",
        },
    ]


def test_source_fetch_error_summary_handles_missing_metadata_and_no_error_data():
    units = [
        _unit("clean", {"url": "https://example.com/ok"}),
        _unit("missing", {"error": "parser failed"}),
    ]

    assert source_fetch_error_summary(units) == [
        {
            "source_id": "missing",
            "source_project": "web",
            "host": None,
            "status_class": None,
            "error_kind": None,
            "count": 1,
            "last_seen_at": "2026-05-01T00:00:00+00:00",
        }
    ]
