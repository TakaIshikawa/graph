from __future__ import annotations

from datetime import timedelta

from graph.store.saved_query_staleness_summary import saved_query_staleness_summary
from graph.types.models import KnowledgeUnit


def _unit(unit_id: str, source_project: str, updated_at: str):
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type="note",
        title=unit_id,
        content="",
        updated_at=updated_at,
    )


def test_saved_query_staleness_summary_reports_graph_updates_never_run_and_max_age():
    rows = saved_query_staleness_summary(
        [
            {"name": "fresh", "filters": {"source_project": "max"}, "last_run_at": "2026-05-03T00:00:00+00:00"},
            {"name": "stale", "filters": {"source_project": "max"}, "last_run_at": "2026-05-01T00:00:00+00:00"},
            {"name": "never", "filters": {}, "last_run_at": None, "created_at": "2026-05-01T00:00:00+00:00"},
            {"name": "old", "filters": {"source_project": "csv"}, "last_run_at": "2026-04-01T00:00:00+00:00"},
        ],
        [
            _unit("a", "max", "2026-05-02T00:00:00+00:00"),
            _unit("b", "csv", "2026-03-01T00:00:00+00:00"),
        ],
        now="2026-05-10T00:00:00+00:00",
        max_age=timedelta(days=7),
    )

    assert rows == [
        {
            "name": "fresh",
            "stale": False,
            "newest_relevant_update_at": "2026-05-02T00:00:00+00:00",
            "refresh_reasons": [],
        },
        {
            "name": "never",
            "stale": True,
            "newest_relevant_update_at": "2026-05-02T00:00:00+00:00",
            "refresh_reasons": ["never_run", "graph_updated"],
        },
        {
            "name": "old",
            "stale": True,
            "newest_relevant_update_at": "2026-03-01T00:00:00+00:00",
            "refresh_reasons": ["max_age_exceeded"],
        },
        {
            "name": "stale",
            "stale": True,
            "newest_relevant_update_at": "2026-05-02T00:00:00+00:00",
            "refresh_reasons": ["graph_updated", "max_age_exceeded"],
        },
    ]
