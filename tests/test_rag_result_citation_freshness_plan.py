from __future__ import annotations

from datetime import date

from graph.rag.result_citation_freshness_plan import plan_result_citation_freshness


def test_classifies_fresh_aging_stale_and_undated_results():
    report = plan_result_citation_freshness(
        "background history",
        [
            {"id": "fresh", "published_at": "2025-12-20"},
            {"id": "aging", "date": "2025-01-01"},
            {"id": "stale", "updated_at": "2020-01-01"},
            {"id": "undated", "title": "No date"},
        ],
        reference_date=date(2026, 1, 1),
    )

    statuses = {row["result_id"]: row["freshness_status"] for row in report["rows"]}
    assert statuses == {"fresh": "fresh", "aging": "aging", "stale": "stale", "undated": "undated"}
    assert report["refresh_needed_count"] == 2


def test_uses_stricter_thresholds_for_latest_queries():
    report = plan_result_citation_freshness(
        "latest policy update",
        [{"id": "r1", "published_at": "2025-10-01"}],
        reference_date=date(2026, 1, 1),
    )

    assert report["rows"][0]["freshness_status"] == "aging"
    assert report["rows"][0]["recommended_action"] == "check_freshness"
