from __future__ import annotations

from graph.rag.result_requery_plan import build_result_requery_plan


def test_result_requery_plan_handles_empty_results():
    payload = build_result_requery_plan("api latency", [], now="2025-06-01")

    assert payload == {
        "requery_needed": True,
        "followup_queries": ["api latency authoritative sources"],
        "reasons": ["empty results"],
        "priority": "high",
    }


def test_result_requery_plan_detects_missing_citation_metadata():
    payload = build_result_requery_plan("api latency", [{"source_id": "a", "date": "2025-01-01"}], now="2025-06-01")

    assert "missing citation metadata" in payload["reasons"]
    assert "api latency cited sources" in payload["followup_queries"]


def test_result_requery_plan_detects_stale_evidence():
    payload = build_result_requery_plan(
        "api latency",
        [{"id": "a", "source_id": "a", "date": "2020-01-01"}, {"id": "b", "source_id": "b", "date": "2020-01-02"}],
        now="2025-06-01",
    )

    assert payload["reasons"] == ["stale or missing dates"]
    assert payload["priority"] == "medium"


def test_result_requery_plan_accepts_sufficient_results():
    payload = build_result_requery_plan(
        "api latency",
        [{"id": "a", "source_id": "a", "date": "2025-01-01"}, {"id": "b", "source_id": "b", "date": "2025-01-02"}],
        now="2025-06-01",
    )

    assert payload == {"requery_needed": False, "followup_queries": [], "reasons": [], "priority": "low"}
