from __future__ import annotations

from graph.rag.answer_source_recency_mix_audit import audit_answer_source_recency_mix


def test_source_recency_mix_classifies_current_heavy():
    report = audit_answer_source_recency_mix("", [{"date": "2025-10-01"}, {"date": "2025-09-01"}, {"date": "2021-01-01"}], now="2026-01-01")

    assert report["recency_mix"] == "current-heavy"
    assert report["passes"] is True


def test_source_recency_mix_detects_acknowledged_mixed_sources():
    report = audit_answer_source_recency_mix(
        "This uses recent sources and older sources.",
        [{"date": "2025-10-01"}, {"date": "2020-01-01"}],
        now="2026-01-01",
    )

    assert report["recency_mix"] == "mixed"
    assert report["acknowledges_recency_mix"] is True
    assert report["passes"] is True


def test_source_recency_mix_unknown_without_dates():
    report = audit_answer_source_recency_mix("", [{"id": "a"}, {"id": "b"}], now="2026-01-01")

    assert report["recency_mix"] == "unknown"
    assert report["unknown_date_count"] == 2
