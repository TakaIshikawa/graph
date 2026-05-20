from __future__ import annotations

from types import SimpleNamespace

import pytest

from graph.rag.source_freshness_risk import analyze_source_freshness_risk


def test_source_freshness_risk_summarizes_sources_deterministically():
    report = analyze_source_freshness_risk(
        [
            {"id": "a", "source_project": "docs", "updated_at": "2026-01-01"},
            ({"id": "b", "metadata": {"source": "docs", "published_at": "2025-01-01"}}, 0.9),
            SimpleNamespace(id="c", source_project="web", metadata={"date": "2024-01-01"}),
            {"id": "d", "source_id": "web"},
        ],
        now="2026-05-01",
        stale_after_days=180,
    )

    assert report["total_results"] == 4
    assert report["stale_count"] == 2
    assert report["missing_date_count"] == 1
    assert report["stale_ratio"] == 0.5
    assert report["sources"][0]["source"] == "web"
    assert report["sources"][0]["stale_ratio"] == 0.5
    assert report["sources"][0]["missing_date_ratio"] == 0.5
    assert report["sources"][1]["source"] == "docs"
    assert "missing_dates" in report["warnings"]
    assert "stale_concentration" in report["warnings"]


def test_source_freshness_risk_empty_and_invalid_threshold():
    assert analyze_source_freshness_risk([])["warnings"] == ["no_results"]
    with pytest.raises(ValueError, match="stale_after_days"):
        analyze_source_freshness_risk([], stale_after_days=0)
