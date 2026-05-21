from __future__ import annotations

from graph.rag.result_evidence_age_mix import analyze_result_evidence_age_mix


def test_result_evidence_age_mix_buckets_dates_deterministically():
    report = analyze_result_evidence_age_mix(
        [
            {"date": "2026-04-20"},
            {"date": "2026-01-01"},
            {"date": "2025-06-01"},
            {"date": "2024-01-01"},
            {"id": "missing"},
        ],
        reference_date="2026-05-01",
    )

    assert report["counts"] == {"fresh": 1, "recent": 1, "aging": 1, "stale": 1, "undated": 1}
    assert report["oldest_date"] == "2024-01-01"
    assert report["newest_date"] == "2026-04-20"


def test_result_evidence_age_mix_warns_for_heavy_mixes():
    undated = analyze_result_evidence_age_mix([{}, {"date": "2026-01-01"}], reference_date="2026-05-01")
    stale = analyze_result_evidence_age_mix([{"date": "2020-01-01"}, {"date": "2021-01-01"}], reference_date="2026-05-01")

    assert undated["warnings"] == ["undated_heavy_evidence_mix"]
    assert stale["warnings"] == ["stale_heavy_evidence_mix"]
