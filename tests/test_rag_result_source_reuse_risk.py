from __future__ import annotations

from graph.rag.result_source_reuse_risk import analyze_result_source_reuse_risk


class Result:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_result_source_reuse_risk_flags_single_source_high_risk():
    summary = analyze_result_source_reuse_risk([{"source": "docs"}, {"source": "Docs"}, {"source": "docs"}])

    assert summary["top_source"] == "docs"
    assert summary["reuse_ratio"] == 1.0
    assert summary["risk_level"] == "high"


def test_result_source_reuse_risk_flags_diverse_low_risk_and_empty():
    assert analyze_result_source_reuse_risk([{"source": "a"}, {"source": "b"}, {"source": "c"}])["risk_level"] == "low"
    assert analyze_result_source_reuse_risk([])["risk_level"] == "low"


def test_result_source_reuse_risk_uses_url_hostname_and_missing_source():
    summary = analyze_result_source_reuse_risk([{"url": "https://Example.com/a"}, Result(metadata={"url": "https://example.com/b"}), {}])

    assert summary["repeated_sources"] == [{"source": "example.com", "count": 2}]
    assert summary["unique_sources"] == 2
