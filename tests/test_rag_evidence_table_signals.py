from __future__ import annotations

from graph.rag.evidence_table_signals import extract_evidence_table_signals


def test_evidence_table_signals_detects_markdown_tables():
    payload = extract_evidence_table_signals(
        "revenue by region",
        [{"id": "r1", "content": "| region | revenue |\n|---|---|\n| us | 10 |"}],
    )

    assert payload["detected_columns"] == ["region", "revenue"]
    assert payload["row_count_estimates"] == {"r1": 2}
    assert payload["suitability_score"] > 0.5


def test_evidence_table_signals_detects_csv_like_snippets():
    payload = extract_evidence_table_signals("latency p95", [{"id": "csv", "text": "service,p50,p95\napi,10,20\nweb,8,15"}])

    assert payload["table_like_results"] == [{"result_id": "csv", "columns": ["service", "p50", "p95"], "row_count": 2}]
    assert payload["detected_columns"] == ["service", "p50", "p95"]


def test_evidence_table_signals_scores_prose_only_evidence_as_unsuitable():
    payload = extract_evidence_table_signals("latency", [{"id": "p", "text": "Latency improved after caching."}])

    assert payload == {
        "table_like_results": [],
        "detected_columns": [],
        "row_count_estimates": {},
        "suitability_score": 0.0,
    }
