from __future__ import annotations

from types import SimpleNamespace

from graph.rag.context_duplication_risk import analyze_context_duplication_risk


def test_context_duplication_risk_detects_url_id_and_overlap_duplicates():
    report = analyze_context_duplication_risk(
        [
            {"id": "a", "url": "https://example.test/one", "content": "alpha beta gamma delta"},
            {"id": "b", "metadata": {"url": "https://example.test/one"}, "content": "different"},
            SimpleNamespace(id="c", metadata={"source_id": "shared"}, content="topic one two"),
            {"id": "d", "source_id": "shared", "content": "topic three"},
            ({"id": "e", "content": "retrieval ranking context window planning"}, 0.8),
            {"id": "f", "snippet": "retrieval ranking context planning budget"},
        ],
        min_overlap_ratio=0.6,
    )

    assert report["total_results"] == 6
    assert report["duplicate_count"] == 6
    assert report["duplicate_ratio"] == 1.0
    assert {"result_ids": ["a", "b"], "reasons": ["url"], "group_size": 2} in report["duplicate_groups"]
    assert any(group["result_ids"] == ["e", "f"] and group["reasons"] == ["content_overlap"] for group in report["duplicate_groups"])
    assert report["warnings"] == ["excessive_repeated_context"]
