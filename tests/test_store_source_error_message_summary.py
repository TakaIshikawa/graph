from __future__ import annotations

from graph.store import summarize_source_error_messages


def test_source_error_message_summary_discovers_nested_common_fields():
    report = summarize_source_error_messages(
        [
            {"id": "a", "source_project": "web", "metadata": {"import": {"error": " timed out "}}},
            {"id": "b", "source_project": "web", "metadata": {"warning": {"message": "low confidence"}}},
            {"id": "c", "source_project": "web", "metadata": {"batch": [{"failure_reason": "timed out"}]}},
        ]
    )

    assert report["messages"] == [
        {"source": "web", "severity": "error", "message": "timed out", "count": 2, "sample_unit_ids": ["a", "c"]},
        {"source": "web", "severity": "warning", "message": "low confidence", "count": 1, "sample_unit_ids": ["b"]},
    ]


def test_source_error_message_summary_uses_explicit_severity_and_sample_limit():
    report = summarize_source_error_messages(
        [
            {"id": "a", "source_project": "api", "metadata": {"errors": [{"message": "retry", "severity": "fatal"}]}},
            {"id": "b", "source_project": "api", "metadata": {"exception": "retry"}},
        ],
        sample_limit=1,
    )

    assert report["messages"][0]["severity"] == "error"
    assert report["messages"][0]["sample_unit_ids"] == ["b"]
    assert report["messages"][1]["severity"] == "fatal"
