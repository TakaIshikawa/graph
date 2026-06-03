from __future__ import annotations

from graph.store import summarize_source_content_type_charsets


def test_content_type_charset_summary_normalizes_and_groups_values():
    summary = summarize_source_content_type_charsets(
        [
            {"id": "a", "headers": {"Content-Type": 'Text/HTML; Charset="UTF-8"'}},
            {"id": "b", "metadata": {"content_type": "application/json"}},
            {"id": "c", "response_headers": {"CONTENT_TYPE": "text/plain; charset="}},
            {"id": "d"},
        ]
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_content_type"] == 3
    assert summary["missing_content_type_count"] == 1
    assert summary["rows"] == [
        {"media_type": "application/json", "charset": "", "charset_status": "missing", "count": 1, "source_ids": ["b"], "examples": ["application/json"]},
        {"media_type": "text/html", "charset": "utf-8", "charset_status": "present", "count": 1, "source_ids": ["a"], "examples": ['Text/HTML; Charset="UTF-8"']},
        {"media_type": "text/plain", "charset": "", "charset_status": "malformed", "count": 1, "source_ids": ["c"], "examples": ["text/plain; charset="]},
    ]
