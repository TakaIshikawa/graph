from __future__ import annotations

from graph.store.unit_content_encoding_issue_summary import (
    summarize_unit_content_encoding_issues,
)


def test_summarize_unit_content_encoding_issues_groups_counts_by_source():
    summary = summarize_unit_content_encoding_issues(
        [
            {"id": "clean", "source": "web", "content": "こんにちは"},
            {"id": "bad-web", "source": "web", "content": "FranÃ§ais �\x01"},
            {"id": "bad-api", "metadata": {"source": "api"}, "content": "Itâ€™s broken\x02"},
        ]
    )

    assert summary == {
        "total_units": 3,
        "affected_unit_count": 2,
        "source_summaries": [
            {
                "source": "api",
                "affected_unit_count": 1,
                "mojibake_issue_count": 2,
                "replacement_character_count": 0,
                "control_character_count": 1,
                "representative_unit_ids": ["bad-api"],
            },
            {
                "source": "web",
                "affected_unit_count": 1,
                "mojibake_issue_count": 2,
                "replacement_character_count": 1,
                "control_character_count": 1,
                "representative_unit_ids": ["bad-web"],
            },
        ],
    }


def test_summarize_unit_content_encoding_issues_caps_representative_unit_ids():
    summary = summarize_unit_content_encoding_issues(
        [
            {"id": f"u{i}", "source_project": "web", "content": "Bad Ã"}
            for i in range(4)
        ],
        sample_limit=2,
    )

    assert summary["affected_unit_count"] == 4
    assert summary["source_summaries"][0]["representative_unit_ids"] == ["u0", "u1"]


def test_summarize_unit_content_encoding_issues_uses_unknown_source():
    summary = summarize_unit_content_encoding_issues([{"id": "u1", "content": "\ufffd"}])

    assert summary["source_summaries"][0]["source"] == "unknown"
    assert summary["source_summaries"][0]["replacement_character_count"] == 1
