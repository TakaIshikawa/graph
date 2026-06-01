from __future__ import annotations

from graph.store.unit_markdown_mention_handle_summary import summarize_unit_markdown_mention_handles


def test_detects_handles_and_ignores_email_and_code():
    summary = summarize_unit_markdown_mention_handles(
        [
            {
                "id": "u1",
                "content": "Ping @alpha-team and @beta.user.\nEmail a@example.com and `@code`.\n```\n@ignored\n```",
            }
        ]
    )

    assert summary["mention_count"] == 2
    assert summary["units_with_mentions"] == 1
    assert summary["handle_counts"] == {"@alpha-team": 1, "@beta.user": 1}
    assert [sample["handle"] for sample in summary["samples"]] == ["@alpha-team", "@beta.user"]
