from __future__ import annotations

from graph.store.source_redirect_hint_summary import summarize_source_redirect_hints


def test_source_redirect_hint_summary_counts_redirect_hints_and_status_codes():
    summary = summarize_source_redirect_hints(
        [
            {"source_id": "b", "metadata": {"original_url": "https://old.test", "final_url": "https://new.test", "status_code": 301}},
            {"source_id": "a", "metadata": {"url": "https://same.test", "redirect_count": 2, "status_code": 302}},
            {"source_id": "plain", "metadata": {"url": "https://plain.test", "redirect_count": 0}},
        ]
    )

    assert summary["total_sources"] == 3
    assert summary["sources_with_redirect_hints"] == 2
    assert summary["redirected_source_count"] == 1
    assert summary["max_redirect_count"] == 2
    assert summary["status_code_counts"] == {"301": 1, "302": 1}
    assert [sample["source_id"] for sample in summary["samples"]] == ["a", "b"]


def test_source_redirect_hint_summary_prefers_top_level_values_and_sample_limit():
    summary = summarize_source_redirect_hints(
        [
            {
                "source_id": "s",
                "original_url": "https://top-old.test",
                "final_url": "https://top-new.test",
                "redirect_count": 3,
                "status_code": 308,
                "metadata": {"original_url": "https://meta-old.test", "final_url": "https://meta-new.test"},
            }
        ],
        sample_limit=1,
    )

    assert summary["samples"] == [
        {
            "source_id": "s",
            "original_url": "https://top-old.test",
            "final_url": "https://top-new.test",
            "redirect_count": "3",
            "status_code": "308",
        }
    ]
