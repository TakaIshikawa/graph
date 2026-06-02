from graph.store import summarize_source_accept_ch_hints


def test_accept_ch_summary_splits_deduplicates_and_counts_hints():
    summary = summarize_source_accept_ch_hints(
        [
            {"id": "b", "headers": {"Accept-CH": "DPR, Width, DPR"}},
            {"id": "a", "metadata": {"accept_ch": "Viewport-Width"}},
            {"id": "c"},
        ]
    )

    assert summary["sources_with_accept_ch"] == 2
    assert summary["missing_accept_ch_count"] == 1
    assert summary["total_accept_ch_hints"] == 3
    assert summary["hint_counts"] == {"dpr": 1, "viewport-width": 1, "width": 1}
    assert summary["samples"][0]["source_id"] == "a"


def test_accept_ch_summary_respects_sample_limit():
    summary = summarize_source_accept_ch_hints([{"id": "a", "Accept-CH": "DPR, Width"}], sample_limit=1)

    assert len(summary["samples"]) == 1
