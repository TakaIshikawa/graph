from graph.store import summarize_source_viewport_meta


def test_viewport_meta_summary_counts_directives_and_special_values():
    summary = summarize_source_viewport_meta(
        [
            {"source_id": "a", "viewport": "width=device-width, initial-scale=1"},
            {"source_id": "b", "metadata": {"meta_viewport": "user-scalable=no, width=320"}},
            {"source_id": "c", "content": '<meta name="viewport" content="width=device-width, user-scalable=0">'},
            {"source_id": "d", "viewport": "  "},
            {"source_id": "e"},
        ],
        sample_limit=2,
    )

    assert summary["total_sources"] == 5
    assert summary["sources_with_viewport_meta"] == 3
    assert summary["width_device_width_count"] == 2
    assert summary["initial_scale_count"] == 1
    assert summary["user_scalable_disabled_count"] == 2
    assert summary["missing_viewport_meta_count"] == 2
    assert summary["directive_counts"] == {"initial-scale": 1, "user-scalable": 2, "width": 3}
    assert [sample["source_id"] for sample in summary["samples"]] == ["a", "b"]
