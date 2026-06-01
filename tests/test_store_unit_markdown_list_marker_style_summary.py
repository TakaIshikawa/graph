from graph.store.unit_markdown_list_marker_style_summary import summarize_unit_markdown_list_marker_styles


def test_list_marker_style_summary_counts_styles_and_mixed_units():
    summary = summarize_unit_markdown_list_marker_styles(
        [
            {"id": "b", "content": "- a\n* b\n+ c\n1. d\nIV) e\n```\n- skip\n```"},
            {"id": "a", "content": "- only\n- same"},
        ]
    )

    assert summary["total_units"] == 2
    assert summary["units_with_lists"] == 2
    assert summary["marker_counts"] == {
        "ordered_numeric": 1,
        "ordered_roman": 1,
        "unordered_dash": 3,
        "unordered_plus": 1,
        "unordered_star": 1,
    }
    assert summary["units_with_mixed_markers"] == 1
    assert summary["examples"] == [
        {
            "unit_id": "b",
            "marker_counts": {
                "ordered_numeric": 1,
                "ordered_roman": 1,
                "unordered_dash": 1,
                "unordered_plus": 1,
                "unordered_star": 1,
            },
        }
    ]
