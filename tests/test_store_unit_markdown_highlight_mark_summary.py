from graph.store.unit_markdown_highlight_mark_summary import summarize_unit_markdown_highlight_marks


def test_summarizes_closed_highlights_and_ignores_unclosed_or_fenced():
    report = summarize_unit_markdown_highlight_marks([
        {"id": "b", "content": "==one== and ==three==\n==unclosed\n```\n==hidden==\n```"},
        {"id": "a", "content": "none"},
    ])

    assert report["total_highlights"] == 2
    assert report["units"] == [
        {
            "unit_id": "a",
            "highlight_count": 0,
            "first_highlight": "",
            "min_highlight_length": 0,
            "max_highlight_length": 0,
            "average_highlight_length": 0,
        },
        {
            "unit_id": "b",
            "highlight_count": 2,
            "first_highlight": "one",
            "min_highlight_length": 3,
            "max_highlight_length": 5,
            "average_highlight_length": 4.0,
        },
    ]
