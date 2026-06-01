from graph.store.unit_markdown_callout_title_summary import summarize_unit_markdown_callout_titles


def test_summarizes_callout_titles_and_ignores_fenced_content():
    report = summarize_unit_markdown_callout_titles([
        {"id": "b", "content": "> [!NOTE] Useful title\n> ordinary\n```\n> [!BUG] Hidden\n```"},
        {"id": "a", "content": "> [!WARNING]\n> [!note]+ Useful title"},
    ])

    assert report["callouts_with_titles"] == 2
    assert report["callouts_without_titles"] == 1
    assert report["title_counts"] == {"Useful title": 2}
    assert report["callout_type_counts"] == {"note": 2, "warning": 1}
    assert report["samples"] == [
        {"unit_id": "a", "line_number": 1, "callout_type": "warning", "title": ""},
        {"unit_id": "a", "line_number": 2, "callout_type": "note", "title": "Useful title"},
        {"unit_id": "b", "line_number": 1, "callout_type": "note", "title": "Useful title"},
    ]
