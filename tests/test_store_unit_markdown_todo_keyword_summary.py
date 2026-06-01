from graph.store.unit_markdown_todo_keyword_summary import summarize_unit_markdown_todo_keywords


def test_counts_default_keywords_outside_fences():
    report = summarize_unit_markdown_todo_keywords([
        {"id": "b", "content": "TODO: fix\nnote this\n```\nBUG hidden\n```"},
        {"id": "a", "content": "FIXME and bug\nHACK TODO"},
    ])

    assert report["keyword_counts"] == {"BUG": 1, "FIXME": 1, "HACK": 1, "NOTE": 1, "TODO": 2}
    assert report["units_with_keywords"] == 2
    assert report["total_keyword_occurrences"] == 6
    assert report["per_unit_top"] == [
        {"unit_id": "a", "keyword": "BUG", "count": 1},
        {"unit_id": "b", "keyword": "NOTE", "count": 1},
    ]


def test_allows_custom_keywords():
    report = summarize_unit_markdown_todo_keywords([{"id": "u", "content": "REVIEW todo REVIEW"}], keywords=["review"])

    assert report["keyword_counts"] == {"REVIEW": 2}
    assert report["total_keyword_occurrences"] == 2
