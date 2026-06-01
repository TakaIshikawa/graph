from graph.store.unit_markdown_escaped_character_summary import summarize_unit_markdown_escaped_characters


def test_summarize_unit_markdown_escaped_characters_counts_markdown_punctuation():
    report = summarize_unit_markdown_escaped_characters([
        {"id": "b", "content": r"\* star \_ under \[ open \] close \# hash"},
        {"id": "a", "content": r"slash \\ and star \*"},
    ])

    assert report["total_units"] == 2
    assert report["escaped_character_count"] == 7
    assert report["escaped_character_counts"] == {"#": 1, "*": 2, "\\": 1, "[": 1, "]": 1, "_": 1}
    assert report["affected_units"] == ["a", "b"]
    assert report["examples"][:2] == [
        {"unit_id": "a", "line_number": 1, "escaped_character": "*"},
        {"unit_id": "a", "line_number": 1, "escaped_character": "\\"},
    ]


def test_summarize_unit_markdown_escaped_characters_ignores_fenced_code_and_limits_examples():
    report = summarize_unit_markdown_escaped_characters(
        [{"id": "u", "content": "\\* one\n```\n\\# hidden\n```\n\\[ two"}],
        sample_limit=1,
    )

    assert report["escaped_character_count"] == 2
    assert report["escaped_character_counts"] == {"*": 1, "[": 1}
    assert report["examples"] == [{"unit_id": "u", "line_number": 1, "escaped_character": "*"}]
