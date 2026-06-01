from graph.store.unit_markdown_superscript_summary import summarize_unit_markdown_superscripts


def test_superscript_summary_counts_spans_and_ignores_fences_empty_and_punctuation():
    summary = summarize_unit_markdown_superscripts(
        [
            {"id": "b", "content": "Line ^z^\n```md\n^skip^\n```\n2 ^ 3 and ^^"},
            {"id": "a", "content": "^alpha^ and ^b^ and x^not^"},
        ],
        sample_limit=2,
    )

    assert summary["total_units"] == 2
    assert summary["units_with_superscript"] == 2
    assert summary["superscript_count"] == 3
    assert summary["most_common_text"] == "alpha"
    assert summary["samples"] == [
        {"unit_id": "a", "line_number": 1, "text": "alpha"},
        {"unit_id": "a", "line_number": 1, "text": "b"},
    ]


def test_superscript_summary_returns_deterministic_most_common_text():
    summary = summarize_unit_markdown_superscripts(
        [
            {"id": "c", "content": "^z^ ^a^"},
            {"id": "b", "content": "^z^"},
            {"id": "a", "content": "^a^"},
        ],
        sample_limit=10,
    )

    assert summary["superscript_count"] == 4
    assert summary["most_common_text"] == "a"
    assert summary["samples"] == [
        {"unit_id": "a", "line_number": 1, "text": "a"},
        {"unit_id": "b", "line_number": 1, "text": "z"},
        {"unit_id": "c", "line_number": 1, "text": "a"},
        {"unit_id": "c", "line_number": 1, "text": "z"},
    ]
