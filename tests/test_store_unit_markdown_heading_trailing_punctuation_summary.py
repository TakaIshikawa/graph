from graph.store.unit_markdown_heading_trailing_punctuation_summary import summarize_unit_markdown_heading_trailing_punctuation


def test_heading_trailing_punctuation_summary_parses_atx_and_closing_hashes():
    summary = summarize_unit_markdown_heading_trailing_punctuation(
        [
            {"id": "b", "content": "# Title:\n#### Question? ###\n####### no\n# Plain"},
            {"id": "a", "content": "## Stop. ##\n### Wow!\n###### Last,"},
        ],
        sample_limit=3,
    )

    assert summary["total_units"] == 2
    assert summary["heading_count"] == 6
    assert summary["headings_with_trailing_punctuation"] == 5
    assert summary["affected_units"] == 2
    assert summary["punctuation_counts"] == {"colon": 1, "comma": 1, "exclamation": 1, "period": 1, "question": 1}
    assert summary["examples"] == [
        {"unit_id": "a", "line": 1, "level": 2, "punctuation": "period", "text": "Stop."},
        {"unit_id": "a", "line": 2, "level": 3, "punctuation": "exclamation", "text": "Wow!"},
        {"unit_id": "a", "line": 3, "level": 6, "punctuation": "comma", "text": "Last,"},
    ]
