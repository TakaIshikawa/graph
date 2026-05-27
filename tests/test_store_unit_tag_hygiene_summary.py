from __future__ import annotations

from graph.store import summarize_unit_tag_hygiene


def test_tag_hygiene_groups_formatting_issues_and_duplicates():
    summary = summarize_unit_tag_hygiene(
        [
            {"id": "clean", "tags": ["research"]},
            {"id": "messy", "tags": [" AI ", "ai", "a  b", "!!!", ""]},
            {"id": "other", "metadata": {"tags": ["AI"]}},
        ]
    )

    rows = {(row["issue_type"], row["normalized_tag"]): row for row in summary["rows"]}
    assert ("surrounding_whitespace", "ai") in rows
    assert rows[("duplicate_normalized_tag", "ai")]["example_unit_ids"] == ["messy"]
    assert rows[("uppercase_variant", "ai")]["unit_count"] == 2
    assert rows[("repeated_whitespace", "a b")]["tag"] == "a  b"
    assert rows[("punctuation_heavy", "!!!")]["unit_count"] == 1
    assert rows[("empty_tag", "")]["unit_count"] == 1
    assert all(row["normalized_tag"] != "research" for row in summary["rows"])
