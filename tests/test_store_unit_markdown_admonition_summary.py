from graph.store import summarize_unit_markdown_admonitions


def test_summary_counts_admonitions_and_callouts():
    summary = summarize_unit_markdown_admonitions([{"content": "!!! note\nx"}, {"content": "??? warning\nx"}, {"content": "> [!TIP]\n> x"}])
    assert summary["total_admonitions"] == 3
    assert summary["units_with_admonitions"] == 3
    assert summary["by_kind"] == {"note": 1, "tip": 1, "warning": 1}
    assert summary["by_syntax"] == {"admonition": 2, "obsidian_callout": 1}


def test_empty_input():
    assert summarize_unit_markdown_admonitions([]) == {"total_admonitions": 0, "units_with_admonitions": 0, "by_kind": {}, "by_syntax": {}}
