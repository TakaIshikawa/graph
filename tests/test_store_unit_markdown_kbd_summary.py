from graph.store import summarize_unit_markdown_kbd_usage


def test_kbd_summary_normalizes_keys_and_counts_sequences():
    summary = summarize_unit_markdown_kbd_usage(
        [{"id": "u", "content": "<kbd> Ctrl   K </kbd> <kbd>Esc</kbd>\n```\n<kbd>Skip</kbd>\n```\n<kbd>Ctrl+P</kbd>"}]
    )
    assert summary["total_units"] == 1
    assert summary["units_with_kbd"] == 1
    assert summary["kbd_count"] == 3
    assert summary["key_counts"] == {"Ctrl K": 1, "Ctrl+P": 1, "Esc": 1}
    assert summary["multi_key_sequence_count"] == 2


def test_kbd_summary_ignores_malformed_tags():
    assert summarize_unit_markdown_kbd_usage([{"content": "<kbd>Open"}])["kbd_count"] == 0
