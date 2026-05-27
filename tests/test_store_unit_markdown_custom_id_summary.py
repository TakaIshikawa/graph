from __future__ import annotations

from graph.store.unit_markdown_custom_id_summary import summarize_unit_markdown_custom_ids


def test_custom_id_summary_groups_counts_and_duplicates_by_source():
    summary = summarize_unit_markdown_custom_ids([
        {"id": "u1", "source": "s", "content": "# A {#a .lead k=v}\n{#a .x}"},
        {"id": "u2", "source": "s", "content": "```\n{#ignored .c k=v}\n```\n{#b}"},
    ])

    assert summary["sources"] == [
        {"source": "s", "unit_count": 2, "units_with_custom_ids": 2, "custom_id_count": 3, "duplicate_custom_id_count": 1, "class_attribute_count": 2, "key_value_attribute_count": 1}
    ]
