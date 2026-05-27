from __future__ import annotations

from graph.store.unit_markdown_block_id_summary import summarize_unit_markdown_block_ids


def test_block_id_summary_counts_duplicates_and_invalid_ids():
    summary = summarize_unit_markdown_block_ids([
        {"id": "u1", "content": "Para ^abc\n^dup\nbad ^bad!"},
        {"id": "u2", "content": "Other ^dup"},
    ])

    assert summary["block_id_count"] == 3
    assert summary["duplicate_block_id_count"] == 1
    assert summary["duplicate_block_id_samples"] == [{"unit_id": "u1", "block_id": "dup", "line_number": 2}, {"unit_id": "u2", "block_id": "dup", "line_number": 1}]
    assert summary["invalid_block_id_samples"] == [{"unit_id": "u1", "block_id": "bad!", "line_number": 3}]
