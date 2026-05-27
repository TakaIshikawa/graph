from __future__ import annotations

from graph.store import summarize_unit_yaml_block_scalars


def test_summarize_unit_yaml_block_scalars_counts_styles_and_chomping():
    summary = summarize_unit_yaml_block_scalars(
        [
            {"id": "a", "content": "---\ndescription: |-\n  one\nsummary: >+\n  two\n---\nBody"},
            {"id": "b", "content": "---\nnote: |\n  three\n---"},
        ]
    )

    assert summary["units_with_block_scalars"] == 2
    assert summary["block_scalar_count"] == 3
    assert summary["style_counts"] == [{"style": "folded", "count": 1}, {"style": "literal", "count": 2}]
    assert summary["chomping_counts"] == [{"chomping": "+", "count": 1}, {"chomping": "-", "count": 1}, {"chomping": "clip", "count": 1}]
