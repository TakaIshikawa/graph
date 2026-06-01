from __future__ import annotations

from graph.store import summarize_unit_markdown_link_title_attributes


def test_link_title_attribute_summary_counts_empty_repeated_and_dense_units():
    summary = summarize_unit_markdown_link_title_attributes(
        [
            {"id": "u1", "content": "[a](https://a.test \"Same\") [b](https://b.test 'Same') [c](https://c.test (  ))"},
            {"id": "u2", "content": "[d](https://d.test (Other)) [plain](https://skip.test)"},
        ],
        high_density_threshold=3,
    )

    assert summary["units_with_title_attributes"] == 2
    assert summary["title_attribute_count"] == 4
    assert summary["empty_title_attribute_count"] == 1
    assert summary["repeated_title_text"] == [{"title_text": "Same", "count": 2}]
    assert summary["high_density_units"] == [{"unit_id": "u1", "title_attribute_count": 3, "empty_title_count": 1}]
