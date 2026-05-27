from graph.store import summarize_unit_markdown_heading_anchors


def test_summary_counts_heading_anchors_and_duplicates():
    summary = summarize_unit_markdown_heading_anchors([{"content": "## A {#same}"}, {"content": "# B {#same}\nText {#no}"}])
    assert summary["total_anchors"] == 2
    assert summary["units_with_anchors"] == 2
    assert summary["duplicate_anchor_ids"] == [{"anchor_id": "same", "count": 2}]
    assert summary["level_counts"] == {1: 1, 2: 1}
    assert summary["anchor_id_counts"] == {"same": 2}
