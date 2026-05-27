from __future__ import annotations

from graph.store.unit_tag_prefix_summary import summarize_unit_tag_prefixes


def test_summarize_unit_tag_prefixes_counts_tags_and_metadata_hashtags():
    summary = summarize_unit_tag_prefixes(
        [
            {"id": "u1", "tags": ["Area/Work", "plain"], "metadata": {"body": "See #project:alpha"}},
            {"id": "u2", "metadata": {"tags": ["area/home", "kind-note"]}},
        ]
    )

    assert summary["tag_count"] == 5
    assert summary["unprefixed_count"] == 1
    assert summary["prefixes"][0]["prefix"] == "Area"
    assert summary["prefixes"][0]["tag_count"] == 2
    assert summary["prefixes"][0]["unit_count"] == 2
