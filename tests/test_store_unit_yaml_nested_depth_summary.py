from __future__ import annotations

from graph.store.unit_yaml_nested_depth_summary import summarize_unit_yaml_nested_depth


def test_summarize_unit_yaml_nested_depth_buckets_nested_metadata():
    summary = summarize_unit_yaml_nested_depth(
        [
            {"id": "empty"},
            {"id": "scalar", "metadata": {"title": "One"}},
            {"id": "deep", "metadata": {"a": [{"b": {"c": 1}}]}},
        ]
    )

    assert summary["total_units"] == 3
    assert summary["max_depth"] == 4
    assert summary["depth_buckets"] == {"0": 1, "1": 1, "4": 1}
    assert summary["deepest_examples"][0] == {"unit_id": "deep", "depth": 4, "key_path": "a[0].b.c"}
