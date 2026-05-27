from __future__ import annotations

from graph.store.relation_metadata_empty_value_summary import summarize_relation_metadata_empty_values


def test_relation_metadata_empty_values_classifies_and_counts():
    summary = summarize_relation_metadata_empty_values(
        [
            {"id": "r1", "metadata": {"evidence": None, "note": " ", "tags": [], "ok": False}},
            {"id": "r2", "metadata": {"evidence": {}, "note": "", "weight": 0}},
            {"id": "r3"},
            {"id": "r4", "metadata": None},
            {"id": "r5", "metadata": {"note": "present", "items": [1]}},
        ]
    )

    assert summary == {
        "relation_count": 5,
        "relations_with_empty_metadata_count": 2,
        "empty_value_count": 5,
        "counts_by_key": [
            {"key": "evidence", "count": 2},
            {"key": "note", "count": 2},
            {"key": "tags", "count": 1},
        ],
        "counts_by_empty_kind": [
            {"empty_kind": "blank_string", "count": 2},
            {"empty_kind": "empty_dict", "count": 1},
            {"empty_kind": "empty_list", "count": 1},
            {"empty_kind": "null", "count": 1},
        ],
        "samples": [
            {"relation_id": "r1", "key": "evidence", "empty_kind": "null"},
            {"relation_id": "r1", "key": "note", "empty_kind": "blank_string"},
            {"relation_id": "r1", "key": "tags", "empty_kind": "empty_list"},
            {"relation_id": "r2", "key": "evidence", "empty_kind": "empty_dict"},
            {"relation_id": "r2", "key": "note", "empty_kind": "blank_string"},
        ],
    }


def test_relation_metadata_empty_values_sorts_keys_by_frequency_then_key():
    summary = summarize_relation_metadata_empty_values(
        [
            {"metadata": {"beta": None, "alpha": None}},
            {"metadata": {"beta": ""}},
        ]
    )

    assert summary["counts_by_key"] == [{"key": "beta", "count": 2}, {"key": "alpha", "count": 1}]
