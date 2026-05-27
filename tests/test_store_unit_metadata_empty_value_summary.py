from __future__ import annotations

from graph.store import summarize_unit_metadata_empty_values


def test_unit_metadata_empty_values_reports_keys_and_units():
    summary = summarize_unit_metadata_empty_values(
        [
            {"id": "u1", "metadata": {"none": None, "blank": " ", "empty_list": [], "ok_false": False}},
            {"id": "u2", "metadata": {"empty_dict": {}, "ok_zero": 0, "ok_list": [1]}},
            {"id": "u3", "metadata": {"blank": "", "ok_text": "value"}},
        ]
    )

    assert summary == {
        "total_units": 3,
        "units_with_empty_metadata_values": 3,
        "key_counts": {"blank": 2, "empty_dict": 1, "empty_list": 1, "none": 1},
        "affected_unit_ids": ["u1", "u2", "u3"],
        "rows": [
            {"key": "blank", "empty_count": 2, "unit_ids": ["u1", "u3"]},
            {"key": "empty_dict", "empty_count": 1, "unit_ids": ["u2"]},
            {"key": "empty_list", "empty_count": 1, "unit_ids": ["u1"]},
            {"key": "none", "empty_count": 1, "unit_ids": ["u1"]},
        ],
    }
