from __future__ import annotations

from types import SimpleNamespace

from graph.store.unit_orphan_summary import summarize_unit_orphans


def test_unit_orphan_summary_groups_by_source_and_entity_type():
    summary = summarize_unit_orphans(
        [
            {"id": "u2", "source_project": "docs", "source_entity_type": "page"},
            {"id": "u1", "source_project": "docs", "source_entity_type": "page"},
            {"id": "u3", "source_project": "crm", "source_entity_type": "contact"},
            {"id": "u4", "source_project": "docs", "source_entity_type": "page"},
        ],
        [{"from_unit_id": "u1", "to_unit_id": "u3"}, {"from_unit_id": "", "to_unit_id": None}],
    )

    assert summary == {
        "rows": [
            {"source": "docs", "entity_type": "page", "orphan_count": 2, "unit_ids": ["u2", "u4"], "total_units": 3}
        ],
        "row_count": 1,
        "unit_count": 4,
    }


def test_unit_orphan_summary_supports_objects_missing_values_and_empty_input():
    assert summarize_unit_orphans([], []) == {"rows": [], "row_count": 0, "unit_count": 0}

    summary = summarize_unit_orphans(
        [
            SimpleNamespace(id="u1", metadata={"source": "api", "entity_type": "note"}),
            {"id": "u2"},
        ],
        [SimpleNamespace(source_unit_id="u1", target_unit_id="missing")],
    )

    assert summary["rows"] == [
        {"source": None, "entity_type": None, "orphan_count": 1, "unit_ids": ["u2"], "total_units": 1}
    ]
