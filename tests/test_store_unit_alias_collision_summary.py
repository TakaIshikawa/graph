from __future__ import annotations

from dataclasses import dataclass

from graph.store.unit_alias_collision_summary import summarize_unit_alias_collisions


@dataclass
class Unit:
    id: str
    metadata: dict[str, object]
    source_project: str = ""
    source_entity_type: str = ""


def test_alias_collision_summary_normalizes_aliases_and_dedupes_per_unit():
    summary = summarize_unit_alias_collisions(
        [
            {"id": "a", "metadata": {"aliases": ["Alpha", " alpha "], "source": "s1", "entity_type": "note"}},
            Unit(id="b", metadata={"alias": "ALPHA"}, source_project="s2", source_entity_type="doc"),
            {"id": "c", "title": "Alpha"},
        ]
    )

    assert summary["collision_count"] == 1
    assert summary["rows"] == [
        {
            "alias": "ALPHA",
            "normalized_alias": "alpha",
            "unit_count": 2,
            "unit_ids": ["a", "b"],
            "sources": ["s1", "s2"],
            "entity_types": ["doc", "note"],
        }
    ]
