from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_edge_reciprocity_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_edge_reciprocity_csv_classifies_reciprocal_one_way_and_self_loop_edges():
    text = export_edge_reciprocity_csv(
        [
            {"source_id": "a", "target_id": "b", "relation": "supports"},
            {"source_id": "b", "target_id": "a", "relation": "cites"},
            {"source_id": "b", "target_id": "a", "relation": "mentions"},
            {"source_id": "a", "target_id": "c", "relation": "related"},
            {"source_id": "d", "target_id": "d", "relation": "self"},
        ]
    )

    assert rows(text) == [
        {"source_id": "a", "target_id": "b", "relation": "supports", "has_reverse": "true", "reverse_relations": "cites; mentions", "reciprocity_group": "reciprocal"},
        {"source_id": "a", "target_id": "c", "relation": "related", "has_reverse": "false", "reverse_relations": "", "reciprocity_group": "one_way"},
        {"source_id": "b", "target_id": "a", "relation": "cites", "has_reverse": "true", "reverse_relations": "supports", "reciprocity_group": "reciprocal"},
        {"source_id": "b", "target_id": "a", "relation": "mentions", "has_reverse": "true", "reverse_relations": "supports", "reciprocity_group": "reciprocal"},
        {"source_id": "d", "target_id": "d", "relation": "self", "has_reverse": "false", "reverse_relations": "", "reciprocity_group": "self_loop"},
    ]
