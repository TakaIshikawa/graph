from __future__ import annotations

from dataclasses import dataclass, field

from graph.store.relation_directionality_summary import summarize_relation_directionality


@dataclass
class Relation:
    relation_type: str
    from_unit_id: str
    to_unit_id: str
    source_project: str = "docs"
    metadata: dict[str, object] = field(default_factory=dict)


def test_summarize_relation_directionality_counts_mapping_relations():
    summary = summarize_relation_directionality(
        [
            {"relation_type": "references", "source": "docs", "source_id": "a", "target_id": "b"},
            {"relation_type": "references", "source": "docs", "source_id": "b", "target_id": "a", "directed": False},
            {"relation_type": "references", "source": "docs", "source_id": "a", "target_id": "a"},
            {"relation_type": "references", "source": "docs", "source_id": "a"},
        ]
    )

    assert summary["rows"] == [
        {
            "relation_type": "references",
            "source": "docs",
            "relation_count": 4,
            "directed_count": 3,
            "undirected_count": 1,
            "self_loop_count": 1,
            "missing_endpoint_count": 1,
            "reciprocal_candidate_count": 1,
        }
    ]
    assert summary["total_relations"] == 4


def test_summarize_relation_directionality_supports_objects_metadata_and_sorting():
    summary = summarize_relation_directionality(
        [
            Relation("blocks", "b", "a", source_project="Beta", metadata={"is_directed": "no"}),
            {"type": "blocks", "metadata": {"source": "alpha", "source_id": "a", "target_id": "b", "directed": "false"}},
            Relation("blocks", "a", "b", source_project="Beta"),
            {"type": "associates", "source": "z", "target": "z", "metadata": {"source": "alpha"}},
        ]
    )

    assert [(row["relation_type"], row["source"]) for row in summary["rows"]] == [
        ("associates", "alpha"),
        ("blocks", "alpha"),
        ("blocks", "Beta"),
    ]
    assert summary["rows"][0]["self_loop_count"] == 1
    assert summary["rows"][1]["undirected_count"] == 1
    assert summary["rows"][2]["directed_count"] == 1
    assert summary["rows"][2]["undirected_count"] == 1
    assert summary["rows"][2]["reciprocal_candidate_count"] == 1
