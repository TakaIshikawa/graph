from __future__ import annotations

from types import SimpleNamespace

from graph.store.relation_self_loop_summary import summarize_relation_self_loops


def test_relation_self_loop_summary_counts_loops_missing_endpoints_and_types():
    summary = summarize_relation_self_loops(
        [
            {"id": "r1", "relation": "mentions", "source_id": "u1", "target_id": "u1", "metadata": {"source": "docs"}},
            {"id": "r2", "relation": "blocks", "source_id": "u2", "target_id": "u2", "metadata": {"source": "imports"}},
            {"id": "r3", "relation": "mentions", "source_id": "u1", "target_id": "u2"},
            {"id": "r4", "relation": "mentions", "source_id": "", "target_id": "u3"},
            {"id": "r5", "relation": "mentions", "source_id": "u4"},
            {"id": "r6", "relation": "mentions", "source_id": "u5", "target_id": "u5", "metadata": {"source": "docs"}},
        ],
        sample_limit=1,
    )

    assert summary == {
        "total_relations": 6,
        "self_loop_count": 3,
        "missing_endpoint_count": 2,
        "relation_type_counts": [{"relation_type": "blocks", "count": 1}, {"relation_type": "mentions", "count": 2}],
        "metadata_source_counts": [{"metadata_source": "docs", "count": 2}, {"metadata_source": "imports", "count": 1}],
        "rows": [
            {
                "relation_type": "blocks",
                "metadata_source": "imports",
                "count": 1,
                "sample_relations": [{"relation_id": "r2", "endpoint_id": "u2"}],
            },
            {
                "relation_type": "mentions",
                "metadata_source": "docs",
                "count": 2,
                "sample_relations": [{"relation_id": "r1", "endpoint_id": "u1"}],
            },
        ],
    }


def test_relation_self_loop_summary_supports_objects_and_metadata_aliases():
    relation = SimpleNamespace(
        edge_id="edge-1",
        metadata={
            "relation_type": "links",
            "source_unit_id": "a",
            "target_unit_id": "a",
            "metadata_source": "crawler",
        },
    )

    assert summarize_relation_self_loops([relation])["rows"] == [
        {
            "relation_type": "links",
            "metadata_source": "crawler",
            "count": 1,
            "sample_relations": [{"relation_id": "edge-1", "endpoint_id": "a"}],
        }
    ]
