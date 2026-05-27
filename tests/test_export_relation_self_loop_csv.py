from __future__ import annotations

from types import SimpleNamespace

from graph.export.relation_self_loop_csv import export_relation_self_loop_csv


def test_export_relation_self_loop_csv_emits_only_true_same_endpoint_loops():
    text = export_relation_self_loop_csv(
        [
            {
                "id": "r1",
                "relation": "mentions",
                "source_id": "u1",
                "target_id": "u1",
                "source_label": "Source",
                "target_label": "Target",
                "metadata": {"source": "docs"},
            },
            {"id": "r2", "relation": "mentions", "source_id": "u1", "target_id": "u2"},
            {"id": "r3", "relation": "mentions", "source_id": "", "target_id": ""},
        ]
    )

    assert text == (
        "relation_id,relation_type,endpoint_id,source_label,target_label,metadata_source\n"
        "r1,mentions,u1,Source,Target,docs\n"
    )


def test_export_relation_self_loop_csv_supports_objects_and_metadata_aliases():
    text = export_relation_self_loop_csv(
        [
            SimpleNamespace(
                edge_id="edge-1",
                metadata={
                    "relation_type": "links",
                    "source_unit_id": "a",
                    "target_unit_id": "a",
                    "metadata_source": "crawler",
                },
            )
        ]
    )

    assert "edge-1,links,a,,,crawler\n" in text
