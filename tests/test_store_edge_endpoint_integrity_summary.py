from __future__ import annotations

from types import SimpleNamespace

from graph.store.edge_endpoint_integrity_summary import edge_endpoint_integrity_summary


def test_edge_endpoint_integrity_summary_counts_statuses_and_supports_objects():
    summary = edge_endpoint_integrity_summary(
        [
            {"id": "ok", "source_id": "u1", "target_id": "u2", "relation": "rel"},
            {"id": "miss", "source_id": "u1", "target_id": "absent", "relation": "rel"},
            {"id": "loop", "source_id": "u1", "target_id": "u1", "relation": "rel"},
            {"id": "dup1", "source_id": "u1", "target_id": "u2", "relation": "dup"},
            {"id": "dup2", "source_id": "u1", "target_id": "u2", "relation": "dup"},
        ],
        [SimpleNamespace(id="u1"), {"id": "u2"}],
    )

    assert summary["total_edges"] == 5
    assert summary["status_counts"] == {"duplicate_edge": 2, "missing_target": 1, "ok": 1, "self_loop": 1}
    assert {row["edge_id"]: row["status"] for row in summary["rows"]}["miss"] == "missing_target"
