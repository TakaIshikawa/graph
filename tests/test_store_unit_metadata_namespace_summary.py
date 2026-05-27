from __future__ import annotations

from graph.store.unit_metadata_namespace_summary import unit_metadata_namespace_summary


class Unit:
    def __init__(self, id: str, metadata: dict):
        self.id = id
        self.metadata = metadata


def test_unit_metadata_namespace_summary_detects_common_prefix_styles():
    rows = unit_metadata_namespace_summary(
        [
            Unit(
                "u1",
                {
                    "http.status": 200,
                    "http:method": "GET",
                    "source_updated_at": "2026-05-01",
                    "title": "A",
                },
            ),
            Unit("u2", {"http.status": 404, "source_created_at": "2026-05-02"}),
        ]
    )

    assert rows == [
        {
            "namespace": "http",
            "key_count": 2,
            "unit_count": 2,
            "keys": ["http.status", "http:method"],
            "sample_unit_ids": ["u1", "u2"],
            "coverage_share": 1.0,
        },
        {
            "namespace": "source",
            "key_count": 2,
            "unit_count": 2,
            "keys": ["source_created_at", "source_updated_at"],
            "sample_unit_ids": ["u1", "u2"],
            "coverage_share": 1.0,
        },
        {
            "namespace": "unscoped",
            "key_count": 1,
            "unit_count": 1,
            "keys": ["title"],
            "sample_unit_ids": ["u1"],
            "coverage_share": 0.5,
        },
    ]


def test_unit_metadata_namespace_summary_supports_mapping_units_and_bounds_samples():
    rows = unit_metadata_namespace_summary(
        [{"id": f"u{i}", "metadata": {"source.id": i, "plain": i}} for i in range(4)],
        sample_limit=2,
    )

    assert rows[0]["namespace"] == "source"
    assert rows[0]["sample_unit_ids"] == ["u0", "u1"]
    assert rows[1]["namespace"] == "unscoped"
    assert rows[1]["sample_unit_ids"] == ["u0", "u1"]
