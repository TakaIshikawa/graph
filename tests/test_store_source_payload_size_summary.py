from __future__ import annotations

from graph.store.source_payload_size_summary import source_payload_size_summary


class Unit:
    def __init__(self, unit_id: str, metadata: dict, content: str = ""):
        self.id = unit_id
        self.metadata = metadata
        self.content = content


def test_source_payload_size_summary_prefers_explicit_sizes_and_groups_by_source():
    rows = source_payload_size_summary(
        [
            {"id": "a", "source_project": "docs", "payload_bytes": 10, "content": "longer than ten"},
            {"id": "b", "source_project": "docs", "metadata": {"content_length": "20"}},
            Unit("c", {"source": "web"}, "hello"),
            Unit("d", {"source": "web", "file_size": 1_000_001}),
        ]
    )

    assert rows == [
        {
            "source": "docs",
            "unit_count": 2,
            "min_bytes": 10,
            "max_bytes": 20,
            "average_bytes": 15.0,
            "total_bytes": 30,
            "oversized_count": 0,
            "sample_unit_ids": ["a", "b"],
        },
        {
            "source": "web",
            "unit_count": 2,
            "min_bytes": 5,
            "max_bytes": 1_000_001,
            "average_bytes": 500003.0,
            "total_bytes": 1_000_006,
            "oversized_count": 1,
            "sample_unit_ids": ["c", "d"],
        },
    ]
