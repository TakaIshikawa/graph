from __future__ import annotations

from graph.store.source_status_code_summary import source_status_code_summary


class Unit:
    def __init__(self, id: str, source_project: str, metadata: dict):
        self.id = id
        self.source_project = source_project
        self.metadata = metadata


def test_source_status_code_summary_groups_status_classes_by_source():
    rows = source_status_code_summary(
        [
            {"id": "a1", "source_project": "api", "metadata": {"status_code": 200}},
            {"id": "a2", "source_project": "api", "metadata": {"http_status": "301"}},
            {"id": "a3", "source_project": "api", "metadata": {"response_status": 404}},
            {"id": "a4", "source_project": "api", "metadata": {"source_status": 503}},
            {
                "id": "a5",
                "source_project": "api",
                "metadata": {"source_status": "queued"},
            },
        ]
    )

    assert rows == [
        {
            "source": "api",
            "status_class": "2xx",
            "count": 1,
            "status_codes": [200],
            "sample_unit_ids": ["a1"],
            "error_share": 0.4,
        },
        {
            "source": "api",
            "status_class": "3xx",
            "count": 1,
            "status_codes": [301],
            "sample_unit_ids": ["a2"],
            "error_share": 0.4,
        },
        {
            "source": "api",
            "status_class": "4xx",
            "count": 1,
            "status_codes": [404],
            "sample_unit_ids": ["a3"],
            "error_share": 0.4,
        },
        {
            "source": "api",
            "status_class": "5xx",
            "count": 1,
            "status_codes": [503],
            "sample_unit_ids": ["a4"],
            "error_share": 0.4,
        },
        {
            "source": "api",
            "status_class": "unknown",
            "count": 1,
            "status_codes": [],
            "sample_unit_ids": ["a5"],
            "error_share": 0.4,
        },
    ]


def test_source_status_code_summary_reads_mapping_object_and_metadata_equivalents():
    rows = source_status_code_summary(
        [
            Unit("u1", "web", {"status_code": 404}),
            {"id": "u2", "metadata": {"source": "web", "http_status": 410}},
            {"id": "u3", "source": "docs", "response_status": 201, "metadata": {}},
        ]
    )

    assert rows == [
        {
            "source": "docs",
            "status_class": "2xx",
            "count": 1,
            "status_codes": [201],
            "sample_unit_ids": ["u3"],
            "error_share": 0.0,
        },
        {
            "source": "web",
            "status_class": "4xx",
            "count": 2,
            "status_codes": [404, 410],
            "sample_unit_ids": ["u1", "u2"],
            "error_share": 1.0,
        },
    ]


def test_source_status_code_summary_bounds_sample_unit_ids():
    rows = source_status_code_summary(
        [
            {"id": f"u{i}", "source_project": "web", "status_code": 500, "metadata": {}}
            for i in range(4)
        ],
        sample_limit=2,
    )

    assert rows[0]["sample_unit_ids"] == ["u0", "u1"]
