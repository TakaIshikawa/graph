from __future__ import annotations

from dataclasses import dataclass

from graph.store.unit_attachment_type_summary import summarize_unit_attachment_types


@dataclass
class Unit:
    id: str
    metadata: dict[str, object]


def test_summarize_unit_attachment_types_groups_types_extensions_and_sizes():
    summary = summarize_unit_attachment_types(
        [
            {
                "id": "u1",
                "metadata": {
                    "attachments": [
                        {"mime_type": "Image/PNG", "size_bytes": 100, "filename": "ignored.png"},
                        {"content_type": "image/png", "bytes": "250"},
                    ]
                },
            },
            Unit(
                id="u2",
                metadata={
                    "assets": [
                        {"filename": "report.pdf", "size": 300},
                        {"type": "text/plain", "content_length": 25},
                    ]
                },
            ),
            {"id": "u3", "metadata": {"files": [{"path": "/tmp/report.PDF", "size": 200}]}},
        ]
    )

    assert summary == {
        "total_units": 3,
        "total_attachments": 5,
        "units_missing_attachment_metadata": 0,
        "attachment_types": [
            {
                "type": "image/png",
                "unit_count": 1,
                "attachment_count": 2,
                "total_bytes": 350,
                "largest_bytes": 250,
            },
            {"type": "pdf", "unit_count": 2, "attachment_count": 2, "total_bytes": 500, "largest_bytes": 300},
            {
                "type": "text/plain",
                "unit_count": 1,
                "attachment_count": 1,
                "total_bytes": 25,
                "largest_bytes": 25,
            },
        ],
    }


def test_summarize_unit_attachment_types_counts_units_with_missing_type_or_size_metadata():
    summary = summarize_unit_attachment_types(
        [
            {"id": "missing-type", "metadata": {"attachments": [{"size": 10}]}},
            {"id": "missing-size", "files": [{"filename": "evidence.csv"}]},
            {"id": "complete", "metadata": {"assets": [{"extension": ".csv", "size": 20}]}},
        ]
    )

    assert summary["units_missing_attachment_metadata"] == 2
    assert summary["attachment_types"] == [
        {"type": "csv", "unit_count": 2, "attachment_count": 2, "total_bytes": 20, "largest_bytes": 20},
        {"type": "unknown", "unit_count": 1, "attachment_count": 1, "total_bytes": 10, "largest_bytes": 10},
    ]
