from __future__ import annotations

from graph.store.unit_attachment_missing_reference_summary import summarize_unit_attachment_missing_references


def test_attachment_missing_reference_summary_compares_attachments_to_content_references():
    summary = summarize_unit_attachment_missing_references(
        [
            {
                "id": "u1",
                "attachments": ["docs/Report.PDF", {"path": "images/chart.png"}, {"name": "notes.txt"}],
                "content": "[report](docs/report.pdf) ![chart](images/chart.png)",
            },
            {
                "id": "u2",
                "metadata": {"attachments": [{"url": "https://files.example.com/deck.PPTX"}, "archive"]},
                "content": "![[deck.PPTX]]",
            },
            {"id": "u3", "attachments": [], "content": ""},
        ]
    )

    assert summary == {
        "unit_count": 3,
        "units_with_unreferenced_attachments_count": 2,
        "unreferenced_attachment_count": 2,
        "counts_by_extension": {"(none)": 1, "txt": 1},
        "samples": [
            {"unit_id": "u1", "attachment": "notes.txt", "extension": "txt"},
            {"unit_id": "u2", "attachment": "archive", "extension": "(none)"},
        ],
    }


def test_attachment_missing_reference_summary_is_zero_safe():
    assert summarize_unit_attachment_missing_references([{"id": "u1"}, {"id": "u2", "content": "[x](file.pdf)"}]) == {
        "unit_count": 2,
        "units_with_unreferenced_attachments_count": 0,
        "unreferenced_attachment_count": 0,
        "counts_by_extension": {},
        "samples": [],
    }
