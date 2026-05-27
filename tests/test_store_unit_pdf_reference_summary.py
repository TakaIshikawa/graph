from __future__ import annotations

from types import SimpleNamespace

from graph.store import summarize_unit_pdf_references


def test_summarize_unit_pdf_references_counts_content_and_metadata():
    summary = summarize_unit_pdf_references(
        [
            {"id": "a", "content": "[paper](https://example.com/a.pdf#page=12)"},
            SimpleNamespace(id="b", content="See https://example.com/b.pdf", metadata={"path": "docs/local.pdf"}),
            {"id": "c", "metadata": {"url": "notes.txt"}},
        ]
    )

    assert summary == {
        "total_units": 3,
        "units_with_pdf_references": 2,
        "pdf_reference_count": 3,
        "remote_pdf_url_count": 2,
        "local_pdf_path_count": 1,
        "page_fragment_reference_count": 1,
        "samples": [
            {"unit_id": "a", "target": "https://example.com/a.pdf#page=12"},
            {"unit_id": "b", "target": "docs/local.pdf"},
            {"unit_id": "b", "target": "https://example.com/b.pdf"},
        ],
    }


def test_summarize_unit_pdf_references_empty_input():
    assert summarize_unit_pdf_references([]) == {
        "total_units": 0,
        "units_with_pdf_references": 0,
        "pdf_reference_count": 0,
        "remote_pdf_url_count": 0,
        "local_pdf_path_count": 0,
        "page_fragment_reference_count": 0,
        "samples": [],
    }
