from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_annotation_density_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_annotation_density_csv_sorts_by_density_then_unit_id():
    text = export_units_to_annotation_density_csv(
        [
            {"id": "b", "title": "Beta", "source_project": "Docs", "content": "x" * 1000, "metadata": {"comments": ["c"]}},
            {"id": "a", "title": "Alpha", "source_project": "Docs", "content": "x" * 100, "metadata": {"annotations": ["a"], "highlights": ["h"], "notes": ["n"]}},
            {"id": "c", "title": "Gamma", "source_project": "Docs", "content": "", "metadata": {}},
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "title": "Alpha",
            "source": "Docs",
            "content_length": "100",
            "annotation_count": "1",
            "highlight_count": "1",
            "note_count": "1",
            "density_per_1k_chars": "30.00",
            "density_bucket": "high",
        },
        {
            "unit_id": "b",
            "title": "Beta",
            "source": "Docs",
            "content_length": "1000",
            "annotation_count": "1",
            "highlight_count": "0",
            "note_count": "0",
            "density_per_1k_chars": "1.00",
            "density_bucket": "low",
        },
        {
            "unit_id": "c",
            "title": "Gamma",
            "source": "Docs",
            "content_length": "0",
            "annotation_count": "0",
            "highlight_count": "0",
            "note_count": "0",
            "density_per_1k_chars": "0.00",
            "density_bucket": "none",
        },
    ]


def test_unit_annotation_density_csv_counts_metadata_shapes_and_writes_path(tmp_path):
    units = [
        {"id": "m", "content": "x" * 500, "metadata": {"annotation_count": 2, "highlight_count": 2, "margin_notes": {"n1": "note"}}}
    ]
    expected = export_units_to_annotation_density_csv(units)
    stats = export_units_to_annotation_density_csv(units, tmp_path / "density.csv")

    assert rows(expected)[0]["density_bucket"] == "medium"
    assert stats["rows_exported"] == 1
    assert (tmp_path / "density.csv").read_text(encoding="utf-8") == expected
