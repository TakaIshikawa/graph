from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_collection_source_diversity_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_collection_source_diversity_groups_sources_with_stable_fallbacks():
    text = export_collection_source_diversity_csv(
        [
            {"id": "a", "metadata": {"collection": "Inbox"}, "source_project": "src1"},
            {"id": "b", "metadata": {"collection": "Inbox"}, "source_project": "src1"},
            {"id": "c", "metadata": {"collection": "Inbox"}, "source_project": "src2"},
            {"id": "d", "metadata": {}, "source_project": ""},
        ]
    )

    assert rows(text) == [
        {
            "collection": "Inbox",
            "unit_count": "3",
            "source_count": "2",
            "dominant_source": "src1",
            "dominant_source_unit_count": "2",
            "dominant_source_share": "66.67",
        },
        {
            "collection": "Unassigned",
            "unit_count": "1",
            "source_count": "1",
            "dominant_source": "Unknown",
            "dominant_source_unit_count": "1",
            "dominant_source_share": "100.00",
        },
    ]


def test_collection_source_diversity_path_mode(tmp_path):
    path = tmp_path / "collections.csv"
    stats = export_collection_source_diversity_csv([{"id": "a", "metadata": {"collection": "A"}, "source_project": "S"}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["collection"] == "A"
    assert stats["rows_exported"] == 1
