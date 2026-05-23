from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_provenance_chain_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_provenance_chain_reads_direct_and_metadata_fields():
    text = export_source_provenance_chain_csv(
        [
            {"id": "s1", "name": "One", "adapter": "api", "metadata": {"imported_from": "feed", "parent_source": "root"}},
            {"source_id": "s2", "title": "Two", "original_url": "https://example.test", "metadata": {"import_batch": "b1"}},
        ]
    )

    assert rows(text) == [
        {
            "source_id": "s1",
            "source_name": "One",
            "adapter": "api",
            "imported_from": "feed",
            "parent_source": "root",
            "original_url": "",
            "import_batch": "",
            "provenance_depth_hint": "2",
        },
        {
            "source_id": "s2",
            "source_name": "Two",
            "adapter": "",
            "imported_from": "",
            "parent_source": "",
            "original_url": "https://example.test",
            "import_batch": "b1",
            "provenance_depth_hint": "1",
        },
    ]


def test_source_provenance_chain_path_mode(tmp_path):
    path = tmp_path / "provenance.csv"
    stats = export_source_provenance_chain_csv([{"id": "s1", "name": "One"}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["provenance_depth_hint"] == "0"
    assert stats["source_count"] == 1
