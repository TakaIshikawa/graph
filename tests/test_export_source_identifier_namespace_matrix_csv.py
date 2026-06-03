from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_identifier_namespace_matrix_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_identifier_namespace_matrix_detects_top_level_and_metadata_namespaces():
    text = export_source_identifier_namespace_matrix_csv(
        [
            {"id": "s2", "doi": "10/test", "metadata": {"isbn": ["1", "2"], "url": "https://example.test"}},
            {"id": "s1", "metadata": {"pmid": "123", "orcid": "0000", "external_id": "ext"}},
        ]
    )

    assert text.splitlines()[0] == "source,doi,external_id,id,isbn,orcid,pmid,url,total_identifiers"
    assert rows(text) == [
        {"source": "s1", "doi": "0", "external_id": "1", "id": "1", "isbn": "0", "orcid": "1", "pmid": "1", "url": "0", "total_identifiers": "4"},
        {"source": "s2", "doi": "1", "external_id": "0", "id": "1", "isbn": "2", "orcid": "0", "pmid": "0", "url": "1", "total_identifiers": "5"},
    ]


def test_source_identifier_namespace_matrix_path_mode(tmp_path):
    path = tmp_path / "ids.csv"
    sources = [{"id": "s", "metadata": {"url": "https://example.test"}}]

    expected = export_source_identifier_namespace_matrix_csv(sources)
    stats = export_source_identifier_namespace_matrix_csv(sources, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["source_count"] == 1
