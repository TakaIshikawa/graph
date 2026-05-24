from __future__ import annotations

from graph.adapters.mendeley_library_csv import MendeleyLibraryCsvAdapter


def test_mendeley_library_csv_parses_quoted_authors_tags_and_document_metadata(tmp_path):
    export = tmp_path / "mendeley.csv"
    export.write_text(
        "Title,Authors,Year,DOI,URL,Publication,Tags,Abstract\n"
        '"Graph Papers","Ada Lovelace; Grace Hopper",2024,10.1000/test,https://example.test/paper,Journal of Graphs,"Graph; Research","A useful abstract."\n',
        encoding="utf-8",
    )

    result = MendeleyLibraryCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "mendeley_library_csv"
    assert unit.source_entity_type == "document"
    assert unit.title == "Graph Papers"
    assert unit.metadata["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert unit.metadata["year"] == 2024
    assert unit.metadata["doi"] == "10.1000/test"
    assert unit.metadata["url"] == "https://example.test/paper"
    assert unit.metadata["publication"] == "Journal of Graphs"
    assert unit.metadata["tags"] == ["graph", "research"]
    assert unit.metadata["abstract"] == "A useful abstract."
    assert "DOI: 10.1000/test" in unit.content


def test_mendeley_library_csv_missing_optional_fields_do_not_crash(tmp_path):
    export = tmp_path / "minimal.csv"
    export.write_text("Title\nSparse Paper\n", encoding="utf-8")

    unit = MendeleyLibraryCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.title == "Sparse Paper"
    assert "doi" not in unit.metadata
