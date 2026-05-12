from __future__ import annotations

from graph.adapters.libby_loans_csv import LibbyLoansCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject


def test_libby_loans_csv_ingests_metadata_tags_and_relationships(tmp_path):
    export = tmp_path / "loans.csv"
    export.write_text(
        "\n".join(
            [
                "Title,Author,Format,Library,Borrowed Date,Returned Date,Series,Subjects,ISBN,Rating",
                "The Book,Ada Lovelace,eBook,City Library,2025-01-01,2025-01-10,Computing,\"History,Technology\",9781234567890,5",
            ]
        ),
        encoding="utf-8",
    )

    result = LibbyLoansCsvAdapter(path=str(export)).ingest()

    loans = [unit for unit in result.units if unit.source_entity_type == "loan"]
    authors = [unit for unit in result.units if unit.source_entity_type == "author"]
    assert len(loans) == 1
    assert len(authors) == 1
    unit = loans[0]
    assert unit.source_project == SourceProject.LIBBY_LOANS_CSV
    assert unit.source_entity_type == "loan"
    assert unit.metadata["borrowed_at"] == "2025-01-01T00:00:00+00:00"
    assert unit.metadata["returned_at"] == "2025-01-10T00:00:00+00:00"
    assert unit.metadata["format"] == "eBook"
    assert unit.metadata["subjects"] == ["History", "Technology"]
    assert unit.metadata["isbn"] == "9781234567890"
    assert {"History", "Technology", "ebook"}.issubset(set(unit.tags))
    assert authors[0].title == "Ada Lovelace"
    assert (unit.source_id, authors[0].source_id) in {(edge.from_unit_id, edge.to_unit_id) for edge in result.edges}
    assert {edge.metadata["kind"] for edge in result.edges} == {"author", "library", "series"}
    assert get_adapter("libby_loans_csv", path=str(export)).name == "libby_loans_csv"


def test_libby_loans_csv_ingests_multiple_authors(tmp_path):
    export = tmp_path / "loans.csv"
    export.write_text("Title,Author\nThe Book,\"Ada Lovelace; Grace Hopper and Mary Shelley\"\n", encoding="utf-8")

    result = LibbyLoansCsvAdapter(path=str(export)).ingest()

    authors = [unit for unit in result.units if unit.source_entity_type == "author"]
    assert sorted(unit.title for unit in authors) == ["Ada Lovelace", "Grace Hopper", "Mary Shelley"]
    assert len([edge for edge in result.edges if edge.metadata["kind"] == "author"]) == 3


def test_libby_loans_csv_skips_blank_and_unknown_authors(tmp_path):
    export = tmp_path / "loans.csv"
    export.write_text("Title,Author\nNo Author,\nUnknown Author,Unknown\n", encoding="utf-8")

    result = LibbyLoansCsvAdapter(path=str(export)).ingest()

    assert [unit.source_entity_type for unit in result.units] == ["loan", "loan"]
    assert [edge for edge in result.edges if edge.metadata["kind"] == "author"] == []
