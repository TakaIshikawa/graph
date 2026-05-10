from __future__ import annotations

import csv
import io
from pathlib import Path

from graph.adapters.zotero_csv import ZoteroCsvAdapter
from graph.types.enums import ContentType, SourceProject


def _write_csv(path: Path, rows: list[dict[str, str]], columns: list[str] | None = None) -> Path:
    if columns is None:
        columns = [
            "Key", "Item Type", "Publication Year", "Author", "Title",
            "Publication Title", "ISBN", "DOI", "Url", "Abstract Note",
            "Date", "Date Added", "Date Modified", "Manual Tags", "Automatic Tags",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    path.write_text(buf.getvalue(), encoding="utf-8")
    return path


def test_parses_journal_article(tmp_path):
    csv_path = _write_csv(tmp_path / "zotero.csv", [
        {
            "Key": "ABC123",
            "Item Type": "journalArticle",
            "Title": "A Study of X",
            "Author": "Smith, John",
            "Abstract Note": "We studied X.",
            "DOI": "10.1234/test",
            "Publication Year": "2023",
        },
    ])

    result = ZoteroCsvAdapter(path=str(csv_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "A Study of X"
    assert unit.source_project == SourceProject.ZOTERO_CSV
    assert unit.source_entity_type == "article"
    assert unit.content == "We studied X."
    assert unit.metadata["doi"] == "10.1234/test"
    assert unit.metadata["author"] == "Smith, John"


def test_maps_item_types_correctly(tmp_path):
    csv_path = _write_csv(tmp_path / "zotero.csv", [
        {"Title": "Book Title", "Item Type": "book"},
        {"Title": "Conf Paper", "Item Type": "conferencePaper"},
        {"Title": "A Thesis", "Item Type": "thesis"},
        {"Title": "A Report", "Item Type": "report"},
    ])

    result = ZoteroCsvAdapter(path=str(csv_path)).ingest()

    types = {u.title: u.source_entity_type for u in result.units}
    assert types["Book Title"] == "book"
    assert types["Conf Paper"] == "conference_paper"
    assert types["A Thesis"] == "thesis"
    assert types["A Report"] == "report"


def test_parses_manual_and_automatic_tags(tmp_path):
    csv_path = _write_csv(tmp_path / "zotero.csv", [
        {
            "Title": "Tagged Item",
            "Item Type": "journalArticle",
            "Manual Tags": "machine learning; deep learning",
            "Automatic Tags": "AI; neural networks",
        },
    ])

    result = ZoteroCsvAdapter(path=str(csv_path)).ingest()

    unit = result.units[0]
    assert "machine learning" in unit.tags
    assert "deep learning" in unit.tags
    assert "ai" in unit.tags
    assert "neural networks" in unit.tags


def test_empty_csv(tmp_path):
    csv_path = _write_csv(tmp_path / "empty.csv", [])
    result = ZoteroCsvAdapter(path=str(csv_path)).ingest()
    assert len(result.units) == 0


def test_no_path_returns_empty():
    result = ZoteroCsvAdapter().ingest()
    assert len(result.units) == 0


def test_deterministic_source_ids(tmp_path):
    csv_path = _write_csv(tmp_path / "zotero.csv", [
        {"Key": "XYZ", "Title": "Deterministic", "Item Type": "book"},
    ])

    r1 = ZoteroCsvAdapter(path=str(csv_path)).ingest()
    r2 = ZoteroCsvAdapter(path=str(csv_path)).ingest()
    assert r1.units[0].source_id == r2.units[0].source_id


def test_entity_type_filtering(tmp_path):
    csv_path = _write_csv(tmp_path / "zotero.csv", [
        {"Title": "Article", "Item Type": "journalArticle"},
        {"Title": "Book", "Item Type": "book"},
    ])

    result = ZoteroCsvAdapter(path=str(csv_path)).ingest(entity_types=["book"])
    assert len(result.units) == 1
    assert result.units[0].source_entity_type == "book"


def test_registry_lookup():
    from graph.adapters.registry import get_adapter

    adapter = get_adapter("zotero_csv", path="/tmp/fake.csv")
    assert adapter.name == "zotero_csv"


def test_title_as_fallback_content(tmp_path):
    csv_path = _write_csv(tmp_path / "zotero.csv", [
        {"Title": "No Abstract", "Item Type": "book"},
    ])

    result = ZoteroCsvAdapter(path=str(csv_path)).ingest()
    assert result.units[0].content == "No Abstract"
