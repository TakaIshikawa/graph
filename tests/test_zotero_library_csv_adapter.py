from __future__ import annotations

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.zotero_library_csv import ZoteroLibraryCsvAdapter
from graph.types.enums import SourceProject


def test_zotero_library_csv_ingests_bibliographic_items(tmp_path):
    export = tmp_path / "zotero.csv"
    export.write_text(
        "Key,Item Type,Publication Year,Author,Title,Publication Title,DOI,ISBN,Url,Abstract Note,Date Added,Date Modified,Tags,Collections\n"
        "ABC123,journalArticle,2024,\"Ada Lovelace; Grace Hopper\",Computing Notes,Journal of Tests,10.123/example,,https://example.com,Important abstract,2026-05-01T10:00:00Z,2026-05-02T10:00:00Z,\"History; Computing\",Research\n"
        "BOOK1,book,1999,Someone,Example Book,, ,978123,,Book abstract,2026-05-03,2026-05-04,Books,\"Library; Favorites\"\n",
        encoding="utf-8",
    )

    result = ZoteroLibraryCsvAdapter(path=str(export)).ingest()

    assert [unit.title for unit in result.units] == ["Computing Notes", "Example Book"]
    article = result.units[0]
    assert article.source_project == SourceProject.ZOTERO_LIBRARY_CSV
    assert article.source_entity_type == "item"
    assert article.metadata["key"] == "ABC123"
    assert article.metadata["item_type"] == "article"
    assert article.metadata["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert article.metadata["publication_year"] == "2024"
    assert article.metadata["publication_title"] == "Journal of Tests"
    assert article.metadata["doi"] == "10.123/example"
    assert article.metadata["url"] == "https://example.com"
    assert article.metadata["abstract"] == "Important abstract"
    assert article.metadata["tags"] == ["history", "computing"]
    assert article.metadata["collections"] == ["Research"]
    assert article.metadata["added_at"] == "2026-05-01T10:00:00+00:00"
    assert article.tags == ["history", "computing"]
    assert "Authors: Ada Lovelace; Grace Hopper" in article.content
    assert "DOI: 10.123/example" in article.content
    assert result.units[1].metadata["item_type"] == "book"
    assert result.units[1].metadata["isbn"] == "978123"


def test_zotero_library_csv_handles_webpages_missing_optional_fields_and_blank_rows(tmp_path):
    export = tmp_path / "minimal.csv"
    export.write_text(
        "Key,Item Type,Title,Url,Abstract Note,Tags,Collections\n"
        ",,,,,,\n"
        ",webpage,Useful Webpage,https://example.org,,Web,Inbox\n"
        "KEYONLY,document,,,,,\n",
        encoding="utf-8",
    )

    result = ZoteroLibraryCsvAdapter(path=str(export)).ingest()

    assert [unit.title for unit in result.units] == ["KEYONLY", "Useful Webpage"]
    webpage = result.units[1]
    assert webpage.metadata["item_type"] == "webpage"
    assert webpage.metadata["url"] == "https://example.org"
    assert webpage.tags == ["web"]
    assert webpage.metadata["collections"] == ["Inbox"]


def test_zotero_library_csv_ids_are_deterministic(tmp_path):
    export = tmp_path / "zotero.csv"
    export.write_text(
        "Key,Item Type,Title,DOI\n"
        "ABC123,journalArticle,Computing Notes,10.123/example\n",
        encoding="utf-8",
    )

    first = ZoteroLibraryCsvAdapter(path=str(export)).ingest().units[0]
    second = ZoteroLibraryCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id


def test_zotero_library_csv_is_registered():
    assert "zotero_library_csv" in list_adapters()
    assert isinstance(get_adapter("zotero-library-csv"), ZoteroLibraryCsvAdapter)
