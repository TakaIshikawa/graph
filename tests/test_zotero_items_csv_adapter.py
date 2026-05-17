from __future__ import annotations

from graph.adapters.zotero_items_csv import ZoteroItemsCsvAdapter
from graph.types.enums import ContentType


def test_zotero_items_csv_ingests_bibliographic_row(tmp_path):
    export = tmp_path / "zotero_items.csv"
    export.write_text(
        "Key,Item Type,Title,Creators,Publication Title,DOI,ISBN,URL,Abstract Note,Date,Tags,Collections,Citation Key,Date Added,Date Modified\n"
        "ABC123,journalArticle,Computing Notes,\"Ada Lovelace; Grace Hopper\",Journal of Tests,10.123/example,9781234567890,https://example.com,Important abstract,2024-04-01,\"History; Computing\",\"Research; Favorites\",lovelace2024,2025-01-01T10:00:00Z,2025-01-02T11:00:00Z\n",
        encoding="utf-8",
    )

    result = ZoteroItemsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "zotero_items_csv"
    assert unit.source_id.startswith("zotero_items_csv:")
    assert unit.source_entity_type == "item"
    assert unit.title == "Computing Notes"
    assert unit.content_type == ContentType.ARTIFACT
    assert "Creators: Ada Lovelace; Grace Hopper" in unit.content
    assert "Publication: Journal of Tests" in unit.content
    assert "Abstract: Important abstract" in unit.content
    assert unit.metadata["key"] == "ABC123"
    assert unit.metadata["citation_key"] == "lovelace2024"
    assert unit.metadata["item_type"] == "article"
    assert unit.metadata["creators"] == ["Ada Lovelace", "Grace Hopper"]
    assert unit.metadata["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert unit.metadata["publication_title"] == "Journal of Tests"
    assert unit.metadata["doi"] == "10.123/example"
    assert unit.metadata["isbn"] == "9781234567890"
    assert unit.metadata["url"] == "https://example.com"
    assert unit.metadata["abstract"] == "Important abstract"
    assert unit.metadata["date"] == "2024-04-01"
    assert unit.metadata["tags"] == ["history", "computing"]
    assert unit.metadata["collections"] == ["Research", "Favorites"]
    assert unit.tags == ["history", "computing"]


def test_zotero_items_csv_handles_citation_key_and_stable_ids(tmp_path):
    export = tmp_path / "minimal.csv"
    export.write_text(
        "Citation Key,Item Type,Title,Author,Url\n"
        "doe2025,webpage,Useful Page,Jane Doe,https://example.org/page\n",
        encoding="utf-8",
    )

    first = ZoteroItemsCsvAdapter(path=str(export)).ingest().units[0]
    second = ZoteroItemsCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.metadata["citation_key"] == "doe2025"
    assert first.metadata["item_type"] == "webpage"
    assert first.metadata["authors"] == ["Jane Doe"]
    assert "Useful Page" in first.content
