from graph.adapters import ZoteroItemsCsvAdapter


def test_zotero_items_csv_ingests_bibliographic_metadata_and_fallback_ids(tmp_path):
    path = tmp_path / "zotero.csv"
    path.write_text("Key,Item Type,Title,Authors,Publication Title,Date,DOI,ISBN,URL,Abstract Note,Tags,Collections\nABC123,article,Paper,\"Ada Lovelace; Grace Hopper\",Journal,2025,10/example,123,https://paper.test,Abstract,\"ai,history\",Reading\n,book,Book,Author,Press,2024,,999,,Summary,,Shelf\n", encoding="utf-8")

    units = ZoteroItemsCsvAdapter(str(path)).ingest().units
    again = ZoteroItemsCsvAdapter(str(path)).ingest().units

    by_title = {unit.title: unit for unit in units}
    assert by_title["Paper"].source_id == "zotero_items_csv:ABC123"
    assert by_title["Paper"].metadata["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert by_title["Paper"].metadata["doi"] == "10/example"
    assert "Abstract" in by_title["Paper"].content
    assert units[1].source_id == again[1].source_id
