from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.wikipedia_reading_list_csv import WikipediaReadingListCsvAdapter


def test_wikipedia_reading_list_csv_ingests_articles(tmp_path):
    export = tmp_path / "reading.csv"
    export.write_text(
        "Page title,Page URL,Wiki,Language,Folder,Extract,Saved timestamp,Archived,Read\n"
        "Ada Lovelace,https://en.wikipedia.org/wiki/Ada_Lovelace,enwiki,en,History,Mathematician,2026-05-01,true,false\n",
        encoding="utf-8",
    )

    unit = WikipediaReadingListCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "wikipedia_reading_list_csv"
    assert unit.source_id == "wikipedia_reading_list_csv:https://en.wikipedia.org/wiki/Ada_Lovelace"
    assert unit.source_entity_type == "article"
    assert unit.metadata["wiki"] == "enwiki"
    assert unit.metadata["language"] == "en"
    assert unit.metadata["list"] == "History"
    assert unit.metadata["description"] == "Mathematician"
    assert unit.metadata["saved_at"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["archived"] is True
    assert unit.metadata["read"] is False
    assert "Language: en" in unit.content


def test_wikipedia_reading_list_csv_aliases_filtering_and_empty_rows(tmp_path):
    export = tmp_path / "reading.csv"
    export.write_text(
        "Title,URL,List,Description,Saved,Read\n"
        ",,,,,\n"
        "Python,https://en.wikipedia.org/wiki/Python_(programming_language),Tech,Language,2026-05-02,yes\n",
        encoding="utf-8",
    )

    result = WikipediaReadingListCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["read"] is True
    assert WikipediaReadingListCsvAdapter(path=str(export)).ingest(entity_types=["page"]).units == []


def test_wikipedia_reading_list_csv_is_registered():
    assert "wikipedia_reading_list_csv" in list_adapters()
    assert isinstance(get_adapter("wikipedia-reading-list-csv"), WikipediaReadingListCsvAdapter)
