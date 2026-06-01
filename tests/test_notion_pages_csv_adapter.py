from graph.adapters.notion_pages_csv import NotionPagesCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters


def test_notion_pages_csv_ingests_page_rows(tmp_path):
    export = tmp_path / "pages.csv"
    export.write_text(
        "Name,URL,Created time,Last edited time,Tags,Status,Parent,Database\n"
        "Roadmap,https://notion.so/roadmap,2026-05-01,2026-05-02,\"work, planning\",Active,Home,Projects\n",
        encoding="utf-8",
    )

    unit = NotionPagesCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "notion_pages_csv"
    assert unit.source_id == "notion_pages_csv:https://notion.so/roadmap"
    assert unit.source_entity_type == "page"
    assert unit.metadata["url"] == "https://notion.so/roadmap"
    assert unit.metadata["tags"] == ["work", "planning"]
    assert unit.metadata["status"] == "Active"
    assert unit.metadata["parent"] == "Home"
    assert unit.metadata["database"] == "Projects"
    assert unit.metadata["source_file"] == "pages.csv"
    assert unit.metadata["row_number"] == 1
    assert "Roadmap" in unit.content


def test_notion_pages_csv_directory_filtering_and_invalid_tolerance(tmp_path):
    (tmp_path / "empty.csv").write_text("bad", encoding="utf-8")
    (tmp_path / "pages.csv").write_text("Title,Tags\nUntitled,\"alpha; beta\"\n,\n", encoding="utf-8")

    result = NotionPagesCsvAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["tags"] == ["alpha", "beta"]
    assert NotionPagesCsvAdapter(path=str(tmp_path)).ingest(entity_types=["database"]).units == []


def test_notion_pages_csv_is_registered():
    assert "notion_pages_csv" in list_adapters()
    assert isinstance(get_adapter("notion-pages-csv"), NotionPagesCsvAdapter)
