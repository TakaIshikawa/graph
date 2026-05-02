from __future__ import annotations

from graph.adapters.csv_rows import CsvRowsAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject


def test_csv_rows_default_columns_create_units_with_row_references(tmp_path):
    csv_path = tmp_path / "notes.csv"
    csv_path.write_text(
        "title,content,tags\n"
        "Solar note,Storage doubled,energy\n"
        "Wind note,Turbines scaled,wind\n",
        encoding="utf-8",
    )

    first = CsvRowsAdapter(path=str(csv_path)).ingest()
    second = CsvRowsAdapter(path=str(csv_path)).ingest()

    assert [unit.title for unit in first.units] == ["Solar note", "Wind note"]
    assert [unit.content for unit in first.units] == [
        "Storage doubled",
        "Turbines scaled",
    ]
    assert [unit.source_id for unit in first.units] == [
        "csv_rows:notes.csv:row-2",
        "csv_rows:notes.csv:row-3",
    ]
    assert [unit.source_id for unit in second.units] == [
        unit.source_id for unit in first.units
    ]
    assert first.units[0].source_project == SourceProject.CSV_ROWS
    assert first.units[0].source_entity_type == "csv_row"
    assert first.units[0].content_type == ContentType.ARTIFACT
    assert first.units[0].metadata["row_number"] == 2
    assert first.units[0].metadata["source_file"] == "notes.csv"


def test_csv_rows_honors_custom_title_and_content_columns(tmp_path):
    csv_path = tmp_path / "research.csv"
    csv_path.write_text(
        "Name,Observation,Status\n"
        "Battery trial,Efficiency improved,reviewed\n",
        encoding="utf-8",
    )

    result = CsvRowsAdapter(
        path=str(csv_path),
        title_column="Name",
        content_column="Observation",
    ).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Battery trial"
    assert unit.content == "Efficiency improved"
    assert unit.metadata["fields"] == {"Status": "reviewed"}


def test_csv_rows_falls_back_when_content_column_is_missing_or_empty(tmp_path):
    missing_content = tmp_path / "missing-content.csv"
    missing_content.write_text(
        "title,status,owner\n"
        "Review notes,open,Taka\n",
        encoding="utf-8",
    )
    empty_content = tmp_path / "empty-content.csv"
    empty_content.write_text(
        "title,content,status\n"
        "Draft note,,pending\n",
        encoding="utf-8",
    )

    missing_result = CsvRowsAdapter(path=str(missing_content)).ingest()
    empty_result = CsvRowsAdapter(path=str(empty_content)).ingest()

    assert missing_result.units[0].content == "status: open\nowner: Taka"
    assert empty_result.units[0].content == "Draft note"


def test_csv_rows_preserves_non_content_columns_as_metadata(tmp_path):
    csv_path = tmp_path / "annotations.csv"
    csv_path.write_text(
        "title,content,url,priority,empty\n"
        "Paper note,Useful result,https://example.com,high,\n",
        encoding="utf-8",
    )

    result = CsvRowsAdapter(path=str(csv_path)).ingest()

    assert result.units[0].metadata == {
        "source_file": "annotations.csv",
        "row_number": 2,
        "fields": {
            "url": "https://example.com",
            "priority": "high",
        },
    }


def test_csv_rows_registry_discovery():
    assert "csv_rows" in list_adapters()
    adapter = get_adapter(
        "csv_rows",
        path="/tmp/rows.csv",
        title_column="Name",
        content_column="Text",
    )
    assert isinstance(adapter, CsvRowsAdapter)
    assert adapter.name == "csv_rows"
