from __future__ import annotations

from graph.adapters.airtable_csv import AirtableCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject


def test_airtable_csv_imports_generic_rows_and_preserves_fields(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text(
        "Name,Status,created_time,last_modified_time\nLaunch plan,Active,2025-01-01T00:00:00Z,2025-01-02T00:00:00Z\n",
        encoding="utf-8",
    )

    unit = AirtableCsvAdapter(path=str(path)).ingest().units[0]

    assert unit.source_project == SourceProject.AIRTABLE_CSV
    assert unit.title == "Launch plan"
    assert unit.content == "Status: Active\ncreated_time: 2025-01-01T00:00:00Z\nlast_modified_time: 2025-01-02T00:00:00Z"
    assert unit.metadata["title_field"] == "Name"
    assert unit.metadata["fields"]["Status"] == "Active"
    assert unit.metadata["created_time"] == "2025-01-01T00:00:00Z"
    assert unit.metadata["last_modified_time"] == "2025-01-02T00:00:00Z"


def test_airtable_csv_title_discovery_prefers_name_title_summary_then_first_non_empty(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text(
        "A,Summary,Title,Name\nfirst,summary,title,name\nfallback,,,\n",
        encoding="utf-8",
    )

    units = AirtableCsvAdapter(path=str(path)).ingest().units

    assert [unit.title for unit in units] == ["name", "fallback"]
    assert [unit.metadata["title_field"] for unit in units] == ["Name", "A"]


def test_airtable_csv_adapter_is_registered():
    assert isinstance(get_adapter("airtable_csv", path="/tmp/table.csv"), AirtableCsvAdapter)
