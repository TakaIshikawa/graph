from __future__ import annotations

from graph.adapters.airtable_bases_csv import AirtableBasesCsvAdapter
from graph.adapters.registry import get_adapter


def test_airtable_bases_csv_ingests_base_inventory(tmp_path):
    path = tmp_path / "bases.csv"
    path.write_text("Base ID,Name,Workspace,Table Count,Collaborator Count,Created At,Updated At,URL,Role\napp1,CRM,Ops,12,5,2026-05-01,2026-05-02,https://airtable.test/app1,creator\n", encoding="utf-8")

    unit = AirtableBasesCsvAdapter(path=str(path)).ingest().units[0]

    assert unit.source_project == "airtable_bases_csv"
    assert unit.source_id == "airtable_bases_csv:app1"
    assert unit.source_entity_type == "base"
    assert unit.metadata["workspace"] == "Ops"
    assert unit.metadata["table_count"] == 12
    assert unit.metadata["collaborator_count"] == 5
    assert "Tables: 12" in unit.content
    assert isinstance(get_adapter("airtable-bases-csv"), AirtableBasesCsvAdapter)
