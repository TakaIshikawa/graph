from __future__ import annotations

from graph.adapters.asana_projects_csv import AsanaProjectsCsvAdapter
from graph.adapters.registry import get_adapter


def test_asana_projects_csv_ingests_project_rows(tmp_path):
    export = tmp_path / "asana.csv"
    export.write_text("GID,Name,Notes,Owner,Team,Workspace,Archived,Created At,Modified At,Due On,Permalink\np1,Migration,Move data,ada,Platform,Acme,true,2026-05-01T00:00:00Z,2026-05-02T00:00:00Z,2026-06-01,https://app.asana.com/0/p1\n", encoding="utf-8")

    unit = AsanaProjectsCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_entity_type == "project"
    assert unit.metadata["project_id"] == "p1"
    assert unit.metadata["archived"] is True
    assert unit.metadata["team"] == "Platform"
    assert "Move data" in unit.content


def test_asana_projects_csv_is_registered():
    assert isinstance(get_adapter("asana-projects-csv"), AsanaProjectsCsvAdapter)
