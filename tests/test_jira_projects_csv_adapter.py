from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.jira_projects_csv import JiraProjectsCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_jira_projects_csv_ingests_aliases_and_project_metadata(tmp_path):
    export = tmp_path / "projects.csv"
    _write_csv(
        export,
        [
            {
                "Project Key": "ENG",
                "Project Name": "Engineering",
                "Type": "software",
                "Project Category": "Internal",
                "Project Lead": "Ada",
                "Project URL": "https://jira.example/projects/ENG",
                "Archived?": "no",
                "Created date": "2026-05-01",
                "Last updated": "2026-05-03T04:05:06Z",
                "Project description": "Build platform tooling",
            }
        ],
    )

    result = JiraProjectsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "jira_projects_csv"
    assert unit.source_id == "jira_projects_csv:ENG"
    assert unit.source_entity_type == "project"
    assert unit.title == "Engineering"
    assert unit.metadata["project_key"] == "ENG"
    assert unit.metadata["name"] == "Engineering"
    assert unit.metadata["project_type"] == "software"
    assert unit.metadata["category"] == "Internal"
    assert unit.metadata["lead"] == "Ada"
    assert unit.metadata["url"] == "https://jira.example/projects/ENG"
    assert unit.metadata["archived"] is False
    assert unit.metadata["created_at"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["updated_at"] == "2026-05-03T04:05:06+00:00"
    assert unit.metadata["description"] == "Build platform tooling"
    assert unit.metadata["source_file"] == "projects.csv"
    assert unit.metadata["source_row"] == 2
    assert "URL: https://jira.example/projects/ENG" in unit.content


def test_jira_projects_csv_directory_bad_files_filters_dedupe_and_registry(tmp_path):
    _write_csv(
        tmp_path / "old.csv",
        [
            {"Key": "OLD", "Name": "Old", "Updated": "2026-04-30"},
            {"Key": "", "Name": "", "Updated": ""},
        ],
    )
    _write_csv(
        tmp_path / "new.csv",
        [
            {"Key": "ENG", "Name": "Engineering", "Updated": "2026-05-02", "Archived": "active"},
            {"Key": "ENG", "Name": "Engineering Duplicate", "Updated": "2026-05-03", "Archived": "archived"},
        ],
    )
    (tmp_path / "bad.csv").write_bytes(b"\xff\xff")
    since = SyncState(source_project="jira_projects_csv", source_entity_type="project", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    result = JiraProjectsCsvAdapter(path=str(tmp_path)).ingest(since=since)
    skipped = JiraProjectsCsvAdapter(path=str(tmp_path)).ingest(entity_types=["issue"])

    assert [unit.source_id for unit in result.units] == ["jira_projects_csv:ENG"]
    assert result.units[0].title == "Engineering Duplicate"
    assert result.units[0].metadata["archived"] is True
    assert skipped.units == []
    assert "jira_projects_csv" in list_adapters()
    assert isinstance(get_adapter("jira-projects-csv", path=str(tmp_path)), JiraProjectsCsvAdapter)


def test_jira_projects_csv_source_id_is_deterministic_without_key(tmp_path):
    export = tmp_path / "projects.csv"
    _write_csv(export, [{"Name": "No Key Project", "URL": "https://jira.example/projects/no-key", "Updated": "2026-05-01"}])

    first = JiraProjectsCsvAdapter(path=str(export)).ingest().units[0]
    second = JiraProjectsCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
