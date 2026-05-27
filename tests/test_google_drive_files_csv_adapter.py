from __future__ import annotations

from graph.adapters.google_drive_files_csv import GoogleDriveFilesCsvAdapter


def test_google_drive_files_csv_ingests_file_metadata(tmp_path):
    export = tmp_path / "drive.csv"
    export.write_text(
        "Name,ID,URL,MIME Type,Owner,Created Time,Modified Time,Size,Starred,Description\nSpec,g1,https://drive.google.com/file,application/pdf,ada@example.com,2024-01-01T00:00:00Z,2024-01-02T00:00:00Z,123,yes,Design spec\n",
        encoding="utf-8",
    )

    unit = GoogleDriveFilesCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.title == "Spec"
    assert unit.metadata["file_id"] == "g1"
    assert unit.metadata["mime_type"] == "application/pdf"
    assert unit.metadata["owner"] == "ada@example.com"
    assert unit.metadata["size"] == 123
    assert unit.metadata["starred"] is True
    assert "Design spec" in unit.content


def test_google_drive_files_csv_allows_missing_url_and_stable_file_id(tmp_path):
    export = tmp_path / "drive.csv"
    export.write_text("Name,ID\nSpec,g1\n", encoding="utf-8")

    first = GoogleDriveFilesCsvAdapter(path=str(export)).ingest().units[0]
    second = GoogleDriveFilesCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert "url" not in first.metadata
