from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from graph.adapters.jsonl_notes import JsonlNotesAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_jsonl_notes_ingests_note_records_with_common_fields(tmp_path):
    path = tmp_path / "notes.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "note-1",
                "title": "First note",
                "text": "Remember the import shape.",
                "tags": ["#Inbox", " Research ", "inbox"],
                "created_at": "2025-04-24T12:00:00Z",
                "updated_at": "2025-04-25T09:30:00Z",
                "metadata": {"source": "export", "rank": 3},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = JsonlNotesAdapter(path=str(path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.JSONL_NOTES
    assert unit.source_entity_type == "jsonl_note"
    assert unit.source_id == "note-1"
    assert unit.title == "First note"
    assert unit.content == "Remember the import shape."
    assert unit.content_type == ContentType.INSIGHT
    assert unit.tags == ["inbox", "research"]
    assert unit.created_at.isoformat() == "2025-04-24T12:00:00+00:00"
    assert unit.updated_at.isoformat() == "2025-04-25T09:30:00+00:00"
    assert unit.metadata == {
        "source": "export",
        "rank": 3,
        "source_file": "notes.jsonl",
        "file_path": str(path),
        "line_number": 1,
    }
    assert result.edges == []


def test_jsonl_notes_accepts_content_field_and_derives_missing_title(tmp_path):
    path = tmp_path / "notes.jsonl"
    path.write_text(
        json.dumps({"id": "note-2", "content": "Derived title\nwith body"})
        + "\n",
        encoding="utf-8",
    )

    result = JsonlNotesAdapter(path=str(path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].title == "Derived title"
    assert result.units[0].content == "Derived title\nwith body"


def test_jsonl_notes_malformed_json_reports_line_number(tmp_path):
    path = tmp_path / "notes.jsonl"
    path.write_text(
        '{"id": "ok", "text": "Imported."}\n'
        "{not json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"line 2"):
        JsonlNotesAdapter(path=str(path)).ingest()


def test_jsonl_notes_skips_empty_content_and_respects_filters(tmp_path):
    path = tmp_path / "notes.jsonl"
    path.write_text(
        json.dumps({"id": "empty", "title": "Empty", "text": " "})
        + "\n"
        + json.dumps({"id": "valid", "title": "Valid", "content": "Body"})
        + "\n",
        encoding="utf-8",
    )

    filtered = JsonlNotesAdapter(path=str(path)).ingest(entity_types=["jsonl_record"])
    result = JsonlNotesAdapter(path=str(path)).ingest(
        since=SyncState(
            source_project="jsonl_notes",
            source_entity_type="jsonl_note",
            last_sync_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
    )

    assert filtered.units == []
    assert filtered.edges == []
    assert [unit.source_id for unit in result.units] == ["valid"]


def test_jsonl_notes_adapter_is_registered():
    assert "jsonl_notes" in list_adapters()
    adapter = get_adapter("jsonl_notes", path="/tmp/notes.jsonl")
    assert isinstance(adapter, JsonlNotesAdapter)
    assert adapter.name == "jsonl_notes"
