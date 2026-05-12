from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from graph.adapters.google_keep import GoogleKeepAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import SyncState


def write_note(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_google_keep_ingests_note_with_labels_checklist_and_metadata(tmp_path):
    note_path = tmp_path / "note.json"
    write_note(
        note_path,
        {
            "id": "note-1",
            "title": "Trip planning",
            "textContent": "Book train tickets",
            "labels": [{"name": "Travel"}, {"name": "#Planning"}, {"name": "travel"}],
            "listContent": [
                {"text": "Reserve hotel", "isChecked": False},
                {"text": "Pack adapter", "isChecked": True},
            ],
            "isArchived": True,
            "isTrashed": False,
            "isPinned": True,
            "color": "RED",
            "createdTimestampUsec": 1_776_211_200_000_000,
            "userEditedTimestampUsec": 1_776_297_600_000_000,
        },
    )

    result = GoogleKeepAdapter(path=str(note_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.GOOGLE_KEEP
    assert unit.source_id == "google_keep:note-1"
    assert unit.source_entity_type == "keep_note"
    assert unit.title == "Trip planning"
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.tags == ["Travel", "Planning"]
    assert unit.content == (
        "Trip planning\n"
        "Book train tickets\n"
        "[ ] Reserve hotel\n"
        "[x] Pack adapter"
    )
    assert unit.created_at == datetime(2026, 4, 15, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2026, 4, 16, tzinfo=timezone.utc)
    assert unit.metadata["source_path"] == str(note_path)
    assert unit.metadata["isArchived"] is True
    assert unit.metadata["isTrashed"] is False
    assert unit.metadata["isPinned"] is True
    assert unit.metadata["color"] == "RED"
    assert unit.metadata["createdTimestampUsec"] == 1_776_211_200_000_000
    assert unit.metadata["userEditedTimestampUsec"] == 1_776_297_600_000_000
    assert unit.metadata["checklist"] == [
        {"text": "Reserve hotel", "checked": False, "position": 1},
        {"text": "Pack adapter", "checked": True, "position": 2},
    ]


def test_google_keep_directory_ingestion_is_sorted_by_path(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    write_note(nested / "b.json", {"id": "b", "title": "B", "textContent": "second"})
    write_note(tmp_path / "a.json", {"id": "a", "title": "A", "textContent": "first"})
    (tmp_path / "skip.txt").write_text("not json", encoding="utf-8")

    result = GoogleKeepAdapter(path=str(tmp_path)).ingest()

    assert [unit.source_id for unit in result.units] == [
        "google_keep:a",
        "google_keep:b",
    ]


def test_google_keep_uses_file_path_for_source_id_when_note_id_is_missing(tmp_path):
    note_path = tmp_path / "no-id.json"
    write_note(note_path, {"title": "", "textContent": "Loose note"})

    unit = GoogleKeepAdapter(path=str(note_path)).ingest().units[0]

    assert unit.source_id == f"google_keep:path:{note_path}"
    assert unit.title == "Untitled Google Keep note"
    assert unit.content == "Untitled Google Keep note\nLoose note"


def test_google_keep_malformed_json_identifies_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")

    with pytest.raises(ValueError, match=f"Malformed Google Keep JSON in {bad}"):
        GoogleKeepAdapter(path=str(bad)).ingest()


def test_google_keep_since_filter_uses_updated_then_created_timestamp(tmp_path):
    write_note(
        tmp_path / "old.json",
        {
            "id": "old",
            "title": "Old",
            "textContent": "Old content",
            "userEditedTimestampUsec": 1_776_038_400_000_000,
        },
    )
    write_note(
        tmp_path / "new.json",
        {
            "id": "new",
            "title": "New",
            "textContent": "New content",
            "createdTimestampUsec": 1_776_211_200_000_000,
        },
    )

    result = GoogleKeepAdapter(path=str(tmp_path)).ingest(
        since=SyncState(
            source_project="google_keep",
            source_entity_type="keep_note",
            last_sync_at=datetime(2026, 4, 14, tzinfo=timezone.utc),
        )
    )
    skipped = GoogleKeepAdapter(path=str(tmp_path)).ingest(entity_types=["other"])

    assert [unit.source_id for unit in result.units] == ["google_keep:new"]
    assert skipped.units == []
    assert skipped.edges == []


def test_google_keep_adapter_is_registered():
    assert "google_keep" in list_adapters()
    adapter = get_adapter("google_keep", path="/tmp/keep")
    assert isinstance(adapter, GoogleKeepAdapter)
    assert adapter.name == "google_keep"


def test_google_keep_reports_checklist_item_entity_type():
    assert GoogleKeepAdapter().entity_types == ["keep_note", "checklist_item"]


def test_google_keep_emits_checklist_items_and_contains_edges(tmp_path):
    note_path = tmp_path / "note.json"
    write_note(
        note_path,
        {
            "id": "note-1",
            "title": "Trip planning",
            "labels": ["Travel"],
            "listContent": [
                {"text": "Reserve hotel", "isChecked": False},
                {"text": "Pack adapter", "isChecked": True},
            ],
        },
    )

    result = GoogleKeepAdapter(path=str(note_path)).ingest(entity_types=["keep_note", "checklist_item"])

    note = next(unit for unit in result.units if unit.source_entity_type == "keep_note")
    items = sorted(
        [unit for unit in result.units if unit.source_entity_type == "checklist_item"],
        key=lambda unit: unit.metadata["position"],
    )
    assert [item.title for item in items] == ["Reserve hotel", "Pack adapter"]
    assert items[0].metadata["checked"] is False
    assert items[0].metadata["position"] == 1
    assert items[0].metadata["parent_note_source_id"] == note.source_id
    assert items[1].metadata["checked"] is True
    assert all(item.source_id.startswith("google_keep:note-1:checklist_item:") for item in items)
    assert len(result.edges) == 2
    assert {(edge.from_unit_id, edge.to_unit_id) for edge in result.edges} == {
        (note.source_id, item.source_id) for item in items
    }
    assert {edge.relation for edge in result.edges} == {EdgeRelation.CONTAINS}
    assert {edge.source for edge in result.edges} == {EdgeSource.SOURCE}


def test_google_keep_checklist_item_filtering(tmp_path):
    note_path = tmp_path / "note.json"
    write_note(
        note_path,
        {
            "id": "note-1",
            "title": "Trip planning",
            "listContent": [{"text": "Reserve hotel", "isChecked": False}],
        },
    )

    default_result = GoogleKeepAdapter(path=str(note_path)).ingest()
    item_only = GoogleKeepAdapter(path=str(note_path)).ingest(entity_types=["checklist_item"])

    assert [unit.source_entity_type for unit in default_result.units] == ["keep_note"]
    assert [unit.source_entity_type for unit in item_only.units] == ["checklist_item"]
    assert item_only.edges == []
