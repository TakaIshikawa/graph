from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter
from graph.adapters.trello_cards_json import TrelloCardsJsonAdapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_trello_cards_json_ingests_board_cards_metadata_and_registry(tmp_path):
    export = tmp_path / "board.json"
    export.write_text(
        json.dumps(
            {
                "name": "Product Board",
                "lists": [{"id": "list-1", "name": "Doing"}],
                "labels": [{"id": "label-1", "name": "Priority"}],
                "members": [{"id": "member-1", "fullName": "Ada Lovelace"}],
                "checklists": [
                    {
                        "id": "check-1",
                        "name": "Launch",
                        "checkItems": [{"id": "item-1", "name": "Write notes", "state": "complete"}],
                    }
                ],
                "cards": [
                    {
                        "id": "card-1",
                        "name": "Ship adapter",
                        "desc": "Normalize Trello cards",
                        "url": "https://trello.com/c/card-1",
                        "idList": "list-1",
                        "idLabels": ["label-1"],
                        "idMembers": ["member-1"],
                        "idChecklists": ["check-1"],
                        "due": "2025-01-05T00:00:00Z",
                        "closed": False,
                        "dateLastActivity": "2025-01-02T00:00:00Z",
                        "actions": [
                            {
                                "id": "comment-1",
                                "type": "commentCard",
                                "date": "2025-01-03T00:00:00Z",
                                "data": {"text": "Looks good"},
                                "memberCreator": {"fullName": "Grace Hopper"},
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = TrelloCardsJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.TRELLO_CARDS_JSON
    assert unit.source_entity_type == "card"
    assert unit.source_id == "trello_cards_json:card-1"
    assert unit.title == "Ship adapter"
    assert unit.metadata["board_name"] == "Product Board"
    assert unit.metadata["list_name"] == "Doing"
    assert unit.metadata["labels"] == ["Priority"]
    assert unit.metadata["members"] == ["Ada Lovelace"]
    assert unit.metadata["checklists"][0]["items"][0]["name"] == "Write notes"
    assert unit.metadata["comments"][0]["text"] == "Looks good"
    assert unit.updated_at == datetime(2025, 1, 2, tzinfo=timezone.utc)
    assert "Priority" in unit.tags
    assert "Looks good" in unit.content
    assert get_adapter("trello_cards_json", path=str(export)).name == "trello_cards_json"


def test_trello_cards_json_archived_since_and_stable_ids(tmp_path):
    (tmp_path / "board.json").write_text(
        json.dumps(
            {
                "cards": [
                    {"id": "old", "name": "Old", "closed": False, "dateLastActivity": "2025-01-01T00:00:00Z"},
                    {"id": "archived", "name": "Archived", "closed": True, "dateLastActivity": "2025-01-03T00:00:00Z"},
                ]
            }
        ),
        encoding="utf-8",
    )
    sync = SyncState(source_project="trello_cards_json", source_entity_type="card", last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
    adapter = TrelloCardsJsonAdapter(path=str(tmp_path))

    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.title for unit in first.units] == ["Archived"]
    assert first.units[0].metadata["closed"] is True
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["list"]).units == []


def test_trello_cards_json_handles_missing_optional_fields(tmp_path):
    export = tmp_path / "cards.json"
    export.write_text(json.dumps([{"id": "minimal", "name": "Minimal card"}]), encoding="utf-8")

    unit = TrelloCardsJsonAdapter(path=str(export)).ingest().units[0]

    assert unit.title == "Minimal card"
    assert unit.metadata["card_id"] == "minimal"
    assert "list_name" not in unit.metadata
    assert "checklists" not in unit.metadata
    assert unit.tags == ["trello", "card"]
