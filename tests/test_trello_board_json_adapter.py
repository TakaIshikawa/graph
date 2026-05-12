from __future__ import annotations

import json

from graph.adapters.registry import get_adapter
from graph.adapters.trello_board_json import TrelloBoardJsonAdapter
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject


def test_trello_board_json_ingests_cards_metadata_and_relationships(tmp_path):
    export = tmp_path / "board.json"
    export.write_text(
        json.dumps(
            {
                "lists": [{"id": "list-1", "name": "Doing"}],
                "labels": [{"id": "label-1", "name": "Import"}],
                "members": [{"id": "member-1", "fullName": "Ada Lovelace"}],
                "checklists": [
                    {
                        "id": "check-1",
                        "name": "Launch",
                        "checkItems": [{"state": "complete"}, {"state": "incomplete"}],
                    }
                ],
                "cards": [
                    {
                        "id": "card-1",
                        "name": "Add Trello import",
                        "desc": "Card body",
                        "idList": "list-1",
                        "idLabels": ["label-1"],
                        "idMembers": ["member-1"],
                        "idChecklists": ["check-1"],
                        "due": "2025-01-10T00:00:00Z",
                        "closed": False,
                        "url": "https://trello.com/c/card-1",
                        "dateLastActivity": "2025-01-02T00:00:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = TrelloBoardJsonAdapter(path=str(export)).ingest(entity_types=["card"])

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.TRELLO_BOARD_JSON
    assert unit.source_id == "trello_board_json:card-1"
    assert unit.metadata["list_name"] == "Doing"
    assert unit.metadata["labels"] == ["Import"]
    assert unit.metadata["members"] == ["Ada Lovelace"]
    assert unit.metadata["checklists"] == [{"name": "Launch", "total": 2, "complete": 1}]
    assert unit.metadata["url"] == "https://trello.com/c/card-1"
    assert {edge.metadata["kind"] for edge in result.edges} == {"list", "label", "member", "checklist"}
    assert get_adapter("trello_board_json", path=str(export)).name == "trello_board_json"


def test_trello_board_json_emits_check_item_units_and_card_edges(tmp_path):
    export = tmp_path / "board.json"
    export.write_text(
        json.dumps(
            {
                "lists": [{"id": "list-1", "name": "Doing"}],
                "labels": [{"id": "label-1", "name": "Import"}],
                "members": [{"id": "member-1", "fullName": "Ada Lovelace"}],
                "checklists": [
                    {
                        "id": "check-1",
                        "name": "Launch",
                        "checkItems": [
                            {
                                "id": "item-1",
                                "name": "Write tests",
                                "state": "complete",
                                "due": "2025-01-09T00:00:00Z",
                                "dueReminder": "1440",
                                "idMembers": ["member-1"],
                            },
                            {"id": "item-2", "name": "Ship adapter", "state": "incomplete"},
                        ],
                    }
                ],
                "cards": [
                    {
                        "id": "card-1",
                        "name": "Add Trello import",
                        "desc": "Card body",
                        "idList": "list-1",
                        "idLabels": ["label-1"],
                        "idMembers": ["member-1"],
                        "idChecklists": ["check-1"],
                        "due": "2025-01-10T00:00:00Z",
                        "url": "https://trello.com/c/card-1",
                        "dateLastActivity": "2025-01-02T00:00:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = TrelloBoardJsonAdapter(path=str(export)).ingest()

    assert TrelloBoardJsonAdapter(path=str(export)).entity_types == ["card", "check_item"]
    assert [unit.source_entity_type for unit in result.units] == ["card", "check_item", "check_item"]
    check_item = next(unit for unit in result.units if unit.metadata.get("item_id") == "item-1")
    assert check_item.source_id == "trello_board_json:card-1:check_item:check-1:item-1"
    assert check_item.title == "Write tests"
    assert check_item.content == "\n".join(
        [
            "Write tests",
            "State: complete",
            "Checklist: Launch",
            "Card: Add Trello import",
            "List: Doing",
            "Due: 2025-01-09T00:00:00Z",
            "URL: https://trello.com/c/card-1",
            "Labels: Import",
            "Member IDs: member-1",
        ]
    )
    assert check_item.metadata["checklist_id"] == "check-1"
    assert check_item.metadata["checklist_name"] == "Launch"
    assert check_item.metadata["item_id"] == "item-1"
    assert check_item.metadata["item_name"] == "Write tests"
    assert check_item.metadata["state"] == "complete"
    assert check_item.metadata["due"] == "2025-01-09T00:00:00Z"
    assert check_item.metadata["dueReminder"] == "1440"
    assert check_item.metadata["member_ids"] == ["member-1"]
    assert check_item.metadata["card_id"] == "card-1"
    assert check_item.metadata["card_name"] == "Add Trello import"
    assert check_item.metadata["card_url"] == "https://trello.com/c/card-1"
    assert check_item.metadata["list_name"] == "Doing"
    assert check_item.metadata["labels"] == ["Import"]
    assert check_item.metadata["source_file"] == "board.json"

    item_edges = [edge for edge in result.edges if edge.metadata["kind"] == "check_item"]
    assert len(item_edges) == 2
    assert item_edges[0].from_unit_id == "trello_board_json:card-1"
    assert item_edges[0].to_unit_id.startswith("trello_board_json:card-1:check_item:check-1:item-")
    assert item_edges[0].relation == EdgeRelation.CONTAINS
    assert item_edges[0].source == EdgeSource.SOURCE
    assert item_edges[0].metadata["relation_type"] == "trello_card_check_item"


def test_trello_board_json_entity_type_filters_check_items(tmp_path):
    export = tmp_path / "board.json"
    export.write_text(
        json.dumps(
            {
                "checklists": [
                    {
                        "id": "check-1",
                        "name": "Launch",
                        "checkItems": [{"id": "item-1", "name": "Write tests", "state": "complete"}],
                    }
                ],
                "cards": [{"id": "card-1", "name": "Add Trello import", "idChecklists": ["check-1"]}],
            }
        ),
        encoding="utf-8",
    )
    adapter = TrelloBoardJsonAdapter(path=str(export))

    cards = adapter.ingest(entity_types=["card"])
    check_items = adapter.ingest(entity_types=["check_item"])

    assert [unit.source_entity_type for unit in cards.units] == ["card"]
    assert {edge.metadata["kind"] for edge in cards.edges} == {"checklist"}
    assert [unit.source_entity_type for unit in check_items.units] == ["check_item"]
    assert check_items.units[0].source_id == "trello_board_json:card-1:check_item:check-1:item-1"
    assert check_items.edges == []


def test_trello_board_json_adds_checklist_items_to_card_metadata_and_content(tmp_path):
    export = tmp_path / "board.json"
    export.write_text(
        json.dumps(
            {
                "checklists": [
                    {
                        "id": "check-1",
                        "name": "Launch",
                        "checkItems": [
                            {"id": "item-1", "name": "Write tests", "state": "complete", "due": "2025-01-09T00:00:00Z"},
                            {"id": "item-2", "name": "Ship adapter", "state": "incomplete"},
                        ],
                    }
                ],
                "cards": [
                    {
                        "id": "card-1",
                        "name": "Add Trello import",
                        "idChecklists": ["check-1"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    card = TrelloBoardJsonAdapter(path=str(export)).ingest(entity_types=["card"]).units[0]

    assert card.metadata["checklist_items"] == [
        {
            "checklist_id": "check-1",
            "checklist_name": "Launch",
            "item_id": "item-1",
            "item_name": "Write tests",
            "state": "complete",
            "complete": True,
            "due": "2025-01-09T00:00:00Z",
            "position": 0,
        },
        {
            "checklist_id": "check-1",
            "checklist_name": "Launch",
            "item_id": "item-2",
            "item_name": "Ship adapter",
            "state": "incomplete",
            "complete": False,
            "position": 1,
        },
    ]
    assert "Checklist item: Write tests (complete; Checklist: Launch; Due: 2025-01-09T00:00:00Z)" in card.content
    assert "Checklist item: Ship adapter (incomplete; Checklist: Launch)" in card.content
