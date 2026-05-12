from __future__ import annotations

import json

from graph.adapters.registry import get_adapter
from graph.adapters.trello_board_json import TrelloBoardJsonAdapter
from graph.types.enums import SourceProject


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

    result = TrelloBoardJsonAdapter(path=str(export)).ingest()

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
