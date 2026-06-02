from datetime import datetime, timezone

from graph.adapters.registry import get_adapter
from graph.adapters.trello_cards_csv import TrelloCardsCsvAdapter
from graph.types.models import SyncState


def test_trello_cards_csv_ingests_card_metadata(tmp_path):
    path = tmp_path / "cards.csv"
    path.write_text("Card ID,Card Name,Description,URL,Board,List,Labels,Members,Due,Date Last Activity,Closed\nc1,Ship adapter,Parse cards,https://trello.test/c1,Graph,Doing,\"backend;csv\",Ada,2026-06-10,2026-06-01T00:00:00Z,false\n", encoding="utf-8")

    unit = TrelloCardsCsvAdapter(str(path)).ingest().units[0]

    assert unit.source_id == "trello_cards_csv:c1"
    assert unit.metadata["board"] == "Graph"
    assert unit.metadata["list"] == "Doing"
    assert unit.metadata["labels"] == ["backend", "csv"]
    assert unit.metadata["closed"] is False
    assert {"trello", "card", "Graph", "Doing", "backend"}.issubset(set(unit.tags))


def test_trello_cards_csv_since_entity_filter_and_registry(tmp_path):
    path = tmp_path / "cards.csv"
    path.write_text("id,name,date last activity\nold,Old,2026-04-01T00:00:00Z\nnew,New,2026-05-02T00:00:00Z\n", encoding="utf-8")
    since = SyncState(source_project="trello_cards_csv", source_entity_type="card", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    result = TrelloCardsCsvAdapter(str(path)).ingest(since=since, entity_types=["card"])

    assert [unit.source_id for unit in result.units] == ["trello_cards_csv:new"]
    assert TrelloCardsCsvAdapter(str(path)).ingest(entity_types=["board"]).units == []
    assert get_adapter("trello_cards_csv", path=str(path)).name == "trello_cards_csv"
