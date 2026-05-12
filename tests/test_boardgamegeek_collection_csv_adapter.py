from __future__ import annotations

import csv

from graph.adapters.boardgamegeek_collection_csv import BoardGameGeekCollectionCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import EdgeRelation, SourceProject


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_boardgamegeek_collection_csv_ingests_export_rows(tmp_path):
    export = tmp_path / "collection.csv"
    _write_csv(
        export,
        [
            {
                "objectid": "174430",
                "objectname": "Gloomhaven",
                "yearpublished": "2017",
                "rating": "9.5",
                "average": "8.6",
                "owned": "1",
                "wishlist": "0",
                "preordered": "false",
                "numplays": "12",
                "designer": "Isaac Childres",
                "publisher": "Cephalofair Games",
                "comment": "Campaign box",
            }
        ],
    )

    result = BoardGameGeekCollectionCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.BOARDGAMEGEEK_COLLECTION_CSV
    assert unit.source_id == "boardgamegeek_collection_csv:174430"
    assert unit.title == "Gloomhaven (2017)"
    assert unit.metadata["bgg_id"] == "174430"
    assert unit.metadata["year_published"] == 2017
    assert unit.metadata["rating"] == 9.5
    assert unit.metadata["average_rating"] == 8.6
    assert unit.metadata["owned"] is True
    assert unit.metadata["wishlist"] is False
    assert unit.metadata["plays"] == 12
    assert unit.metadata["designers"] == ["Isaac Childres"]
    assert unit.metadata["collection_comments"] == "Campaign box"
    assert "owned" in unit.tags
    assert result.edges == []


def test_boardgamegeek_collection_csv_aliases_booleans_and_registry(tmp_path):
    export = tmp_path / "collection.csv"
    _write_csv(
        export,
        [
            {
                "BGG ID": "13",
                "Name": "Catan",
                "Year Published": "1995",
                "User Rating": "N/A",
                "Wishlist": "yes",
                "Preordered": "no",
                "Plays": "3.0",
                "Designers": "Klaus Teuber; Someone Else",
            }
        ],
    )

    result = BoardGameGeekCollectionCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["wishlist"] is True
    assert unit.metadata["preordered"] is False
    assert "rating" not in unit.metadata
    assert unit.metadata["plays"] == 3
    assert unit.metadata["designers"] == ["Klaus Teuber", "Someone Else"]
    assert get_adapter("boardgamegeek_collection_csv", path=str(export)).name == "boardgamegeek_collection_csv"


def test_boardgamegeek_collection_csv_emits_publisher_aggregates_and_edges(tmp_path):
    export = tmp_path / "collection.csv"
    _write_csv(
        export,
        [
            {
                "objectid": "1",
                "objectname": "One",
                "yearpublished": "2020",
                "rating": "8",
                "owned": "1",
                "numplays": "2",
                "publisher": "Cephalofair Games; Other Publisher",
            },
            {
                "objectid": "2",
                "objectname": "Two",
                "yearpublished": "2022",
                "rating": "10",
                "owned": "0",
                "numplays": "0",
                "publisher": "cephalofair games",
            },
            {
                "objectid": "3",
                "objectname": "Three",
                "yearpublished": "2019",
                "rating": "N/A",
                "owned": "yes",
                "numplays": "5",
                "publisher": "Other Publisher",
            },
        ],
    )

    result = BoardGameGeekCollectionCsvAdapter(path=str(export)).ingest(entity_types=["publisher", "board_game"])

    publishers = [unit for unit in result.units if unit.source_entity_type == "publisher"]
    games = [unit for unit in result.units if unit.source_entity_type == "board_game"]
    assert BoardGameGeekCollectionCsvAdapter().entity_types == ["board_game", "publisher"]
    assert len(publishers) == 2

    cephalofair = next(unit for unit in publishers if unit.metadata["normalized_publisher"] == "cephalofair games")
    cephalofair_games = [game for game in games if "Cephalofair Games" in game.metadata.get("publishers", []) or "cephalofair games" in game.metadata.get("publishers", [])]
    assert cephalofair.source_id.startswith("boardgamegeek_collection_csv:publisher:")
    assert cephalofair.metadata["publisher"] == "Cephalofair Games"
    assert cephalofair.metadata["game_count"] == 2
    assert cephalofair.metadata["game_source_ids"] == sorted(game.source_id for game in cephalofair_games)
    assert cephalofair.metadata["owned_count"] == 1
    assert cephalofair.metadata["played_count"] == 1
    assert cephalofair.metadata["average_user_rating"] == 9.0
    assert cephalofair.metadata["year_range"] == [2020, 2022]
    assert {
        (edge.from_unit_id, edge.to_unit_id, edge.relation, edge.metadata["relation_type"])
        for edge in result.edges
        if edge.from_unit_id == cephalofair.source_id
    } == {
        (cephalofair.source_id, game.source_id, EdgeRelation.CONTAINS, "publisher_contains_game")
        for game in cephalofair_games
    }


def test_boardgamegeek_collection_csv_publisher_filtering(tmp_path):
    export = tmp_path / "collection.csv"
    _write_csv(export, [{"objectid": "1", "objectname": "One", "publisher": "Cephalofair Games"}])

    publisher_only = BoardGameGeekCollectionCsvAdapter(path=str(export)).ingest(entity_types=["publisher"])
    game_only = BoardGameGeekCollectionCsvAdapter(path=str(export)).ingest(entity_types=["board_game"])

    assert [unit.source_entity_type for unit in publisher_only.units] == ["publisher"]
    assert publisher_only.edges == []
    assert [unit.source_entity_type for unit in game_only.units] == ["board_game"]
    assert game_only.edges == []
