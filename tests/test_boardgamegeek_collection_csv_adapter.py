from __future__ import annotations

import csv

from graph.adapters.boardgamegeek_collection_csv import BoardGameGeekCollectionCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject


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
