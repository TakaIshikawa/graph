from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.foursquare_checkins_csv import FoursquareCheckinsCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_foursquare_checkins_csv_ingests_checkin_metadata_and_registry(tmp_path):
    export = tmp_path / "checkins.csv"
    export.write_text(
        "\n".join(
            [
                "Venue,Category,City,Address,Latitude,Longitude,Created At,Shout,URL",
                "Cafe,Food,Tokyo,1 Main,35.1,139.2,2025-01-02T03:04:05Z,Great coffee,https://4sq.com/checkin",
            ]
        ),
        encoding="utf-8",
    )

    result = FoursquareCheckinsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.FOURSQUARE_CHECKINS_CSV
    assert unit.source_entity_type == "checkin"
    assert unit.metadata["venue"] == "Cafe"
    assert unit.metadata["category"] == "Food"
    assert unit.metadata["city"] == "Tokyo"
    assert unit.metadata["address"] == "1 Main"
    assert unit.metadata["latitude"] == 35.1
    assert unit.metadata["longitude"] == 139.2
    assert unit.metadata["created_at"] == "2025-01-02T03:04:05+00:00"
    assert unit.metadata["note"] == "Great coffee"
    assert unit.metadata["source_file"] == "checkins.csv"
    assert unit.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert get_adapter("foursquare_checkins_csv", path=str(export)).name == "foursquare_checkins_csv"


def test_foursquare_checkins_csv_aliases_directory_invalid_coordinates_and_filters(tmp_path):
    (tmp_path / "one.csv").write_text("Name,Venue Category,Town,Location,Lat,Lng,Date,Comment\nOld,Park,Tokyo,Park St,95,200,2025-01-01,Old\n", encoding="utf-8")
    (tmp_path / "two.csv").write_text("Place,Category,City,Address,Latitude,Longitude,Checkin Date,Note,Link\nNew,Museum,Kyoto,2 Main,35.5,135.7,2025-01-03,New,https://example.com/new\n", encoding="utf-8")
    (tmp_path / "bad.csv").write_bytes(b"\xff\xfe\x00")

    adapter = FoursquareCheckinsCsvAdapter(path=str(tmp_path))
    sync = SyncState(source_project="foursquare_checkins_csv", source_entity_type="checkin", last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)
    all_units = adapter.ingest().units

    assert [unit.title for unit in first.units] == ["New"]
    assert first.units[0].metadata["latitude"] == 35.5
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    old = next(unit for unit in all_units if unit.title == "Old")
    assert "latitude" not in old.metadata
    assert "longitude" not in old.metadata
    assert adapter.ingest(entity_types=["venue"]).units == []
