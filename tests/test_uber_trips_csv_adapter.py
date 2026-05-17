from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.uber_trips_csv import UberTripsCsvAdapter
from graph.types.models import SyncState


def test_uber_trips_csv_ingests_trip_metadata_and_tags(tmp_path):
    export = tmp_path / "uber_trips.csv"
    export.write_text(
        "Trip UUID,Trip/Request Date,Completed Date,City,Product Type,Status,Fare,Currency,Distance (mi),Duration,Pickup Address,Dropoff Address,Driver\n"
        "trip-1,2026-05-01T08:00:00Z,2026-05-01T08:24:00Z,Tokyo,UberX,completed,$18.40,USD,7.2,24 min,1 Start St,9 End Ave,Ada\n",
        encoding="utf-8",
    )

    result = UberTripsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "uber_trips_csv"
    assert unit.source_id == "uber_trips_csv:trip-1"
    assert unit.source_entity_type == "trip"
    assert unit.metadata["requested_at"] == "2026-05-01T08:00:00+00:00"
    assert unit.metadata["completed_at"] == "2026-05-01T08:24:00+00:00"
    assert unit.metadata["city"] == "Tokyo"
    assert unit.metadata["product_type"] == "UberX"
    assert unit.metadata["status"] == "completed"
    assert unit.metadata["fare"] == 18.4
    assert unit.metadata["currency"] == "USD"
    assert unit.metadata["distance"] == 7.2
    assert unit.metadata["distance_unit"] == "mi"
    assert unit.metadata["duration_seconds"] == 1440
    assert unit.metadata["pickup_address"] == "1 Start St"
    assert unit.metadata["dropoff_address"] == "9 End Ave"
    assert unit.metadata["driver"] == "Ada"
    assert unit.created_at == datetime(2026, 5, 1, 8, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2026, 5, 1, 8, 24, tzinfo=timezone.utc)
    assert {"uber", "trip", "Tokyo", "UberX", "completed"}.issubset(set(unit.tags))
    assert "Fare: 18.4 USD" in unit.content


def test_uber_trips_csv_directory_aliases_invalid_files_since_and_entity_filter(tmp_path):
    (tmp_path / "old.csv").write_text(
        "Date,End Time,Trip City,Product,Trip Status,Total,Fare Currency,Distance (km),Duration Seconds,Origin,Destination\n"
        "2026-05-01,2026-05-01 00:30:00,Paris,Uber Green,completed,12.00,EUR,4.5,1800,Old pickup,Old dropoff\n",
        encoding="utf-8",
    )
    (tmp_path / "new.csv").write_text(
        "Request Date,Completed At,City,Service,Status,Fare,Currency,Distance Unit,Trip Distance,Trip Duration,From,To\n"
        "2026-05-03 09:00,2026-05-03 09:18,Kyoto,Black,completed,2600,JPY,km,8.1,00:18:00,Station,Hotel\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("not csv", encoding="utf-8")
    (tmp_path / "bad.csv").write_bytes(b"\xff\xfe\x00")

    sync = SyncState(source_project="uber_trips_csv", source_entity_type="trip", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))
    adapter = UberTripsCsvAdapter(path=str(tmp_path))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)
    all_units = adapter.ingest().units

    assert [unit.metadata["city"] for unit in first.units] == ["Kyoto"]
    assert first.units[0].metadata["distance"] == 8.1
    assert first.units[0].metadata["distance_unit"] == "km"
    assert first.units[0].metadata["duration_seconds"] == 1080
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert [unit.metadata["city"] for unit in all_units] == ["Paris", "Kyoto"]
    assert adapter.ingest(entity_types=["ride"]).units == []


def test_uber_trips_csv_tolerates_empty_and_sparse_rows(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    sparse = tmp_path / "sparse.csv"
    sparse.write_text("Trip UUID,City,Fare\n, ,\n,New York,$33.10\n", encoding="utf-8")

    result = UberTripsCsvAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["city"] == "New York"
    assert result.units[0].metadata["fare"] == 33.1
    assert result.units[0].source_id.startswith("uber_trips_csv:")
