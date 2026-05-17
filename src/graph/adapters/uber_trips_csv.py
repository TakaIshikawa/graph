"""Adapter for Uber trip history CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_duration_seconds, parse_float, read_csv_rows
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class UberTripsCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "uber_trips_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["trip"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "trip" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None

        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _unit(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        trip_uuid = first(row, "Trip UUID", "Trip Uuid", "UUID", "Trip ID", "Trip Id", "Trip Identifier")
        requested_text = first(row, "Trip/Request Date", "Request Date", "Requested At", "Date", "Trip Date", "Begin Trip Time", "Start Time")
        completed_text = first(row, "Completed Date", "Completed At", "Dropoff Time", "End Time", "End Date")
        requested_at = parse_datetime(requested_text)
        completed_at = parse_datetime(completed_text)
        city = first(row, "City", "Trip City")
        product_type = first(row, "Product Type", "Product", "Service", "Vehicle Type")
        status = first(row, "Status", "Trip Status")
        fare = parse_float(first(row, "Fare", "Fare Amount", "Total", "Total Fare", "Amount"))
        currency = first(row, "Currency", "Fare Currency", "ISO Currency Code")
        distance = parse_float(first(row, "Distance", "Trip Distance", "Distance (mi)", "Distance (km)"))
        distance_unit = self._distance_unit(row)
        duration_seconds = parse_duration_seconds(first(row, "Duration", "Trip Duration", "Duration (seconds)", "Duration Seconds"))
        pickup_address = first(row, "Pickup Address", "Pickup Location", "Begin Trip Address", "Origin", "From")
        dropoff_address = first(row, "Dropoff Address", "Drop Off Address", "Drop-off Address", "Dropoff Location", "Destination", "To")
        driver = first(row, "Driver", "Driver Name")

        if not any([trip_uuid, requested_text, completed_text, city, product_type, status, fare is not None, pickup_address, dropoff_address, driver]):
            return None

        now = datetime.now(timezone.utc)
        trip_time = completed_at or requested_at
        metadata = clean_metadata(
            {
                "trip_uuid": trip_uuid,
                "requested_at": requested_at.isoformat() if requested_at else requested_text,
                "completed_at": completed_at.isoformat() if completed_at else completed_text,
                "city": city,
                "product_type": product_type,
                "status": status,
                "fare": fare,
                "currency": currency,
                "distance": distance,
                "distance_unit": distance_unit,
                "duration_seconds": duration_seconds,
                "pickup_address": pickup_address,
                "dropoff_address": dropoff_address,
                "driver": driver,
                "source_file": source_file,
            }
        )

        return KnowledgeUnit(
            source_project="uber_trips_csv",
            source_id=f"uber_trips_csv:{trip_uuid}" if trip_uuid else digest_source_id("uber_trips_csv", requested_text, completed_text, pickup_address, dropoff_address, fare, index),
            source_entity_type="trip",
            title=self._title(city, product_type, pickup_address, dropoff_address),
            content=self._content(metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(tag for tag in ["uber", "trip", city, product_type, status] if tag)),
            created_at=requested_at or completed_at or now,
            updated_at=trip_time or now,
        )

    def _distance_unit(self, row: dict[str, Any]) -> str:
        for key in row:
            lowered = key.casefold()
            if "distance" not in lowered:
                continue
            if "km" in lowered or "kilometer" in lowered:
                return "km"
            if "mi" in lowered or "mile" in lowered:
                return "mi"
        return first(row, "Distance Unit", "Distance Units")

    def _title(self, city: str, product_type: str, pickup_address: str, dropoff_address: str) -> str:
        route = " to ".join(part for part in [pickup_address, dropoff_address] if part)
        parts = [part for part in [product_type, city, route] if part]
        return " - ".join(parts) or "Uber trip"

    def _content(self, metadata: dict[str, Any]) -> str:
        parts = []
        for key, label in (
            ("requested_at", "Requested"),
            ("completed_at", "Completed"),
            ("city", "City"),
            ("product_type", "Product"),
            ("status", "Status"),
            ("fare", "Fare"),
            ("distance", "Distance"),
            ("duration_seconds", "Duration seconds"),
            ("pickup_address", "Pickup"),
            ("dropoff_address", "Dropoff"),
            ("driver", "Driver"),
        ):
            if metadata.get(key) is not None:
                value = metadata[key]
                if key == "fare" and metadata.get("currency"):
                    value = f"{value} {metadata['currency']}"
                if key == "distance" and metadata.get("distance_unit"):
                    value = f"{value} {metadata['distance_unit']}"
                parts.append(f"{label}: {value}")
        return "\n".join(parts)
