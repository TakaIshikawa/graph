from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.google_photos_takeout import GooglePhotosTakeoutAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import SourceProject


def test_google_photos_takeout_ingests_photo_sidecar_with_metadata(tmp_path):
    sidecar = tmp_path / "Vacation" / "IMG_0001.JPG.json"
    sidecar.parent.mkdir()
    sidecar.write_text(
        json.dumps(
            {
                "title": "IMG_0001.JPG",
                "description": "Sunset at the overlook",
                "imageViews": "42",
                "creationTime": {"timestamp": "1710000000", "formatted": "Mar 9, 2024, 4:00:00 PM UTC"},
                "photoTakenTime": {"timestamp": "1709996400", "formatted": "Mar 9, 2024, 3:00:00 PM UTC"},
                "geoData": {"latitude": 37.7793, "longitude": -122.4192, "altitude": 12.5},
                "geoDataExif": {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0},
                "url": "https://photos.google.com/photo/photo-id",
                "people": [{"name": "Alice", "url": "https://photos.google.com/people/alice"}],
            }
        ),
        encoding="utf-8",
    )

    result = GooglePhotosTakeoutAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.GOOGLE_PHOTOS_TAKEOUT
    assert unit.source_entity_type == "photo"
    assert unit.source_id == GooglePhotosTakeoutAdapter(path=str(tmp_path)).ingest().units[0].source_id
    assert unit.title == "IMG_0001.JPG"
    assert unit.content == "Sunset at the overlook"
    assert unit.created_at == datetime(2024, 3, 9, 15, 0, tzinfo=timezone.utc)
    assert unit.metadata["description"] == "Sunset at the overlook"
    assert unit.metadata["imageViews"] == 42
    assert unit.metadata["url"] == "https://photos.google.com/photo/photo-id"
    assert unit.metadata["photoTakenTime"]["timestamp"] == "1709996400"
    assert unit.metadata["creationTime"]["timestamp"] == "1710000000"
    assert unit.metadata["latitude"] == 37.7793
    assert unit.metadata["longitude"] == -122.4192
    assert unit.metadata["altitude"] == 12.5
    assert unit.metadata["geo_source"] == "geoData"
    assert unit.metadata["people"] == [{"name": "Alice", "url": "https://photos.google.com/people/alice"}]
    assert unit.metadata["album"] == "Vacation"
    assert "album:Vacation" in unit.tags


def test_google_photos_takeout_ingests_video_and_caller_album_context(tmp_path):
    sidecar = tmp_path / "VID_0002.MP4.json"
    sidecar.write_text(
        json.dumps(
            {
                "title": "VID_0002.MP4",
                "creationTime": {"timestamp": "1710000000"},
                "photoTakenTime": {"timestamp": "1709996400"},
                "geoData": {"latitude": 0.0, "longitude": 0.0},
                "geoDataExif": {"latitude": 35.6812, "longitude": 139.7671},
            }
        ),
        encoding="utf-8",
    )

    result = GooglePhotosTakeoutAdapter(path=str(sidecar), album="Camera Uploads").ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_entity_type == "video"
    assert unit.metadata["album"] == "Camera Uploads"
    assert unit.metadata["latitude"] == 35.6812
    assert unit.metadata["longitude"] == 139.7671
    assert unit.metadata["geo_source"] == "geoDataExif"


def test_google_photos_takeout_missing_optional_fields_do_not_fail(tmp_path):
    sidecar = tmp_path / "minimal.json"
    sidecar.write_text(json.dumps({"title": "minimal.png"}), encoding="utf-8")

    result = GooglePhotosTakeoutAdapter(path=str(sidecar)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "minimal.png"
    assert unit.source_entity_type == "photo"
    assert unit.metadata["description"] == ""
    assert unit.metadata["people"] == []
    assert "latitude" not in unit.metadata


def test_google_photos_takeout_registry():
    assert "google_photos_takeout" in list_adapters()
    assert get_adapter("google_photos_takeout", path="/tmp/photos").name == "google_photos_takeout"
