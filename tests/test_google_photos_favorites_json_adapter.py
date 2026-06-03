from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.google_photos_favorites_json import GooglePhotosFavoritesJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject


def test_google_photos_favorites_json_ingests_favorite_photo_and_registry(tmp_path):
    export = tmp_path / "photos.json"
    export.write_text(
        json.dumps(
            [
                {
                    "title": "beach.jpg",
                    "description": "Sunset",
                    "favorite": True,
                    "photoTakenTime": {"timestamp": "2025-01-01T10:00:00Z"},
                    "creationTime": {"timestamp": "2025-01-02T10:00:00Z"},
                    "geoData": {"latitude": 1.5, "longitude": 2.5},
                    "url": "https://photos.example/beach",
                    "people": [{"name": "Ada"}],
                    "album": "Favorites",
                    "filename": "beach.jpg",
                },
                {"title": "ignored.jpg", "favorite": False},
            ]
        ),
        encoding="utf-8",
    )

    result = GooglePhotosFavoritesJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.GOOGLE_PHOTOS_FAVORITES_JSON
    assert unit.title == "beach.jpg"
    assert unit.metadata["media_type"] == "photo"
    assert unit.metadata["geo_data"] == {"latitude": 1.5, "longitude": 2.5}
    assert unit.metadata["people"] == ["Ada"]
    assert unit.metadata["album"] == "Favorites"
    assert unit.created_at == datetime(2025, 1, 1, 10, tzinfo=timezone.utc)
    assert "Sunset" in unit.content
    assert get_adapter("google_photos_favorites_json", path=str(export)).name == "google_photos_favorites_json"


def test_google_photos_favorites_json_favorites_export_imports_all_and_video(tmp_path):
    export = tmp_path / "favorites.json"
    export.write_text(json.dumps({"items": [{"title": "clip.mp4", "mimeType": "video/mp4", "creationTime": {"timestamp": "2025-01-03T00:00:00Z"}}]}), encoding="utf-8")

    unit = GooglePhotosFavoritesJsonAdapter(path=str(export)).ingest().units[0]

    assert unit.metadata["media_type"] == "video"
    assert unit.metadata["favorite"] is True
    assert "geo_data" not in unit.metadata


def test_google_photos_favorites_json_stable_ids_and_filters(tmp_path):
    export = tmp_path / "photos.json"
    export.write_text(json.dumps([{"title": "one.jpg", "favorited": "yes"}, {"title": "two.jpg"}]), encoding="utf-8")
    adapter = GooglePhotosFavoritesJsonAdapter(path=str(export))

    first = adapter.ingest()
    second = adapter.ingest()

    assert [unit.title for unit in first.units] == ["one.jpg"]
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["album"]).units == []
