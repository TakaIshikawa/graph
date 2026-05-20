"""Adapter for Steam wishlist CSV exports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any

from graph.adapters._personal_exports import clean_metadata, digest_source_id, ensure_utc, first, iter_paths, parse_datetime, parse_float, parse_int, parse_money, read_csv_rows, split_values
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class SteamWishlistCsvAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "steam_wishlist_csv"

    @property
    def entity_types(self) -> list[str]:
        return ["wishlisted_game"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "wishlisted_game" not in entity_types:
            return result
        sync_at = ensure_utc(since.last_sync_at) if since else None

        for path in iter_paths(self.path, {".csv"}):
            try:
                rows = read_csv_rows(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            for index, row in enumerate(rows):
                unit = self._unit_from_row(row, path.name, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)

        result.units.sort(key=lambda unit: (unit.updated_at, unit.source_id))
        return result

    def _unit_from_row(self, row: dict[str, Any], source_file: str, index: int) -> KnowledgeUnit | None:
        app_id = first(row, "App ID", "AppID", "appid", "Application ID", "steam_appid", "Game ID")
        title = first(row, "Title", "Name", "Game", "Game Title")
        store_url = first(row, "Store URL", "URL", "Link", "Store Link")
        if not store_url and app_id:
            store_url = f"https://store.steampowered.com/app/{app_id}/"
        if not any([app_id, title, store_url]):
            return None

        release_text = first(row, "Release Date", "Released", "Release")
        added_text = first(row, "Added Date", "Wishlisted Date", "Date Added", "Added", "Wishlist Date", "Added On")
        modified_text = first(row, "Modified Time", "Updated Time", "Updated At", "Last Updated")
        release_date = parse_datetime(release_text)
        added_at = parse_datetime(added_text)
        modified_at = parse_datetime(modified_text)
        review_score = parse_float(first(row, "Review Score", "Reviews Score", "User Review Score", "Score", "Review %"))
        review_count = parse_int(first(row, "Review Count", "Reviews Count", "Total Reviews", "Reviews"))
        price = parse_money(first(row, "Price", "Current Price", "Final Price"))
        original_price = parse_money(first(row, "Original Price", "Base Price", "List Price"))
        discount_percent = parse_int(first(row, "Discount", "Discount Percent", "Sale Discount"))
        ranking = parse_int(first(row, "Ranking", "Rank", "Wishlist Rank", "Order"))
        genres = self._labels(first(row, "Genres", "Genre", "Tags", "Tag"))
        platforms = self._labels(first(row, "Platforms", "Platform", "Available Platforms"))
        review_summary = first(row, "Review Summary", "Reviews Summary", "Reviews Text", "Recent Reviews")
        notes = first(row, "Notes", "Note", "Comment")
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "app_id": app_id,
                "store_url": store_url,
                "release_date": release_date.isoformat() if release_date else release_text,
                "review_score": review_score,
                "review_count": review_count,
                "review_summary": review_summary,
                "price": price,
                "original_price": original_price,
                "discount_percent": discount_percent,
                "genres": genres,
                "platforms": platforms,
                "ranking": ranking,
                "added_at": added_at.isoformat() if added_at else added_text,
                "modified_at": modified_at.isoformat() if modified_at else modified_text,
                "notes": notes,
                "source_file": source_file,
                "source_row": dict(row),
            }
        )
        timestamp = added_at or release_date or modified_at or now
        sync_timestamp = modified_at or added_at or release_date or now
        return KnowledgeUnit(
            source_project="steam_wishlist_csv",
            source_id=f"steam_wishlist_csv:{app_id}" if app_id else digest_source_id("steam_wishlist_csv", title, store_url, added_text, ranking, source_file, index),
            source_entity_type="wishlisted_game",
            title=title or f"Steam app {app_id}",
            content=self._content(title or f"Steam app {app_id}", metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=list(dict.fromkeys(["steam", "wishlist", *genres, *platforms])),
            created_at=timestamp,
            updated_at=sync_timestamp,
        )

    def _labels(self, value: str) -> list[str]:
        labels: list[str] = []
        for item in split_values(value):
            label = " ".join(item.casefold().split())
            if label and label not in labels:
                labels.append(label)
        return labels

    def _content(self, title: str, metadata: dict[str, Any]) -> str:
        parts = [
            title,
            f"Store URL: {metadata.get('store_url')}" if metadata.get("store_url") else "",
            f"Release date: {metadata.get('release_date')}" if metadata.get("release_date") else "",
            f"Price: {metadata.get('price')}" if metadata.get("price") is not None else "",
            f"Discount: {metadata.get('discount_percent')}%" if metadata.get("discount_percent") is not None else "",
            f"Genres: {', '.join(metadata.get('genres') or [])}" if metadata.get("genres") else "",
            f"Platforms: {', '.join(metadata.get('platforms') or [])}" if metadata.get("platforms") else "",
            metadata.get("notes", ""),
        ]
        return "\n".join(part for part in parts if part)
