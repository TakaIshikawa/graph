"""Adapter for Firefox places.sqlite browser history."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class FirefoxPlacesAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "firefox_places"

    @property
    def entity_types(self) -> list[str]:
        return ["page_visit"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "page_visit" not in entity_types:
            return result
        path = Path(self.path).expanduser() if self.path else None
        if path is None or not path.is_file():
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        try:
            rows = self._read_rows(path)
        except sqlite3.Error:
            return result

        units: list[KnowledgeUnit] = []
        for row in rows:
            unit = self._unit_from_row(row)
            if unit is None:
                continue
            if sync_at and unit.created_at <= sync_at:
                continue
            units.append(unit)
        result.units.extend(sorted(units, key=lambda unit: (unit.created_at, unit.source_id)))
        return result

    def _read_rows(self, path: Path) -> list[sqlite3.Row]:
        uri = f"file:{quote(str(path), safe='/')}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            if not self._table_exists(conn, "moz_places"):
                return []
            has_bookmarks = self._table_exists(conn, "moz_bookmarks")
            place_columns = self._columns(conn, "moz_places")
            history_columns = self._columns(conn, "moz_historyvisits") if self._table_exists(conn, "moz_historyvisits") else set()
            bookmark_columns = self._columns(conn, "moz_bookmarks") if has_bookmarks else set()
            visit_count_expr = "p.visit_count" if "visit_count" in place_columns else "NULL"
            frecency_expr = "p.frecency" if "frecency" in place_columns else "NULL"
            typed_expr = "p.typed" if "typed" in place_columns else "0"
            last_visit_expr = "p.last_visit_date" if "last_visit_date" in place_columns else "NULL"
            history_visit_expr = "MAX(h.visit_date)" if "visit_date" in history_columns else "NULL"
            history_count_expr = "COUNT(h.id)" if "id" in history_columns else "0"
            history_join = "LEFT JOIN moz_historyvisits h ON h.place_id = p.id" if history_columns else ""
            bookmark_select = (
                "EXISTS(SELECT 1 FROM moz_bookmarks b WHERE b.fk = p.id AND b.type = 1) AS bookmarked"
                if has_bookmarks
                else "0 AS bookmarked"
            )
            bookmark_date_select = (
                "MAX(b.dateAdded) AS bookmark_date"
                if has_bookmarks and "dateAdded" in bookmark_columns
                else "NULL AS bookmark_date"
            )
            bookmark_join = (
                "LEFT JOIN moz_bookmarks b ON b.fk = p.id AND b.type = 1"
                if has_bookmarks
                else ""
            )
            query = f"""
                SELECT
                    p.id,
                    p.url,
                    p.title,
                    {visit_count_expr} AS visit_count,
                    {frecency_expr} AS frecency,
                    {typed_expr} AS typed,
                    {last_visit_expr} AS last_visit_date,
                    {history_visit_expr} AS visit_date,
                    {history_count_expr} AS history_visit_count,
                    {bookmark_select},
                    {bookmark_date_select}
                FROM moz_places p
                {history_join}
                {bookmark_join}
                WHERE p.url IS NOT NULL
                GROUP BY p.id
            """
            return list(conn.execute(query))

    def _unit_from_row(self, row: sqlite3.Row) -> KnowledgeUnit | None:
        url = str(row["url"] or "").strip()
        if not url:
            return None
        last_visit_at = self._firefox_datetime(row["visit_date"] or row["last_visit_date"] or row["bookmark_date"])
        if last_visit_at is None:
            return None
        title = str(row["title"] or url).strip()
        metadata = {
            "url": url,
            "title": title,
            "visit_count": row["visit_count"],
            "history_visit_count": row["history_visit_count"],
            "frecency": row["frecency"],
            "last_visit_at": last_visit_at.isoformat(),
            "typed": bool(row["typed"]),
            "bookmarked": bool(row["bookmarked"]),
            "bookmark_date": self._parse_int(row["bookmark_date"]),
            "place_id": row["id"],
        }
        return KnowledgeUnit(
            source_project=SourceProject.FIREFOX_PLACES,
            source_id=self._source_id(url),
            source_entity_type="page_visit",
            title=title,
            content=url,
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["firefox", "browser_history"],
            created_at=last_visit_at,
            updated_at=last_visit_at,
        )

    def _source_id(self, url: str) -> str:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        return f"firefox_places:{digest}"

    def _firefox_datetime(self, value: Any) -> datetime | None:
        if value is None:
            return None
        try:
            micros = int(value)
        except (TypeError, ValueError):
            return None
        return datetime.fromtimestamp(micros / 1_000_000, tz=timezone.utc)

    def _table_exists(self, conn: sqlite3.Connection, name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        return row is not None

    def _columns(self, conn: sqlite3.Connection, table: str) -> set[str]:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}

    def _parse_int(self, value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
