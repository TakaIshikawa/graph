"""Adapter for Safari History.db SQLite databases."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class SafariHistoryAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "safari_history"

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

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        units: list[KnowledgeUnit] = []
        for path in self._iter_paths():
            try:
                rows = self._read_rows(path)
            except sqlite3.Error:
                continue
            for row in rows:
                unit = self._unit_from_row(row, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                units.append(unit)

        result.units.extend(sorted(units, key=lambda unit: (unit.updated_at, unit.source_id)))
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file():
            return [root]
        if not root.is_dir():
            return []
        return sorted(
            (
                child
                for child in root.rglob("*")
                if child.is_file() and child.name in {"History.db", "History.sqlite", "History"}
            ),
            key=lambda child: str(child.relative_to(root)),
        )

    def _read_rows(self, path: Path) -> list[sqlite3.Row]:
        uri = f"file:{quote(str(path), safe='/')}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            if not self._table_exists(conn, "history_items") or not self._table_exists(conn, "history_visits"):
                return []

            item_columns = self._columns(conn, "history_items")
            visit_columns = self._columns(conn, "history_visits")
            item_title_expr = "i.title" if "title" in item_columns else "NULL"
            domain_expr = "i.domain_expansion" if "domain_expansion" in item_columns else "NULL"
            visit_count_expr = "i.visit_count" if "visit_count" in item_columns else "NULL"
            visit_title_expr = "v.title" if "title" in visit_columns else "NULL"
            load_successful_expr = "v.load_successful" if "load_successful" in visit_columns else "NULL"
            http_non_get_expr = "v.http_non_get" if "http_non_get" in visit_columns else "NULL"
            synthesized_expr = "v.synthesized" if "synthesized" in visit_columns else "NULL"
            redirect_source_expr = "v.redirect_source" if "redirect_source" in visit_columns else "NULL"
            redirect_destination_expr = "v.redirect_destination" if "redirect_destination" in visit_columns else "NULL"
            origin_expr = "v.origin" if "origin" in visit_columns else "NULL"
            attributes_expr = "v.attributes" if "attributes" in visit_columns else "NULL"
            score_expr = "v.score" if "score" in visit_columns else "NULL"

            query = f"""
                SELECT
                    v.id AS visit_id,
                    v.history_item AS history_item_id,
                    v.visit_time AS visit_time,
                    {visit_title_expr} AS visit_title,
                    {load_successful_expr} AS load_successful,
                    {http_non_get_expr} AS http_non_get,
                    {synthesized_expr} AS synthesized,
                    {redirect_source_expr} AS redirect_source,
                    {redirect_destination_expr} AS redirect_destination,
                    {origin_expr} AS origin,
                    {attributes_expr} AS attributes,
                    {score_expr} AS score,
                    i.id AS item_id,
                    i.url AS url,
                    {item_title_expr} AS item_title,
                    {domain_expr} AS domain_expansion,
                    {visit_count_expr} AS visit_count
                FROM history_visits v
                INNER JOIN history_items i ON i.id = v.history_item
                WHERE i.url IS NOT NULL
            """
            return list(conn.execute(query))

    def _unit_from_row(self, row: sqlite3.Row, source_file: str) -> KnowledgeUnit | None:
        url = str(row["url"] or "").strip()
        if not url:
            return None
        visited_at = self._safari_datetime(row["visit_time"])
        if visited_at is None:
            return None

        title = str(row["visit_title"] or row["item_title"] or "").strip() or url
        metadata = {
            "url": url,
            "title": title,
            "visit_time": self._parse_float(row["visit_time"]),
            "visited_at": visited_at.isoformat(),
            "browser": "safari",
            "source_name": "Safari History",
            "source_file": source_file,
            "history_item_id": self._parse_int(row["history_item_id"]),
            "visit_id": self._parse_int(row["visit_id"]),
            "domain_expansion": row["domain_expansion"],
            "visit_count": self._parse_int(row["visit_count"]),
            "load_successful": self._parse_bool(row["load_successful"]),
            "http_non_get": self._parse_bool(row["http_non_get"]),
            "synthesized": self._parse_bool(row["synthesized"]),
            "redirect_source": self._parse_int(row["redirect_source"]),
            "redirect_destination": self._parse_int(row["redirect_destination"]),
            "origin": row["origin"],
            "attributes": self._parse_int(row["attributes"]),
            "score": self._parse_float(row["score"]),
        }
        return KnowledgeUnit(
            source_project=SourceProject.SAFARI_HISTORY,
            source_id=self._source_id(url, row["visit_id"], row["visit_time"]),
            source_entity_type="page_visit",
            title=title,
            content=self._content(title, url, visited_at),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["safari", "browser_history"],
            created_at=visited_at,
            updated_at=visited_at,
        )

    def _content(self, title: str, url: str, visited_at: datetime) -> str:
        return "\n".join([title, f"URL: {url}", f"Visited: {visited_at.isoformat()}"])

    def _source_id(self, url: str, visit_id: Any, visit_time: Any) -> str:
        payload = repr((url, self._parse_int(visit_id), self._parse_float(visit_time)))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"safari_history:{digest}"

    def _safari_datetime(self, value: Any) -> datetime | None:
        seconds = self._parse_float(value)
        if seconds is None:
            return None
        try:
            return datetime(2001, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)
        except OverflowError:
            return None

    def _parse_bool(self, value: Any) -> bool | None:
        parsed = self._parse_int(value)
        if parsed is None:
            return None
        return bool(parsed)

    def _parse_float(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_int(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _table_exists(self, conn: sqlite3.Connection, name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        return row is not None

    def _columns(self, conn: sqlite3.Connection, table: str) -> set[str]:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
