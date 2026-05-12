"""Adapter for Firefox places.sqlite browser history."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote_plus, urlsplit

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class FirefoxPlacesAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "firefox_places"

    @property
    def entity_types(self) -> list[str]:
        return ["page_visit", "search_term"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        requested = set(entity_types or self.entity_types)
        if not requested.intersection(self.entity_types):
            return result
        path = Path(self.path).expanduser() if self.path else None
        if path is None or not path.is_file():
            return result

        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        try:
            rows, search_rows = self._read_rows(path)
        except sqlite3.Error:
            return result

        units: list[KnowledgeUnit] = []
        page_rows = rows if "page_visit" in requested else []
        for row in page_rows:
            unit = self._unit_from_row(row)
            if unit is None:
                continue
            if sync_at and unit.created_at <= sync_at:
                continue
            units.append(unit)
        requested_search_rows = search_rows if "search_term" in requested else []
        for row in requested_search_rows:
            unit = self._search_unit_from_row(row)
            if unit is None:
                continue
            if sync_at and unit.created_at <= sync_at:
                continue
            units.append(unit)
        result.units.extend(sorted(units, key=lambda unit: (unit.created_at, unit.source_id)))
        return result

    def _read_rows(self, path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        uri = f"file:{quote(str(path), safe='/')}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            if not self._table_exists(conn, "moz_places"):
                return [], []
            has_bookmarks = self._table_exists(conn, "moz_bookmarks")
            place_columns = self._columns(conn, "moz_places")
            has_history = self._table_exists(conn, "moz_historyvisits")
            bookmark_columns = self._columns(conn, "moz_bookmarks") if has_bookmarks else set()
            visit_count_expr = "p.visit_count" if "visit_count" in place_columns else "NULL"
            frecency_expr = "p.frecency" if "frecency" in place_columns else "NULL"
            typed_expr = "p.typed" if "typed" in place_columns else "0"
            last_visit_expr = "p.last_visit_date" if "last_visit_date" in place_columns else "NULL"
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
                    {bookmark_select},
                    {bookmark_date_select}
                FROM moz_places p
                {bookmark_join}
                WHERE p.url IS NOT NULL
                GROUP BY p.id
            """
            rows = [dict(row) for row in conn.execute(query)]
            history = self._history_aggregates(conn) if has_history else {}
            for row in rows:
                aggregate = history.get(row["id"], {})
                row["first_visit_date"] = aggregate.get("first_visit_date")
                row["visit_date"] = aggregate.get("last_visit_date")
                row["history_visit_count"] = aggregate.get("history_visit_count", 0)
                row["transition_counts"] = aggregate.get("transition_counts", {})
            return rows, self._search_rows(conn, rows, history, path.name)

    def _history_aggregates(self, conn: sqlite3.Connection) -> dict[int, dict[str, Any]]:
        columns = self._columns(conn, "moz_historyvisits")
        if "place_id" not in columns:
            return {}
        visit_date_expr = "visit_date" if "visit_date" in columns else "NULL"
        count_expr = "COUNT(id)" if "id" in columns else "COUNT(*)"
        rows = conn.execute(
            f"""
            SELECT
                place_id,
                MIN({visit_date_expr}) AS first_visit_date,
                MAX({visit_date_expr}) AS last_visit_date,
                {count_expr} AS history_visit_count
            FROM moz_historyvisits
            GROUP BY place_id
            """
        )
        aggregates = {
            int(row["place_id"]): {
                "first_visit_date": row["first_visit_date"],
                "last_visit_date": row["last_visit_date"],
                "history_visit_count": row["history_visit_count"],
                "transition_counts": {},
            }
            for row in rows
        }
        if "transition" in columns:
            transition_rows = conn.execute(
                """
                SELECT place_id, transition, COUNT(*) AS count
                FROM moz_historyvisits
                GROUP BY place_id, transition
                """
            )
            for row in transition_rows:
                place_id = int(row["place_id"])
                transition = str(row["transition"])
                aggregates.setdefault(place_id, {"transition_counts": {}})
                aggregates[place_id].setdefault("transition_counts", {})[transition] = row["count"]
        return aggregates

    def _unit_from_row(self, row: dict[str, Any]) -> KnowledgeUnit | None:
        url = str(row["url"] or "").strip()
        if not url:
            return None
        last_visit_at = self._firefox_datetime(row["visit_date"] or row["last_visit_date"] or row["bookmark_date"])
        if last_visit_at is None:
            return None
        first_visit_at = self._firefox_datetime(row["first_visit_date"]) or last_visit_at
        title = str(row["title"] or url).strip()
        metadata = {
            "url": url,
            "title": title,
            "visit_count": row["visit_count"],
            "history_visit_count": row["history_visit_count"],
            "frecency": row["frecency"],
            "first_visit_at": first_visit_at.isoformat(),
            "last_visit_at": last_visit_at.isoformat(),
            "transition_counts": row["transition_counts"],
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

    def _search_rows(
        self,
        conn: sqlite3.Connection,
        place_rows: list[dict[str, Any]],
        history: dict[int, dict[str, Any]],
        source_file: str,
    ) -> list[dict[str, Any]]:
        aggregates: dict[str, dict[str, Any]] = {}
        if self._table_exists(conn, "moz_keywords"):
            self._add_keyword_search_rows(conn, history, source_file, aggregates)
        for row in place_rows:
            query = self._query_from_url(row.get("url"))
            if not query:
                continue
            aggregate = history.get(int(row["id"]), {})
            first_seen = self._firefox_datetime(aggregate.get("first_visit_date"))
            last_seen = self._firefox_datetime(
                aggregate.get("last_visit_date") or row.get("last_visit_date")
            )
            visit_count = int(aggregate.get("history_visit_count") or row.get("visit_count") or 1)
            self._add_search_aggregate(
                aggregates,
                query=query,
                visit_count=visit_count,
                first_seen_at=first_seen or last_seen,
                last_seen_at=last_seen or first_seen,
                source_table="moz_places",
                source_file=source_file,
            )
        return sorted(aggregates.values(), key=lambda item: (item["query"], item["source_table"]))

    def _add_keyword_search_rows(
        self,
        conn: sqlite3.Connection,
        history: dict[int, dict[str, Any]],
        source_file: str,
        aggregates: dict[str, dict[str, Any]],
    ) -> None:
        columns = self._columns(conn, "moz_keywords")
        keyword_expr = "keyword" if "keyword" in columns else "NULL"
        place_expr = "place_id" if "place_id" in columns else ("fk" if "fk" in columns else "NULL")
        for row in conn.execute(
            f"SELECT {keyword_expr} AS keyword, {place_expr} AS place_id FROM moz_keywords"
        ):
            query = self._normalize_query(row["keyword"])
            if not query:
                continue
            place_id = self._parse_int(row["place_id"])
            aggregate = history.get(place_id, {}) if place_id is not None else {}
            first_seen = self._firefox_datetime(aggregate.get("first_visit_date"))
            last_seen = self._firefox_datetime(aggregate.get("last_visit_date"))
            visit_count = int(aggregate.get("history_visit_count") or 1)
            self._add_search_aggregate(
                aggregates,
                query=query,
                visit_count=visit_count,
                first_seen_at=first_seen or last_seen,
                last_seen_at=last_seen or first_seen,
                source_table="moz_keywords",
                source_file=source_file,
            )

    def _add_search_aggregate(
        self,
        aggregates: dict[str, dict[str, Any]],
        *,
        query: str,
        visit_count: int,
        first_seen_at: datetime | None,
        last_seen_at: datetime | None,
        source_table: str,
        source_file: str,
    ) -> None:
        existing = aggregates.setdefault(
            query,
            {
                "query": query,
                "visit_count": 0,
                "first_seen_at": first_seen_at,
                "last_seen_at": last_seen_at,
                "source_table": source_table,
                "source_file": source_file,
            },
        )
        existing["visit_count"] += max(1, visit_count)
        if first_seen_at and (existing["first_seen_at"] is None or first_seen_at < existing["first_seen_at"]):
            existing["first_seen_at"] = first_seen_at
        if last_seen_at and (existing["last_seen_at"] is None or last_seen_at > existing["last_seen_at"]):
            existing["last_seen_at"] = last_seen_at
        tables = set(str(existing["source_table"]).split(","))
        tables.add(source_table)
        existing["source_table"] = ",".join(sorted(tables))

    def _search_unit_from_row(self, row: dict[str, Any]) -> KnowledgeUnit | None:
        query = self._normalize_query(row.get("query"))
        if not query:
            return None
        last_seen_at = row.get("last_seen_at") or row.get("first_seen_at")
        if not isinstance(last_seen_at, datetime):
            last_seen_at = datetime.now(timezone.utc)
        first_seen_at = row.get("first_seen_at") if isinstance(row.get("first_seen_at"), datetime) else last_seen_at
        metadata = {
            "query": query,
            "visit_count": row["visit_count"],
            "first_seen_at": first_seen_at.isoformat(),
            "last_seen_at": last_seen_at.isoformat(),
            "source_table": row["source_table"],
            "source_file": row["source_file"],
        }
        return KnowledgeUnit(
            source_project=SourceProject.FIREFOX_PLACES,
            source_id=self._search_source_id(query),
            source_entity_type="search_term",
            title=query,
            content=query,
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["firefox", "search"],
            created_at=last_seen_at,
            updated_at=last_seen_at,
        )

    def _source_id(self, url: str) -> str:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        return f"firefox_places:{digest}"

    def _search_source_id(self, query: str) -> str:
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:24]
        return f"firefox_places:search:{digest}"

    def _query_from_url(self, value: Any) -> str:
        url = str(value or "").strip()
        if not url:
            return ""
        parsed = urlsplit(url)
        host = (parsed.hostname or "").casefold()
        path = parsed.path.casefold()
        if "search" not in host and "search" not in path and host not in {"www.google.com", "google.com", "duckduckgo.com", "www.bing.com", "bing.com"}:
            return ""
        params = parse_qs(parsed.query)
        for key in ("q", "query", "p", "text"):
            values = params.get(key)
            if values:
                return self._normalize_query(values[0])
        return ""

    def _normalize_query(self, value: Any) -> str:
        return " ".join(unquote_plus(str(value or "")).strip().casefold().split())

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
