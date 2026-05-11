"""Adapter for Chrome/Chromium History SQLite databases."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class ChromeHistoryAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "chrome_history"

    @property
    def entity_types(self) -> list[str]:
        return ["page_visit"]

    def __init__(self, path: str = "", include_internal_urls: bool = False) -> None:
        self.path = path
        self.include_internal_urls = include_internal_urls

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
        units: dict[str, KnowledgeUnit] = {}
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
                units.setdefault(unit.source_id, unit)

        result.units.extend(sorted(units.values(), key=lambda unit: (unit.updated_at, unit.source_id)))
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
                if child.is_file() and child.name in {"History", "History.db", "History.sqlite"}
            ),
            key=lambda child: str(child.relative_to(root)),
        )

    def _read_rows(self, path: Path) -> list[sqlite3.Row]:
        uri = f"file:{quote(str(path), safe='/')}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            if not self._table_exists(conn, "urls"):
                return []

            url_columns = self._columns(conn, "urls")
            visits_columns = self._columns(conn, "visits") if self._table_exists(conn, "visits") else set()
            last_visit_expr = "u.last_visit_time" if "last_visit_time" in url_columns else "NULL"
            hidden_expr = "u.hidden" if "hidden" in url_columns else "0"
            visit_count_expr = "u.visit_count" if "visit_count" in url_columns else "NULL"
            typed_count_expr = "u.typed_count" if "typed_count" in url_columns else "NULL"

            if visits_columns:
                transition_expr = "latest.transition" if "transition" in visits_columns else "NULL"
                visit_time_expr = "latest.visit_time" if "visit_time" in visits_columns else "NULL"
                visit_duration_expr = "latest.visit_duration" if "visit_duration" in visits_columns else "NULL"
                from_visit_expr = "latest.from_visit" if "from_visit" in visits_columns else "NULL"
                visit_join = """
                    LEFT JOIN (
                        SELECT v.*
                        FROM visits v
                        INNER JOIN (
                            SELECT url, MAX(visit_time) AS visit_time
                            FROM visits
                            GROUP BY url
                        ) mv ON mv.url = v.url AND mv.visit_time = v.visit_time
                    ) latest ON latest.url = u.id
                """
            else:
                transition_expr = "NULL"
                visit_time_expr = "NULL"
                visit_duration_expr = "NULL"
                from_visit_expr = "NULL"
                visit_join = ""

            query = f"""
                SELECT
                    u.id,
                    u.url,
                    u.title,
                    {visit_count_expr} AS visit_count,
                    {typed_count_expr} AS typed_count,
                    {last_visit_expr} AS last_visit_time,
                    {hidden_expr} AS hidden,
                    {visit_time_expr} AS visit_time,
                    {transition_expr} AS transition,
                    {visit_duration_expr} AS visit_duration,
                    {from_visit_expr} AS from_visit
                FROM urls u
                {visit_join}
                WHERE u.url IS NOT NULL
            """
            return list(conn.execute(query))

    def _unit_from_row(self, row: sqlite3.Row, source_file: str) -> KnowledgeUnit | None:
        url = str(row["url"] or "").strip()
        normalized_url = self._normalize_url(url)
        if not normalized_url:
            return None
        if not self.include_internal_urls and self._is_internal_url(normalized_url):
            return None

        last_visit_at = self._chrome_datetime(row["last_visit_time"] or row["visit_time"])
        if last_visit_at is None:
            return None

        title = str(row["title"] or "").strip() or self._title_from_url(normalized_url)
        domain = urlsplit(normalized_url).hostname or ""
        transition = self._transition_metadata(row["transition"])
        metadata = {
            "url": url,
            "normalized_url": normalized_url,
            "domain": domain,
            "title": title,
            "visit_count": self._parse_int(row["visit_count"]),
            "typed_count": self._parse_int(row["typed_count"]),
            "last_visit_time": self._parse_int(row["last_visit_time"]),
            "last_visit_at": last_visit_at.isoformat(),
            "hidden": bool(row["hidden"]),
            "visit_time": self._parse_int(row["visit_time"]),
            "visit_duration": self._parse_int(row["visit_duration"]),
            "from_visit": self._parse_int(row["from_visit"]),
            "transition": transition,
            "source_file": source_file,
            "url_id": self._parse_int(row["id"]),
        }
        return KnowledgeUnit(
            source_project=SourceProject.CHROME_HISTORY,
            source_id=self._source_id(normalized_url),
            source_entity_type="page_visit",
            title=title,
            content=self._content(title, normalized_url, metadata),
            content_type=ContentType.METADATA,
            metadata=metadata,
            tags=["chrome", "browser_history"],
            created_at=last_visit_at,
            updated_at=last_visit_at,
        )

    def _normalize_url(self, value: str) -> str:
        text = value.strip()
        if not text:
            return ""
        parsed = urlsplit(text)
        if not parsed.scheme and not parsed.netloc:
            parsed = urlsplit(f"https://{text}")

        scheme = (parsed.scheme or "https").lower()
        if scheme in {"chrome", "about"}:
            path = parsed.netloc + parsed.path
            return f"{scheme}:{path}"
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            return ""
        port = parsed.port
        netloc = hostname
        if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
            netloc = f"{hostname}:{port}"
        path = parsed.path or "/"
        return urlunsplit((scheme, netloc, path, parsed.query, ""))

    def _is_internal_url(self, normalized_url: str) -> bool:
        return normalized_url.startswith(("chrome:", "about:"))

    def _title_from_url(self, normalized_url: str) -> str:
        parsed = urlsplit(normalized_url)
        if parsed.hostname:
            suffix = parsed.path.strip("/")
            if suffix:
                return f"{parsed.hostname}/{suffix}"
            return parsed.hostname
        return normalized_url

    def _source_id(self, normalized_url: str) -> str:
        digest = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:24]
        return f"chrome_history:{digest}"

    def _content(self, title: str, normalized_url: str, metadata: dict[str, Any]) -> str:
        parts = [title, f"URL: {normalized_url}", f"Last visited: {metadata['last_visit_at']}"]
        if metadata["visit_count"] is not None:
            parts.append(f"Visit count: {metadata['visit_count']}")
        if metadata["transition"]["type"]:
            parts.append(f"Transition: {metadata['transition']['type']}")
        return "\n".join(parts)

    def _transition_metadata(self, value: Any) -> dict[str, Any]:
        transition = self._parse_int(value)
        if transition is None:
            return {"raw": None, "core": None, "type": "", "qualifiers": []}

        core = transition & 0xFF
        transition_type = {
            0: "link",
            1: "typed",
            2: "auto_bookmark",
            3: "auto_subframe",
            4: "manual_subframe",
            5: "generated",
            6: "auto_toplevel",
            7: "form_submit",
            8: "reload",
            9: "keyword",
            10: "keyword_generated",
        }.get(core, "unknown")
        qualifiers = [
            name
            for bit, name in (
                (0x01000000, "forward_back"),
                (0x02000000, "from_address_bar"),
                (0x10000000, "chain_start"),
                (0x20000000, "chain_end"),
                (0x40000000, "client_redirect"),
                (0x80000000, "server_redirect"),
            )
            if transition & bit
        ]
        return {"raw": transition, "core": core, "type": transition_type, "qualifiers": qualifiers}

    def _chrome_datetime(self, value: Any) -> datetime | None:
        micros = self._parse_int(value)
        if micros is None or micros <= 0:
            return None
        try:
            return datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=micros)
        except OverflowError:
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
