"""Adapter for simple SQLite query logs."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class SqliteQueryLogAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "sqlite_query_log"

    @property
    def entity_types(self) -> list[str]:
        return ["query_log"]

    def __init__(
        self,
        db_path: str = "",
        *,
        table: str = "queries",
        query_column: str = "query",
        created_column: str = "created_at",
        result_count_column: str | None = None,
    ) -> None:
        self.db_path = db_path
        self.table = table
        self.query_column = query_column
        self.created_column = created_column
        self.result_count_column = result_count_column

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "query_log" not in entity_types:
            return result
        if not self.db_path or not Path(self.db_path).expanduser().exists():
            return result

        sync_at = self._sync_datetime(since) if since else None
        with sqlite3.connect(str(Path(self.db_path).expanduser())) as conn:
            conn.row_factory = sqlite3.Row
            for row in self._read_rows(conn):
                query = self._text(row["query"])
                if not query:
                    continue

                created_at = self._parse_datetime(row["created_at"])
                if sync_at and created_at and created_at <= sync_at:
                    continue

                result.units.append(self._unit(row, query, created_at))

        return result

    def _read_rows(self, conn: sqlite3.Connection) -> list[sqlite3.Row]:
        columns = [
            f"rowid AS {self._quote_identifier('_rowid')}",
            f"{self._quote_identifier(self.query_column)} AS {self._quote_identifier('query')}",
            f"{self._quote_identifier(self.created_column)} AS {self._quote_identifier('created_at')}",
        ]
        if self.result_count_column:
            columns.append(
                f"{self._quote_identifier(self.result_count_column)} AS {self._quote_identifier('result_count')}"
            )

        return conn.execute(
            f"SELECT {', '.join(columns)} FROM {self._quote_identifier(self.table)}"
        ).fetchall()

    def _unit(
        self,
        row: sqlite3.Row,
        query: str,
        created_at: datetime | None,
    ) -> KnowledgeUnit:
        now = datetime.now(timezone.utc)
        metadata: dict[str, Any] = {
            "table": self.table,
            "query_column": self.query_column,
            "created_column": self.created_column,
            "rowid": row["_rowid"],
            "raw_created_at": row["created_at"],
        }
        if self.result_count_column:
            metadata["result_count_column"] = self.result_count_column
            metadata["result_count"] = row["result_count"]

        return KnowledgeUnit(
            source_project=SourceProject.SQLITE_QUERY_LOG,
            source_id=self._source_id(row["_rowid"], query),
            source_entity_type="query_log",
            title=query,
            content=query,
            content_type=ContentType.METADATA,
            metadata=metadata,
            created_at=created_at or now,
            updated_at=created_at or now,
        )

    def _source_id(self, rowid: Any, query: str) -> str:
        digest = hashlib.sha256(f"{rowid}\0{query}".encode()).hexdigest()[:12]
        return f"sqlite_query_log:{self.table}:{rowid}:{digest}"

    def _text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _quote_identifier(self, value: str) -> str:
        if not value:
            raise ValueError("SQLite identifier must not be empty")
        return '"' + value.replace('"', '""') + '"'

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return self._ensure_utc(value)
        if isinstance(value, int | float):
            return self._from_unix_timestamp(float(value))

        text = str(value).strip()
        if not text:
            return None
        try:
            return self._from_unix_timestamp(float(text))
        except ValueError:
            pass

        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return self._ensure_utc(parsed)

    def _from_unix_timestamp(self, value: float) -> datetime | None:
        magnitude = abs(value)
        if magnitude >= 1_000_000_000_000_000_000:
            value /= 1_000_000_000
        elif magnitude >= 1_000_000_000_000_000:
            value /= 1_000_000
        elif magnitude >= 1_000_000_000_000:
            value /= 1_000
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        parsed = self._parse_datetime(value)
        if parsed is None:
            raise ValueError(f"Invalid sync timestamp: {value}")
        return parsed

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
