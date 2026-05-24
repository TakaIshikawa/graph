"""Read-only adapter for macOS Messages chat.db SQLite exports."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from graph.adapters._personal_exports import clean_metadata, digest_source_id
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState

APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)


class ImessageChatSqliteAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "imessage_chat_sqlite"

    @property
    def entity_types(self) -> list[str]:
        return ["message"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if entity_types is not None and "message" not in entity_types:
            return result
        db_path = self._db_path()
        if db_path is None:
            return result
        sync_at = since.last_sync_at.astimezone(timezone.utc) if since else None
        try:
            rows = self._read_rows(db_path)
        except sqlite3.Error:
            return result
        for row in rows:
            unit = self._unit_from_row(row, db_path.name)
            if unit and (sync_at is None or unit.updated_at > sync_at):
                result.units.append(unit)
        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        return result

    def _db_path(self) -> Path | None:
        if not self.path:
            return None
        path = Path(self.path).expanduser()
        if path.is_dir():
            path = path / "chat.db"
        return path if path.is_file() else None

    def _read_rows(self, path: Path) -> list[dict[str, Any]]:
        uri = f"file:{quote(str(path), safe='/')}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            if not self._table_exists(conn, "message"):
                return []
            attachment_expr = "''"
            if self._table_exists(conn, "message_attachment_join") and self._table_exists(conn, "attachment"):
                attachment_expr = "(SELECT group_concat(a.filename, ';') FROM message_attachment_join maj JOIN attachment a ON a.ROWID = maj.attachment_id WHERE maj.message_id = m.ROWID)"
            chat_expr = "''"
            if self._table_exists(conn, "chat_message_join") and self._table_exists(conn, "chat"):
                chat_expr = "(SELECT group_concat(c.chat_identifier, ';') FROM chat_message_join cmj JOIN chat c ON c.ROWID = cmj.chat_id WHERE cmj.message_id = m.ROWID)"
            has_handle = self._table_exists(conn, "handle") and "handle_id" in self._columns(conn, "message")
            handle_join = "LEFT JOIN handle h ON h.ROWID = m.handle_id" if has_handle else ""
            handle_expr = "h.id" if has_handle else "NULL"
            cols = self._columns(conn, "message")
            def col(name: str) -> str:
                return f"m.{name}" if name in cols else f"NULL AS {name}"
            query = f"""
                SELECT m.ROWID AS rowid, {col('guid')}, {col('text')}, {col('date')}, {col('date_read')},
                       {col('is_from_me')}, {col('is_read')}, {col('service')},
                       {handle_expr} AS handle, {chat_expr} AS chat_ids, {attachment_expr} AS attachments
                FROM message m
                {handle_join}
                ORDER BY m.ROWID
            """
            return [dict(row) for row in conn.execute(query)]

    def _unit_from_row(self, row: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        text = str(row.get("text") or "")
        attachments = [item for item in str(row.get("attachments") or "").split(";") if item]
        if not text and not attachments:
            return None
        sent_at = self._apple_time(row.get("date"))
        read_at = self._apple_time(row.get("date_read"))
        now = datetime.now(timezone.utc)
        metadata = clean_metadata(
            {
                "guid": row.get("guid"),
                "handle": row.get("handle"),
                "chat_ids": [item for item in str(row.get("chat_ids") or "").split(";") if item],
                "text": text,
                "service": row.get("service"),
                "is_from_me": bool(row.get("is_from_me")),
                "is_read": bool(row.get("is_read")),
                "sent_at": sent_at.isoformat() if sent_at else "",
                "read_at": read_at.isoformat() if read_at else "",
                "attachments": attachments,
                "source_file": source_file,
            }
        )
        return KnowledgeUnit(
            source_project="imessage_chat_sqlite",
            source_id=f"imessage_chat_sqlite:message:{row.get('guid')}" if row.get("guid") else digest_source_id("imessage_chat_sqlite", row.get("rowid")),
            source_entity_type="message",
            title=self._title(text, attachments),
            content=self._content(text, attachments, row.get("handle"), row.get("service")),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=["imessage", str(row.get("service") or "").casefold()],
            created_at=sent_at or now,
            updated_at=read_at or sent_at or now,
        )

    def _apple_time(self, value: Any) -> datetime | None:
        if value in (None, ""):
            return None
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        if abs(number) > 10_000_000_000:
            number = number // 1_000_000_000
        return APPLE_EPOCH + timedelta(seconds=number)

    def _table_exists(self, conn: sqlite3.Connection, name: str) -> bool:
        return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None

    def _columns(self, conn: sqlite3.Connection, table: str) -> set[str]:
        return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")}

    def _title(self, text: str, attachments: list[str]) -> str:
        if text:
            return text[:80]
        return f"Message attachment: {attachments[0]}"

    def _content(self, text: str, attachments: list[str], handle: Any, service: Any) -> str:
        parts = [text] if text else []
        if handle:
            parts.append(f"Handle: {handle}")
        if service:
            parts.append(f"Service: {service}")
        if attachments:
            parts.append(f"Attachments: {', '.join(attachments)}")
        return "\n".join(parts)
