"""Adapter for Calibre metadata.db libraries."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import quote

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class CalibreSqliteAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "calibre_sqlite"

    @property
    def entity_types(self) -> list[str]:
        return ["book"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "book" not in entity_types:
            return result

        db_path = self._db_path()
        if db_path is None:
            return result

        sync_at = self._sync_datetime(since) if since else None
        try:
            rows = self._read_rows(db_path)
        except sqlite3.Error:
            return result

        units: list[KnowledgeUnit] = []
        for row in rows:
            unit = self._unit_from_row(row, db_path.parent)
            if unit is None:
                continue
            comparable_at = unit.updated_at or unit.created_at
            if sync_at and comparable_at <= sync_at:
                continue
            units.append(unit)

        result.units.extend(sorted(units, key=lambda unit: (unit.updated_at, unit.source_id)))
        return result

    def _db_path(self) -> Path | None:
        if not self.path:
            return None
        path = Path(self.path).expanduser()
        if path.is_dir():
            path = path / "metadata.db"
        if path.is_file():
            return path
        return None

    def _read_rows(self, path: Path) -> list[dict[str, Any]]:
        uri = f"file:{quote(str(path), safe='/')}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            if not self._table_exists(conn, "books"):
                return []
            books = [dict(row) for row in conn.execute(f"SELECT {self._book_select(conn)} FROM books ORDER BY id")]
            book_ids = [self._int_or_none(book.get("id")) for book in books]
            valid_book_ids = [book_id for book_id in book_ids if book_id is not None]
            authors = self._linked_values(conn, "authors", "books_authors_link", "author", "name", valid_book_ids)
            tags = self._linked_values(conn, "tags", "books_tags_link", "tag", "name", valid_book_ids)
            publishers = self._linked_values(
                conn,
                "publishers",
                "books_publishers_link",
                "publisher",
                "name",
                valid_book_ids,
            )
            ratings = self._linked_values(conn, "ratings", "books_ratings_link", "rating", "rating", valid_book_ids)
            identifiers = self._identifiers(conn, valid_book_ids)
            comments = self._comments(conn, valid_book_ids)
            formats = self._formats(conn, valid_book_ids)

        for book in books:
            book_id = self._int_or_none(book.get("id"))
            if book_id is None:
                continue
            book["authors"] = authors.get(book_id, [])
            book["tags"] = tags.get(book_id, [])
            book["publishers"] = publishers.get(book_id, [])
            book["ratings"] = ratings.get(book_id, [])
            book["identifiers"] = identifiers.get(book_id, {})
            book["comments"] = comments.get(book_id, "")
            book["formats"] = formats.get(book_id, [])
        return books

    def _book_select(self, conn: sqlite3.Connection) -> str:
        columns = self._columns(conn, "books")
        expressions = ["id", "title"]
        for column in ("timestamp", "last_modified", "pubdate", "path", "uuid", "isbn", "author_sort", "sort"):
            expressions.append(f"{column} AS {column}" if column in columns else f"NULL AS {column}")
        return ", ".join(expressions)

    def _unit_from_row(self, row: dict[str, Any], library_root: Path) -> KnowledgeUnit | None:
        book_id = self._int_or_none(row.get("id"))
        title = self._clean(row.get("title"))
        if book_id is None or not title:
            return None

        authors = [self._clean(author) for author in row.get("authors", []) if self._clean(author)]
        tags = [self._clean(tag) for tag in row.get("tags", []) if self._clean(tag)]
        identifiers = dict(row.get("identifiers", {}))
        isbn = self._clean(row.get("isbn"))
        if isbn and "isbn" not in identifiers:
            identifiers["isbn"] = isbn
        publisher = self._first_text(row.get("publishers", []))
        rating = self._rating(row.get("ratings", []))
        added_at = self._parse_datetime(row.get("timestamp"))
        updated_at = self._parse_datetime(row.get("last_modified")) or added_at
        publication_date = self._parse_datetime(row.get("pubdate"))
        now = datetime.now(timezone.utc)
        created_at = added_at or updated_at or now
        modified_at = updated_at or created_at
        library_path = self._library_path(library_root, row.get("path"))
        comments = self._comment_text(row.get("comments"))
        formats = [self._clean(fmt).upper() for fmt in row.get("formats", []) if self._clean(fmt)]

        metadata = {
            "book_id": book_id,
            "title": title,
            "authors": authors,
            "author_sort": self._clean(row.get("author_sort")),
            "sort": self._clean(row.get("sort")),
            "tags": tags,
            "formats": formats,
            "identifiers": identifiers,
            "publisher": publisher,
            "rating": rating,
            "publication_date": publication_date.isoformat() if publication_date else None,
            "added_at": added_at.isoformat() if added_at else None,
            "updated_at": updated_at.isoformat() if updated_at else None,
            "library_path": library_path,
            "relative_path": self._clean(row.get("path")),
            "uuid": self._clean(row.get("uuid")),
            "comments": comments,
        }
        return KnowledgeUnit(
            source_project=SourceProject.CALIBRE_SQLITE,
            source_id=f"calibre_sqlite:{book_id}",
            source_entity_type="book",
            title=self._format_title(title, authors),
            content=self._content(title, authors, tags, formats, identifiers, publisher, rating, comments),
            content_type=ContentType.ARTIFACT,
            metadata=metadata,
            tags=tags,
            created_at=created_at,
            updated_at=modified_at,
        )

    def _linked_values(
        self,
        conn: sqlite3.Connection,
        value_table: str,
        link_table: str,
        link_column: str,
        value_column: str,
        book_ids: list[int],
    ) -> dict[int, list[Any]]:
        if not book_ids or not self._table_exists(conn, value_table) or not self._table_exists(conn, link_table):
            return {}
        value_columns = self._columns(conn, value_table)
        link_columns = self._columns(conn, link_table)
        if value_column not in value_columns or "book" not in link_columns or link_column not in link_columns:
            return {}
        order_expr = "l.id" if "id" in link_columns else f"v.{value_column}"
        placeholders = ", ".join("?" for _ in book_ids)
        query = f"""
            SELECT l.book, v.{value_column} AS value
            FROM {link_table} l
            JOIN {value_table} v ON v.id = l.{link_column}
            WHERE l.book IN ({placeholders})
            ORDER BY l.book, {order_expr}
        """
        values: dict[int, list[Any]] = {}
        for row in conn.execute(query, book_ids):
            values.setdefault(int(row["book"]), []).append(row["value"])
        return values

    def _identifiers(self, conn: sqlite3.Connection, book_ids: list[int]) -> dict[int, dict[str, str]]:
        if not book_ids or not self._table_exists(conn, "identifiers"):
            return {}
        columns = self._columns(conn, "identifiers")
        if not {"book", "type", "val"}.issubset(columns):
            return {}
        placeholders = ", ".join("?" for _ in book_ids)
        query = f"""
            SELECT book, type, val
            FROM identifiers
            WHERE book IN ({placeholders})
            ORDER BY book, type
        """
        identifiers: dict[int, dict[str, str]] = {}
        for row in conn.execute(query, book_ids):
            key = self._clean(row["type"]).lower()
            value = self._clean(row["val"])
            if key and value:
                identifiers.setdefault(int(row["book"]), {})[key] = value
        return identifiers

    def _comments(self, conn: sqlite3.Connection, book_ids: list[int]) -> dict[int, str]:
        if not book_ids or not self._table_exists(conn, "comments"):
            return {}
        columns = self._columns(conn, "comments")
        if not {"book", "text"}.issubset(columns):
            return {}
        placeholders = ", ".join("?" for _ in book_ids)
        query = f"SELECT book, text FROM comments WHERE book IN ({placeholders})"
        return {int(row["book"]): self._comment_text(row["text"]) for row in conn.execute(query, book_ids)}

    def _formats(self, conn: sqlite3.Connection, book_ids: list[int]) -> dict[int, list[str]]:
        if not book_ids or not self._table_exists(conn, "data"):
            return {}
        columns = self._columns(conn, "data")
        if not {"book", "format"}.issubset(columns):
            return {}
        placeholders = ", ".join("?" for _ in book_ids)
        query = f"SELECT book, format FROM data WHERE book IN ({placeholders}) ORDER BY book, format"
        formats: dict[int, list[str]] = {}
        for row in conn.execute(query, book_ids):
            value = self._clean(row["format"]).upper()
            if value:
                formats.setdefault(int(row["book"]), []).append(value)
        return formats

    def _content(
        self,
        title: str,
        authors: list[str],
        tags: list[str],
        formats: list[str],
        identifiers: dict[str, str],
        publisher: str,
        rating: int | None,
        comments: str,
    ) -> str:
        parts = [f"Title: {title}"]
        if authors:
            parts.append(f"Authors: {', '.join(authors)}")
        if publisher:
            parts.append(f"Publisher: {publisher}")
        if rating is not None:
            parts.append(f"Rating: {rating}/10")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        if formats:
            parts.append(f"Formats: {', '.join(formats)}")
        if identifiers:
            parts.append("Identifiers: " + ", ".join(f"{key}:{value}" for key, value in sorted(identifiers.items())))
        if comments:
            parts.append(f"\nComments:\n{comments}")
        return "\n".join(parts)

    def _library_path(self, library_root: Path, relative_path: Any) -> str:
        relative = self._clean(relative_path)
        if not relative:
            return str(library_root)
        return str(library_root / relative)

    def _rating(self, values: list[Any]) -> int | None:
        for value in values:
            rating = self._int_or_none(value)
            if rating is not None:
                return rating
        return None

    def _format_title(self, title: str, authors: list[str]) -> str:
        if authors:
            return f"{title} by {', '.join(authors)}"
        return title

    def _first_text(self, values: list[Any]) -> str:
        for value in values:
            text = self._clean(value)
            if text:
                return text
        return ""

    def _comment_text(self, value: Any) -> str:
        text = unescape(self._clean(value))
        text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", text)
        text = re.sub(r"(?i)</\s*p\s*>", "\n", text)
        text = re.sub(r"<[^>]+>", "", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def _clean(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _parse_datetime(self, value: Any) -> datetime | None:
        text = self._clean(value)
        if not text:
            return None
        if text.startswith("0101-01-01"):
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _int_or_none(self, value: Any) -> int | None:
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
