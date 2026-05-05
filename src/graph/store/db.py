"""SQLite store for the knowledge graph."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
import csv
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import date
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit, urlunsplit

from graph.store.migrations import SCHEMA_VERSION, ensure_schema
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState

SAVED_QUERIES_SCHEMA_VERSION = 2
COLLECTIONS_SCHEMA_VERSION = 1
SUPPORTED_QUERY_SCHEDULES = {"daily", "weekly", "monthly"}
COLLECTION_ACTIVITY_BUCKETS = {"day", "week", "month", "year"}
COLLECTION_ACTIVITY_FIELDS = {"created_at", "ingested_at", "updated_at"}
UNIT_ACTIVITY_BUCKETS = {"day", "week", "month"}
UNIT_ACTIVITY_FIELDS = {"created_at", "ingested_at", "updated_at"}
REQUIRED_SQLITE_BACKUP_OBJECTS = {
    "schema_version",
    "knowledge_units",
    "edges",
    "knowledge_fts",
}
_MAX_METADATA_INVENTORY_EXAMPLES = 3
_MAX_METADATA_INVENTORY_EXAMPLE_LENGTH = 80
_MAX_TAG_USAGE_EXAMPLES = 3
_DUPLICATE_EXTERNAL_URL_KEYS = frozenset(
    {
        "url",
        "source_url",
        "canonical_url",
        "external_url",
        "link",
        "permalink",
        "uri",
        "normalized_url",
    }
)
_CONTENT_URL_RE = re.compile(r"https?://[^\s<>\[\]{}\"']+", re.IGNORECASE)

if TYPE_CHECKING:
    from graph.adapters.base import IngestResult


class MetadataPathError(ValueError):
    """Raised when a dotted metadata path cannot be applied safely."""


class DatabaseBackupError(ValueError):
    """Raised when a SQLite backup or restore cannot be completed safely."""


class _EmbeddingStatus(dict):
    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict) and "percent_fresh" not in other:
            return {key: value for key, value in self.items() if key != "percent_fresh"} == other
        return super().__eq__(other)


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_datetime(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _datetime_filter_sql(
    field: str,
    *,
    after: datetime | str | None = None,
    before: datetime | str | None = None,
) -> tuple[list[str], list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    try:
        parsed_after = _parse_datetime(after)
    except ValueError as exc:
        raise ValueError(f"{field}_after must be an ISO-8601 date or datetime.") from exc
    try:
        parsed_before = _parse_datetime(before)
    except ValueError as exc:
        raise ValueError(f"{field}_before must be an ISO-8601 date or datetime.") from exc
    if parsed_after and parsed_before and parsed_after > parsed_before:
        raise ValueError(f"{field}_after must be on or before {field}_before.")
    if parsed_after:
        clauses.append(f"{field}_at >= ?")
        params.append(parsed_after.isoformat())
    if parsed_before:
        clauses.append(f"{field}_at <= ?")
        params.append(parsed_before.isoformat())
    return clauses, params


def _json_value(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _excerpt(text: str, query: str, *, length: int = 160) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""

    terms = [
        term.lower()
        for term in re.findall(r"[\w-]+", query)
        if term.upper() not in {"AND", "OR", "NOT", "NEAR"}
    ]
    lower_text = text.lower()
    positions = [lower_text.find(term) for term in terms if lower_text.find(term) >= 0]
    if positions:
        start = max(min(positions) - length // 3, 0)
    else:
        start = 0
    snippet = text[start : start + length].strip()
    if start > 0:
        snippet = "..." + snippet
    if start + length < len(text):
        snippet += "..."
    return snippet


def _fallback_search_terms(query: str) -> list[str]:
    terms = [
        term
        for term in re.findall(r"[\w-]+", query)
        if term.upper() not in {"AND", "OR", "NOT", "NEAR"}
    ]
    return terms or [query]


def _requires_exact_single_term_filter(query: str) -> bool:
    stripped = query.strip()
    return bool(stripped) and not re.search(r"\s", stripped) and bool(re.search(r"[-_/]", stripped))


def _row_to_unit(row: sqlite3.Row) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=row["id"],
        source_project=row["source_project"],
        source_id=row["source_id"],
        source_entity_type=row["source_entity_type"],
        title=row["title"],
        content=row["content"],
        content_type=row["content_type"],
        metadata=json.loads(row["metadata"]),
        tags=json.loads(row["tags"]),
        confidence=row["confidence"],
        utility_score=row["utility_score"],
        created_at=row["created_at"],
        ingested_at=row["ingested_at"],
        updated_at=row["updated_at"],
    )


def _row_to_edge(row: sqlite3.Row) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=row["id"],
        from_unit_id=row["from_unit_id"],
        to_unit_id=row["to_unit_id"],
        relation=row["relation"],
        weight=row["weight"],
        source=row["source"],
        metadata=json.loads(row["metadata"]),
        created_at=row["created_at"],
    )


def _row_to_saved_query(row: sqlite3.Row) -> dict:
    return {
        "name": row["name"],
        "query": row["query"],
        "mode": row["mode"],
        "limit": row["limit"],
        "filters": json.loads(row["filters"]),
        "schedule": row["schedule"],
        "last_run_at": row["last_run_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _row_to_saved_query_run(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "saved_query_name": row["saved_query_name"],
        "run_at": row["run_at"],
        "effective_limit": row["effective_limit"],
        "mode": row["mode"],
        "filters": json.loads(row["filters"]),
        "result_count": row["result_count"],
        "top_result_ids": json.loads(row["top_result_ids"]),
    }


def _row_to_collection(row: sqlite3.Row) -> dict:
    data = {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "metadata": json.loads(row["metadata"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    if "unit_count" in row.keys():
        data["unit_count"] = row["unit_count"]
    return data


def _normalize_query_schedule(schedule: str | None) -> str | None:
    if schedule is None:
        return None
    normalized = str(schedule).strip().lower()
    if not normalized:
        return None
    if normalized not in SUPPORTED_QUERY_SCHEDULES:
        valid = ", ".join(sorted(SUPPORTED_QUERY_SCHEDULES))
        raise ValueError(f"Unknown saved query schedule: {schedule}. Use one of: {valid}.")
    return normalized


def _metadata_path_parts(path: str) -> list[str]:
    parts = path.split(".")
    if not path or any(part == "" for part in parts):
        raise MetadataPathError("Metadata path must be a non-empty dotted path.")
    for part in parts:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", part):
            raise MetadataPathError(
                "Metadata path parts may only contain letters, numbers, underscores, and hyphens."
            )
    return parts


def _format_metadata_path(parts: list[str]) -> str:
    return ".".join(parts)


def _metadata_json_path(path: str) -> str:
    return "$" + "".join(f'."{part}"' for part in _metadata_path_parts(path))


def _metadata_inventory_path_part(value: Any) -> str:
    return str(value).replace(".", "\\.")


def _flatten_metadata_inventory(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        if not prefix:
            items: list[tuple[str, Any]] = []
        elif not value:
            return [(prefix, value)]
        else:
            items = []
        for raw_key, child in sorted(value.items(), key=lambda item: str(item[0])):
            key = _metadata_inventory_path_part(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            items.extend(_flatten_metadata_inventory(child, path))
        return items

    if isinstance(value, list | tuple):
        if not value:
            return [(prefix, value)] if prefix else []
        items = []
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            items.extend(_flatten_metadata_inventory(child, path))
        return items

    return [(prefix, value)] if prefix else []


def _metadata_inventory_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"
    if isinstance(value, Enum):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list | tuple | set):
        return "array"
    return type(value).__name__


def _metadata_inventory_normalized_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _metadata_inventory_normalized_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list | tuple):
        return [_metadata_inventory_normalized_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_metadata_inventory_normalized_value(item) for item in value), key=str)
    return value


def _metadata_inventory_example_value(value: Any) -> str:
    normalized = _metadata_inventory_normalized_value(value)
    if isinstance(normalized, str):
        text = normalized
    elif normalized is None or isinstance(normalized, int | float | bool):
        text = str(normalized).lower() if isinstance(normalized, bool) else str(normalized)
    else:
        text = json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > _MAX_METADATA_INVENTORY_EXAMPLE_LENGTH:
        text = f"{text[: _MAX_METADATA_INVENTORY_EXAMPLE_LENGTH - 1].rstrip()}..."
    return text


def _metadata_inventory_counter_values(counter: Counter[str]) -> list[str]:
    return [
        key
        for key, _count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _metadata_inventory_distinct_key(value: Any) -> tuple[str, str]:
    normalized = _metadata_inventory_normalized_value(value)
    value_type = _metadata_inventory_value_type(value)
    return (
        value_type,
        json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str),
    )


def _inline_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_external_url(value: Any) -> str | None:
    text = _inline_text(value)
    if not text:
        return None
    text = text.rstrip(".,;:")
    parsed = urlsplit(text)
    if not parsed.scheme and not parsed.netloc:
        parsed = urlsplit(f"https://{text}")
    if parsed.scheme.casefold() not in {"http", "https"}:
        return None

    scheme = parsed.scheme.casefold() or "https"
    hostname = (parsed.hostname or "").casefold()
    if not hostname:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    netloc = hostname
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{hostname}:{port}"
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def _iter_metadata_external_url_values(value: Any, key: str | None = None):
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            yield from _iter_metadata_external_url_values(child_value, str(child_key))
        return
    if isinstance(value, list | tuple | set):
        for child in value:
            yield from _iter_metadata_external_url_values(child, key)
        return
    if key is not None and key.casefold() in _DUPLICATE_EXTERNAL_URL_KEYS:
        yield value


def _extract_content_external_urls(content: str) -> set[str]:
    urls: set[str] = set()
    for match in _CONTENT_URL_RE.finditer(content or ""):
        text = match.group(0).rstrip(".,;:!?")
        while text.endswith(")") and text.count("(") < text.count(")"):
            text = text[:-1]
        normalized = _normalize_external_url(text)
        if normalized is not None:
            urls.add(normalized)
    return urls


def _sorted_counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _activity_bucket_label(value: datetime, bucket: str) -> str:
    if bucket == "day":
        return value.date().isoformat()
    if bucket == "week":
        week_start = value.date() - timedelta(days=value.weekday())
        return week_start.isoformat()
    if bucket == "month":
        return f"{value.year:04d}-{value.month:02d}"
    if bucket == "year":
        return f"{value.year:04d}"
    raise ValueError(f"Unsupported collection activity bucket: {bucket}")


def _activity_bucket_start(value: datetime, bucket: str) -> date:
    value_date = value.date()
    if bucket == "day":
        return value_date
    if bucket == "week":
        return value_date - timedelta(days=value_date.weekday())
    if bucket == "month":
        return date(value_date.year, value_date.month, 1)
    raise ValueError(f"Unsupported activity bucket: {bucket}")


def _activity_bucket_label_from_date(value: date, bucket: str) -> str:
    if bucket in {"day", "week"}:
        return value.isoformat()
    if bucket == "month":
        return f"{value.year:04d}-{value.month:02d}"
    raise ValueError(f"Unsupported activity bucket: {bucket}")


def _next_activity_bucket(value: date, bucket: str) -> date:
    if bucket == "day":
        return value + timedelta(days=1)
    if bucket == "week":
        return value + timedelta(days=7)
    if bucket == "month":
        if value.month == 12:
            return date(value.year + 1, 1, 1)
        return date(value.year, value.month + 1, 1)
    raise ValueError(f"Unsupported activity bucket: {bucket}")


def _activity_empty_bucket_labels(
    start: datetime,
    end: datetime,
    bucket: str,
) -> list[str]:
    current = _activity_bucket_start(start, bucket)
    final = _activity_bucket_start(end, bucket)
    labels: list[str] = []
    while current <= final:
        labels.append(_activity_bucket_label_from_date(current, bucket))
        current = _next_activity_bucket(current, bucket)
    return labels


def metadata_path_value(metadata: dict, path: str):
    current = metadata
    for part in _metadata_path_parts(path):
        if not isinstance(current, dict):
            return None
        if part not in current:
            return None
        current = current[part]
    return current


def _metadata_path_lookup(metadata: dict, path: str) -> tuple[bool, Any]:
    current: Any = metadata
    for part in _metadata_path_parts(path):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _metadata_value_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping | list):
        return bool(value)
    return True


def metadata_path_matches(metadata: dict, path: str, value: object) -> bool:
    return metadata_path_value(metadata, path) == value


def _metadata_filter_sql(
    field: str,
    *,
    metadata_key: str | None = None,
    metadata_value: object | None = None,
) -> tuple[str, list[object]]:
    has_key = metadata_key is not None
    has_value = metadata_value is not None
    if has_key != has_value:
        raise ValueError("metadata_key and metadata_value must be supplied together.")
    if not has_key:
        return "", []
    return f" AND json_extract({field}, ?) = ?", [_metadata_json_path(str(metadata_key)), metadata_value]


class Store:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        ensure_schema(self.conn)

    def close(self) -> None:
        self.conn.close()

    def backup_database(self, destination: str | Path, *, force: bool = False) -> dict:
        destination_path = Path(destination)
        if destination_path.exists() and not force:
            raise DatabaseBackupError(f"Backup destination already exists: {destination_path}")
        if self.db_path.resolve() == destination_path.resolve():
            raise DatabaseBackupError("Backup destination must be different from the active database.")

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists():
            destination_path.unlink()

        self.conn.commit()
        target = sqlite3.connect(str(destination_path))
        try:
            self.conn.backup(target)
            target.commit()
        finally:
            target.close()

        return {
            "source_path": str(self.db_path),
            "destination_path": str(destination_path),
            "copied_file_size": destination_path.stat().st_size,
        }

    @classmethod
    def validate_database_backup(cls, source: str | Path) -> None:
        source_path = Path(source)
        if not source_path.exists() or not source_path.is_file():
            raise DatabaseBackupError(f"Restore source does not exist: {source_path}")

        try:
            conn = sqlite3.connect(f"{source_path.resolve().as_uri()}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise DatabaseBackupError(
                f"Restore source is not a readable SQLite database: {source_path}"
            ) from exc

        try:
            quick_check = conn.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or quick_check[0] != "ok":
                raise DatabaseBackupError("Restore source failed SQLite integrity checks.")

            objects = {
                row[0]
                for row in conn.execute(
                    """
                    SELECT name
                    FROM sqlite_schema
                    WHERE type IN ('table', 'view')
                    """
                ).fetchall()
            }
            missing = sorted(REQUIRED_SQLITE_BACKUP_OBJECTS - objects)
            if missing:
                raise DatabaseBackupError(
                    "Restore source is missing required graph tables: " + ", ".join(missing)
                )

            version_row = conn.execute(
                "SELECT version FROM schema_version ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            if version_row is None or version_row[0] != SCHEMA_VERSION:
                found = None if version_row is None else version_row[0]
                raise DatabaseBackupError(
                    f"Restore source schema version {found!r} is not supported; "
                    f"expected {SCHEMA_VERSION}."
                )
        except sqlite3.DatabaseError as exc:
            raise DatabaseBackupError(
                f"Restore source is not a valid SQLite database: {source_path}"
            ) from exc
        finally:
            conn.close()

    def restore_database(self, source: str | Path, *, force: bool = False) -> dict:
        source_path = Path(source)
        if self.db_path.exists() and not force:
            raise DatabaseBackupError(
                f"Refusing to overwrite active database without --force: {self.db_path}"
            )
        if source_path.resolve() == self.db_path.resolve():
            raise DatabaseBackupError("Restore source must be different from the active database.")

        self.validate_database_backup(source_path)
        self.close()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.db_path.with_name(f".{self.db_path.name}.restore-{uuid.uuid4().hex}.tmp")
        source_conn = sqlite3.connect(f"{source_path.resolve().as_uri()}?mode=ro", uri=True)
        target_conn = sqlite3.connect(str(temp_path))
        try:
            source_conn.backup(target_conn)
            target_conn.commit()
        finally:
            target_conn.close()
            source_conn.close()

        for candidate in (
            self.db_path,
            self.db_path.with_name(self.db_path.name + "-wal"),
            self.db_path.with_name(self.db_path.name + "-shm"),
        ):
            candidate.unlink(missing_ok=True)
        temp_path.replace(self.db_path)

        return {
            "source_path": str(source_path),
            "destination_path": str(self.db_path),
            "copied_file_size": self.db_path.stat().st_size,
        }

    # --- Unit CRUD ---

    def insert_unit(self, unit: KnowledgeUnit) -> KnowledgeUnit:
        if not unit.id:
            unit.id = _new_id()
        now = _utcnow_iso()
        self.conn.execute(
            """INSERT INTO knowledge_units
               (id, source_project, source_id, source_entity_type,
                title, content, content_type, metadata, tags,
                confidence, utility_score, embedding, embedding_updated_at,
                created_at, ingested_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(source_project, source_id, source_entity_type)
               DO UPDATE SET
                   title = excluded.title,
                   content = excluded.content,
                   metadata = excluded.metadata,
                   tags = excluded.tags,
                   confidence = excluded.confidence,
                   utility_score = excluded.utility_score,
                   updated_at = excluded.updated_at
            """,
            (
                unit.id,
                unit.source_project,
                unit.source_id,
                unit.source_entity_type,
                unit.title,
                unit.content,
                unit.content_type,
                json.dumps(unit.metadata),
                json.dumps(unit.tags),
                unit.confidence,
                unit.utility_score,
                None,
                None,
                unit.created_at.isoformat()
                if isinstance(unit.created_at, datetime)
                else str(unit.created_at),
                now,
                unit.updated_at.isoformat()
                if isinstance(unit.updated_at, datetime)
                else str(unit.updated_at),
            ),
        )
        self.conn.commit()
        return unit

    def get_unit(self, unit_id: str) -> KnowledgeUnit | None:
        row = self.conn.execute("SELECT * FROM knowledge_units WHERE id = ?", (unit_id,)).fetchone()
        return _row_to_unit(row) if row else None

    def get_unit_by_source(
        self, source_project: str, source_id: str, source_entity_type: str
    ) -> KnowledgeUnit | None:
        row = self.conn.execute(
            """SELECT * FROM knowledge_units
               WHERE source_project = ? AND source_id = ? AND source_entity_type = ?""",
            (source_project, source_id, source_entity_type),
        ).fetchone()
        return _row_to_unit(row) if row else None

    def get_all_units(self, *, limit: int = 10000) -> list[KnowledgeUnit]:
        rows = self.conn.execute(
            "SELECT * FROM knowledge_units ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_unit(r) for r in rows]

    def get_units_by_metadata_range(
        self,
        field_name: str,
        min_value: object,
        max_value: object,
    ) -> list[KnowledgeUnit]:
        """Query units where a metadata field value falls within a specified range.

        Supports numeric (int, float) and ISO date string comparisons.

        Args:
            field_name: Dotted path to the metadata field (e.g., "score" or "metrics.rating")
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)

        Returns:
            List of units where metadata[field_name] is between min_value and max_value

        Raises:
            ValueError: If min_value > max_value
        """
        if not isinstance(field_name, str) or not field_name:
            raise ValueError("field_name must be a non-empty string")

        # Validate that min_value <= max_value
        try:
            if min_value > max_value:
                raise ValueError("min_value must be less than or equal to max_value")
        except TypeError as exc:
            raise ValueError(
                f"min_value and max_value must be comparable types: {type(min_value).__name__} "
                f"and {type(max_value).__name__}"
            ) from exc

        # Build JSON path
        json_path = _metadata_json_path(field_name)

        # Query units where the metadata field is within range
        query = """
            SELECT * FROM knowledge_units
            WHERE json_extract(metadata, ?) >= ?
              AND json_extract(metadata, ?) <= ?
            ORDER BY created_at DESC
        """
        rows = self.conn.execute(
            query,
            (json_path, min_value, json_path, max_value),
        ).fetchall()
        return [_row_to_unit(r) for r in rows]

    def metadata_key_inventory(
        self,
        prefix: str | None = None,
        *,
        limit: int | None = None,
    ) -> list[dict]:
        """Return deterministic usage rows for flattened unit metadata paths."""
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
        ):
            raise ValueError("limit must be a positive integer.")

        normalized_prefix = str(prefix).strip() if prefix is not None else None
        if normalized_prefix == "":
            normalized_prefix = None

        counts: Counter[str] = Counter()
        value_types: dict[str, Counter[str]] = defaultdict(Counter)
        source_projects: dict[str, Counter[str]] = defaultdict(Counter)
        example_values: dict[str, set[str]] = defaultdict(set)

        rows = self.conn.execute(
            """SELECT id, source_project, source_id, title, metadata
               FROM knowledge_units
               ORDER BY source_project, source_id, title, id"""
        ).fetchall()
        for row in rows:
            metadata = json.loads(row["metadata"])
            if not isinstance(metadata, Mapping):
                continue
            for path, value in _flatten_metadata_inventory(metadata):
                if normalized_prefix is not None and not path.startswith(normalized_prefix):
                    continue
                counts[path] += 1
                value_types[path][_metadata_inventory_value_type(value)] += 1
                source_projects[path][str(row["source_project"])] += 1
                if len(example_values[path]) < _MAX_METADATA_INVENTORY_EXAMPLES:
                    example_values[path].add(_metadata_inventory_example_value(value))

        inventory_rows = [
            {
                "path": path,
                "count": counts[path],
                "value_types": _metadata_inventory_counter_values(value_types[path]),
                "example_values": sorted(example_values[path])[
                    :_MAX_METADATA_INVENTORY_EXAMPLES
                ],
                "source_projects": _metadata_inventory_counter_values(source_projects[path]),
            }
            for path in sorted(counts, key=lambda item: (-counts[item], item))
        ]
        return inventory_rows[:limit] if limit is not None else inventory_rows

    def metadata_key_profile(
        self,
        prefix: str | None = None,
        *,
        limit: int | None = None,
        sample_size: int = _MAX_METADATA_INVENTORY_EXAMPLES,
    ) -> list[dict]:
        """Profile flattened unit metadata paths by count, values, and simple types."""
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
        ):
            raise ValueError("limit must be a positive integer.")
        if (
            not isinstance(sample_size, int)
            or isinstance(sample_size, bool)
            or sample_size <= 0
        ):
            raise ValueError("sample_size must be a positive integer.")

        normalized_prefix = str(prefix).strip() if prefix is not None else None
        if normalized_prefix == "":
            normalized_prefix = None

        counts: Counter[str] = Counter()
        value_types: dict[str, Counter[str]] = defaultdict(Counter)
        distinct_values: dict[str, dict[tuple[str, str], Any]] = defaultdict(dict)

        rows = self.conn.execute(
            """SELECT metadata
               FROM knowledge_units
               ORDER BY source_project, source_id, title, id"""
        ).fetchall()
        for row in rows:
            metadata = json.loads(row["metadata"])
            if not isinstance(metadata, Mapping):
                continue
            for path, value in _flatten_metadata_inventory(metadata):
                if normalized_prefix is not None and not path.startswith(normalized_prefix):
                    continue
                counts[path] += 1
                value_types[path][_metadata_inventory_value_type(value)] += 1
                distinct_key = _metadata_inventory_distinct_key(value)
                distinct_values[path].setdefault(
                    distinct_key,
                    _metadata_inventory_normalized_value(value),
                )

        profile_rows = []
        for path in sorted(counts, key=lambda item: (-counts[item], item)):
            samples = [
                value
                for _sort_key, value in sorted(
                    distinct_values[path].items(),
                    key=lambda item: item[0],
                )[:sample_size]
            ]
            profile_rows.append(
                {
                    "key": path,
                    "occurrence_count": counts[path],
                    "distinct_value_count": len(distinct_values[path]),
                    "value_types": _metadata_inventory_counter_values(value_types[path]),
                    "sample_values": samples,
                }
            )

        return profile_rows[:limit] if limit is not None else profile_rows

    def metadata_value_histogram(
        self,
        path: str,
        *,
        source_project: str | None = None,
        limit: int | None = None,
    ) -> dict:
        """Count scalar metadata values at a dotted path across knowledge units."""
        normalized_path = _format_metadata_path(_metadata_path_parts(path))
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
        ):
            raise ValueError("limit must be a positive integer.")

        where_parts, params = self._unit_filter_parts(source_project=source_project)
        query = "SELECT source_project, metadata FROM knowledge_units"
        if where_parts:
            query += " WHERE " + " AND ".join(where_parts)
        query += " ORDER BY source_project, id"

        counts: Counter[tuple[str, str, Any]] = Counter()
        missing_count = 0
        unit_count = 0
        value_count = 0

        for row in self.conn.execute(query, params).fetchall():
            unit_count += 1
            metadata = json.loads(row["metadata"])
            found, raw_value = _metadata_path_lookup(metadata, normalized_path)
            if not found:
                missing_count += 1
                continue

            values = raw_value if isinstance(raw_value, list) else [raw_value]
            for value in values:
                if isinstance(value, Mapping) or isinstance(value, list):
                    continue
                value_type = _metadata_inventory_value_type(value)
                counts[(value_type, json.dumps(value, sort_keys=True), value)] += 1
                value_count += 1

        sorted_items = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
        if limit is not None:
            sorted_items = sorted_items[:limit]

        return {
            "path": normalized_path,
            "source_project": source_project,
            "unit_count": unit_count,
            "missing_count": missing_count,
            "value_count": value_count,
            "values": [
                {
                    "value": value,
                    "value_type": value_type,
                    "count": count,
                }
                for (value_type, _sort_value, value), count in sorted_items
            ],
        }

    def metadata_completeness_summary(
        self,
        required_keys: list[str],
        *,
        source_project: str | None = None,
        source_entity_type: str | None = None,
        content_type: str | None = None,
        limit: int | None = None,
    ) -> dict:
        """Summarize required metadata path presence across knowledge units."""
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit < 0
        ):
            raise ValueError("limit must be a non-negative integer or None")
        normalized_keys = sorted(
            {_format_metadata_path(_metadata_path_parts(str(key))) for key in required_keys}
        )
        present_counts: dict[str, int] = {key: 0 for key in normalized_keys}
        missing_unit_ids: dict[str, list[str]] = {key: [] for key in normalized_keys}

        where_parts, params = self._unit_filter_parts(
            source_project=source_project,
            source_entity_type=source_entity_type,
            content_type=content_type,
        )
        query = "SELECT id, source_project, source_id, title, metadata FROM knowledge_units"
        if where_parts:
            query += " WHERE " + " AND ".join(where_parts)
        query += " ORDER BY source_project, source_id, title, id"

        rows = self.conn.execute(query, params).fetchall()
        total_units = len(rows)

        for row in rows:
            metadata = json.loads(row["metadata"])
            for key in normalized_keys:
                found, value = _metadata_path_lookup(metadata, key)
                if found and _metadata_value_is_present(value):
                    present_counts[key] += 1
                else:
                    if limit is None or len(missing_unit_ids[key]) < limit:
                        missing_unit_ids[key].append(row["id"])

        missing_counts = {
            key: total_units - present_counts[key] for key in normalized_keys
        }
        keys = [
            {
                "key": key,
                "present_count": present_counts[key],
                "missing_count": missing_counts[key],
                "missing_unit_ids": missing_unit_ids[key],
            }
            for key in normalized_keys
        ]

        return {
            "total_units": total_units,
            "required_keys": normalized_keys,
            "source_project": source_project,
            "source_entity_type": source_entity_type,
            "content_type": content_type,
            "keys": keys,
            "present_counts": present_counts,
            "missing_counts": missing_counts,
            "missing_unit_ids": missing_unit_ids,
        }

    def metadata_key_usage_frequency(self) -> dict[str, int]:
        """Return frequency counts of top-level metadata keys across all units."""
        key_counts: Counter[str] = Counter()

        rows = self.conn.execute(
            """SELECT metadata
               FROM knowledge_units
               ORDER BY id"""
        ).fetchall()

        for row in rows:
            metadata = json.loads(row["metadata"])
            if not isinstance(metadata, Mapping):
                continue
            for key in metadata.keys():
                key_counts[key] += 1

        return dict(sorted(key_counts.items(), key=lambda item: (-item[1], item[0])))

    def source_freshness_histogram(
        self,
        *,
        fresh_days: object = 30,
        stale_days: object = 90,
    ) -> list[dict]:
        """Bucket unit updated_at freshness counts by source project."""
        if (
            not isinstance(fresh_days, int)
            or isinstance(fresh_days, bool)
            or not isinstance(stale_days, int)
            or isinstance(stale_days, bool)
        ):
            raise ValueError("fresh_days and stale_days must be non-negative integers.")
        if fresh_days < 0 or stale_days < 0:
            raise ValueError("fresh_days and stale_days must be non-negative integers.")
        if fresh_days > stale_days:
            raise ValueError("fresh_days must be less than or equal to stale_days.")

        now = datetime.now(timezone.utc)
        buckets: dict[str, Counter[str]] = defaultdict(Counter)
        rows = self.conn.execute(
            """SELECT source_project, updated_at
               FROM knowledge_units
               ORDER BY source_project, id"""
        ).fetchall()

        for row in rows:
            source_project = str(row["source_project"])
            buckets[source_project]["total"] += 1
            raw_updated_at = row["updated_at"]
            if raw_updated_at is None or str(raw_updated_at).strip() == "":
                buckets[source_project]["unknown"] += 1
                continue
            try:
                updated_at = _parse_datetime(raw_updated_at)
            except ValueError:
                buckets[source_project]["unknown"] += 1
                continue
            if updated_at is None:
                buckets[source_project]["unknown"] += 1
                continue

            age_days = (now - updated_at).total_seconds() / 86400
            if age_days <= fresh_days:
                buckets[source_project]["fresh"] += 1
            elif age_days <= stale_days:
                buckets[source_project]["aging"] += 1
            else:
                buckets[source_project]["stale"] += 1

        return [
            {
                "source_project": source_project,
                "total": buckets[source_project]["total"],
                "fresh": buckets[source_project]["fresh"],
                "aging": buckets[source_project]["aging"],
                "stale": buckets[source_project]["stale"],
                "unknown": buckets[source_project]["unknown"],
            }
            for source_project in sorted(buckets)
        ]

    def find_source_id_collisions(
        self,
        *,
        source_project: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Find duplicate non-empty source ids grouped by source project."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit must be a non-negative integer.")

        where_parts = ["TRIM(source_id) != ''"]
        params: list[object] = []
        if source_project:
            where_parts.append("source_project = ?")
            params.append(source_project)
        where = " AND ".join(where_parts)

        rows = self.conn.execute(
            f"""WITH duplicate_groups AS (
                    SELECT source_project, source_id, COUNT(*) AS duplicate_count
                    FROM knowledge_units
                    WHERE {where}
                    GROUP BY source_project, source_id
                    HAVING COUNT(*) > 1
                )
                SELECT
                    duplicate_groups.source_project,
                    duplicate_groups.source_id,
                    duplicate_groups.duplicate_count,
                    (
                        SELECT json_group_array(id)
                        FROM (
                            SELECT id
                            FROM knowledge_units
                            WHERE source_project = duplicate_groups.source_project
                              AND source_id = duplicate_groups.source_id
                            ORDER BY id
                        )
                    ) AS unit_ids,
                    (
                        SELECT json_group_array(title)
                        FROM (
                            SELECT title
                            FROM knowledge_units
                            WHERE source_project = duplicate_groups.source_project
                              AND source_id = duplicate_groups.source_id
                            ORDER BY id
                        )
                    ) AS titles
                FROM duplicate_groups
                ORDER BY
                    duplicate_groups.duplicate_count DESC,
                    duplicate_groups.source_project,
                    duplicate_groups.source_id
                LIMIT ?""",
            [*params, limit],
        ).fetchall()
        return [
            {
                "source_project": row["source_project"],
                "source_id": row["source_id"],
                "count": row["duplicate_count"],
                "unit_ids": json.loads(row["unit_ids"]),
                "titles": json.loads(row["titles"]),
            }
            for row in rows
        ]

    def find_duplicate_external_urls(self, *, limit: int = 50) -> list[dict]:
        """Find normalized external URLs referenced by two or more units."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit must be a non-negative integer.")

        rows = self.conn.execute(
            """SELECT id, source_project, source_id, source_entity_type, title, metadata, content
               FROM knowledge_units
               ORDER BY source_project, source_id, title, id"""
        ).fetchall()
        units_by_url: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)

        for row in rows:
            urls = _extract_content_external_urls(row["content"])
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                metadata = {}
            if isinstance(metadata, Mapping):
                for value in _iter_metadata_external_url_values(metadata):
                    normalized = _normalize_external_url(value)
                    if normalized is not None:
                        urls.add(normalized)

            if not urls:
                continue

            unit_record = {
                "id": row["id"],
                "source_project": row["source_project"],
                "source_id": row["source_id"],
                "source_entity_type": row["source_entity_type"],
                "title": row["title"],
            }
            for url in urls:
                units_by_url[url][row["id"]] = unit_record

        duplicate_rows = []
        for url in sorted(units_by_url):
            units = sorted(
                units_by_url[url].values(),
                key=lambda unit: (
                    unit["source_project"],
                    unit["source_id"],
                    unit["title"],
                    unit["id"],
                ),
            )
            if len(units) < 2:
                continue
            duplicate_rows.append({"url": url, "count": len(units), "units": units})

        duplicate_rows.sort(key=lambda row: (-row["count"], row["url"]))
        return duplicate_rows[:limit]

    def unit_activity_summary(
        self,
        *,
        field: str = "created_at",
        bucket: str = "month",
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        include_empty: bool = False,
    ) -> dict:
        """Bucket knowledge units by a timestamp field with per-bucket breakdowns."""
        field = str(field).strip().lower()
        bucket = str(bucket).strip().lower()
        if field not in UNIT_ACTIVITY_FIELDS:
            valid = ", ".join(sorted(UNIT_ACTIVITY_FIELDS))
            raise ValueError(f"Unsupported unit activity field: {field}. Use one of: {valid}.")
        if bucket not in UNIT_ACTIVITY_BUCKETS:
            valid = ", ".join(sorted(UNIT_ACTIVITY_BUCKETS))
            raise ValueError(f"Unsupported unit activity bucket: {bucket}. Use one of: {valid}.")
        if not isinstance(include_empty, bool):
            raise ValueError("include_empty must be a boolean.")

        try:
            parsed_start = _parse_datetime(start)
        except ValueError as exc:
            raise ValueError("start must be an ISO-8601 date or datetime.") from exc
        try:
            parsed_end = _parse_datetime(end)
        except ValueError as exc:
            raise ValueError("end must be an ISO-8601 date or datetime.") from exc
        if parsed_start is not None and parsed_end is not None and parsed_start > parsed_end:
            raise ValueError("start must be on or before end.")

        rows = self.conn.execute(
            f"""SELECT id, source_project, content_type, {field} AS activity_at
                FROM knowledge_units
                ORDER BY {field} ASC, id ASC"""
        ).fetchall()

        bucket_counts: Counter[str] = Counter()
        source_project_counts: dict[str, Counter[str]] = defaultdict(Counter)
        content_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
        seen_values: list[datetime] = []

        for row in rows:
            activity_at = _parse_datetime(row["activity_at"])
            if activity_at is None:
                continue
            if parsed_start is not None and activity_at < parsed_start:
                continue
            if parsed_end is not None and activity_at > parsed_end:
                continue

            label = _activity_bucket_label(activity_at, bucket)
            seen_values.append(activity_at)
            bucket_counts[label] += 1
            source_project_counts[label][str(row["source_project"])] += 1
            content_type_counts[label][str(row["content_type"])] += 1

        labels = set(bucket_counts)
        if include_empty and parsed_start is not None and parsed_end is not None:
            labels.update(_activity_empty_bucket_labels(parsed_start, parsed_end, bucket))

        bucket_rows = [
            {
                "bucket": label,
                "count": bucket_counts[label],
                "source_project_counts": _sorted_counter_dict(source_project_counts[label]),
                "content_type_counts": _sorted_counter_dict(content_type_counts[label]),
            }
            for label in sorted(labels)
        ]

        return {
            "field": field,
            "bucket": bucket,
            "start": parsed_start.isoformat() if parsed_start is not None else None,
            "end": parsed_end.isoformat() if parsed_end is not None else None,
            "include_empty": include_empty,
            "buckets": bucket_rows,
            "first_seen_at": min(seen_values).isoformat() if seen_values else None,
            "last_seen_at": max(seen_values).isoformat() if seen_values else None,
        }

    def tag_vocabulary(self, *, exclude_unit_id: str | None = None) -> dict[str, int]:
        """Return existing graph tags and counts, optionally excluding one unit."""
        query = """SELECT json_each.value AS tag, COUNT(*) AS count
                   FROM knowledge_units, json_each(knowledge_units.tags)"""
        params: list[object] = []
        where_parts = ["TRIM(json_each.value) != ''"]
        if exclude_unit_id is not None:
            where_parts.append("knowledge_units.id != ?")
            params.append(exclude_unit_id)
        query += " WHERE " + " AND ".join(where_parts)
        query += " GROUP BY json_each.value ORDER BY tag"

        rows = self.conn.execute(query, params).fetchall()
        return {str(row["tag"]): row["count"] for row in rows}

    def tag_usage_summary(
        self,
        *,
        limit: int = 50,
        include_examples: bool = True,
    ) -> dict:
        """Return deterministic tag assignment counts and top tag usage rows."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit must be a non-negative integer.")

        total_row = self.conn.execute(
            """SELECT
                   COUNT(DISTINCT json_each.value) AS total_distinct_tags,
                   COUNT(*) AS total_tag_assignments
               FROM knowledge_units, json_each(knowledge_units.tags)
               WHERE json_each.type = 'text'
                 AND TRIM(json_each.value) != ''"""
        ).fetchone()
        total_distinct_tags = total_row["total_distinct_tags"] or 0
        total_tag_assignments = total_row["total_tag_assignments"] or 0

        rows = self.conn.execute(
            """SELECT json_each.value AS tag, COUNT(*) AS count
               FROM knowledge_units, json_each(knowledge_units.tags)
               WHERE json_each.type = 'text'
                 AND TRIM(json_each.value) != ''
               GROUP BY json_each.value
               ORDER BY count DESC, tag
               LIMIT ?""",
            (limit,),
        ).fetchall()

        tags = []
        for row in rows:
            count = row["count"]
            tag = str(row["tag"])
            payload = {
                "tag": tag,
                "count": count,
                "percentage": round((count / total_tag_assignments) * 100, 2)
                if total_tag_assignments
                else 0.0,
            }
            if include_examples:
                example_rows = self.conn.execute(
                    """SELECT id, title, source_project, source_id, source_entity_type
                       FROM knowledge_units
                       WHERE EXISTS (
                           SELECT 1
                           FROM json_each(knowledge_units.tags)
                           WHERE json_each.value = ?
                       )
                       ORDER BY title COLLATE NOCASE, title, source_project, source_id, id
                       LIMIT ?""",
                    (tag, _MAX_TAG_USAGE_EXAMPLES),
                ).fetchall()
                payload["examples"] = [
                    {
                        "id": example["id"],
                        "title": example["title"],
                        "source": {
                            "project": example["source_project"],
                            "id": example["source_id"],
                            "entity_type": example["source_entity_type"],
                        },
                    }
                    for example in example_rows
                ]
            tags.append(payload)

        return {
            "total_distinct_tags": total_distinct_tags,
            "total_tag_assignments": total_tag_assignments,
            "tags": tags,
        }

    def add_unit_alias(
        self,
        unit_id: str,
        alias: str,
        *,
        source: str | None = None,
    ) -> dict:
        alias = alias.strip()
        if not alias:
            raise ValueError("alias must not be empty.")

        unit = self.get_unit(unit_id)
        if unit is None:
            return {
                "unit_id": unit_id,
                "alias": alias,
                "added": False,
                "error": "unit_not_found",
                "message": f"Unit not found: {unit_id}",
            }

        existing = self.conn.execute(
            """SELECT unit_id, alias, source, created_at
               FROM unit_aliases
               WHERE unit_id = ? AND alias = ?""",
            (unit_id, alias),
        ).fetchone()
        if existing is not None:
            return {
                "unit_id": unit_id,
                "alias": alias,
                "source": existing["source"],
                "created_at": existing["created_at"],
                "added": False,
            }

        now = _utcnow_iso()
        self.conn.execute(
            """INSERT INTO unit_aliases (unit_id, alias, source, created_at)
               VALUES (?, ?, ?, ?)""",
            (unit_id, alias, source, now),
        )
        self.conn.commit()
        self.fts_index_unit(unit)
        return {
            "unit_id": unit_id,
            "alias": alias,
            "source": source,
            "created_at": now,
            "added": True,
        }

    def remove_unit_alias(self, unit_id: str, alias: str) -> dict:
        alias = alias.strip()
        if not alias:
            raise ValueError("alias must not be empty.")

        unit = self.get_unit(unit_id)
        if unit is None:
            return {
                "unit_id": unit_id,
                "alias": alias,
                "removed": False,
                "error": "unit_not_found",
                "message": f"Unit not found: {unit_id}",
            }

        cursor = self.conn.execute(
            "DELETE FROM unit_aliases WHERE unit_id = ? AND alias = ?",
            (unit_id, alias),
        )
        self.conn.commit()
        self.fts_index_unit(unit)
        return {"unit_id": unit_id, "alias": alias, "removed": cursor.rowcount > 0}

    def list_unit_aliases(self, unit_id: str) -> dict:
        unit = self.get_unit(unit_id)
        if unit is None:
            return {
                "unit_id": unit_id,
                "aliases": [],
                "count": 0,
                "error": "unit_not_found",
                "message": f"Unit not found: {unit_id}",
            }

        rows = self.conn.execute(
            """SELECT unit_id, alias, source, created_at
               FROM unit_aliases
               WHERE unit_id = ?
               ORDER BY lower(alias), alias""",
            (unit_id,),
        ).fetchall()
        aliases = [
            {
                "unit_id": row["unit_id"],
                "alias": row["alias"],
                "source": row["source"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        return {"unit_id": unit_id, "aliases": aliases, "count": len(aliases)}

    def _unit_alias_texts(self, unit_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT alias FROM unit_aliases WHERE unit_id = ? ORDER BY lower(alias), alias",
            (unit_id,),
        ).fetchall()
        return [str(row["alias"]) for row in rows]

    def get_units(
        self,
        *,
        source_project: str | None = None,
        content_type: str | None = None,
        limit: int | None = None,
    ) -> list[KnowledgeUnit]:
        where_parts, params = self._unit_filter_parts(
            source_project=source_project,
            content_type=content_type,
        )
        query = "SELECT * FROM knowledge_units"
        if where_parts:
            query += " WHERE " + " AND ".join(where_parts)
        query += " ORDER BY created_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, limit))
        rows = self.conn.execute(query, params).fetchall()
        return [_row_to_unit(r) for r in rows]

    def get_units_by_source_project(
        self,
        source_project: SourceProject,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[KnowledgeUnit]:
        """Query units filtered by SourceProject enum value.

        Args:
            source_project: SourceProject enum value to filter by
            limit: Maximum number of units to return (optional)
            offset: Number of units to skip (optional)

        Returns:
            List of units matching the source project, ordered by created_at DESC
        """
        if not isinstance(source_project, SourceProject):
            raise TypeError(
                f"source_project must be a SourceProject enum, got {type(source_project).__name__}"
            )

        query = "SELECT * FROM knowledge_units WHERE source_project = ? ORDER BY created_at DESC"
        params: list[object] = [source_project.value]

        # Validate parameters
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 0):
            raise ValueError("limit must be a non-negative integer")

        if offset is not None and (not isinstance(offset, int) or isinstance(offset, bool) or offset < 0):
            raise ValueError("offset must be a non-negative integer")

        # SQL requires LIMIT before OFFSET
        # If offset is provided without limit, we need to use a large default limit
        if offset is not None and limit is None:
            query += " LIMIT -1 OFFSET ?"
            params.append(offset)
        elif limit is not None and offset is None:
            query += " LIMIT ?"
            params.append(limit)
        elif limit is not None and offset is not None:
            query += " LIMIT ? OFFSET ?"
            params.append(limit)
            params.append(offset)

        rows = self.conn.execute(query, params).fetchall()
        return [_row_to_unit(r) for r in rows]

    def get_units_with_embeddings(
        self,
        *,
        source_project: str | None = None,
        content_type: str | None = None,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        updated_after: datetime | str | None = None,
        updated_before: datetime | str | None = None,
        metadata_key: str | None = None,
        metadata_value: object | None = None,
    ) -> list[tuple[KnowledgeUnit, bytes]]:
        query = "SELECT * FROM knowledge_units WHERE embedding IS NOT NULL"
        params: list = []
        if source_project:
            query += " AND source_project = ?"
            params.append(source_project)
        if content_type:
            query += " AND content_type = ?"
            params.append(content_type)
        for clause in (
            _datetime_filter_sql("created", after=created_after, before=created_before),
            _datetime_filter_sql("updated", after=updated_after, before=updated_before),
        ):
            for where_part in clause[0]:
                query += f" AND {where_part}"
            params.extend(clause[1])
        metadata_sql, metadata_params = _metadata_filter_sql(
            "metadata",
            metadata_key=metadata_key,
            metadata_value=metadata_value,
        )
        query += metadata_sql
        params.extend(metadata_params)
        rows = self.conn.execute(query, params).fetchall()
        return [(_row_to_unit(r), r["embedding"]) for r in rows]

    def update_embedding(self, unit_id: str, embedding: bytes) -> None:
        self.conn.execute(
            "UPDATE knowledge_units SET embedding = ?, embedding_updated_at = ? WHERE id = ?",
            (embedding, _utcnow_iso(), unit_id),
        )
        self.conn.commit()

    def get_embedding_status(
        self,
        *,
        source_project: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, int | float]:
        where, params = self._unit_filter_sql(
            source_project=source_project,
            content_type=content_type,
        )
        row = self.conn.execute(
            f"""SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN embedding IS NULL THEN 1 ELSE 0 END) AS missing,
                    SUM(CASE
                        WHEN embedding IS NOT NULL
                         AND embedding_updated_at IS NOT NULL
                         AND embedding_updated_at >= updated_at
                        THEN 1 ELSE 0 END) AS fresh,
                    SUM(CASE
                        WHEN embedding IS NOT NULL
                         AND (embedding_updated_at IS NULL OR updated_at > embedding_updated_at)
                        THEN 1 ELSE 0 END) AS stale
                FROM knowledge_units
                {where}""",
            params,
        ).fetchone()
        total = row["total"] or 0
        fresh = row["fresh"] or 0
        return _EmbeddingStatus({
            "total": total,
            "missing": row["missing"] or 0,
            "fresh": fresh,
            "stale": row["stale"] or 0,
            "percent_fresh": round((fresh / total) * 100, 2) if total else 0.0,
        })

    def get_embedding_status_groups(
        self,
        group_by: str,
        *,
        source_project: str | None = None,
        content_type: str | None = None,
    ) -> list[dict[str, int | float | str]]:
        if group_by not in {"source_project", "content_type"}:
            raise ValueError("group_by must be source_project or content_type")
        where, params = self._unit_filter_sql(
            source_project=source_project,
            content_type=content_type,
        )
        rows = self.conn.execute(
            f"""SELECT
                    {group_by} AS group_value,
                    COUNT(*) AS total,
                    SUM(CASE WHEN embedding IS NULL THEN 1 ELSE 0 END) AS missing,
                    SUM(CASE
                        WHEN embedding IS NOT NULL
                         AND embedding_updated_at IS NOT NULL
                         AND embedding_updated_at >= updated_at
                        THEN 1 ELSE 0 END) AS fresh,
                    SUM(CASE
                        WHEN embedding IS NOT NULL
                         AND (embedding_updated_at IS NULL OR updated_at > embedding_updated_at)
                        THEN 1 ELSE 0 END) AS stale
                FROM knowledge_units
                {where}
                GROUP BY {group_by}
                ORDER BY {group_by}""",
            params,
        ).fetchall()
        return [self._embedding_status_group_dict(row, group_by) for row in rows]

    def get_embedding_status_matrix(
        self,
        *,
        source_project: str | None = None,
        content_type: str | None = None,
    ) -> list[dict[str, int | float | str]]:
        where, params = self._unit_filter_sql(
            source_project=source_project,
            content_type=content_type,
        )
        rows = self.conn.execute(
            f"""SELECT
                    source_project,
                    content_type,
                    COUNT(*) AS total,
                    SUM(CASE WHEN embedding IS NULL THEN 1 ELSE 0 END) AS missing,
                    SUM(CASE
                        WHEN embedding IS NOT NULL
                         AND embedding_updated_at IS NOT NULL
                         AND embedding_updated_at >= updated_at
                        THEN 1 ELSE 0 END) AS fresh,
                    SUM(CASE
                        WHEN embedding IS NOT NULL
                         AND (embedding_updated_at IS NULL OR updated_at > embedding_updated_at)
                        THEN 1 ELSE 0 END) AS stale
                FROM knowledge_units
                {where}
                GROUP BY source_project, content_type
                ORDER BY source_project, content_type""",
            params,
        ).fetchall()
        return [self._embedding_status_group_dict(row) for row in rows]

    def get_embedding_refresh_status(
        self,
        *,
        source_project: str | None = None,
        content_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, str | None]]:
        where_parts, params = self._unit_filter_parts(
            source_project=source_project,
            content_type=content_type,
        )
        where_parts.append(
            """(embedding IS NULL
                OR embedding_updated_at IS NULL
                OR updated_at > embedding_updated_at)"""
        )
        query = f"""SELECT
                       id,
                       title,
                       source_project,
                       content_type,
                       updated_at,
                       embedding_updated_at,
                       CASE
                           WHEN embedding IS NULL THEN 'missing_embedding'
                           WHEN embedding_updated_at IS NULL THEN 'missing_embedding_timestamp'
                           ELSE 'stale_embedding'
                       END AS reason
                   FROM knowledge_units
                   WHERE {" AND ".join(where_parts)}
                   ORDER BY updated_at DESC, created_at DESC
                   LIMIT ?"""
        params.append(max(0, limit))
        rows = self.conn.execute(query, params).fetchall()
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "source_project": row["source_project"],
                "content_type": row["content_type"],
                "reason": row["reason"],
                "updated_at": row["updated_at"],
                "embedding_updated_at": row["embedding_updated_at"],
            }
            for row in rows
        ]

    def _embedding_status_group_dict(
        self, row: sqlite3.Row, group_by: str | None = None
    ) -> dict[str, int | float | str]:
        total = row["total"] or 0
        fresh = row["fresh"] or 0
        payload: dict[str, int | float | str] = {}
        if group_by is not None:
            payload[group_by] = row["group_value"]
        else:
            payload["source_project"] = row["source_project"]
            payload["content_type"] = row["content_type"]
        payload.update(
            {
                "total": total,
                "missing": row["missing"] or 0,
                "fresh": fresh,
                "stale": row["stale"] or 0,
                "percent_fresh": round((fresh / total) * 100, 2) if total else 0.0,
            }
        )
        return payload

    def get_units_for_embedding_refresh(
        self,
        *,
        source_project: str | None = None,
        content_type: str | None = None,
        force: bool = False,
        stale_only: bool = False,
        limit: int | None = None,
    ) -> list[KnowledgeUnit]:
        where_parts, params = self._unit_filter_parts(
            source_project=source_project,
            content_type=content_type,
        )
        if force:
            pass
        elif stale_only:
            where_parts.append(
                """(embedding IS NULL
                    OR embedding_updated_at IS NULL
                    OR updated_at > embedding_updated_at)"""
            )
        else:
            where_parts.append("embedding IS NULL")

        query = "SELECT * FROM knowledge_units"
        if where_parts:
            query += " WHERE " + " AND ".join(where_parts)
        query += " ORDER BY created_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, limit))

        rows = self.conn.execute(query, params).fetchall()
        return [_row_to_unit(r) for r in rows]

    def _unit_filter_parts(
        self,
        *,
        source_project: str | None = None,
        source_entity_type: str | None = None,
        content_type: str | None = None,
    ) -> tuple[list[str], list]:
        where_parts: list[str] = []
        params: list = []
        if source_project:
            where_parts.append("source_project = ?")
            params.append(source_project)
        if source_entity_type:
            where_parts.append("source_entity_type = ?")
            params.append(source_entity_type)
        if content_type:
            where_parts.append("content_type = ?")
            params.append(content_type)
        return where_parts, params

    def _unit_filter_sql(
        self,
        *,
        source_project: str | None = None,
        source_entity_type: str | None = None,
        content_type: str | None = None,
    ) -> tuple[str, list]:
        where_parts, params = self._unit_filter_parts(
            source_project=source_project,
            source_entity_type=source_entity_type,
            content_type=content_type,
        )
        where = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        return where, params

    def update_unit_fields(
        self,
        unit_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        content_type: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> KnowledgeUnit | None:
        unit = self.get_unit(unit_id)
        if unit is None:
            return None

        if title is not None:
            unit.title = title
        if content is not None:
            unit.content = content
        if content_type is not None:
            unit.content_type = content_type
        if tags:
            for tag in tags:
                if tag not in unit.tags:
                    unit.tags.append(tag)
        if metadata:
            unit.metadata = {**unit.metadata, **metadata}

        now = _utcnow_iso()
        self.conn.execute(
            """UPDATE knowledge_units
               SET title = ?,
                   content = ?,
                   content_type = ?,
                   metadata = ?,
                   tags = ?,
                   updated_at = ?
               WHERE id = ?""",
            (
                unit.title,
                unit.content,
                unit.content_type,
                json.dumps(unit.metadata),
                json.dumps(unit.tags),
                now,
                unit.id,
            ),
        )
        self.conn.commit()
        updated = self.get_unit(unit_id)
        if updated is not None:
            self.fts_index_unit(updated)
        return updated

    def set_unit_metadata_path(
        self,
        unit_id: str,
        path: str,
        value: object,
    ) -> KnowledgeUnit | None:
        unit = self.get_unit(unit_id)
        if unit is None:
            return None

        parts = _metadata_path_parts(path)
        metadata = dict(unit.metadata)
        current = metadata
        for index, part in enumerate(parts[:-1]):
            next_value = current.get(part)
            if next_value is None:
                next_value = {}
                current[part] = next_value
            if not isinstance(next_value, dict):
                traversed = _format_metadata_path(parts[: index + 1])
                raise MetadataPathError(
                    f"Metadata path cannot traverse non-object value at '{traversed}'."
                )
            current = next_value
        current[parts[-1]] = value
        return self._replace_unit_metadata(unit_id, metadata)

    def remove_unit_metadata_path(self, unit_id: str, path: str) -> KnowledgeUnit | None:
        unit = self.get_unit(unit_id)
        if unit is None:
            return None

        parts = _metadata_path_parts(path)
        metadata = dict(unit.metadata)
        current = metadata
        for index, part in enumerate(parts[:-1]):
            next_value = current.get(part)
            if next_value is None:
                return unit
            if not isinstance(next_value, dict):
                traversed = _format_metadata_path(parts[: index + 1])
                raise MetadataPathError(
                    f"Metadata path cannot traverse non-object value at '{traversed}'."
                )
            current = next_value
        if parts[-1] not in current:
            return unit
        current.pop(parts[-1])
        return self._replace_unit_metadata(unit_id, metadata)

    def _replace_unit_metadata(self, unit_id: str, metadata: dict) -> KnowledgeUnit | None:
        now = _utcnow_iso()
        self.conn.execute(
            """UPDATE knowledge_units
               SET metadata = ?, updated_at = ?
               WHERE id = ?""",
            (json.dumps(metadata), now, unit_id),
        )
        self.conn.commit()
        updated = self.get_unit(unit_id)
        if updated is not None:
            self.fts_index_unit(updated)
        return updated

    def pin_unit(self, unit_id: str, *, reason: str | None = None) -> KnowledgeUnit | None:
        unit = self.get_unit(unit_id)
        if unit is None:
            return None

        metadata = dict(unit.metadata)
        now = _utcnow_iso()
        metadata["pinned"] = True
        metadata["pinned_at"] = now
        if reason is not None:
            metadata["pin_reason"] = reason
        else:
            metadata.pop("pin_reason", None)

        self.conn.execute(
            """UPDATE knowledge_units
               SET metadata = ?, updated_at = ?
               WHERE id = ?""",
            (json.dumps(metadata), now, unit.id),
        )
        self.conn.commit()
        updated = self.get_unit(unit_id)
        if updated is not None:
            self.fts_index_unit(updated)
        return updated

    def unpin_unit(self, unit_id: str) -> KnowledgeUnit | None:
        unit = self.get_unit(unit_id)
        if unit is None:
            return None

        metadata = dict(unit.metadata)
        for key in ("pinned", "pinned_at", "pin_reason"):
            metadata.pop(key, None)

        now = _utcnow_iso()
        self.conn.execute(
            """UPDATE knowledge_units
               SET metadata = ?, updated_at = ?
               WHERE id = ?""",
            (json.dumps(metadata), now, unit.id),
        )
        self.conn.commit()
        updated = self.get_unit(unit_id)
        if updated is not None:
            self.fts_index_unit(updated)
        return updated

    def merge_units(self, source_id: str, target_id: str, dry_run: bool = False) -> dict:
        if source_id == target_id:
            raise ValueError("source_id and target_id must be different.")

        source = self.get_unit(source_id)
        target = self.get_unit(target_id)
        missing_ids = [
            unit_id
            for unit_id, unit in ((source_id, source), (target_id, target))
            if unit is None
        ]
        if missing_ids:
            return {
                "source_id": source_id,
                "target_id": target_id,
                "dry_run": dry_run,
                "merged": False,
                "error": "unit_not_found",
                "missing_unit_ids": missing_ids,
                "message": "Unit not found: " + ", ".join(missing_ids),
            }

        assert source is not None
        assert target is not None

        merged_tags = list(target.tags)
        added_tags: list[str] = []
        for tag in source.tags:
            if tag not in merged_tags:
                merged_tags.append(tag)
                added_tags.append(tag)

        merged_metadata = dict(target.metadata)
        metadata_keys: list[str] = []
        metadata_conflicts: list[str] = []
        source_conflict_metadata: dict = {}
        for key in sorted(source.metadata):
            value = source.metadata[key]
            metadata_keys.append(key)
            if key not in merged_metadata or merged_metadata[key] == value:
                merged_metadata[key] = value
            else:
                metadata_conflicts.append(key)
                source_conflict_metadata[key] = value

        if source_conflict_metadata:
            merged_from_units = dict(merged_metadata.get("merged_from_units") or {})
            existing_source_entry = dict(merged_from_units.get(source_id) or {})
            existing_source_entry["metadata"] = {
                **dict(existing_source_entry.get("metadata") or {}),
                **source_conflict_metadata,
            }
            merged_from_units[source_id] = existing_source_entry
            merged_metadata["merged_from_units"] = merged_from_units

        existing_keys = {
            (row["from_unit_id"], row["to_unit_id"], row["relation"])
            for row in self.conn.execute(
                """SELECT from_unit_id, to_unit_id, relation
                   FROM edges
                   WHERE from_unit_id != ? AND to_unit_id != ?""",
                (source_id, source_id),
            ).fetchall()
        }
        source_edge_rows = self.conn.execute(
            """SELECT * FROM edges
               WHERE from_unit_id = ? OR to_unit_id = ?
               ORDER BY created_at, id""",
            (source_id, source_id),
        ).fetchall()

        rewired_edges: list[dict] = []
        skipped_duplicate_edges: list[dict] = []
        skipped_self_edges: list[dict] = []
        rewired_edge_counts = {"incoming": 0, "outgoing": 0, "total": 0}

        for row in source_edge_rows:
            edge = _row_to_edge(row)
            new_from = target_id if edge.from_unit_id == source_id else edge.from_unit_id
            new_to = target_id if edge.to_unit_id == source_id else edge.to_unit_id
            planned = {
                "edge_id": edge.id,
                "from_unit_id": edge.from_unit_id,
                "to_unit_id": edge.to_unit_id,
                "relation": edge.relation,
                "new_from_unit_id": new_from,
                "new_to_unit_id": new_to,
            }
            if new_from == new_to:
                skipped_self_edges.append(planned)
                continue

            edge_key = (new_from, new_to, edge.relation)
            if edge_key in existing_keys:
                skipped_duplicate_edges.append(planned)
                continue

            existing_keys.add(edge_key)
            rewired_edges.append(planned)
            if edge.from_unit_id == source_id:
                rewired_edge_counts["outgoing"] += 1
            if edge.to_unit_id == source_id:
                rewired_edge_counts["incoming"] += 1
            rewired_edge_counts["total"] += 1

        deleted_unit_id = None if dry_run else source_id
        summary = {
            "source_id": source_id,
            "target_id": target_id,
            "dry_run": dry_run,
            "merged": not dry_run,
            "merged_tags": merged_tags,
            "added_tags": added_tags,
            "metadata_keys": metadata_keys,
            "metadata_conflicts": metadata_conflicts,
            "rewired_edge_counts": rewired_edge_counts,
            "rewired_edges": rewired_edges,
            "skipped_duplicate_edges": skipped_duplicate_edges,
            "skipped_self_edges": skipped_self_edges,
            "deleted_unit_id": deleted_unit_id,
        }

        if dry_run:
            return summary

        now = _utcnow_iso()
        with self.conn:
            self.conn.execute(
                """UPDATE knowledge_units
                   SET metadata = ?, tags = ?, updated_at = ?
                   WHERE id = ?""",
                (json.dumps(merged_metadata), json.dumps(merged_tags), now, target_id),
            )
            for edge in rewired_edges:
                self.conn.execute(
                    """UPDATE edges
                       SET from_unit_id = ?, to_unit_id = ?
                       WHERE id = ?""",
                    (edge["new_from_unit_id"], edge["new_to_unit_id"], edge["edge_id"]),
                )
            skipped_edge_ids = [
                edge["edge_id"] for edge in [*skipped_duplicate_edges, *skipped_self_edges]
            ]
            for edge_id in skipped_edge_ids:
                self.conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
            self.conn.execute("DELETE FROM knowledge_fts WHERE unit_id = ?", (source_id,))
            self.conn.execute("DELETE FROM unit_aliases WHERE unit_id = ?", (source_id,))
            self.conn.execute("DELETE FROM knowledge_units WHERE id = ?", (source_id,))

        updated_target = self.get_unit(target_id)
        if updated_target is not None:
            self.fts_index_unit(updated_target)
        return summary

    def get_pinned_units(
        self,
        *,
        source_project: str | None = None,
        content_type: str | None = None,
        tag: str | None = None,
        limit: int | None = None,
    ) -> list[KnowledgeUnit]:
        where_parts = ["json_extract(metadata, '$.pinned') = 1"]
        params: list = []
        if source_project:
            where_parts.append("source_project = ?")
            params.append(source_project)
        if content_type:
            where_parts.append("content_type = ?")
            params.append(content_type)
        if tag:
            where_parts.append(
                "EXISTS (SELECT 1 FROM json_each(knowledge_units.tags) WHERE value = ?)"
            )
            params.append(tag)

        query = (
            "SELECT * FROM knowledge_units "
            "WHERE "
            + " AND ".join(where_parts)
            + " ORDER BY json_extract(metadata, '$.pinned_at') DESC, updated_at DESC"
        )
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, limit))

        rows = self.conn.execute(query, params).fetchall()
        return [_row_to_unit(row) for row in rows]

    def rename_tag(
        self,
        old_tag: str,
        new_tag: str,
        *,
        dry_run: bool = False,
        source_project: str | None = None,
        content_type: str | None = None,
    ) -> dict:
        old_tag = old_tag.strip()
        new_tag = new_tag.strip()
        if not old_tag:
            raise ValueError("old_tag must not be empty.")
        if not new_tag:
            raise ValueError("new_tag must not be empty.")

        where, params = self._unit_filter_sql(
            source_project=source_project,
            content_type=content_type,
        )
        rows = self.conn.execute(
            f"SELECT * FROM knowledge_units {where} ORDER BY created_at DESC",
            params,
        ).fetchall()
        units = [_row_to_unit(row) for row in rows]

        matched_units: list[KnowledgeUnit] = []
        changed: list[tuple[KnowledgeUnit, list[str]]] = []
        for unit in units:
            if old_tag not in unit.tags:
                continue
            matched_units.append(unit)

            renamed_tags: list[str] = []
            for tag in unit.tags:
                candidate = new_tag if tag == old_tag else tag
                if candidate not in renamed_tags:
                    renamed_tags.append(candidate)

            if renamed_tags != unit.tags:
                changed.append((unit, renamed_tags))

        changed_units = [
            {
                "id": unit.id,
                "title": unit.title,
                "source_project": str(unit.source_project),
                "source_entity_type": unit.source_entity_type,
                "content_type": str(unit.content_type),
                "old_tags": unit.tags,
                "new_tags": renamed_tags,
            }
            for unit, renamed_tags in changed
        ]

        if not dry_run and changed:
            now = _utcnow_iso()
            with self.conn:
                for unit, renamed_tags in changed:
                    self.conn.execute(
                        """UPDATE knowledge_units
                           SET tags = ?, updated_at = ?
                           WHERE id = ?""",
                        (json.dumps(renamed_tags), now, unit.id),
                    )
                    unit.tags = renamed_tags
                    unit.updated_at = now
                    self.conn.execute("DELETE FROM knowledge_fts WHERE unit_id = ?", (unit.id,))
                    self.conn.execute(
                        "INSERT INTO knowledge_fts (unit_id, title, content, tags) VALUES (?, ?, ?, ?)",
                        (unit.id, unit.title, unit.content, self._fts_tags_text_for_unit(unit)),
                    )

        return {
            "old_tag": old_tag,
            "new_tag": new_tag,
            "dry_run": dry_run,
            "matched_count": len(matched_units),
            "updated_count": len(changed_units),
            "changed_count": len(changed_units),
            "affected_count": len(changed_units),
            "affected_unit_ids": [unit["id"] for unit in changed_units],
            "changed_units": changed_units,
            "affected_units": changed_units,
            "filters": {
                "source_project": source_project,
                "content_type": content_type,
            },
        }

    def preview_tag_rename(
        self,
        old_tag: str,
        new_tag: str,
        *,
        limit: int = 50,
    ) -> dict:
        """Return a deterministic dry-run report for an exact tag rename."""
        old_tag = old_tag.strip()
        new_tag = new_tag.strip()
        if not old_tag:
            raise ValueError("old_tag must not be empty.")
        if not new_tag:
            raise ValueError("new_tag must not be empty.")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit must be a non-negative integer.")

        rows = self.conn.execute(
            """SELECT *
               FROM knowledge_units
               WHERE EXISTS (
                   SELECT 1
                   FROM json_each(knowledge_units.tags)
                   WHERE value = ?
               )
               ORDER BY title COLLATE NOCASE, title, source_project, source_id, id""",
            (old_tag,),
        ).fetchall()

        affected_units = []
        for unit in [_row_to_unit(row) for row in rows]:
            after_tags: list[str] = []
            for tag in unit.tags:
                candidate = new_tag if tag == old_tag else tag
                if candidate not in after_tags:
                    after_tags.append(candidate)

            if after_tags == unit.tags:
                continue

            affected_units.append(
                {
                    "id": unit.id,
                    "title": unit.title,
                    "source_project": str(unit.source_project),
                    "source_id": unit.source_id,
                    "source_entity_type": unit.source_entity_type,
                    "content_type": str(unit.content_type),
                    "before_tags": unit.tags,
                    "after_tags": after_tags,
                }
            )

        return {
            "old_tag": old_tag,
            "new_tag": new_tag,
            "affected_count": len(affected_units),
            "returned_units": affected_units[:limit],
        }

    def remove_tag(
        self,
        tag: str,
        *,
        dry_run: bool = False,
        source_project: str | None = None,
        content_type: str | None = None,
        limit: int | None = None,
    ) -> dict:
        tag = tag.strip()
        if not tag:
            raise ValueError("tag must not be empty.")

        where_parts, params = self._unit_filter_parts(
            source_project=source_project,
            content_type=content_type,
        )
        where_parts.append("EXISTS (SELECT 1 FROM json_each(knowledge_units.tags) WHERE value = ?)")
        params.append(tag)

        query = "SELECT * FROM knowledge_units WHERE " + " AND ".join(where_parts)
        query += " ORDER BY created_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, limit))

        rows = self.conn.execute(query, params).fetchall()
        matched_units = [_row_to_unit(row) for row in rows]

        changed: list[tuple[KnowledgeUnit, list[str]]] = []
        for unit in matched_units:
            next_tags = [unit_tag for unit_tag in unit.tags if unit_tag != tag]
            if next_tags != unit.tags:
                changed.append((unit, next_tags))

        changed_units = [
            {
                "id": unit.id,
                "title": unit.title,
                "source_project": str(unit.source_project),
                "source_entity_type": unit.source_entity_type,
                "content_type": str(unit.content_type),
                "old_tags": unit.tags,
                "new_tags": next_tags,
            }
            for unit, next_tags in changed
        ]

        if not dry_run and changed:
            now = _utcnow_iso()
            with self.conn:
                for unit, next_tags in changed:
                    self.conn.execute(
                        """UPDATE knowledge_units
                           SET tags = ?, updated_at = ?
                           WHERE id = ?""",
                        (json.dumps(next_tags), now, unit.id),
                    )
                    unit.tags = next_tags
                    unit.updated_at = now
                    self.conn.execute("DELETE FROM knowledge_fts WHERE unit_id = ?", (unit.id,))
                    self.conn.execute(
                        "INSERT INTO knowledge_fts (unit_id, title, content, tags) VALUES (?, ?, ?, ?)",
                        (unit.id, unit.title, unit.content, self._fts_tags_text_for_unit(unit)),
                    )

        return {
            "tag": tag,
            "dry_run": dry_run,
            "limit": limit,
            "matched_count": len(matched_units),
            "removed_count": len(changed_units),
            "changed_count": len(changed_units),
            "affected_count": len(changed_units),
            "affected_unit_ids": [unit["id"] for unit in changed_units],
            "changed_units": changed_units,
            "affected_units": changed_units,
            "representative_units": changed_units,
            "filters": {
                "source_project": source_project,
                "content_type": content_type,
            },
        }

    def apply_tags_to_units(
        self,
        unit_ids: list[str],
        *,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict:
        add_tags = list(dict.fromkeys(tag.strip() for tag in (add_tags or []) if tag.strip()))
        remove_tags = list(dict.fromkeys(tag.strip() for tag in (remove_tags or []) if tag.strip()))
        if not add_tags and not remove_tags:
            raise ValueError("At least one --add or --remove tag is required.")
        overlap = sorted(set(add_tags) & set(remove_tags))
        if overlap:
            raise ValueError("Tags cannot be both added and removed: " + ", ".join(overlap))

        ordered_ids = list(dict.fromkeys(unit_ids))
        units = [unit for unit_id in ordered_ids if (unit := self.get_unit(unit_id))]

        changed: list[tuple[KnowledgeUnit, list[str]]] = []
        remove_set = set(remove_tags)
        for unit in units:
            next_tags = [tag for tag in unit.tags if tag not in remove_set]
            for tag in add_tags:
                if tag not in next_tags:
                    next_tags.append(tag)
            if next_tags != unit.tags:
                changed.append((unit, next_tags))

        changed_units = [
            {
                "id": unit.id,
                "title": unit.title,
                "source_project": str(unit.source_project),
                "source_entity_type": unit.source_entity_type,
                "content_type": str(unit.content_type),
                "old_tags": unit.tags,
                "new_tags": next_tags,
            }
            for unit, next_tags in changed
        ]

        if not dry_run and changed:
            now = _utcnow_iso()
            with self.conn:
                for unit, next_tags in changed:
                    self.conn.execute(
                        """UPDATE knowledge_units
                           SET tags = ?, updated_at = ?
                           WHERE id = ?""",
                        (json.dumps(next_tags), now, unit.id),
                    )
                    unit.tags = next_tags
                    unit.updated_at = now
                    self.conn.execute("DELETE FROM knowledge_fts WHERE unit_id = ?", (unit.id,))
                    self.conn.execute(
                        "INSERT INTO knowledge_fts (unit_id, title, content, tags) VALUES (?, ?, ?, ?)",
                        (unit.id, unit.title, unit.content, self._fts_tags_text_for_unit(unit)),
                    )

        return {
            "add_tags": add_tags,
            "remove_tags": remove_tags,
            "dry_run": dry_run,
            "matched_count": len(units),
            "changed_count": len(changed_units),
            "affected_count": len(changed_units),
            "changed_units": changed_units,
            "affected_units": changed_units,
        }

    def delete_unit(self, unit_id: str) -> dict:
        unit = self.get_unit(unit_id)
        if unit is None:
            return {"unit_id": unit_id, "deleted": False, "edges_deleted": 0}

        edge_cursor = self.conn.execute(
            "DELETE FROM edges WHERE from_unit_id = ? OR to_unit_id = ?",
            (unit_id, unit_id),
        )
        self.conn.execute("DELETE FROM knowledge_fts WHERE unit_id = ?", (unit_id,))
        self.conn.execute("DELETE FROM unit_aliases WHERE unit_id = ?", (unit_id,))
        unit_cursor = self.conn.execute("DELETE FROM knowledge_units WHERE id = ?", (unit_id,))
        self.conn.commit()
        return {
            "unit_id": unit_id,
            "deleted": unit_cursor.rowcount > 0,
            "edges_deleted": edge_cursor.rowcount,
        }

    # --- Collections ---

    def create_collection(
        self,
        name: str,
        *,
        description: str = "",
        metadata: dict | None = None,
    ) -> dict:
        name = name.strip()
        if not name:
            raise ValueError("collection name must not be empty.")

        existing = self.get_collection(name)
        if existing is not None:
            return {**existing, "created": False}

        now = _utcnow_iso()
        collection_id = _new_id()
        self.conn.execute(
            """INSERT INTO collections
               (id, name, description, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                collection_id,
                name,
                description,
                json.dumps(metadata or {}, sort_keys=True),
                now,
                now,
            ),
        )
        self.conn.commit()
        created = self.get_collection(name)
        if created is None:
            raise RuntimeError(f"Collection was not written: {name}")
        return {**created, "created": True}

    def get_collection(self, name: str) -> dict | None:
        row = self.conn.execute(
            """SELECT c.*, COUNT(cu.unit_id) AS unit_count
               FROM collections c
               LEFT JOIN collection_units cu ON cu.collection_id = c.id
               WHERE c.name = ?
               GROUP BY c.id""",
            (name,),
        ).fetchone()
        return _row_to_collection(row) if row else None

    def list_collections(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT c.*, COUNT(cu.unit_id) AS unit_count
               FROM collections c
               LEFT JOIN collection_units cu ON cu.collection_id = c.id
               GROUP BY c.id
               ORDER BY c.name"""
        ).fetchall()
        return [_row_to_collection(row) for row in rows]

    def rename_collection(self, old_name: str, new_name: str) -> dict:
        old_name = old_name.strip()
        new_name = new_name.strip()
        if not old_name:
            raise ValueError("old collection name must not be empty.")
        if not new_name:
            raise ValueError("new collection name must not be empty.")

        collection = self.get_collection(old_name)
        if collection is None:
            return {
                "old_name": old_name,
                "new_name": new_name,
                "renamed": False,
                "error": "collection_not_found",
                "message": f"Collection not found: {old_name}",
            }
        if old_name == new_name:
            return {"old_name": old_name, "new_name": new_name, "renamed": True, "collection": collection}
        if self.get_collection(new_name) is not None:
            return {
                "old_name": old_name,
                "new_name": new_name,
                "renamed": False,
                "error": "collection_exists",
                "message": f"Collection already exists: {new_name}",
            }

        now = _utcnow_iso()
        self.conn.execute(
            "UPDATE collections SET name = ?, updated_at = ? WHERE id = ?",
            (new_name, now, collection["id"]),
        )
        self.conn.commit()
        renamed = self.get_collection(new_name)
        return {
            "old_name": old_name,
            "new_name": new_name,
            "renamed": True,
            "collection": renamed,
        }

    def delete_collection(self, name: str) -> dict:
        collection = self.get_collection(name)
        if collection is None:
            return {
                "name": name,
                "deleted": False,
                "memberships_deleted": 0,
                "error": "collection_not_found",
                "message": f"Collection not found: {name}",
            }

        membership_count = collection.get("unit_count", 0)
        cursor = self.conn.execute("DELETE FROM collections WHERE id = ?", (collection["id"],))
        self.conn.commit()
        return {
            "name": name,
            "deleted": cursor.rowcount > 0,
            "memberships_deleted": membership_count,
        }

    def add_unit_to_collection(self, collection_name: str, unit_id: str) -> dict:
        collection = self.get_collection(collection_name)
        if collection is None:
            return {
                "collection": collection_name,
                "unit_id": unit_id,
                "added": False,
                "error": "collection_not_found",
                "message": f"Collection not found: {collection_name}",
            }
        unit = self.get_unit(unit_id)
        if unit is None:
            return {
                "collection": collection_name,
                "unit_id": unit_id,
                "added": False,
                "error": "unit_not_found",
                "message": f"Unit not found: {unit_id}",
            }

        before = self.conn.total_changes
        self.conn.execute(
            """INSERT INTO collection_units (collection_id, unit_id, added_at)
               VALUES (?, ?, ?)
               ON CONFLICT(collection_id, unit_id) DO NOTHING""",
            (collection["id"], unit_id, _utcnow_iso()),
        )
        self.conn.commit()
        inserted = self.conn.total_changes > before
        return {
            "collection": self.get_collection(collection_name),
            "unit_id": unit_id,
            "added": inserted,
            "already_member": not inserted,
            "unit": self.collection_unit_summary(unit),
        }

    def remove_unit_from_collection(self, collection_name: str, unit_id: str) -> dict:
        collection = self.get_collection(collection_name)
        if collection is None:
            return {
                "collection": collection_name,
                "unit_id": unit_id,
                "removed": False,
                "error": "collection_not_found",
                "message": f"Collection not found: {collection_name}",
            }
        cursor = self.conn.execute(
            "DELETE FROM collection_units WHERE collection_id = ? AND unit_id = ?",
            (collection["id"], unit_id),
        )
        self.conn.commit()
        return {
            "collection": self.get_collection(collection_name),
            "unit_id": unit_id,
            "removed": cursor.rowcount > 0,
        }

    def list_collection_members(self, collection_name: str, *, limit: int | None = None) -> dict:
        collection = self.get_collection(collection_name)
        if collection is None:
            return {
                "collection": collection_name,
                "members": [],
                "error": "collection_not_found",
                "message": f"Collection not found: {collection_name}",
            }

        params: list[object] = [collection["id"]]
        query = """SELECT ku.*, cu.added_at
                   FROM collection_units cu
                   JOIN knowledge_units ku ON ku.id = cu.unit_id
                   WHERE cu.collection_id = ?
                   ORDER BY cu.added_at DESC, ku.created_at DESC"""
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, limit))
        rows = self.conn.execute(query, params).fetchall()
        return {
            "collection": collection,
            "members": [
                {
                    **self.collection_unit_summary(_row_to_unit(row)),
                    "added_at": row["added_at"],
                }
                for row in rows
            ],
        }

    def collection_activity_summary(
        self,
        collection_name: str,
        *,
        bucket: str = "month",
        field: str = "created_at",
        limit: int = 24,
    ) -> dict:
        bucket = str(bucket).strip().lower()
        field = str(field).strip().lower()
        if bucket not in COLLECTION_ACTIVITY_BUCKETS:
            valid = ", ".join(sorted(COLLECTION_ACTIVITY_BUCKETS))
            raise ValueError(f"Unsupported collection activity bucket: {bucket}. Use one of: {valid}.")
        if field not in COLLECTION_ACTIVITY_FIELDS:
            valid = ", ".join(sorted(COLLECTION_ACTIVITY_FIELDS))
            raise ValueError(f"Unsupported collection activity field: {field}. Use one of: {valid}.")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be a positive integer.")

        collection = self.get_collection(collection_name)
        if collection is None:
            return {
                "collection": collection_name,
                "buckets": [],
                "source_project_counts": {},
                "content_type_counts": {},
                "tag_counts": {},
                "first_seen_at": None,
                "last_seen_at": None,
                "error": "collection_not_found",
                "message": f"Collection not found: {collection_name}",
            }

        rows = self.conn.execute(
            """SELECT ku.*
               FROM collection_units cu
               JOIN knowledge_units ku ON ku.id = cu.unit_id
               WHERE cu.collection_id = ?""",
            (collection["id"],),
        ).fetchall()

        bucket_counts: Counter[str] = Counter()
        source_project_counts: Counter[str] = Counter()
        content_type_counts: Counter[str] = Counter()
        tag_counts: Counter[str] = Counter()
        seen_values: list[datetime] = []

        for row in rows:
            unit = _row_to_unit(row)
            seen_at = _parse_datetime(getattr(unit, field))
            if seen_at is None:
                continue
            seen_values.append(seen_at)
            bucket_counts[_activity_bucket_label(seen_at, bucket)] += 1
            source_project_counts[str(unit.source_project)] += 1
            content_type_counts[str(unit.content_type)] += 1
            tag_counts.update(str(tag) for tag in unit.tags)

        bucket_items = sorted(bucket_counts.items(), key=lambda item: item[0])
        if len(bucket_items) > limit:
            bucket_items = bucket_items[-limit:]

        return {
            "collection": collection,
            "bucket": bucket,
            "field": field,
            "limit": limit,
            "buckets": [
                {"bucket": bucket_name, "count": count}
                for bucket_name, count in bucket_items
            ],
            "source_project_counts": _sorted_counter_dict(source_project_counts),
            "content_type_counts": _sorted_counter_dict(content_type_counts),
            "tag_counts": _sorted_counter_dict(tag_counts),
            "first_seen_at": min(seen_values).isoformat() if seen_values else None,
            "last_seen_at": max(seen_values).isoformat() if seen_values else None,
        }

    def collection_diff(
        self,
        left_name: str,
        right_name: str,
        *,
        limit: int | None = None,
    ) -> dict:
        left = self.get_collection(left_name)
        right = self.get_collection(right_name)
        missing = [
            name
            for name, collection in ((left_name, left), (right_name, right))
            if collection is None
        ]
        if missing:
            raise ValueError(f"Collection not found: {', '.join(missing)}")

        def member_ids(collection_id: str) -> set[str]:
            rows = self.conn.execute(
                "SELECT unit_id FROM collection_units WHERE collection_id = ?",
                (collection_id,),
            ).fetchall()
            return {row["unit_id"] for row in rows}

        def member_summaries(unit_ids: set[str]) -> list[dict]:
            if not unit_ids:
                return []
            placeholders = ", ".join("?" for _ in unit_ids)
            params: list[object] = sorted(unit_ids)
            query = f"""SELECT *
                        FROM knowledge_units
                        WHERE id IN ({placeholders})
                        ORDER BY lower(title), title, id"""
            if limit is not None:
                query += " LIMIT ?"
                params.append(max(0, limit))
            rows = self.conn.execute(query, params).fetchall()
            return [self.collection_unit_summary(_row_to_unit(row)) for row in rows]

        left_ids = member_ids(left["id"])
        right_ids = member_ids(right["id"])
        left_only_ids = left_ids - right_ids
        right_only_ids = right_ids - left_ids
        both_ids = left_ids & right_ids

        return {
            "left": left,
            "right": right,
            "left_only": member_summaries(left_only_ids),
            "right_only": member_summaries(right_only_ids),
            "both": member_summaries(both_ids),
            "counts": {
                "left": len(left_ids),
                "right": len(right_ids),
                "left_only": len(left_only_ids),
                "right_only": len(right_only_ids),
                "both": len(both_ids),
            },
        }

    def collection_unit_summary(self, unit: KnowledgeUnit) -> dict:
        return {
            "id": unit.id,
            "title": unit.title,
            "source_project": str(unit.source_project),
            "source_id": unit.source_id,
            "source_entity_type": unit.source_entity_type,
            "content_type": str(unit.content_type),
            "tags": unit.tags,
            "created_at": _json_value(unit.created_at),
            "updated_at": _json_value(unit.updated_at),
        }

    def export_collections(self) -> dict:
        """Return a portable JSON-serializable collection backup."""
        rows = self.conn.execute(
            """SELECT c.id, c.name, c.description, c.metadata
               FROM collections c
               ORDER BY c.name ASC"""
        ).fetchall()
        collections = []
        for row in rows:
            member_rows = self.conn.execute(
                """SELECT unit_id
                   FROM collection_units
                   WHERE collection_id = ?
                   ORDER BY unit_id ASC""",
                (row["id"],),
            ).fetchall()
            collections.append(
                {
                    "name": row["name"],
                    "description": row["description"],
                    "metadata": json.loads(row["metadata"]),
                    "unit_ids": [member["unit_id"] for member in member_rows],
                }
            )
        return {
            "schema_version": COLLECTIONS_SCHEMA_VERSION,
            "exported_at": _utcnow_iso(),
            "collections": collections,
        }

    def import_collections(
        self,
        payload: dict,
        missing_units: Literal["skip", "strict"] = "skip",
    ) -> dict:
        """Import collection definitions and existing memberships idempotently."""
        if missing_units not in {"skip", "strict"}:
            raise ValueError("missing_units must be one of: skip, strict")
        schema_version = payload.get("schema_version")
        if schema_version != COLLECTIONS_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported collections schema_version {schema_version!r}; "
                f"expected {COLLECTIONS_SCHEMA_VERSION}"
            )

        collections = payload.get("collections", [])
        if not isinstance(collections, list):
            raise ValueError("Collections payload must contain a collections array")

        normalized: list[dict] = []
        referenced_unit_ids: set[str] = set()
        for data in collections:
            if not isinstance(data, dict):
                raise ValueError("Each collection must be a JSON object")
            name = str(data.get("name") or "").strip()
            if not name:
                raise ValueError("Each collection must include a non-empty name")
            metadata = data.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise ValueError(f"Collection {name!r} metadata must be a JSON object")
            unit_ids = data.get("unit_ids", data.get("members", []))
            if not isinstance(unit_ids, list):
                raise ValueError(f"Collection {name!r} unit_ids must be an array")
            normalized_unit_ids = [str(unit_id) for unit_id in unit_ids]
            referenced_unit_ids.update(normalized_unit_ids)
            normalized.append(
                {
                    "name": name,
                    "description": str(data.get("description") or ""),
                    "metadata": metadata,
                    "unit_ids": normalized_unit_ids,
                }
            )

        existing_unit_ids = {
            row["id"]
            for row in self.conn.execute(
                "SELECT id FROM knowledge_units WHERE id IN ({})".format(
                    ",".join("?" for _ in referenced_unit_ids) or "NULL"
                ),
                tuple(referenced_unit_ids),
            ).fetchall()
        }
        missing = sorted(referenced_unit_ids - existing_unit_ids)
        if missing and missing_units == "strict":
            raise ValueError(
                "Collections import references missing unit IDs: " + ", ".join(missing)
            )

        collections_inserted = 0
        collections_updated = 0
        collections_skipped = 0
        memberships_added = 0
        memberships_existing = 0

        try:
            self.conn.execute("BEGIN")
            for data in normalized:
                existing = self.get_collection(data["name"])
                now = _utcnow_iso()
                metadata_json = json.dumps(data["metadata"], sort_keys=True)
                if existing is None:
                    collection_id = _new_id()
                    self.conn.execute(
                        """INSERT INTO collections
                           (id, name, description, metadata, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            collection_id,
                            data["name"],
                            data["description"],
                            metadata_json,
                            now,
                            now,
                        ),
                    )
                    collections_inserted += 1
                else:
                    collection_id = existing["id"]
                    imported_values = {
                        "description": data["description"],
                        "metadata": data["metadata"],
                    }
                    existing_values = {
                        "description": existing["description"],
                        "metadata": existing["metadata"],
                    }
                    if imported_values == existing_values:
                        collections_skipped += 1
                    else:
                        self.conn.execute(
                            """UPDATE collections
                               SET description = ?, metadata = ?, updated_at = ?
                               WHERE id = ?""",
                            (data["description"], metadata_json, now, collection_id),
                        )
                        collections_updated += 1

                for unit_id in data["unit_ids"]:
                    if unit_id not in existing_unit_ids:
                        continue
                    before = self.conn.total_changes
                    self.conn.execute(
                        """INSERT INTO collection_units (collection_id, unit_id, added_at)
                           VALUES (?, ?, ?)
                           ON CONFLICT(collection_id, unit_id) DO NOTHING""",
                        (collection_id, unit_id, now),
                    )
                    if self.conn.total_changes > before:
                        memberships_added += 1
                    else:
                        memberships_existing += 1
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

        return {
            "collections_inserted": collections_inserted,
            "collections_updated": collections_updated,
            "collections_skipped": collections_skipped,
            "memberships_added": memberships_added,
            "memberships_existing": memberships_existing,
            "missing_units": missing,
            "missing_units_count": len(missing),
        }

    # --- JSON import/export ---

    def export_json(self) -> dict:
        """Return a portable JSON-serializable graph backup."""
        units = [
            {
                "id": unit.id,
                "source_project": str(unit.source_project),
                "source_id": unit.source_id,
                "source_entity_type": unit.source_entity_type,
                "title": unit.title,
                "content": unit.content,
                "content_type": str(unit.content_type),
                "metadata": unit.metadata,
                "tags": unit.tags,
                "confidence": unit.confidence,
                "utility_score": unit.utility_score,
                "created_at": _json_value(unit.created_at),
                "ingested_at": _json_value(unit.ingested_at),
                "updated_at": _json_value(unit.updated_at),
            }
            for unit in self.get_all_units(limit=1000000000)
        ]
        edges = [
            {
                "id": edge.id,
                "from_unit_id": edge.from_unit_id,
                "to_unit_id": edge.to_unit_id,
                "relation": str(edge.relation),
                "weight": edge.weight,
                "source": str(edge.source),
                "metadata": edge.metadata,
                "created_at": _json_value(edge.created_at),
            }
            for edge in self.get_all_edges()
        ]
        alias_rows = self.conn.execute(
            """SELECT unit_id, alias, source, created_at
               FROM unit_aliases
               ORDER BY unit_id, lower(alias), alias"""
        ).fetchall()
        aliases = [
            {
                "unit_id": row["unit_id"],
                "alias": row["alias"],
                "source": row["source"],
                "created_at": row["created_at"],
            }
            for row in alias_rows
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "exported_at": _utcnow_iso(),
            "units": units,
            "edges": edges,
            "aliases": aliases,
        }

    def export_jsonl_records(
        self,
        *,
        record_type: Literal["both", "units", "edges"] = "both",
    ) -> list[dict]:
        """Return JSONL-ready graph records ordered deterministically."""
        if record_type not in {"both", "units", "edges"}:
            raise ValueError("record_type must be one of: both, units, edges")

        records: list[dict] = []
        if record_type in {"both", "units"}:
            rows = self.conn.execute(
                "SELECT * FROM knowledge_units ORDER BY created_at ASC, id ASC"
            ).fetchall()
            records.extend(
                {
                    "record_type": "unit",
                    "id": unit.id,
                    "source_project": str(unit.source_project),
                    "source_id": unit.source_id,
                    "source_entity_type": unit.source_entity_type,
                    "title": unit.title,
                    "content": unit.content,
                    "content_type": str(unit.content_type),
                    "metadata": unit.metadata,
                    "tags": unit.tags,
                    "confidence": unit.confidence,
                    "utility_score": unit.utility_score,
                    "created_at": _json_value(unit.created_at),
                    "ingested_at": _json_value(unit.ingested_at),
                    "updated_at": _json_value(unit.updated_at),
                }
                for unit in (_row_to_unit(row) for row in rows)
            )

        if record_type in {"both", "edges"}:
            rows = self.conn.execute(
                "SELECT * FROM edges ORDER BY created_at ASC, id ASC"
            ).fetchall()
            records.extend(
                {
                    "record_type": "edge",
                    "id": edge.id,
                    "from_unit_id": edge.from_unit_id,
                    "to_unit_id": edge.to_unit_id,
                    "relation": str(edge.relation),
                    "weight": edge.weight,
                    "source": str(edge.source),
                    "metadata": edge.metadata,
                    "created_at": _json_value(edge.created_at),
                }
                for edge in (_row_to_edge(row) for row in rows)
            )

        return sorted(
            records,
            key=lambda record: (record["created_at"] or "", record["id"]),
        )

    def import_json(self, payload: dict) -> dict:
        """Import a portable graph backup idempotently.

        Unit IDs are preserved for new rows. If a unit already exists by source
        identity, that database ID is kept and imported edges are remapped to it.
        """
        schema_version = payload.get("schema_version")
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported graph JSON schema_version {schema_version!r}; "
                f"expected {SCHEMA_VERSION}"
            )

        units_inserted = 0
        units_updated = 0
        edges_inserted = 0
        edges_skipped = 0
        imported_to_graph_id: dict[str, str] = {}

        for data in payload.get("units", []):
            unit = KnowledgeUnit(**data)
            existing = self.get_unit_by_source(
                str(unit.source_project), unit.source_id, unit.source_entity_type
            )
            if existing:
                unit.id = existing.id
                units_updated += 1
            else:
                units_inserted += 1

            saved = self.insert_unit(unit)
            actual_id = unit.id or saved.id
            imported_to_graph_id[data["id"]] = actual_id
            fetched = self.get_unit(actual_id)
            if fetched:
                self.fts_index_unit(fetched)

        alias_unit_ids: set[str] = set()
        for data in payload.get("aliases", []):
            unit_id = imported_to_graph_id.get(data["unit_id"], data["unit_id"])
            if self.get_unit(unit_id) is None:
                continue
            alias = str(data.get("alias") or "").strip()
            if not alias:
                continue
            self.conn.execute(
                """INSERT INTO unit_aliases (unit_id, alias, source, created_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(unit_id, alias) DO UPDATE SET
                       source = excluded.source""",
                (
                    unit_id,
                    alias,
                    data.get("source"),
                    data.get("created_at") or _utcnow_iso(),
                ),
            )
            alias_unit_ids.add(unit_id)
        self.conn.commit()
        for unit_id in alias_unit_ids:
            unit = self.get_unit(unit_id)
            if unit:
                self.fts_index_unit(unit)

        for data in payload.get("edges", []):
            from_id = imported_to_graph_id.get(data["from_unit_id"], data["from_unit_id"])
            to_id = imported_to_graph_id.get(data["to_unit_id"], data["to_unit_id"])
            before = self.conn.total_changes
            edge = KnowledgeEdge(
                id=data.get("id", ""),
                from_unit_id=from_id,
                to_unit_id=to_id,
                relation=data["relation"],
                weight=data.get("weight", 1.0),
                source=data.get("source", EdgeSource.INFERRED),
                metadata=data.get("metadata", {}),
                created_at=data.get("created_at") or _utcnow_iso(),
            )
            self.insert_edge(edge)
            if self.conn.total_changes > before:
                edges_inserted += 1
            else:
                edges_skipped += 1

        return {
            "units_inserted": units_inserted,
            "units_updated": units_updated,
            "edges_inserted": edges_inserted,
            "edges_skipped": edges_skipped,
        }

    def count_units(
        self,
        *,
        source_project: str | None = None,
        source_entity_type: str | None = None,
    ) -> int:
        where_parts: list[str] = []
        params: list = []
        if source_project:
            where_parts.append("source_project = ?")
            params.append(source_project)
        if source_entity_type:
            where_parts.append("source_entity_type = ?")
            params.append(source_entity_type)
        where = " WHERE " + " AND ".join(where_parts) if where_parts else ""
        row = self.conn.execute(f"SELECT COUNT(*) FROM knowledge_units{where}", params).fetchone()
        return row[0]

    def count_units_by_source(
        self,
        *,
        time_range: tuple[datetime, datetime] | None = None,
        tags: list[str] | None = None,
        metadata_key: str | None = None,
    ) -> dict[SourceProject, int]:
        """Return unit counts grouped by source project type with optional filters.

        Args:
            time_range: Optional tuple of (start, end) datetime to filter by created_at
            tags: Optional list of tags - units must have ALL specified tags
            metadata_key: Optional metadata key that must be present

        Returns:
            Dictionary mapping SourceProject to count
        """
        where_parts: list[str] = []
        params: list[object] = []

        # Time range filter
        if time_range is not None:
            if (
                not isinstance(time_range, tuple)
                or len(time_range) != 2
                or not all(isinstance(dt, datetime) for dt in time_range)
            ):
                raise ValueError("time_range must be a tuple of two datetime objects")
            start, end = time_range
            if start > end:
                raise ValueError("time_range start must be before or equal to end")
            where_parts.append("created_at >= ?")
            params.append(start.isoformat())
            where_parts.append("created_at <= ?")
            params.append(end.isoformat())

        # Tags filter - units must have ALL specified tags
        if tags is not None:
            if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
                raise ValueError("tags must be a list of strings")
            for tag in tags:
                # Use JSON functions to check if tag is in the tags array
                where_parts.append("EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?)")
                params.append(tag)

        # Metadata key filter
        if metadata_key is not None:
            if not isinstance(metadata_key, str):
                raise ValueError("metadata_key must be a string")
            json_path = _metadata_json_path(metadata_key)
            where_parts.append("json_extract(metadata, ?) IS NOT NULL")
            params.append(json_path)

        # Build query
        query = "SELECT source_project, COUNT(*) as count FROM knowledge_units"
        if where_parts:
            query += " WHERE " + " AND ".join(where_parts)
        query += " GROUP BY source_project"

        rows = self.conn.execute(query, params).fetchall()

        # Convert to dict with SourceProject enum keys
        result: dict[SourceProject, int] = {}
        for row in rows:
            try:
                source_project = SourceProject(row["source_project"])
                result[source_project] = row["count"]
            except ValueError:
                # Skip unknown source projects
                continue

        return result

    def get_content_length_distribution_stats(
        self,
        *,
        source_project: SourceProject | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute distribution statistics for unit content length (character count).

        Args:
            source_project: Optional SourceProject to filter by
            tags: Optional list of tags - units must have ALL specified tags

        Returns:
            Dictionary with keys: p50, p75, p90, p95, p99, mean, median, max
        """
        where_parts: list[str] = []
        params: list[object] = []

        # Source project filter
        if source_project is not None:
            if not isinstance(source_project, SourceProject):
                raise TypeError("source_project must be a SourceProject enum")
            where_parts.append("source_project = ?")
            params.append(source_project.value)

        # Tags filter
        if tags is not None:
            if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
                raise ValueError("tags must be a list of strings")
            for tag in tags:
                where_parts.append("EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?)")
                params.append(tag)

        # Build query to get content lengths
        query = "SELECT LENGTH(content) as content_length FROM knowledge_units"
        if where_parts:
            query += " WHERE " + " AND ".join(where_parts)
        query += " ORDER BY content_length"

        rows = self.conn.execute(query, params).fetchall()

        if not rows:
            # Return zeros for empty result
            return {
                "p50": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "max": 0.0,
            }

        lengths = [row["content_length"] for row in rows]
        count = len(lengths)

        # Calculate statistics
        total = sum(lengths)
        mean = total / count
        max_length = max(lengths)

        # Calculate percentiles
        def percentile(p: float) -> float:
            """Calculate the p-th percentile (0 <= p <= 1)."""
            if count == 1:
                return float(lengths[0])
            index = p * (count - 1)
            lower = int(index)
            upper = min(lower + 1, count - 1)
            weight = index - lower
            return lengths[lower] * (1 - weight) + lengths[upper] * weight

        p50 = percentile(0.50)
        p75 = percentile(0.75)
        p90 = percentile(0.90)
        p95 = percentile(0.95)
        p99 = percentile(0.99)
        median = p50

        return {
            "p50": p50,
            "p75": p75,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "mean": mean,
            "median": median,
            "max": float(max_length),
        }

    def count_units_ingested_since(
        self,
        since: datetime,
        *,
        source_project: str | None = None,
        source_entity_type: str | None = None,
    ) -> int:
        where_parts = ["ingested_at >= ?"]
        params: list = [since.isoformat()]
        if source_project:
            where_parts.append("source_project = ?")
            params.append(source_project)
        if source_entity_type:
            where_parts.append("source_entity_type = ?")
            params.append(source_entity_type)
        row = self.conn.execute(
            "SELECT COUNT(*) FROM knowledge_units WHERE " + " AND ".join(where_parts),
            params,
        ).fetchone()
        return row[0]

    def freshness_report(
        self,
        targets: list[tuple[str, str]],
        *,
        days: int = 7,
        now: datetime | None = None,
    ) -> list[dict]:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        days = max(0, days)
        recent_since = now - timedelta(days=days)

        report = []
        for source_project, source_entity_type in targets:
            state = self.get_sync_state(source_project, source_entity_type)
            last_sync_at = _parse_datetime(state.last_sync_at) if state else None
            age_days = None
            if last_sync_at is not None:
                age_days = max(0.0, (now - last_sync_at).total_seconds() / 86400)

            report.append(
                {
                    "source_project": source_project,
                    "source_entity_type": source_entity_type,
                    "last_sync_at": last_sync_at.isoformat() if last_sync_at else None,
                    "age_days": age_days,
                    "recent_unit_count": self.count_units_ingested_since(
                        recent_since,
                        source_project=source_project,
                        source_entity_type=source_entity_type,
                    ),
                    "total_unit_count": self.count_units(
                        source_project=source_project,
                        source_entity_type=source_entity_type,
                    ),
                    "stale": last_sync_at is None or age_days is None or age_days > days,
                }
            )
        return report

    # --- Saved queries ---

    def save_query(
        self,
        *,
        name: str,
        query: str,
        mode: str = "fulltext",
        limit: int = 10,
        filters: dict | None = None,
        schedule: str | None = None,
    ) -> dict:
        now = _utcnow_iso()
        normalized_filters = filters or {}
        normalized_schedule = _normalize_query_schedule(schedule)
        self.conn.execute(
            """INSERT INTO saved_queries
               (name, query, mode, "limit", filters, schedule, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(name)
               DO UPDATE SET
                   query = excluded.query,
                   mode = excluded.mode,
                   "limit" = excluded."limit",
                   filters = excluded.filters,
                   schedule = excluded.schedule,
                   updated_at = excluded.updated_at
            """,
            (
                name,
                query,
                mode,
                limit,
                json.dumps(normalized_filters, sort_keys=True),
                normalized_schedule,
                now,
                now,
            ),
        )
        self.conn.commit()
        saved = self.get_saved_query(name)
        if saved is None:
            raise RuntimeError(f"Saved query was not written: {name}")
        return saved

    def get_saved_query(self, name: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM saved_queries WHERE name = ?", (name,)).fetchone()
        return _row_to_saved_query(row) if row else None

    def list_saved_queries(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM saved_queries ORDER BY name").fetchall()
        return [_row_to_saved_query(row) for row in rows]

    def mark_saved_query_run(
        self,
        name: str,
        *,
        effective_limit: int | None = None,
        mode: str | None = None,
        filters: dict | None = None,
        result_count: int | None = None,
        top_result_ids: list[str] | None = None,
    ) -> dict | None:
        saved = self.get_saved_query(name)
        if saved is None:
            return None

        now = _utcnow_iso()
        if (
            effective_limit is not None
            or mode is not None
            or filters is not None
            or result_count is not None
            or top_result_ids is not None
        ):
            self.record_saved_query_run(
                name,
                run_at=now,
                effective_limit=effective_limit if effective_limit is not None else saved["limit"],
                mode=mode or saved["mode"],
                filters=filters if filters is not None else saved["filters"],
                result_count=result_count if result_count is not None else 0,
                top_result_ids=top_result_ids or [],
                commit=False,
            )
        cursor = self.conn.execute(
            "UPDATE saved_queries SET last_run_at = ?, updated_at = ? WHERE name = ?",
            (now, now, name),
        )
        self.conn.commit()
        if cursor.rowcount == 0:
            return None
        return self.get_saved_query(name)

    def record_saved_query_run(
        self,
        name: str,
        *,
        effective_limit: int,
        mode: str,
        filters: dict | None,
        result_count: int,
        top_result_ids: list[str],
        run_at: str | datetime | None = None,
        commit: bool = True,
    ) -> dict:
        saved = self.get_saved_query(name)
        if saved is None:
            raise ValueError(f"Saved query not found: {name}")

        if run_at is None:
            normalized_run_at = _utcnow_iso()
        elif isinstance(run_at, datetime):
            normalized_run_at = (
                run_at if run_at.tzinfo is not None else run_at.replace(tzinfo=timezone.utc)
            ).astimezone(timezone.utc).isoformat()
        else:
            normalized_run_at = str(run_at)

        normalized_filters = filters or {}
        normalized_ids = [str(unit_id) for unit_id in top_result_ids]
        cursor = self.conn.execute(
            """INSERT INTO saved_query_runs
               (saved_query_name, run_at, effective_limit, mode, filters, result_count, top_result_ids)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                name,
                normalized_run_at,
                effective_limit,
                mode,
                json.dumps(normalized_filters, sort_keys=True),
                result_count,
                json.dumps(normalized_ids),
            ),
        )
        if commit:
            self.conn.commit()
        row = self.conn.execute(
            "SELECT * FROM saved_query_runs WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
        return _row_to_saved_query_run(row)

    def list_saved_query_runs(
        self,
        *,
        name: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        params: list[object] = []
        query = "SELECT * FROM saved_query_runs"
        if name is not None:
            query += " WHERE saved_query_name = ?"
            params.append(name)
        query += " ORDER BY run_at DESC, id DESC LIMIT ?"
        params.append(max(0, limit))
        rows = self.conn.execute(query, params).fetchall()
        return [_row_to_saved_query_run(row) for row in rows]

    def export_saved_queries(self) -> dict:
        """Return a JSON-serializable saved query backup."""
        return {
            "schema_version": SAVED_QUERIES_SCHEMA_VERSION,
            "exported_at": _utcnow_iso(),
            "queries": self.list_saved_queries(),
        }

    def import_saved_queries(self, payload: dict) -> dict:
        """Import saved queries idempotently by name."""
        schema_version = payload.get("schema_version")
        if schema_version not in (1, SAVED_QUERIES_SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported saved queries schema_version {schema_version!r}; "
                f"expected {SAVED_QUERIES_SCHEMA_VERSION}"
            )

        queries = payload.get("queries", [])
        if not isinstance(queries, list):
            raise ValueError("Saved queries payload must contain a queries array")

        inserted = 0
        updated = 0
        skipped = 0

        for data in queries:
            if not isinstance(data, dict):
                raise ValueError("Each saved query must be a JSON object")
            name = data.get("name")
            query = data.get("query")
            if not name or not query:
                raise ValueError("Each saved query must include name and query")

            filters = data.get("filters") or {}
            if not isinstance(filters, dict):
                raise ValueError(f"Saved query {name!r} filters must be a JSON object")

            existing = self.get_saved_query(name)
            now = _utcnow_iso()
            imported = {
                "name": name,
                "query": query,
                "mode": data.get("mode", "fulltext"),
                "limit": data.get("limit", 10),
                "filters": filters,
                "schedule": _normalize_query_schedule(data.get("schedule")),
                "last_run_at": data.get("last_run_at"),
                "created_at": data.get("created_at") or (existing["created_at"] if existing else now),
                "updated_at": data.get("updated_at")
                or data.get("created_at")
                or (existing["updated_at"] if existing else now),
            }

            if existing == imported:
                skipped += 1
                continue

            self.conn.execute(
                """INSERT INTO saved_queries
                   (name, query, mode, "limit", filters, schedule, last_run_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(name)
                   DO UPDATE SET
                       query = excluded.query,
                       mode = excluded.mode,
                       "limit" = excluded."limit",
                       filters = excluded.filters,
                       schedule = excluded.schedule,
                       last_run_at = excluded.last_run_at,
                       created_at = excluded.created_at,
                       updated_at = excluded.updated_at
                """,
                (
                    imported["name"],
                    imported["query"],
                    imported["mode"],
                    imported["limit"],
                    json.dumps(imported["filters"], sort_keys=True),
                    imported["schedule"],
                    imported["last_run_at"],
                    imported["created_at"],
                    imported["updated_at"],
                ),
            )
            if existing:
                updated += 1
            else:
                inserted += 1

        self.conn.commit()
        return {
            "inserted": inserted,
            "updated": updated,
            "skipped": skipped,
        }

    def delete_saved_query(self, name: str) -> bool:
        cursor = self.conn.execute("DELETE FROM saved_queries WHERE name = ?", (name,))
        self.conn.commit()
        return cursor.rowcount > 0

    # --- Edge CRUD ---

    def insert_edge(self, edge: KnowledgeEdge) -> KnowledgeEdge:
        if not edge.id:
            edge.id = _new_id()
        self.conn.execute(
            """INSERT INTO edges
               (id, from_unit_id, to_unit_id, relation, weight, source, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(from_unit_id, to_unit_id, relation) DO NOTHING
            """,
            (
                edge.id,
                edge.from_unit_id,
                edge.to_unit_id,
                edge.relation,
                edge.weight,
                edge.source,
                json.dumps(edge.metadata),
                edge.created_at.isoformat()
                if isinstance(edge.created_at, datetime)
                else str(edge.created_at),
            ),
        )
        self.conn.commit()
        return edge

    def import_edges_csv(self, path: str | Path, *, dry_run: bool = False) -> dict:
        """Import curated graph edges from a CSV file, validating row-by-row."""
        csv_path = Path(path)
        required_columns = {"from_unit_id", "to_unit_id", "relation"}
        optional_columns = {"weight", "source", "metadata_json"}
        allowed_columns = required_columns | optional_columns

        result = {
            "path": str(csv_path),
            "dry_run": dry_run,
            "inserted": 0,
            "skipped_existing": 0,
            "invalid": [],
            "inserted_rows": [],
            "skipped_existing_rows": [],
        }
        planned_keys: set[tuple[str, str, str]] = set()

        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing_columns = sorted(required_columns - fieldnames)
            unknown_columns = sorted(fieldnames - allowed_columns)

            for row_number, row in enumerate(reader, start=2):
                errors: list[str] = []
                if missing_columns:
                    errors.append(f"missing required columns: {', '.join(missing_columns)}")
                if unknown_columns:
                    errors.append(f"unknown columns: {', '.join(unknown_columns)}")

                from_unit_id = (row.get("from_unit_id") or "").strip()
                to_unit_id = (row.get("to_unit_id") or "").strip()
                relation_value = (row.get("relation") or "").strip()
                weight_value = (row.get("weight") or "").strip()
                source_value = (row.get("source") or "").strip()
                metadata_value = (row.get("metadata_json") or "").strip()

                if not from_unit_id:
                    errors.append("from_unit_id is required")
                elif self.get_unit(from_unit_id) is None:
                    errors.append(f"from_unit_id not found: {from_unit_id}")

                if not to_unit_id:
                    errors.append("to_unit_id is required")
                elif self.get_unit(to_unit_id) is None:
                    errors.append(f"to_unit_id not found: {to_unit_id}")

                try:
                    relation = EdgeRelation(relation_value)
                except ValueError:
                    relation = None
                    errors.append(f"unknown relation: {relation_value or '<blank>'}")

                if weight_value:
                    try:
                        weight = float(weight_value)
                    except ValueError:
                        weight = 1.0
                        errors.append(f"weight must be numeric: {weight_value}")
                else:
                    weight = 1.0

                if source_value:
                    try:
                        source = EdgeSource(source_value)
                    except ValueError:
                        source = EdgeSource.MANUAL
                        errors.append(f"unknown source: {source_value}")
                else:
                    source = EdgeSource.MANUAL

                if metadata_value:
                    try:
                        metadata = json.loads(metadata_value)
                    except json.JSONDecodeError as exc:
                        metadata = {}
                        errors.append(f"metadata_json must be valid JSON: {exc.msg}")
                    else:
                        if not isinstance(metadata, dict):
                            errors.append("metadata_json must be a JSON object")
                else:
                    metadata = {}

                if errors:
                    result["invalid"].append({"row_number": row_number, "reasons": errors})
                    continue

                assert relation is not None
                edge_key = (from_unit_id, to_unit_id, relation.value)
                if edge_key in planned_keys or self.edge_exists(from_unit_id, to_unit_id, relation.value):
                    result["skipped_existing"] += 1
                    result["skipped_existing_rows"].append(row_number)
                    continue

                planned_keys.add(edge_key)
                if not dry_run:
                    self.conn.execute(
                        """INSERT INTO edges
                           (id, from_unit_id, to_unit_id, relation, weight, source, metadata, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(from_unit_id, to_unit_id, relation) DO NOTHING
                        """,
                        (
                            _new_id(),
                            from_unit_id,
                            to_unit_id,
                            relation.value,
                            weight,
                            source.value,
                            json.dumps(metadata),
                            _utcnow_iso(),
                        ),
                    )
                result["inserted"] += 1
                result["inserted_rows"].append(row_number)

        if not dry_run:
            self.conn.commit()
        return result

    def get_all_edges(self) -> list[KnowledgeEdge]:
        rows = self.conn.execute("SELECT * FROM edges").fetchall()
        return [_row_to_edge(r) for r in rows]

    def get_edge(self, edge_id: str) -> KnowledgeEdge | None:
        row = self.conn.execute("SELECT * FROM edges WHERE id = ?", (edge_id,)).fetchone()
        return _row_to_edge(row) if row else None

    def get_edges_for_unit(self, unit_id: str) -> list[KnowledgeEdge]:
        rows = self.conn.execute(
            """SELECT * FROM edges
               WHERE from_unit_id = ? OR to_unit_id = ?
               ORDER BY created_at DESC, id""",
            (unit_id, unit_id),
        ).fetchall()
        return [_row_to_edge(r) for r in rows]

    def find_edges(
        self,
        *,
        relation: str | None = None,
        source: str | None = None,
        from_unit_id: str | None = None,
        to_unit_id: str | None = None,
        source_project: str | None = None,
        limit: int | None = None,
    ) -> list[KnowledgeEdge]:
        clauses: list[str] = []
        params: list[object] = []
        joins = ""

        if relation is not None:
            clauses.append("e.relation = ?")
            params.append(relation)
        if source is not None:
            clauses.append("e.source = ?")
            params.append(source)
        if from_unit_id is not None:
            clauses.append("e.from_unit_id = ?")
            params.append(from_unit_id)
        if to_unit_id is not None:
            clauses.append("e.to_unit_id = ?")
            params.append(to_unit_id)
        if source_project is not None:
            joins = """
               JOIN knowledge_units from_unit ON from_unit.id = e.from_unit_id
               JOIN knowledge_units to_unit ON to_unit.id = e.to_unit_id
            """
            clauses.append("(from_unit.source_project = ? OR to_unit.source_project = ?)")
            params.extend([source_project, source_project])

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT e.*
            FROM edges e
            {joins}
            {where}
            ORDER BY e.created_at DESC, e.id
        """
        if limit is not None:
            capped_limit = max(0, limit)
            query += " LIMIT ?"
            params.append(capped_limit)

        rows = self.conn.execute(query, params).fetchall()
        return [_row_to_edge(r) for r in rows]

    def edge_exists(self, from_unit_id: str, to_unit_id: str, relation: str) -> bool:
        row = self.conn.execute(
            """SELECT 1 FROM edges
               WHERE from_unit_id = ? AND to_unit_id = ? AND relation = ?
               LIMIT 1""",
            (from_unit_id, to_unit_id, relation),
        ).fetchone()
        return row is not None

    def rename_edge_relation(
        self,
        old_relation: str,
        new_relation: str,
        *,
        dry_run: bool = False,
        sample_limit: int = 10,
    ) -> dict:
        old_relation = old_relation.strip()
        new_relation = new_relation.strip()
        if not old_relation:
            raise ValueError("old_relation must not be empty.")
        if not new_relation:
            raise ValueError("new_relation must not be empty.")
        if old_relation == new_relation:
            raise ValueError("old_relation and new_relation must be different.")

        rows = self.conn.execute(
            """SELECT *
               FROM edges
               WHERE relation = ?
               ORDER BY created_at DESC, id""",
            (old_relation,),
        ).fetchall()
        edges = [_row_to_edge(row) for row in rows]
        changed_edges = [
            {
                "id": edge.id,
                "from_unit_id": edge.from_unit_id,
                "to_unit_id": edge.to_unit_id,
                "old_relation": old_relation,
                "new_relation": new_relation,
                "weight": edge.weight,
                "source": str(edge.source),
                "metadata": edge.metadata,
                "created_at": edge.created_at,
            }
            for edge in edges
        ]

        if not dry_run and edges:
            columns = {
                row["name"]
                for row in self.conn.execute("PRAGMA table_info(edges)").fetchall()
            }
            if "updated_at" in columns:
                self.conn.execute(
                    """UPDATE edges
                       SET relation = ?, updated_at = ?
                       WHERE relation = ?""",
                    (new_relation, _utcnow_iso(), old_relation),
                )
            else:
                self.conn.execute(
                    "UPDATE edges SET relation = ? WHERE relation = ?",
                    (new_relation, old_relation),
                )
            self.conn.commit()

        sample_edges = changed_edges[:sample_limit]
        sample_edge_ids = [edge["id"] for edge in sample_edges]
        return {
            "old_relation": old_relation,
            "new_relation": new_relation,
            "dry_run": dry_run,
            "matched_count": len(edges),
            "updated_count": len(edges),
            "changed_count": len(edges),
            "affected_count": len(edges),
            "affected_edge_ids": [edge["id"] for edge in changed_edges],
            "sample_edge_ids": sample_edge_ids,
            "changed_edges": changed_edges,
            "affected_edges": changed_edges,
            "sample_edges": sample_edges,
            "sample_limit": sample_limit,
        }

    def update_edge_fields(
        self,
        edge_id: str,
        *,
        relation: str | None = None,
        weight: float | None = None,
        source: str | None = None,
        metadata: dict | None = None,
    ) -> KnowledgeEdge | None:
        edge = self.get_edge(edge_id)
        if edge is None:
            return None

        if relation is not None:
            edge.relation = relation
        if weight is not None:
            edge.weight = weight
        if source is not None:
            edge.source = source
        if metadata:
            edge.metadata = {**edge.metadata, **metadata}

        self.conn.execute(
            """UPDATE edges
               SET relation = ?,
                   weight = ?,
                   source = ?,
                   metadata = ?
               WHERE id = ?""",
            (
                edge.relation,
                edge.weight,
                edge.source,
                json.dumps(edge.metadata),
                edge.id,
            ),
        )
        self.conn.commit()
        return self.get_edge(edge_id)

    def delete_edge(self, edge_id: str) -> dict:
        cursor = self.conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
        self.conn.commit()
        return {"edge_id": edge_id, "deleted": cursor.rowcount > 0}

    def delete_edges(
        self,
        *,
        relation: str | None = None,
        source: str | None = None,
        from_unit_id: str | None = None,
        to_unit_id: str | None = None,
        source_project: str | None = None,
        limit: int | None = None,
    ) -> list[KnowledgeEdge]:
        edges = self.find_edges(
            relation=relation,
            source=source,
            from_unit_id=from_unit_id,
            to_unit_id=to_unit_id,
            source_project=source_project,
            limit=limit,
        )
        if not edges:
            return []

        edge_ids = [edge.id for edge in edges]
        placeholders = ", ".join("?" for _ in edge_ids)
        self.conn.execute(f"DELETE FROM edges WHERE id IN ({placeholders})", edge_ids)
        self.conn.commit()
        return edges

    def edge_relation_distribution(self) -> dict[EdgeRelation, int]:
        """Return distribution statistics of edge relations across the graph."""
        relation_counts: Counter[EdgeRelation] = Counter()

        rows = self.conn.execute(
            """SELECT relation
               FROM edges
               ORDER BY id"""
        ).fetchall()

        for row in rows:
            try:
                relation = EdgeRelation(row["relation"])
                relation_counts[relation] += 1
            except ValueError:
                # Skip invalid relation values
                continue

        return dict(sorted(relation_counts.items(), key=lambda item: (-item[1], item[0].value)))

    # --- Integrity audit helpers ---

    def find_dangling_edges(self, *, limit: int = 20) -> dict:
        row = self.conn.execute(
            """SELECT COUNT(*) AS count
               FROM edges e
               LEFT JOIN knowledge_units from_unit ON from_unit.id = e.from_unit_id
               LEFT JOIN knowledge_units to_unit ON to_unit.id = e.to_unit_id
               WHERE from_unit.id IS NULL OR to_unit.id IS NULL"""
        ).fetchone()
        rows = self.conn.execute(
            """SELECT e.id, e.from_unit_id, e.to_unit_id, e.relation,
                      from_unit.id IS NULL AS missing_from,
                      to_unit.id IS NULL AS missing_to
               FROM edges e
               LEFT JOIN knowledge_units from_unit ON from_unit.id = e.from_unit_id
               LEFT JOIN knowledge_units to_unit ON to_unit.id = e.to_unit_id
               WHERE from_unit.id IS NULL OR to_unit.id IS NULL
               ORDER BY e.created_at DESC, e.id
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return {
            "count": row["count"] or 0,
            "examples": [
                {
                    "edge_id": r["id"],
                    "from_unit_id": r["from_unit_id"],
                    "to_unit_id": r["to_unit_id"],
                    "relation": r["relation"],
                    "missing_from": bool(r["missing_from"]),
                    "missing_to": bool(r["missing_to"]),
                }
                for r in rows
            ],
        }

    def find_self_loop_edges(self, *, limit: int = 20) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) AS count FROM edges WHERE from_unit_id = to_unit_id"
        ).fetchone()
        rows = self.conn.execute(
            """SELECT id, from_unit_id, to_unit_id, relation
               FROM edges
               WHERE from_unit_id = to_unit_id
               ORDER BY created_at DESC, id
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return {
            "count": row["count"] or 0,
            "examples": [
                {
                    "edge_id": r["id"],
                    "unit_id": r["from_unit_id"],
                    "relation": r["relation"],
                }
                for r in rows
            ],
        }

    def find_duplicate_edge_triples(self, *, limit: int = 20) -> dict:
        row = self.conn.execute(
            """SELECT COUNT(*) AS count
               FROM (
                   SELECT 1
                   FROM edges
                   GROUP BY from_unit_id, to_unit_id, relation
                   HAVING COUNT(*) > 1
               )"""
        ).fetchone()
        rows = self.conn.execute(
            """SELECT from_unit_id, to_unit_id, relation, COUNT(*) AS duplicate_count,
                      json_group_array(id) AS edge_ids
               FROM edges
               GROUP BY from_unit_id, to_unit_id, relation
               HAVING COUNT(*) > 1
               ORDER BY duplicate_count DESC, from_unit_id, to_unit_id, relation
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return {
            "count": row["count"] or 0,
            "examples": [
                {
                    "from_unit_id": r["from_unit_id"],
                    "to_unit_id": r["to_unit_id"],
                    "relation": r["relation"],
                    "duplicate_count": r["duplicate_count"],
                    "edge_ids": json.loads(r["edge_ids"]),
                }
                for r in rows
            ],
        }

    def find_units_missing_fts_rows(self, *, limit: int = 20) -> dict:
        row = self.conn.execute(
            """SELECT COUNT(*) AS count
               FROM knowledge_units u
               LEFT JOIN knowledge_fts f ON f.unit_id = u.id
               WHERE f.unit_id IS NULL"""
        ).fetchone()
        rows = self.conn.execute(
            """SELECT u.id, u.title, u.source_project, u.source_entity_type
               FROM knowledge_units u
               LEFT JOIN knowledge_fts f ON f.unit_id = u.id
               WHERE f.unit_id IS NULL
               ORDER BY u.created_at DESC, u.id
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return {
            "count": row["count"] or 0,
            "examples": [
                {
                    "unit_id": r["id"],
                    "title": r["title"],
                    "source_project": r["source_project"],
                    "source_entity_type": r["source_entity_type"],
                }
                for r in rows
            ],
        }

    def find_stale_fts_rows(self, *, limit: int = 20) -> dict:
        row = self.conn.execute(
            """SELECT COUNT(*) AS count
               FROM knowledge_fts f
               LEFT JOIN knowledge_units u ON u.id = f.unit_id
               WHERE u.id IS NULL"""
        ).fetchone()
        rows = self.conn.execute(
            """SELECT f.rowid, f.unit_id, f.title
               FROM knowledge_fts f
               LEFT JOIN knowledge_units u ON u.id = f.unit_id
               WHERE u.id IS NULL
               ORDER BY f.rowid
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return {
            "count": row["count"] or 0,
            "examples": [
                {"fts_rowid": r["rowid"], "unit_id": r["unit_id"], "title": r["title"]}
                for r in rows
            ],
        }

    def find_invalid_json_rows(self, *, limit: int = 20) -> dict:
        row = self.conn.execute(
            """SELECT
                   (SELECT COUNT(*) FROM knowledge_units
                    WHERE NOT json_valid(metadata) OR NOT json_valid(tags))
                 + (SELECT COUNT(*) FROM edges WHERE NOT json_valid(metadata))
                 AS count"""
        ).fetchone()
        rows = self.conn.execute(
            """SELECT 'knowledge_units' AS table_name, id, NULL AS edge_id,
                      NOT json_valid(metadata) AS invalid_metadata,
                      NOT json_valid(tags) AS invalid_tags
               FROM knowledge_units
               WHERE NOT json_valid(metadata) OR NOT json_valid(tags)
               UNION ALL
               SELECT 'edges' AS table_name, NULL AS id, id AS edge_id,
                      NOT json_valid(metadata) AS invalid_metadata,
                      0 AS invalid_tags
               FROM edges
               WHERE NOT json_valid(metadata)
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return {
            "count": row["count"] or 0,
            "examples": [
                {
                    "table": r["table_name"],
                    "unit_id": r["id"],
                    "edge_id": r["edge_id"],
                    "invalid_metadata": bool(r["invalid_metadata"]),
                    "invalid_tags": bool(r["invalid_tags"]),
                }
                for r in rows
            ],
        }

    def find_blank_units(self, *, limit: int = 20) -> dict:
        row = self.conn.execute(
            """SELECT COUNT(*) AS count
               FROM knowledge_units
               WHERE trim(title) = '' OR trim(content) = ''"""
        ).fetchone()
        rows = self.conn.execute(
            """SELECT id, title, source_project, source_entity_type,
                      trim(title) = '' AS blank_title,
                      trim(content) = '' AS blank_content
               FROM knowledge_units
               WHERE trim(title) = '' OR trim(content) = ''
               ORDER BY created_at DESC, id
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return {
            "count": row["count"] or 0,
            "examples": [
                {
                    "unit_id": r["id"],
                    "title": r["title"],
                    "source_project": r["source_project"],
                    "source_entity_type": r["source_entity_type"],
                    "blank_title": bool(r["blank_title"]),
                    "blank_content": bool(r["blank_content"]),
                }
                for r in rows
            ],
        }

    def repair_fts_index_integrity(self) -> dict:
        stale_cursor = self.conn.execute(
            """DELETE FROM knowledge_fts
               WHERE unit_id NOT IN (SELECT id FROM knowledge_units)"""
        )
        rows = self.conn.execute(
            """SELECT u.*
               FROM knowledge_units u
               LEFT JOIN knowledge_fts f ON f.unit_id = u.id
               WHERE f.unit_id IS NULL"""
        ).fetchall()
        for row in rows:
            unit = _row_to_unit(row)
            self.conn.execute(
                "INSERT INTO knowledge_fts (unit_id, title, content, tags) VALUES (?, ?, ?, ?)",
                (unit.id, unit.title, unit.content, self._fts_tags_text_for_unit(unit)),
            )
        self.conn.commit()
        return {
            "fts_rows_inserted": len(rows),
            "fts_rows_deleted": stale_cursor.rowcount,
        }

    def rebuild_fts_index(self) -> dict:
        rows = self.conn.execute("SELECT * FROM knowledge_units").fetchall()
        fts_row = self.conn.execute("SELECT COUNT(*) AS count FROM knowledge_fts").fetchone()
        rows_deleted = fts_row["count"] or 0

        with self.conn:
            self.conn.execute("DELETE FROM knowledge_fts")
            for row in rows:
                unit = _row_to_unit(row)
                self.conn.execute(
                    "INSERT INTO knowledge_fts (unit_id, title, content, tags) VALUES (?, ?, ?, ?)",
                    (unit.id, unit.title, unit.content, self._fts_tags_text_for_unit(unit)),
                )

        return {"rows_deleted": rows_deleted, "rows_inserted": len(rows)}

    def get_backlinks(
        self,
        unit_id: str,
        *,
        direction: str = "incoming",
        relation: str | None = None,
        source_project: str | None = None,
        content_type: str | None = None,
        tag: str | None = None,
        limit: int = 20,
    ) -> dict:
        center = self.get_unit(unit_id)
        if center is None:
            return {"center": None, "links": []}

        if direction not in ("incoming", "outgoing", "both"):
            raise ValueError("direction must be incoming, outgoing, or both")

        limit = max(0, limit)
        where = []
        params: list = []
        if direction in ("incoming", "both"):
            where.append("(e.to_unit_id = ?)")
            params.append(unit_id)
        if direction in ("outgoing", "both"):
            where.append("(e.from_unit_id = ?)")
            params.append(unit_id)

        query = f"""
            SELECT
                e.id AS edge_id,
                e.from_unit_id,
                e.to_unit_id,
                e.relation,
                e.weight,
                e.source,
                e.metadata AS edge_metadata,
                e.created_at AS edge_created_at,
                u.*
            FROM edges e
            JOIN knowledge_units u
              ON u.id = CASE
                  WHEN e.from_unit_id = ? THEN e.to_unit_id
                  ELSE e.from_unit_id
              END
            WHERE ({" OR ".join(where)})
        """
        query_params: list = [unit_id, *params]
        if relation:
            query += " AND e.relation = ?"
            query_params.append(relation)
        if source_project:
            query += " AND u.source_project = ?"
            query_params.append(source_project)
        if content_type:
            query += " AND u.content_type = ?"
            query_params.append(content_type)
        if tag:
            query += " AND EXISTS (SELECT 1 FROM json_each(u.tags) WHERE value = ?)"
            query_params.append(tag)
        query += " ORDER BY e.weight DESC, u.updated_at DESC, e.id LIMIT ?"
        query_params.append(limit)

        rows = self.conn.execute(query, query_params).fetchall()
        links = []
        for row in rows:
            edge = KnowledgeEdge(
                id=row["edge_id"],
                from_unit_id=row["from_unit_id"],
                to_unit_id=row["to_unit_id"],
                relation=row["relation"],
                weight=row["weight"],
                source=row["source"],
                metadata=json.loads(row["edge_metadata"]),
                created_at=row["edge_created_at"],
            )
            links.append(
                {
                    "direction": "outgoing" if row["from_unit_id"] == unit_id else "incoming",
                    "relation": str(edge.relation),
                    "edge": edge,
                    "unit": _row_to_unit(row),
                }
            )
        return {"center": center, "links": links}

    def edge_exists_between(self, left_unit_id: str, right_unit_id: str) -> bool:
        """Return true if any direct edge exists between two units in either direction."""
        row = self.conn.execute(
            """SELECT 1 FROM edges
               WHERE (from_unit_id = ? AND to_unit_id = ?)
                  OR (from_unit_id = ? AND to_unit_id = ?)
               LIMIT 1""",
            (left_unit_id, right_unit_id, right_unit_id, left_unit_id),
        ).fetchone()
        return row is not None

    # --- Sync state ---

    def get_sync_state(self, source_project: str, source_entity_type: str) -> SyncState | None:
        row = self.conn.execute(
            """SELECT * FROM sync_state
               WHERE source_project = ? AND source_entity_type = ?""",
            (source_project, source_entity_type),
        ).fetchone()
        if not row:
            return None
        return SyncState(
            source_project=row["source_project"],
            source_entity_type=row["source_entity_type"],
            last_sync_at=row["last_sync_at"],
            last_source_id=row["last_source_id"],
            items_synced=row["items_synced"],
        )

    def get_all_sync_state(self) -> list[SyncState]:
        rows = self.conn.execute(
            """SELECT * FROM sync_state
               ORDER BY source_project ASC, source_entity_type ASC"""
        ).fetchall()
        return [
            SyncState(
                source_project=row["source_project"],
                source_entity_type=row["source_entity_type"],
                last_sync_at=row["last_sync_at"],
                last_source_id=row["last_source_id"],
                items_synced=row["items_synced"],
            )
            for row in rows
        ]

    def upsert_sync_state(self, state: SyncState) -> None:
        self.conn.execute(
            """INSERT INTO sync_state
               (source_project, source_entity_type, last_sync_at, last_source_id, items_synced)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(source_project, source_entity_type)
               DO UPDATE SET
                   last_sync_at = excluded.last_sync_at,
                   last_source_id = excluded.last_source_id,
                   items_synced = sync_state.items_synced + excluded.items_synced
            """,
            (
                state.source_project,
                state.source_entity_type,
                state.last_sync_at.isoformat()
                if isinstance(state.last_sync_at, datetime)
                else str(state.last_sync_at),
                state.last_source_id,
                state.items_synced,
            ),
        )
        self.conn.commit()

    # --- FTS ---

    def _fts_tags_text_for_unit(self, unit: KnowledgeUnit) -> str:
        return " ".join([*map(str, unit.tags), *self._unit_alias_texts(unit.id)])

    def fts_index_unit(self, unit: KnowledgeUnit) -> None:
        self.conn.execute("DELETE FROM knowledge_fts WHERE unit_id = ?", (unit.id,))
        self.conn.execute(
            "INSERT INTO knowledge_fts (unit_id, title, content, tags) VALUES (?, ?, ?, ?)",
            (unit.id, unit.title, unit.content, self._fts_tags_text_for_unit(unit)),
        )
        self.conn.commit()

    def fts_search(
        self,
        query: str,
        *,
        limit: int = 20,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        updated_after: datetime | str | None = None,
        updated_before: datetime | str | None = None,
        metadata_key: str | None = None,
        metadata_value: object | None = None,
    ) -> list[dict]:
        date_clauses: list[str] = []
        date_params: list[object] = []
        for clauses, params in (
            _datetime_filter_sql("created", after=created_after, before=created_before),
            _datetime_filter_sql("updated", after=updated_after, before=updated_before),
        ):
            date_clauses.extend(clauses)
            date_params.extend(params)
        date_sql = "".join(f" AND ku.{clause}" for clause in date_clauses)
        metadata_sql = ""
        metadata_params: list[object] = []
        metadata_sql, metadata_params = _metadata_filter_sql(
            "ku.metadata",
            metadata_key=metadata_key,
            metadata_value=metadata_value,
        )
        try:
            rows = self.conn.execute(
                """SELECT unit_id,
                          rank,
                          snippet(knowledge_fts, -1, '[', ']', '...', 24) AS snippet
                   FROM knowledge_fts
                   JOIN knowledge_units ku ON ku.id = knowledge_fts.unit_id
                   WHERE knowledge_fts MATCH ?
                   """ + date_sql + metadata_sql + """
                   ORDER BY rank
                   LIMIT ?""",
                (query, *date_params, *metadata_params, limit),
            ).fetchall()
            results = [
                {
                    "unit_id": r["unit_id"],
                    "rank": r["rank"],
                    "snippet": r["snippet"] or "",
                }
                for r in rows
            ]
            if _requires_exact_single_term_filter(query):
                exact = query.strip().lower()
                filtered = []
                for result in results:
                    unit = self.get_unit(result["unit_id"])
                    if unit is None:
                        continue
                    haystacks = [unit.title, unit.content, *unit.tags, *self._unit_alias_texts(unit.id)]
                    if any(exact in str(value).lower() for value in haystacks):
                        filtered.append(result)
                return filtered
            return results
        except sqlite3.OperationalError:
            # Fallback to LIKE search if FTS query syntax is invalid
            terms = _fallback_search_terms(query)
            clauses = " OR ".join(
                [
                    """title LIKE ? OR content LIKE ? OR tags LIKE ?
                       OR EXISTS (
                           SELECT 1 FROM unit_aliases ua
                           WHERE ua.unit_id = knowledge_units.id AND ua.alias LIKE ?
                       )"""
                    for _ in terms
                ]
            )
            params: list[object] = []
            for term in terms:
                pattern = f"%{term}%"
                params.extend([pattern, pattern, pattern, pattern])
            params.extend(date_params)
            params.extend(metadata_params)
            params.append(limit)
            date_sql = "".join(f" AND {clause}" for clause in date_clauses)
            rows = self.conn.execute(
                f"""SELECT id as unit_id, content, -1.0 as rank
                   FROM knowledge_units
                   WHERE ({clauses}){date_sql}{metadata_sql.replace("ku.", "")}
                   LIMIT ?""",
                params,
            ).fetchall()
            return [
                {
                    "unit_id": r["unit_id"],
                    "rank": r["rank"],
                    "snippet": _excerpt(r["content"], query),
                }
                for r in rows
            ]

    # --- Ingestion orchestration ---

    def ingest(self, result: "IngestResult", source_project: str) -> dict:
        """Ingest adapter results: insert units, remap edge IDs, insert edges.

        Returns stats dict with units_inserted, units_skipped, edges_inserted.
        """
        units_inserted = 0
        units_skipped = 0

        # Build mapping from source_id -> graph unit id for edge remapping
        source_to_graph_id: dict[str, str] = {}

        for unit in result.units:
            existing = self.get_unit_by_source(
                unit.source_project, unit.source_id, unit.source_entity_type
            )
            if existing:
                # Update existing unit
                unit.id = existing.id
                self.insert_unit(unit)  # UPSERT
                self.fts_index_unit(unit)
                source_to_graph_id[unit.source_id] = existing.id
                units_skipped += 1
            else:
                inserted = self.insert_unit(unit)
                source_to_graph_id[unit.source_id] = inserted.id
                self.fts_index_unit(inserted)
                units_inserted += 1

        # Remap and insert edges
        edges_inserted = 0
        for edge in result.edges:
            edge_source_project = edge.metadata.get("source_project", source_project)
            # Resolve source-local IDs to graph IDs
            from_id = source_to_graph_id.get(edge.from_unit_id)
            to_id = source_to_graph_id.get(edge.to_unit_id)

            if not from_id:
                # Try finding in existing graph data
                from_unit = self.get_unit_by_source(
                    edge_source_project, edge.from_unit_id, self._guess_entity_type(edge, "from")
                )
                from_id = from_unit.id if from_unit else None

            if not to_id:
                to_unit = self.get_unit_by_source(
                    edge_source_project, edge.to_unit_id, self._guess_entity_type(edge, "to")
                )
                to_id = to_unit.id if to_unit else None

            if from_id and to_id:
                edge.from_unit_id = from_id
                edge.to_unit_id = to_id
                self.insert_edge(edge)
                edges_inserted += 1

        return {
            "units_inserted": units_inserted,
            "units_skipped": units_skipped,
            "edges_inserted": edges_inserted,
        }

    def _guess_entity_type(self, edge: KnowledgeEdge, direction: str) -> str:
        """Guess entity type for edge resolution based on source metadata."""
        explicit_type = edge.metadata.get(f"{direction}_entity_type")
        if explicit_type:
            return explicit_type
        source_project = edge.metadata.get("source_project", "")
        if source_project == "max":
            return "insight" if direction == "from" else "buildable_unit"
        if source_project == "forty_two" or edge.source == EdgeSource.SOURCE:
            return "knowledge_node"
        return "knowledge_node"
