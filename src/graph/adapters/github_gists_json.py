"""Adapter for GitHub Gists JSON exports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class GithubGistsJsonAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "github_gists_json"

    @property
    def entity_types(self) -> list[str]:
        return ["gist"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "gist" not in set(entity_types or self.entity_types):
            return result
        sync_at = self._ensure_utc(since.last_sync_at) if since else None
        for path in self._iter_paths():
            try:
                records = self._read_records(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            for record in records:
                unit = self._unit_from_record(record, path.name)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
        result.units = sorted({unit.source_id: unit for unit in result.units}.values(), key=lambda unit: unit.source_id)
        return result

    def _iter_paths(self) -> list[Path]:
        if not self.path:
            return []
        root = Path(self.path).expanduser()
        if root.is_file() and root.suffix.lower() == ".json":
            return [root]
        if not root.is_dir():
            return []
        return sorted(child for child in root.rglob("*.json") if child.is_file())

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            for key in ("gists", "items", "data", "results"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [parsed]
        return []

    def _unit_from_record(self, record: dict[str, Any], source_file: str) -> KnowledgeUnit | None:
        gist_id = self._text(record.get("id"))
        description = self._text(record.get("description"))
        html_url = self._text(record.get("html_url") or record.get("url"))
        files = self._files(record.get("files"))
        created_at = self._parse_datetime(record.get("created_at"))
        updated_at = self._parse_datetime(record.get("updated_at")) or created_at
        owner = self._owner(record.get("owner") or record.get("user"))
        public = self._parse_bool(record.get("public"))
        comments = self._parse_int(record.get("comments") or record.get("comments_count"))
        if not gist_id and not description and not files and not html_url:
            return None
        metadata = {
            "id": gist_id,
            "description": description,
            "files": [{key: value for key, value in file.items() if key != "content"} for file in files],
            "file_names": [file["filename"] for file in files if file.get("filename")],
            "languages": sorted({file["language"] for file in files if file.get("language")}),
            "public": public,
            "owner": owner,
            "comments_count": comments,
            "html_url": html_url,
            "created_at": created_at.isoformat() if created_at else self._text(record.get("created_at")),
            "updated_at": updated_at.isoformat() if updated_at else self._text(record.get("updated_at")),
            "source_file": source_file,
            "record": record,
        }
        now = datetime.now(timezone.utc)
        return KnowledgeUnit(
            source_project=SourceProject.GITHUB_GISTS_JSON,
            source_id=self._source_id(gist_id, html_url, description),
            source_entity_type="gist",
            title=description or html_url or gist_id or "GitHub gist",
            content=self._content(description, html_url, files),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value not in ("", None, [])},
            tags=list(dict.fromkeys(["github", "gist", *metadata["languages"]])),
            created_at=created_at or now,
            updated_at=updated_at or created_at or now,
        )

    def _files(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, dict):
            raw = value.values()
        elif isinstance(value, list):
            raw = value
        else:
            raw = []
        files: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            filename = self._text(item.get("filename") or item.get("name"))
            language = self._text(item.get("language"))
            content = self._text(item.get("content") or item.get("truncated_content"))
            files.append(
                {
                    "filename": filename,
                    "language": language,
                    "type": self._text(item.get("type")),
                    "size": self._parse_int(item.get("size")),
                    "raw_url": self._text(item.get("raw_url")),
                    "content": content,
                }
            )
        return files

    def _content(self, description: str, html_url: str, files: list[dict[str, Any]]) -> str:
        parts = [description]
        if files:
            summary = ", ".join(file["filename"] for file in files if file.get("filename"))
            if summary:
                parts.append(f"Files: {summary}")
        if html_url:
            parts.append(f"URL: {html_url}")
        for file in files:
            snippet = self._text(file.get("content"))
            if snippet:
                parts.append(f"{file.get('filename') or 'file'}:\n{snippet[:500]}")
        return "\n\n".join(part for part in parts if part)

    def _source_id(self, gist_id: str, html_url: str, description: str) -> str:
        raw = gist_id or html_url or description
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"github_gists_json:{digest}"

    def _owner(self, value: Any) -> str:
        if isinstance(value, dict):
            return self._text(value.get("login") or value.get("name"))
        return self._text(value)

    def _parse_int(self, value: Any) -> int | None:
        if value in ("", None):
            return None
        try:
            return int(float(str(value).strip()))
        except ValueError:
            return None

    def _parse_bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        text = self._text(value).casefold()
        if text in {"true", "yes", "1", "public"}:
            return True
        if text in {"false", "no", "0", "private"}:
            return False
        return None

    def _parse_datetime(self, value: Any) -> datetime | None:
        text = self._text(value)
        if not text:
            return None
        try:
            return self._ensure_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
        except ValueError:
            return None

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value).strip()
