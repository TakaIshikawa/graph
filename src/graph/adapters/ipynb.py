"""Adapter for local Jupyter notebooks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit, SyncState


class IpynbAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "ipynb"

    @property
    def entity_types(self) -> list[str]:
        return ["notebook"]

    def __init__(self, root_path: str = "") -> None:
        self.root_path = root_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "notebook" not in entity_types:
            return result

        root = Path(self.root_path).expanduser()
        if not root.exists() or not root.is_dir():
            return result

        sync_at = self._sync_timestamp(since) if since else None
        files = sorted(path for path in root.rglob("*") if self._is_notebook(path))
        for path in files:
            try:
                stat = path.stat()
            except OSError:
                continue
            if sync_at is not None and stat.st_mtime <= sync_at:
                continue

            notebook = self._read_notebook(path)
            if notebook is None:
                continue

            source_id = path.relative_to(root).as_posix()
            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.ME,
                    source_id=source_id,
                    source_entity_type="notebook",
                    title=self._title(notebook, path),
                    content=self._content(notebook),
                    content_type=ContentType.ARTIFACT,
                    metadata=self._metadata(notebook, source_id, stat.st_size),
                    tags=self._tags(notebook),
                    created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
                    updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                )
            )

        return result

    def _is_notebook(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() == ".ipynb"

    def _read_notebook(self, path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or not isinstance(data.get("cells"), list):
            return None
        return data

    def _title(self, notebook: dict[str, Any], path: Path) -> str:
        metadata = notebook.get("metadata")
        if isinstance(metadata, dict):
            title = metadata.get("title")
            if isinstance(title, str) and title.strip():
                return title.strip()
        return path.stem

    def _content(self, notebook: dict[str, Any]) -> str:
        parts: list[str] = []
        for index, cell in enumerate(notebook.get("cells", []), start=1):
            if not isinstance(cell, dict):
                continue
            cell_type = cell.get("cell_type")
            source = self._source_text(cell.get("source"))
            if cell_type == "markdown":
                text = source.strip()
                if text:
                    parts.append(text)
            elif cell_type == "code":
                summary = self._code_summary(index, source, notebook)
                if summary:
                    parts.append(summary)
        return "\n\n".join(parts)

    def _code_summary(
        self,
        index: int,
        source: str,
        notebook: dict[str, Any],
    ) -> str:
        lines = [line.rstrip() for line in source.splitlines()]
        meaningful = [line.strip() for line in lines if line.strip()]
        if not meaningful:
            return ""

        language = self._language(notebook)
        preview = "\n".join(meaningful[:5])
        if len(meaningful) > 5:
            preview += "\n..."
        return f"Code cell {index} ({language}, {len(meaningful)} lines):\n{preview}"

    def _metadata(
        self,
        notebook: dict[str, Any],
        source_id: str,
        file_size: int,
    ) -> dict[str, Any]:
        cells = [cell for cell in notebook.get("cells", []) if isinstance(cell, dict)]
        markdown_count = sum(1 for cell in cells if cell.get("cell_type") == "markdown")
        code_count = sum(1 for cell in cells if cell.get("cell_type") == "code")
        raw_count = sum(1 for cell in cells if cell.get("cell_type") == "raw")
        notebook_metadata = notebook.get("metadata")
        if not isinstance(notebook_metadata, dict):
            notebook_metadata = {}

        return {
            "path": source_id,
            "file_size": file_size,
            "cell_count": len(cells),
            "markdown_cell_count": markdown_count,
            "code_cell_count": code_count,
            "raw_cell_count": raw_count,
            "kernelspec": self._jsonable(notebook_metadata.get("kernelspec", {})),
            "language_info": self._jsonable(notebook_metadata.get("language_info", {})),
            "language": self._language(notebook),
            "notebook_tags": self._tags(notebook),
        }

    def _tags(self, notebook: dict[str, Any]) -> list[str]:
        metadata = notebook.get("metadata")
        if not isinstance(metadata, dict):
            return []

        raw_tags = metadata.get("tags", [])
        if isinstance(raw_tags, str):
            raw_tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
        if not isinstance(raw_tags, list):
            return []

        tags: list[str] = []
        for tag in raw_tags:
            normalized = str(tag).strip().removeprefix("#").strip()
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _language(self, notebook: dict[str, Any]) -> str:
        metadata = notebook.get("metadata")
        if not isinstance(metadata, dict):
            return "unknown"
        language_info = metadata.get("language_info")
        if isinstance(language_info, dict):
            name = language_info.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        kernelspec = metadata.get("kernelspec")
        if isinstance(kernelspec, dict):
            language = kernelspec.get("language")
            if isinstance(language, str) and language.strip():
                return language.strip()
        return "unknown"

    def _source_text(self, value: Any) -> str:
        if isinstance(value, list):
            return "".join(str(part) for part in value)
        if isinstance(value, str):
            return value
        return ""

    def _sync_timestamp(self, since: SyncState) -> float:
        if isinstance(since.last_sync_at, datetime):
            return since.last_sync_at.timestamp()
        return datetime.fromisoformat(str(since.last_sync_at)).timestamp()

    def _jsonable(self, value: Any) -> Any:
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)
