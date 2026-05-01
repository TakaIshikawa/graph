"""Adapter for local Org-mode files.

This is intentionally a small dependency-free parser. It supports headings,
end-of-heading tags, immediate property drawers, and common bracket links. It
does not attempt to implement the full Org grammar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


HEADING_RE = re.compile(r"^(?P<stars>\*+)\s+(?P<title>.*)$")
TAG_SUFFIX_RE = re.compile(r"\s+:(?P<tags>[A-Za-z0-9_@#%:.-]+):\s*$")
PROPERTY_RE = re.compile(r"^:(?P<key>[A-Za-z0-9_@#%+-]+):\s*(?P<value>.*)$")
LINK_RE = re.compile(r"\[\[(?P<target>[^\]]+)\](?:\[[^\]]*\])?\]")
TODO_KEYWORDS = {"TODO", "DONE", "WAIT", "WAITING", "NEXT", "CANCELLED", "CANCELED"}


@dataclass
class OrgHeading:
    source_id: str
    path: str
    title: str
    level: int
    tags: list[str]
    properties: dict[str, str]
    content: str
    mtime: float
    ctime: float
    line: int
    links: list[str] = field(default_factory=list)


class OrgAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "org"

    @property
    def entity_types(self) -> list[str]:
        return ["org_heading"]

    def __init__(self, root_path: str = "") -> None:
        self.root_path = root_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "org_heading" not in entity_types:
            return result

        root = Path(self.root_path).expanduser()
        if not root.exists() or not root.is_dir():
            return result

        files = sorted(path for path in root.rglob("*.org") if path.is_file())
        all_headings: list[OrgHeading] = []
        for path in files:
            all_headings.extend(self._read_file(root, path))

        source_ids = {heading.source_id for heading in all_headings}
        link_index = self._build_link_index(all_headings)
        sync_at = self._sync_timestamp(since) if since else None

        for heading in all_headings:
            if sync_at is not None and heading.mtime <= sync_at:
                continue

            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.ORG,
                    source_id=heading.source_id,
                    source_entity_type="org_heading",
                    title=heading.title,
                    content=heading.content,
                    content_type=ContentType.INSIGHT,
                    metadata={
                        "path": heading.path,
                        "heading_level": heading.level,
                        "line": heading.line,
                        "properties": heading.properties,
                    },
                    tags=heading.tags,
                    created_at=datetime.fromtimestamp(heading.ctime, tz=timezone.utc),
                    updated_at=datetime.fromtimestamp(heading.mtime, tz=timezone.utc),
                )
            )

            for target in self._resolve_links(heading, link_index, source_ids):
                if target == heading.source_id:
                    continue
                result.edges.append(
                    KnowledgeEdge(
                        from_unit_id=heading.source_id,
                        to_unit_id=target,
                        relation=EdgeRelation.RELATES_TO,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": str(SourceProject.ORG),
                            "from_entity_type": "org_heading",
                            "to_entity_type": "org_heading",
                            "relation_type": "org_link",
                        },
                    )
                )

        return result

    def _read_file(self, root: Path, path: Path) -> list[OrgHeading]:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []

        stat = path.stat()
        rel_path = path.relative_to(root).as_posix()
        lines = text.splitlines()
        heading_rows: list[tuple[int, int, str]] = []
        for index, line in enumerate(lines):
            match = HEADING_RE.match(line)
            if match:
                heading_rows.append((index, len(match.group("stars")), match.group("title")))

        headings: list[OrgHeading] = []
        used_source_ids: set[str] = set()
        for position, (line_index, level, raw_title) in enumerate(heading_rows):
            next_heading_index = (
                heading_rows[position + 1][0] if position + 1 < len(heading_rows) else len(lines)
            )
            title, tags = self._parse_title(raw_title)
            properties, body_start = self._parse_properties(lines, line_index + 1, next_heading_index)
            body_lines = lines[body_start:next_heading_index]
            content = "\n".join([title, *body_lines]).rstrip()
            if text.endswith("\n") and content:
                content += "\n"
            source_id = self._unique_source_id(rel_path, title, used_source_ids)
            used_source_ids.add(source_id)
            headings.append(
                OrgHeading(
                    source_id=source_id,
                    path=rel_path,
                    title=title,
                    level=level,
                    tags=tags,
                    properties=properties,
                    content=content,
                    mtime=stat.st_mtime,
                    ctime=stat.st_ctime,
                    line=line_index + 1,
                    links=[match.group("target").strip() for match in LINK_RE.finditer(content)],
                )
            )
        return headings

    def _parse_title(self, raw_title: str) -> tuple[str, list[str]]:
        tags: list[str] = []
        title = raw_title.strip()
        tag_match = TAG_SUFFIX_RE.search(title)
        if tag_match:
            title = title[: tag_match.start()].rstrip()
            tags = [tag for tag in tag_match.group("tags").split(":") if tag]

        parts = title.split(maxsplit=1)
        if parts and parts[0] in TODO_KEYWORDS:
            title = parts[1] if len(parts) > 1 else parts[0]
        return title or "Untitled", tags

    def _parse_properties(
        self, lines: list[str], start: int, stop: int
    ) -> tuple[dict[str, str], int]:
        index = start
        while index < stop and not lines[index].strip():
            index += 1
        if index >= stop or lines[index].strip().upper() != ":PROPERTIES:":
            return {}, start

        properties: dict[str, str] = {}
        index += 1
        while index < stop:
            stripped = lines[index].strip()
            if stripped.upper() == ":END:":
                return properties, index + 1
            match = PROPERTY_RE.match(stripped)
            if match:
                properties[match.group("key")] = match.group("value").strip()
            index += 1
        return properties, start

    def _build_link_index(self, headings: list[OrgHeading]) -> dict[str, str]:
        index: dict[str, str] = {}
        first_by_file: dict[str, str] = {}
        for heading in headings:
            first_by_file.setdefault(heading.path, heading.source_id)
            keys = {
                heading.source_id,
                f"{heading.path}::{heading.title}",
                f"{heading.path}::*{heading.title}",
                heading.title,
                f"*{heading.title}",
            }
            for property_key in ("CUSTOM_ID", "ID"):
                value = heading.properties.get(property_key)
                if value:
                    keys.add(f"#{value}")
                    keys.add(f"{heading.path}::#{value}")
                    keys.add(f"id:{value}")
            for key in keys:
                index.setdefault(self._normalize_target(key), heading.source_id)

        for path, source_id in first_by_file.items():
            index.setdefault(self._normalize_target(path), source_id)
            index.setdefault(self._normalize_target(f"file:{path}"), source_id)
        return index

    def _resolve_links(
        self,
        heading: OrgHeading,
        link_index: dict[str, str],
        source_ids: set[str],
    ) -> list[str]:
        targets: list[str] = []
        for raw_target in heading.links:
            for candidate in self._link_candidates(raw_target, heading.path):
                source_id = candidate if candidate in source_ids else link_index.get(candidate)
                if source_id and source_id not in targets:
                    targets.append(source_id)
                    break
        return targets

    def _link_candidates(self, raw_target: str, current_path: str) -> list[str]:
        target = self._normalize_target(raw_target)
        candidates = [target]
        if target.startswith("file:"):
            file_target = target.removeprefix("file:")
            candidates.append(file_target)
            candidates.append(self._relative_file_target(file_target, current_path))
            if "::" in file_target:
                path, anchor = file_target.split("::", 1)
                candidates.extend([f"{path}::{anchor}", path])
                relative_path = self._relative_file_target(path, current_path)
                candidates.extend([f"{relative_path}::{anchor}", relative_path])
        elif target.startswith("*") or target.startswith("#"):
            candidates.append(f"{current_path}::{target}")
        elif "::" not in target and not target.startswith("id:"):
            candidates.append(f"{current_path}::{target}")
            candidates.append(f"{current_path}::*{target}")
        return candidates

    def _relative_file_target(self, target: str, current_path: str) -> str:
        if target.startswith("/") or target.startswith("../") or target.startswith("./"):
            return target.removeprefix("./")
        current_parent = PurePosixPath(current_path).parent
        if str(current_parent) == ".":
            return target
        return (current_parent / target).as_posix()

    def _unique_source_id(self, rel_path: str, title: str, used: set[str]) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "heading"
        base = f"{rel_path}#{slug}"
        source_id = base
        counter = 2
        while source_id in used:
            source_id = f"{base}-{counter}"
            counter += 1
        return source_id

    def _normalize_target(self, target: str) -> str:
        return unquote(target.strip()).replace("\\ ", " ")

    def _sync_timestamp(self, since: SyncState) -> float:
        if isinstance(since.last_sync_at, datetime):
            return since.last_sync_at.timestamp()
        return datetime.fromisoformat(str(since.last_sync_at)).timestamp()
