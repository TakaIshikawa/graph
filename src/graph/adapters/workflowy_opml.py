"""Adapter for Workflowy OPML exports."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from graph.adapters._personal_exports import clean_metadata, digest_source_id, iter_paths
from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit, SyncState


class WorkflowyOpmlAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "workflowy_opml"

    @property
    def entity_types(self) -> list[str]:
        return ["bullet"]

    def __init__(self, path: str = "") -> None:
        self.path = path

    def ingest(self, *, since: SyncState | None = None, entity_types: list[str] | None = None) -> IngestResult:
        result = IngestResult()
        if "bullet" not in set(entity_types or self.entity_types):
            return result
        for path in iter_paths(self.path, {".opml", ".xml"}):
            try:
                root = ET.parse(path).getroot()
            except (OSError, ET.ParseError):
                continue
            body = root.find(".//body")
            if body is None:
                continue
            for index, outline in enumerate(list(body), start=1):
                self._walk(result, outline, path, (index,), ())
        result.units.sort(key=lambda unit: unit.source_id)
        return result

    def _walk(self, result: IngestResult, node: ET.Element, source: Path, position: tuple[int, ...], parents: tuple[str, ...]) -> None:
        text = _clean_text(node.attrib.get("text") or node.attrib.get("title") or "")
        note = _clean_text(node.attrib.get("_note") or node.attrib.get("note") or "")
        path_titles = (*parents, text) if text else parents
        child_count = len([child for child in list(node) if _local(child.tag) == "outline"])
        if text or note:
            path_value = "/".join(str(part) for part in position)
            metadata = clean_metadata({"depth": len(position), "path": path_value, "parent_path": "/".join(str(part) for part in position[:-1]), "note": note, "completed": _completed(node), "child_count": child_count, "sibling_order": position[-1], "source_file": str(source), "title_path": list(path_titles)})
            result.units.append(KnowledgeUnit(source_project=self.name, source_id=digest_source_id(self.name, source, path_value, text), source_entity_type="bullet", title=text[:120] or note[:120], content="\n".join(part for part in (text, note) if part), content_type=ContentType.ARTIFACT, metadata=metadata, tags=["workflowy", "bullet"], created_at=datetime.now(timezone.utc)))
        for index, child in enumerate([child for child in list(node) if _local(child.tag) == "outline"], start=1):
            self._walk(result, child, source, (*position, index), path_titles)


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", value)).strip()


def _completed(node: ET.Element) -> bool | None:
    value = (node.attrib.get("_complete") or node.attrib.get("complete") or node.attrib.get("completed") or "").strip().lower()
    if value in {"true", "1", "yes", "complete", "completed"}:
        return True
    if value in {"false", "0", "no", "open"}:
        return False
    return None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
