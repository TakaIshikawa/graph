"""Adapter for Tana JSON exports with supertags and fields."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState

# Tana inline references: [[node name]] or [[^node-id]]
NODE_REF_RE = re.compile(r"\[\[(?:\^)?([^\]]+)\]\]")

# Tana tags: #tag or #[[tag name]]
TAG_RE = re.compile(r"#\[\[([^\]]+)\]\]|(?<![\w/])#([\w][\w-]*)", re.UNICODE)

# ISO-ish date patterns Tana uses in calendar nodes
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


class TanaAdapter(SourceAdapter):
    """Import Tana JSON exports preserving supertags, fields, and hierarchy."""

    @property
    def name(self) -> str:
        return "tana"

    @property
    def entity_types(self) -> list[str]:
        return ["node", "supertag"]

    def __init__(
        self,
        file_path: str = "",
        root_path: str = "",
        path: str = "",
    ) -> None:
        self.file_path = file_path or path
        self.root_path = root_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        allowed_types = set(entity_types or self.entity_types)
        if not allowed_types.intersection(self.entity_types):
            return result

        paths = self._json_paths()
        sync_at = self._sync_datetime(since) if since else None

        # Collect all nodes and supertag definitions
        supertag_defs: dict[str, dict[str, Any]] = {}  # tag_id -> definition
        nodes: list[dict[str, Any]] = []

        for p in paths:
            data = self._read_json(p)
            self._extract_data(data, supertag_defs, nodes, p)

        # Build lookup indices
        node_id_map: dict[str, str] = {}  # tana node id -> source_id
        node_name_map: dict[str, str] = {}  # lowercase name -> source_id
        included_ids: set[str] = set()

        # First pass: create supertag units
        if "supertag" in allowed_types:
            for tag_id, tag_def in supertag_defs.items():
                tag_name = tag_def.get("name", tag_id)
                source_id = self._source_id("supertag", tag_id)
                created_at = self._parse_datetime(tag_def.get("createdAt"))
                updated_at = self._parse_datetime(tag_def.get("updatedAt"))

                if sync_at and updated_at and updated_at <= sync_at:
                    continue

                fields_schema = self._extract_field_schema(tag_def)
                metadata: dict[str, Any] = {
                    "tana_id": tag_id,
                    "tag_name": tag_name,
                }
                if fields_schema:
                    metadata["fields_schema"] = fields_schema
                if tag_def.get("description"):
                    metadata["description"] = tag_def["description"]
                if tag_def.get("extends"):
                    metadata["extends"] = tag_def["extends"]

                unit = KnowledgeUnit(
                    source_project=SourceProject.TANA,
                    source_id=source_id,
                    source_entity_type="supertag",
                    title=tag_name,
                    content=tag_def.get("description", tag_name),
                    content_type=ContentType.METADATA,
                    metadata=metadata,
                    tags=[tag_name.lower()],
                    created_at=created_at or datetime.now(timezone.utc),
                    updated_at=updated_at or created_at or datetime.now(timezone.utc),
                )
                result.units.append(unit)
                node_id_map[tag_id] = source_id
                node_name_map[tag_name.lower()] = source_id
                included_ids.add(source_id)

        # Second pass: create node units
        if "node" in allowed_types:
            for node in nodes:
                node_id = node.get("id", "")
                node_name = node.get("name") or node.get("title") or ""
                if not node_id and not node_name:
                    continue

                source_id = self._source_id("node", node_id or node_name)
                created_at = self._parse_datetime(node.get("createdAt"))
                updated_at = self._parse_datetime(node.get("updatedAt"))

                if sync_at and updated_at and updated_at <= sync_at:
                    continue

                # Extract content from node
                content = node_name
                if node.get("description"):
                    content = f"{node_name}\n\n{node['description']}"

                # Extract supertags applied to this node
                node_tags = self._extract_node_tags(node, supertag_defs)

                # Extract field values
                field_values = self._extract_field_values(node)

                # Extract calendar date if present
                calendar_date = self._extract_calendar_date(node)

                metadata = {
                    "tana_id": node_id,
                }
                if node_tags:
                    metadata["supertags"] = node_tags
                if field_values:
                    metadata["fields"] = field_values
                if calendar_date:
                    metadata["date"] = calendar_date
                if node.get("parentId"):
                    metadata["parent_id"] = node["parentId"]

                tags = [t.lower() for t in node_tags]
                # Also extract inline tags from content
                for m in TAG_RE.finditer(content):
                    tag = (m.group(1) or m.group(2)).strip().lower()
                    if tag and tag not in tags:
                        tags.append(tag)

                unit = KnowledgeUnit(
                    source_project=SourceProject.TANA,
                    source_id=source_id,
                    source_entity_type="node",
                    title=node_name or "Untitled",
                    content=content or "Untitled",
                    content_type=ContentType.ARTIFACT,
                    metadata=metadata,
                    tags=sorted(tags),
                    created_at=created_at or datetime.now(timezone.utc),
                    updated_at=updated_at or created_at or datetime.now(timezone.utc),
                )
                result.units.append(unit)
                node_id_map[node_id] = source_id
                if node_name:
                    node_name_map[node_name.lower()] = source_id
                included_ids.add(source_id)

        # Build edges: parent-child hierarchy
        if "node" in allowed_types:
            emitted_edges: set[tuple[str, str, str]] = set()
            for node in nodes:
                node_id = node.get("id", "")
                source_id = node_id_map.get(node_id)
                if not source_id or source_id not in included_ids:
                    continue

                # Parent-child
                parent_id = node.get("parentId")
                if parent_id:
                    parent_source_id = node_id_map.get(parent_id)
                    if parent_source_id and parent_source_id in included_ids:
                        edge_key = (parent_source_id, source_id, "contains")
                        if edge_key not in emitted_edges:
                            emitted_edges.add(edge_key)
                            result.edges.append(
                                KnowledgeEdge(
                                    id=self._edge_id(parent_source_id, source_id, "contains"),
                                    from_unit_id=parent_source_id,
                                    to_unit_id=source_id,
                                    relation=EdgeRelation.CONTAINS,
                                    source=EdgeSource.SOURCE,
                                    metadata={
                                        "source_project": SourceProject.TANA.value,
                                        "relation_type": "hierarchy",
                                    },
                                )
                            )

                # Inline references
                content = node.get("name", "") or ""
                if node.get("description"):
                    content += " " + node["description"]
                for ref_match in NODE_REF_RE.finditer(content):
                    ref = ref_match.group(1).strip()
                    target_id = node_id_map.get(ref) or node_name_map.get(ref.lower())
                    if target_id and target_id != source_id and target_id in included_ids:
                        edge_key = (source_id, target_id, "references")
                        if edge_key not in emitted_edges:
                            emitted_edges.add(edge_key)
                            result.edges.append(
                                KnowledgeEdge(
                                    id=self._edge_id(source_id, target_id, "references"),
                                    from_unit_id=source_id,
                                    to_unit_id=target_id,
                                    relation=EdgeRelation.REFERENCES,
                                    source=EdgeSource.SOURCE,
                                    metadata={
                                        "source_project": SourceProject.TANA.value,
                                        "relation_type": "inline_ref",
                                    },
                                )
                            )

        result.units.sort(key=lambda u: (u.source_entity_type, u.source_id))
        result.edges.sort(key=lambda e: (e.from_unit_id, e.to_unit_id))
        return result

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _extract_data(
        self,
        data: Any,
        supertag_defs: dict[str, dict[str, Any]],
        nodes: list[dict[str, Any]],
        path: Path,
    ) -> None:
        """Walk Tana export structure to collect supertags and nodes."""
        if isinstance(data, dict):
            # Check for supertag definitions section
            for tag in data.get("supertags", []):
                if isinstance(tag, dict) and tag.get("id"):
                    supertag_defs[tag["id"]] = tag

            # Check for nodes section
            for node in data.get("nodes", []):
                if isinstance(node, dict):
                    node.setdefault("_source_path", str(path))
                    nodes.append(node)
                    # Recurse into children
                    self._collect_child_nodes(node, nodes, path)

            # Handle flat list format where top-level is workspace
            if "children" in data and "nodes" not in data:
                for child in data.get("children", []):
                    if isinstance(child, dict):
                        child.setdefault("_source_path", str(path))
                        nodes.append(child)
                        self._collect_child_nodes(child, nodes, path)

        elif isinstance(data, list):
            # Array of nodes at top level
            for item in data:
                if isinstance(item, dict):
                    item.setdefault("_source_path", str(path))
                    # Check if it's a supertag definition
                    if item.get("type") == "supertag" or item.get("isTag"):
                        supertag_defs[item.get("id", item.get("name", ""))] = item
                    else:
                        nodes.append(item)
                    self._collect_child_nodes(item, nodes, path)

    def _collect_child_nodes(
        self,
        parent: dict[str, Any],
        nodes: list[dict[str, Any]],
        path: Path,
    ) -> None:
        """Recursively collect child nodes, setting parentId."""
        parent_id = parent.get("id", "")
        for child in parent.get("children", []):
            if isinstance(child, dict):
                child.setdefault("parentId", parent_id)
                child.setdefault("_source_path", str(path))
                nodes.append(child)
                self._collect_child_nodes(child, nodes, path)

    def _extract_field_schema(self, tag_def: dict[str, Any]) -> list[dict[str, str]]:
        """Extract field definitions from a supertag definition."""
        fields = []
        for field in tag_def.get("fields", []):
            if isinstance(field, dict):
                entry: dict[str, str] = {}
                if field.get("name"):
                    entry["name"] = field["name"]
                if field.get("type"):
                    entry["type"] = field["type"]
                if field.get("id"):
                    entry["id"] = field["id"]
                if entry:
                    fields.append(entry)
        return fields

    def _extract_node_tags(
        self, node: dict[str, Any], supertag_defs: dict[str, dict[str, Any]]
    ) -> list[str]:
        """Get supertag names applied to a node."""
        tags: list[str] = []
        for tag_ref in node.get("supertags", []):
            if isinstance(tag_ref, str):
                tag_def = supertag_defs.get(tag_ref)
                tag_name = tag_def["name"] if tag_def else tag_ref
                tags.append(tag_name)
            elif isinstance(tag_ref, dict):
                tag_id = tag_ref.get("id") or tag_ref.get("tagId", "")
                tag_def = supertag_defs.get(tag_id)
                tag_name = tag_ref.get("name") or (tag_def["name"] if tag_def else tag_id)
                tags.append(tag_name)
        return tags

    def _extract_field_values(
        self, node: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract field values from a node's structured attributes."""
        field_values: dict[str, Any] = {}

        # Direct fields on the node
        for fv in node.get("fields", []):
            if isinstance(fv, dict):
                name = fv.get("name") or fv.get("fieldId", "")
                value = fv.get("value") or fv.get("values")
                if name and value is not None:
                    field_values[name] = value

        # Props/attributes style
        for key, value in node.get("props", {}).items():
            if key not in ("id", "name", "title", "children", "supertags", "parentId"):
                field_values[key] = value

        return field_values

    def _extract_calendar_date(self, node: dict[str, Any]) -> str | None:
        """Extract date from a Tana calendar/date node."""
        # Tana calendar nodes have a date field or date-like name
        if node.get("date"):
            raw = str(node["date"])
            m = DATE_RE.search(raw)
            if m:
                return m.group(1)
            return raw

        # Check name for date pattern
        name = node.get("name") or ""
        m = DATE_RE.search(name)
        if m:
            return m.group(1)

        return None

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------

    def _json_paths(self) -> list[Path]:
        if self.file_path:
            path = Path(self.file_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Tana export path does not exist: {path}")
            if not path.is_file():
                raise ValueError(f"Tana file_path must be a JSON file: {path}")
            return [path]

        if self.root_path:
            root = Path(self.root_path).expanduser()
            if not root.exists():
                raise FileNotFoundError(f"Tana export root does not exist: {root}")
            if not root.is_dir():
                raise ValueError(f"Tana root_path must be a directory: {root}")
            paths = sorted(p for p in root.rglob("*.json") if p.is_file())
            if not paths:
                raise FileNotFoundError(f"No Tana JSON export files found under: {root}")
            return paths

        raise ValueError("TanaAdapter requires file_path or root_path")

    def _read_json(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed Tana JSON export at {path}: {exc.msg}") from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"Unable to read Tana JSON export at {path}: {exc}") from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _source_id(self, entity_type: str, raw_id: str) -> str:
        digest = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
        return f"tana:{entity_type}:{digest}"

    def _edge_id(self, from_id: str, to_id: str, relation_type: str) -> str:
        raw = "|".join([SourceProject.TANA.value, relation_type, from_id, to_id])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"tana-{relation_type}-{digest}"

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        if isinstance(value, (int, float)):
            # Tana may use epoch millis
            if value > 1e12:
                value = value / 1000
            return datetime.fromtimestamp(value, tz=timezone.utc)
        raw = str(value).strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except (ValueError, TypeError):
            return None

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
