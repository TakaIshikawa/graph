"""Adapter for Obsidian Canvas .canvas JSON files."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


NODE_ENTITY_TYPES = {
    "text": "canvas_text",
    "file": "canvas_file",
    "link": "canvas_link",
    "group": "canvas_group",
}


class ObsidianCanvasAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "obsidian_canvas"

    @property
    def entity_types(self) -> list[str]:
        return list(NODE_ENTITY_TYPES.values())

    def __init__(
        self,
        path: str = "",
        *,
        root_path: str = "",
        source_id_root: str | None = None,
    ) -> None:
        self.path = path or root_path
        self.source_id_root = source_id_root

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

        root = Path(self.path).expanduser() if self.path else None
        if root is None or not root.exists():
            return result

        source_root = Path(self.source_id_root).expanduser() if self.source_id_root else root
        if root.is_file() and not self.source_id_root:
            source_root = root.parent

        sync_at = self._sync_timestamp(since) if since else None
        for path in self._canvas_paths(root):
            stat = path.stat()
            if sync_at is not None and stat.st_mtime <= sync_at:
                continue

            relative_path = self._relative_path(path, source_root)
            canvas = self._read_canvas(path)
            nodes = self._nodes(canvas, path)
            source_ids = self._source_ids(nodes, relative_path)
            included_source_ids: set[str] = set()

            for node in nodes:
                entity_type = NODE_ENTITY_TYPES.get(self._text(node.get("type")))
                if entity_type is None or entity_type not in allowed_types:
                    continue
                unit = self._unit(node, relative_path, source_ids[node["id"]], stat)
                result.units.append(unit)
                included_source_ids.add(unit.source_id)

            result.edges.extend(
                self._explicit_edges(
                    canvas,
                    relative_path,
                    source_ids,
                    included_source_ids,
                )
            )
            result.edges.extend(
                self._group_edges(
                    nodes,
                    relative_path,
                    source_ids,
                    included_source_ids,
                )
            )

        result.units.sort(key=lambda unit: (unit.source_id, unit.source_entity_type))
        result.edges.sort(key=lambda edge: (edge.from_unit_id, edge.to_unit_id, edge.id))
        return result

    def _canvas_paths(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() == ".canvas" else []
        if not root.is_dir():
            return []
        return sorted(path for path in root.rglob("*.canvas") if path.is_file())

    def _read_canvas(self, path: Path) -> dict[str, Any]:
        try:
            parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed Obsidian Canvas JSON in {path}: {exc.msg}") from exc
        except UnicodeDecodeError as exc:
            raise ValueError(f"Could not decode Obsidian Canvas file {path}") from exc
        except OSError as exc:
            raise ValueError(f"Could not read Obsidian Canvas file {path}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(f"Obsidian Canvas file {path} must contain one object")
        return parsed

    def _nodes(self, canvas: dict[str, Any], path: Path) -> list[dict[str, Any]]:
        raw_nodes = canvas.get("nodes", [])
        if not isinstance(raw_nodes, list):
            raise ValueError(f"Obsidian Canvas nodes must be a list in {path}")

        nodes: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        for index, node in enumerate(raw_nodes):
            if not isinstance(node, dict):
                continue
            node_id = self._text(node.get("id")) or f"node-{index + 1}"
            if node_id in used_ids:
                node_id = f"{node_id}-{index + 1}"
            used_ids.add(node_id)
            normalized = dict(node)
            normalized["id"] = node_id
            nodes.append(normalized)
        return nodes

    def _unit(
        self,
        node: dict[str, Any],
        relative_path: str,
        source_id: str,
        stat: Any,
    ) -> KnowledgeUnit:
        node_type = self._text(node.get("type"))
        title = self._title(node)
        content = self._content(node, title)
        return KnowledgeUnit(
            source_project=SourceProject.OBSIDIAN_CANVAS,
            source_id=source_id,
            source_entity_type=NODE_ENTITY_TYPES[node_type],
            title=title,
            content=content,
            content_type=ContentType.ARTIFACT,
            metadata=self._node_metadata(node, relative_path),
            tags=["obsidian", "canvas", node_type],
            created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )

    def _explicit_edges(
        self,
        canvas: dict[str, Any],
        relative_path: str,
        source_ids: dict[str, str],
        included_source_ids: set[str],
    ) -> list[KnowledgeEdge]:
        raw_edges = canvas.get("edges", [])
        if not isinstance(raw_edges, list):
            return []

        edges: list[KnowledgeEdge] = []
        emitted: set[tuple[str, str, str]] = set()
        for index, edge in enumerate(raw_edges):
            if not isinstance(edge, dict):
                continue
            from_source_id = source_ids.get(self._text(edge.get("fromNode")))
            to_source_id = source_ids.get(self._text(edge.get("toNode")))
            if (
                not from_source_id
                or not to_source_id
                or from_source_id not in included_source_ids
                or to_source_id not in included_source_ids
            ):
                continue

            edge_id = self._text(edge.get("id")) or f"edge-{index + 1}"
            key = (from_source_id, to_source_id, edge_id)
            if key in emitted:
                continue
            emitted.add(key)
            edges.append(
                KnowledgeEdge(
                    id=self._edge_id("relates", edge_id, from_source_id, to_source_id),
                    from_unit_id=from_source_id,
                    to_unit_id=to_source_id,
                    relation=EdgeRelation.RELATES_TO,
                    source=EdgeSource.SOURCE,
                    metadata=self._edge_metadata(edge, edge_id, relative_path),
                )
            )
        return edges

    def _group_edges(
        self,
        nodes: list[dict[str, Any]],
        relative_path: str,
        source_ids: dict[str, str],
        included_source_ids: set[str],
    ) -> list[KnowledgeEdge]:
        groups = [node for node in nodes if self._text(node.get("type")) == "group"]
        edges: list[KnowledgeEdge] = []
        emitted: set[tuple[str, str]] = set()
        for group in groups:
            group_source_id = source_ids[group["id"]]
            if group_source_id not in included_source_ids:
                continue
            for node in nodes:
                if node["id"] == group["id"] or self._text(node.get("type")) == "group":
                    continue
                child_source_id = source_ids.get(node["id"])
                if not child_source_id or child_source_id not in included_source_ids:
                    continue
                if not self._contains(group, node):
                    continue
                key = (group_source_id, child_source_id)
                if key in emitted:
                    continue
                emitted.add(key)
                edges.append(
                    KnowledgeEdge(
                        id=self._edge_id("contains", group["id"], group_source_id, child_source_id),
                        from_unit_id=group_source_id,
                        to_unit_id=child_source_id,
                        relation=EdgeRelation.CONTAINS,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.OBSIDIAN_CANVAS.value,
                            "relation_type": "canvas_group_contains",
                            "canvas_path": relative_path,
                            "group_node_id": group["id"],
                            "child_node_id": node["id"],
                            "group_label": self._text(group.get("label")),
                        },
                    )
                )
        return edges

    def _source_ids(self, nodes: list[dict[str, Any]], relative_path: str) -> dict[str, str]:
        return {
            node["id"]: f"obsidian_canvas:{relative_path}#{node['id']}"
            for node in nodes
            if self._text(node.get("type")) in NODE_ENTITY_TYPES
        }

    def _node_metadata(self, node: dict[str, Any], relative_path: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "source_project": SourceProject.OBSIDIAN_CANVAS.value,
            "canvas_path": relative_path,
            "node_id": node["id"],
            "type": self._text(node.get("type")),
        }
        preserved_keys = (
            "x",
            "y",
            "width",
            "height",
            "color",
            "label",
            "file",
            "path",
            "url",
            "subpath",
        )
        for key in preserved_keys:
            if key in node:
                metadata[key] = self._jsonable(node[key])
        return metadata

    def _edge_metadata(
        self,
        edge: dict[str, Any],
        edge_id: str,
        relative_path: str,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "source_project": SourceProject.OBSIDIAN_CANVAS.value,
            "relation_type": "canvas_edge",
            "canvas_path": relative_path,
            "edge_id": edge_id,
            "from_node": self._text(edge.get("fromNode")),
            "to_node": self._text(edge.get("toNode")),
        }
        for key in ("fromSide", "toSide", "color", "label"):
            if key in edge:
                metadata[key] = self._jsonable(edge[key])
        return metadata

    def _title(self, node: dict[str, Any]) -> str:
        node_type = self._text(node.get("type"))
        label = self._text(node.get("label"))
        if label:
            return label
        if node_type == "text":
            first_line = next(
                (
                    line.strip()
                    for line in self._text(node.get("text")).splitlines()
                    if line.strip()
                ),
                "",
            )
            return first_line[:80] or "Untitled Canvas text"
        if node_type == "file":
            file_path = self._text(node.get("file"))
            return Path(file_path).name or "Untitled Canvas file"
        if node_type == "link":
            return self._text(node.get("url")) or "Untitled Canvas link"
        if node_type == "group":
            return "Untitled Canvas group"
        return "Untitled Canvas node"

    def _content(self, node: dict[str, Any], title: str) -> str:
        node_type = self._text(node.get("type"))
        if node_type == "text":
            return self._text(node.get("text")) or title
        if node_type == "file":
            parts = [self._text(node.get("file"))]
            subpath = self._text(node.get("subpath"))
            if subpath:
                parts.append(subpath)
            return "\n".join(part for part in parts if part) or title
        if node_type == "link":
            return self._text(node.get("url")) or title
        return title

    def _contains(self, group: dict[str, Any], node: dict[str, Any]) -> bool:
        group_box = self._box(group)
        node_box = self._box(node)
        if group_box is None or node_box is None:
            return False
        gx, gy, gw, gh = group_box
        nx, ny, nw, nh = node_box
        return nx >= gx and ny >= gy and nx + nw <= gx + gw and ny + nh <= gy + gh

    def _box(self, node: dict[str, Any]) -> tuple[float, float, float, float] | None:
        values = [self._float(node.get(key)) for key in ("x", "y", "width", "height")]
        if any(value is None for value in values):
            return None
        x, y, width, height = values
        return (x or 0.0, y or 0.0, width or 0.0, height or 0.0)

    def _edge_id(
        self,
        relation_type: str,
        edge_id: str,
        from_source_id: str,
        to_source_id: str,
    ) -> str:
        raw = "|".join(
            [
                SourceProject.OBSIDIAN_CANVAS.value,
                relation_type,
                edge_id,
                from_source_id,
                to_source_id,
            ]
        )
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"obsidian-canvas-{relation_type}-{digest}"

    def _relative_path(self, path: Path, source_root: Path) -> str:
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _sync_timestamp(self, since: SyncState) -> float:
        if isinstance(since.last_sync_at, datetime):
            return since.last_sync_at.timestamp()
        return datetime.fromisoformat(str(since.last_sync_at).replace("Z", "+00:00")).timestamp()

    def _text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True)
        return str(value).strip()

    def _float(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _jsonable(self, value: Any) -> Any:
        try:
            json.dumps(value)
        except TypeError:
            return str(value)
        return value
