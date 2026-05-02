"""Graph service using NetworkX for in-memory graph algorithms."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
import hashlib
from itertools import combinations
import html
import json
import math
from pathlib import Path
import re
import shutil
from urllib.parse import quote, urlsplit, urlunsplit

import networkx as nx
import yaml

from graph.store.db import Store
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


_NORMALIZED_TEXT_RE = re.compile(r"[^a-z0-9]+")
_TTL_LOCAL_NAME_RE = re.compile(r"[^A-Za-z0-9_]")
_EXTERNAL_URL_RE = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)
_TRAILING_URL_PUNCTUATION = ".,;:!?)]}'\""
_TIMELINE_BUCKETS = {"day", "week", "month", "year"}
_TIMELINE_FIELDS = {"created_at", "ingested_at", "updated_at"}
_MERMAID_WHITESPACE_RE = re.compile(r"\s+")
_MARKDOWN_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_EDGE_SUGGESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}
_TAG_SUGGESTION_STOPWORDS = _EDGE_SUGGESTION_STOPWORDS | {
    "about",
    "after",
    "all",
    "also",
    "can",
    "into",
    "its",
    "more",
    "not",
    "our",
    "their",
    "there",
    "these",
    "they",
    "was",
    "will",
    "you",
    "your",
}
_REFERENCE_URL_METADATA_FIELDS = {"url", "link", "canonical_url", "source_url", "source_id"}
_DUPLICATE_URL_METADATA_FIELDS = {"canonical_url", "link"}


def _normalize_text(value: str) -> str:
    return _NORMALIZED_TEXT_RE.sub(" ", value.lower()).strip()


def _singularize_token(value: str) -> str:
    if len(value) > 4 and value.endswith("ies"):
        return f"{value[:-3]}y"
    if len(value) > 4 and value.endswith("ses"):
        return value[:-2]
    if len(value) > 3 and value.endswith("s"):
        return value[:-1]
    return value


def _normalize_tag_variant(value: str) -> str:
    tokens = [_singularize_token(token) for token in _normalize_text(value).split()]
    return " ".join(token for token in tokens if token)


def _tag_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def _content_tokens(value: str) -> Counter[str]:
    return Counter(_normalize_text(value).split())


def _edge_suggestion_tokens(*values: str) -> set[str]:
    tokens = set()
    for value in values:
        without_urls = _EXTERNAL_URL_RE.sub(" ", value)
        for token in _normalize_text(without_urls).split():
            if len(token) < 3 or token in _EDGE_SUGGESTION_STOPWORDS:
                continue
            tokens.add(_singularize_token(token))
    return tokens


def _tag_suggestion_tokens(*values: str) -> set[str]:
    tokens = set()
    for value in values:
        for token in _normalize_text(value or "").split():
            normalized = _singularize_token(token)
            if len(normalized) < 2 or normalized in _TAG_SUGGESTION_STOPWORDS:
                continue
            tokens.add(normalized)
    return tokens


def _counter_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = sum((left & right).values())
    total = max(sum(left.values()), sum(right.values()))
    return overlap / total if total else 0.0


def _turtle_literal(value: object) -> str:
    text = str(value)
    escaped = []
    for char in text:
        codepoint = ord(char)
        if char == "\\":
            escaped.append("\\\\")
        elif char == '"':
            escaped.append('\\"')
        elif char == "\n":
            escaped.append("\\n")
        elif char == "\r":
            escaped.append("\\r")
        elif char == "\t":
            escaped.append("\\t")
        elif codepoint < 0x20:
            escaped.append(f"\\u{codepoint:04X}")
        else:
            escaped.append(char)
    return f'"{"".join(escaped)}"'


def _turtle_local_name(value: object) -> str:
    name = _TTL_LOCAL_NAME_RE.sub("_", str(value)).strip("_")
    if not name or not re.match(r"[A-Za-z_]", name):
        name = f"rel_{name}"
    return name


def _unit_uri(base_uri: str, unit_id: str) -> str:
    return f"<{base_uri}{quote(unit_id, safe='')}>"


def _normalize_external_url(value: str) -> str | None:
    url = value.rstrip(_TRAILING_URL_PUNCTUATION)
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        return None
    netloc = parsed.netloc.lower()
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path, parsed.query, parsed.fragment))


def _external_url_domain(url: str) -> str | None:
    hostname = urlsplit(url).hostname
    return hostname.lower().rstrip(".") if hostname else None


def _json_value(value: object) -> object:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _markdown_filename_stem(title: str) -> str:
    stem = _MARKDOWN_FILENAME_RE.sub("-", title.strip().lower())
    stem = re.sub(r"-{2,}", "-", stem).strip(" .-_")
    return stem[:80].strip(" .-_") or "untitled"


def _mermaid_label(value: object) -> str:
    text = _MERMAID_WHITESPACE_RE.sub(" ", str(value)).strip()
    return html.escape(text, quote=True).replace("|", "&#124;")


def _markdown_inline(value: object) -> str:
    return " ".join(str(value).split())


def _markdown_heading(value: object) -> str:
    text = _markdown_inline(value)
    return text.replace("#", r"\#") or "Untitled"


def _metadata_strings(value: object, path: str = "metadata") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        strings = []
        for key, child in value.items():
            strings.extend(_metadata_strings(child, f"{path}.{key}"))
        return strings
    if isinstance(value, list):
        strings = []
        for index, child in enumerate(value):
            strings.extend(_metadata_strings(child, f"{path}[{index}]"))
        return strings
    return []


def _unit_external_urls(unit) -> set[str]:
    urls = set()
    fields = [("content", unit.content)]
    fields.extend(_metadata_strings(unit.metadata))
    for _, text in fields:
        for match in _EXTERNAL_URL_RE.finditer(text):
            url = _normalize_external_url(match.group(0))
            if url is not None:
                urls.add(url)
    return urls


def _extract_urls_from_text(text: str) -> set[str]:
    urls = set()
    for match in _EXTERNAL_URL_RE.finditer(text or ""):
        url = _normalize_external_url(match.group(0))
        if url is not None:
            urls.add(url)
    return urls


def _metadata_url_field_values(value: object, path: str = "metadata") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        strings = []
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if (
                str(key).lower() in _REFERENCE_URL_METADATA_FIELDS
                and isinstance(child, str)
            ):
                strings.append((child_path, child))
            strings.extend(_metadata_url_field_values(child, child_path))
        return strings
    if isinstance(value, list):
        strings = []
        for index, child in enumerate(value):
            strings.extend(_metadata_url_field_values(child, f"{path}[{index}]"))
        return strings
    return []


def _metadata_duplicate_url_values(value: object, path: str = "metadata") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        strings = []
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() in _DUPLICATE_URL_METADATA_FIELDS and isinstance(child, str):
                strings.append((str(key).lower(), child))
            strings.extend(_metadata_duplicate_url_values(child, child_path))
        return strings
    if isinstance(value, list):
        strings = []
        for index, child in enumerate(value):
            strings.extend(_metadata_duplicate_url_values(child, f"{path}[{index}]"))
        return strings
    return []


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_timeline_datetime(value: str | datetime | None, *, name: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value).strip()
        if raw.endswith("Z"):
            raw = f"{raw[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO-8601 date or datetime.") from exc
    return _ensure_aware(parsed)


def _timeline_bucket_start(value: datetime, bucket: str) -> datetime:
    value = _ensure_aware(value)
    if bucket == "day":
        return value.replace(hour=0, minute=0, second=0, microsecond=0)
    if bucket == "week":
        day = value.replace(hour=0, minute=0, second=0, microsecond=0)
        return day - timedelta(days=day.weekday())
    if bucket == "month":
        return value.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if bucket == "year":
        return value.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(f"Unsupported timeline bucket: {bucket}. Use day, week, month, or year.")


def _timeline_bucket_end(start: datetime, bucket: str) -> datetime:
    if bucket == "day":
        return start + timedelta(days=1)
    if bucket == "week":
        return start + timedelta(days=7)
    if bucket == "month":
        if start.month == 12:
            return start.replace(year=start.year + 1, month=1)
        return start.replace(month=start.month + 1)
    if bucket == "year":
        return start.replace(year=start.year + 1)
    raise ValueError(f"Unsupported timeline bucket: {bucket}. Use day, week, month, or year.")


def _timeline_bucket_label(start: datetime, bucket: str) -> str:
    if bucket == "day":
        return start.date().isoformat()
    if bucket == "week":
        year, week, _ = start.isocalendar()
        return f"{year}-W{week:02d}"
    if bucket == "month":
        return f"{start.year:04d}-{start.month:02d}"
    if bucket == "year":
        return f"{start.year:04d}"
    raise ValueError(f"Unsupported timeline bucket: {bucket}. Use day, week, month, or year.")


class GraphService:
    """In-memory NetworkX graph built from SQLite for graph algorithms."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.G: nx.DiGraph = nx.DiGraph()

    def rebuild(self) -> int:
        """Rebuild NetworkX graph from SQLite. Returns node count."""
        self.G.clear()
        units = self.store.get_all_units()
        for u in units:
            self.G.add_node(
                u.id,
                title=u.title,
                source_project=u.source_project,
                source_entity_type=u.source_entity_type,
                content_type=u.content_type,
                confidence=u.confidence or 0.0,
                utility_score=u.utility_score or 0.0,
                tags=u.tags,
                created_at=_json_value(u.created_at),
                updated_at=_json_value(u.updated_at),
            )
        edges = self.store.get_all_edges()
        for e in edges:
            if e.from_unit_id in self.G and e.to_unit_id in self.G:
                self.G.add_edge(
                    e.from_unit_id,
                    e.to_unit_id,
                    relation=e.relation,
                    weight=e.weight,
                    source=e.source,
                    created_at=_json_value(e.created_at),
                    id=e.id,
                )
        return len(self.G.nodes)

    def build_export_graph(self) -> nx.DiGraph:
        """Build a GraphML-safe graph containing scalar export attributes only."""
        export_graph = nx.DiGraph()
        for node_id, data in self.G.nodes(data=True):
            tags = data.get("tags") or []
            if isinstance(tags, list):
                tags_value = ",".join(str(tag) for tag in tags)
            else:
                tags_value = str(tags)
            export_graph.add_node(
                node_id,
                title=str(data.get("title", "")),
                source_project=str(data.get("source_project", "")),
                source_entity_type=str(data.get("source_entity_type", "")),
                content_type=str(data.get("content_type", "")),
                tags=tags_value,
                confidence=float(data.get("confidence", 0.0) or 0.0),
                utility_score=float(data.get("utility_score", 0.0) or 0.0),
                created_at=str(data.get("created_at", "")),
                updated_at=str(data.get("updated_at", "")),
            )
        for from_id, to_id, data in self.G.edges(data=True):
            export_graph.add_edge(
                from_id,
                to_id,
                relation=str(data.get("relation", "")),
                weight=float(data.get("weight", 1.0) or 0.0),
                source=str(data.get("source", "")),
                created_at=str(data.get("created_at", "")),
            )
        return export_graph

    def export_graphml(self, path: str | Path) -> dict:
        """Write the current graph to a GraphML file and return export stats."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_graph = self.build_export_graph()
        nx.write_graphml(export_graph, output_path)
        return {
            "path": str(output_path),
            "node_count": export_graph.number_of_nodes(),
            "edge_count": export_graph.number_of_edges(),
        }

    def export_gexf(self, path: str | Path) -> dict:
        """Write the current graph to a GEXF file for Gephi and return export stats."""
        if not self.G:
            self.rebuild()

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_graph = self.build_export_graph()
        nx.write_gexf(export_graph, output_path)
        return {
            "path": str(output_path),
            "nodes_exported": export_graph.number_of_nodes(),
            "edges_exported": export_graph.number_of_edges(),
            "bytes_written": output_path.stat().st_size,
        }

    def export_mermaid(
        self,
        path: str | Path,
        *,
        unit_id: str | None = None,
        depth: int = 1,
        limit: int = 100,
    ) -> dict:
        """Write a Markdown Mermaid graph block and return export stats."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        capped_limit = max(1, int(limit))

        if unit_id is not None:
            payload = self.build_neighborhood_export(unit_id, depth=depth)
            center_id = payload["center"]["id"] if payload["center"] else unit_id
            units_by_id = {unit["id"]: unit for unit in payload["units"]}
            distances = nx.single_source_shortest_path_length(
                self.G.to_undirected(), center_id, cutoff=payload["depth"]
            )
            ordered_ids = sorted(
                units_by_id,
                key=lambda found_id: (distances.get(found_id, payload["depth"] + 1), found_id),
            )
            selected_ids = set(ordered_ids[:capped_limit])
            units = [
                units_by_id[found_id] for found_id in ordered_ids if found_id in selected_ids
            ]
            edges = [
                edge
                for edge in payload["edges"]
                if edge["from_unit_id"] in selected_ids and edge["to_unit_id"] in selected_ids
            ]
            capped = len(payload["units"]) > len(units)
            depth_used = payload["depth"]
        else:
            all_units = sorted(self.store.get_all_units(limit=1000000000), key=lambda unit: unit.id)
            units = [self._unit_export_data(unit) for unit in all_units[:capped_limit]]
            selected_ids = {unit["id"] for unit in units}
            edges = [
                self._edge_export_data(edge)
                for edge in sorted(
                    self.store.get_all_edges(),
                    key=lambda edge: (
                        edge.from_unit_id,
                        edge.to_unit_id,
                        str(edge.relation),
                        edge.id,
                    ),
                )
                if edge.from_unit_id in selected_ids and edge.to_unit_id in selected_ids
            ]
            capped = len(all_units) > len(units)
            depth_used = None

        aliases = {unit["id"]: f"n{index}" for index, unit in enumerate(units)}
        lines = ["```mermaid", "graph TD"]
        for unit in units:
            lines.append(f'    {aliases[unit["id"]]}["{_mermaid_label(unit["title"])}"]')
        for edge in edges:
            from_alias = aliases[edge["from_unit_id"]]
            to_alias = aliases[edge["to_unit_id"]]
            relation = _mermaid_label(edge["relation"])
            lines.append(f"    {from_alias} -->|{relation}| {to_alias}")
        lines.extend(["```", ""])

        output_path.write_text("\n".join(lines), encoding="utf-8")
        stats = {
            "path": str(output_path),
            "node_count": len(units),
            "edge_count": len(edges),
            "capped": capped,
        }
        if unit_id is not None:
            stats["depth"] = depth_used
            stats["center_unit_id"] = unit_id
        return stats

    def export_cytoscape(
        self,
        path: str | Path,
        *,
        unit_id: str | None = None,
        depth: int = 1,
        limit: int = 100,
    ) -> dict:
        """Write Cytoscape.js elements JSON and return export stats."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        capped_limit = max(1, int(limit))

        if unit_id is not None:
            payload = self.build_neighborhood_export(unit_id, depth=depth)
            units = sorted(payload["units"], key=lambda unit: unit["id"])
            selected_ids = {unit["id"] for unit in units}
            edges = [
                edge
                for edge in payload["edges"]
                if edge["from_unit_id"] in selected_ids and edge["to_unit_id"] in selected_ids
            ]
            mode = "neighborhood"
            capped = False
            depth_used = payload["depth"]
        else:
            all_units = sorted(self.store.get_all_units(limit=1000000000), key=lambda unit: unit.id)
            units = [self._unit_export_data(unit) for unit in all_units[:capped_limit]]
            selected_ids = {unit["id"] for unit in units}
            edges = [
                self._edge_export_data(edge)
                for edge in sorted(
                    self.store.get_all_edges(),
                    key=lambda edge: (
                        edge.from_unit_id,
                        edge.to_unit_id,
                        str(edge.relation),
                        edge.id,
                    ),
                )
                if edge.from_unit_id in selected_ids and edge.to_unit_id in selected_ids
            ]
            mode = "whole_graph"
            capped = len(all_units) > len(units)
            depth_used = None

        elements = {
            "nodes": [
                {
                    "data": {
                        "id": unit["id"],
                        "label": unit["title"],
                        "title": unit["title"],
                        "source_project": unit["source_project"],
                        "content_type": unit["content_type"],
                        "tags": unit["tags"],
                        "utility_score": unit["utility_score"],
                        "confidence": unit["confidence"],
                        "created_at": unit["created_at"],
                        "updated_at": unit["updated_at"],
                    }
                }
                for unit in units
            ],
            "edges": [
                {
                    "data": {
                        "id": edge["id"],
                        "source": edge["from_unit_id"],
                        "target": edge["to_unit_id"],
                        "relation": edge["relation"],
                        "weight": edge["weight"],
                        "edge_source": edge["source"],
                        "created_at": edge["created_at"],
                    }
                }
                for edge in edges
            ],
        }
        output_path.write_text(
            json.dumps({"elements": elements}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        stats = {
            "path": str(output_path),
            "node_count": len(units),
            "edge_count": len(edges),
            "mode": mode,
            "capped": capped,
        }
        if unit_id is not None:
            stats["depth"] = depth_used
            stats["center_unit_id"] = unit_id
        return stats

    def export_link_markdown(
        self,
        path: str | Path,
        *,
        unit_id: str | None = None,
        depth: int = 1,
    ) -> dict:
        """Write a deterministic Markdown report of incoming and outgoing links."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if unit_id is not None:
            payload = self.build_neighborhood_export(unit_id, depth=depth)
            center_unit_id = unit_id
            units = list(payload["units"])
            edges = list(payload["edges"])
            mode = "neighborhood"
            depth_used = payload["depth"]
        else:
            center_unit_id = None
            all_units = sorted(
                self.store.get_all_units(limit=1000000000),
                key=lambda unit: unit.id,
            )
            units = [self._unit_export_data(unit) for unit in all_units]
            unit_ids = {unit["id"] for unit in units}
            edges = [
                self._edge_export_data(edge)
                for edge in self.store.get_all_edges()
                if edge.from_unit_id in unit_ids and edge.to_unit_id in unit_ids
            ]
            mode = "whole_graph"
            depth_used = None

        units = sorted(
            units,
            key=lambda unit: (_markdown_inline(unit["title"]).lower(), unit["id"]),
        )
        units_by_id = {unit["id"]: unit for unit in units}
        edges = sorted(
            edges,
            key=lambda edge: (
                _markdown_inline(edge["relation"]).lower(),
                _markdown_inline(units_by_id[edge["from_unit_id"]]["title"]).lower(),
                edge["from_unit_id"],
                _markdown_inline(units_by_id[edge["to_unit_id"]]["title"]).lower(),
                edge["to_unit_id"],
                edge["id"],
            ),
        )

        outgoing_edges: dict[str, list[dict]] = {unit["id"]: [] for unit in units}
        incoming_edges: dict[str, list[dict]] = {unit["id"]: [] for unit in units}
        for edge in edges:
            outgoing_edges[edge["from_unit_id"]].append(edge)
            incoming_edges[edge["to_unit_id"]].append(edge)

        def edge_line(edge: dict, linked_unit_id: str, arrow: str) -> str:
            linked_unit = units_by_id[linked_unit_id]
            relation = _markdown_inline(edge["relation"])
            linked_title = _markdown_inline(linked_unit["title"])
            return f"- `{relation}` {arrow} {linked_title} (`{linked_unit_id}`)"

        lines = [
            "# Graph Link Report",
            "",
            f"- Scope: {mode}",
            f"- Units exported: {len(units)}",
            f"- Edges exported: {len(edges)}",
        ]
        if unit_id is not None:
            lines.extend(
                [
                    f"- Center unit ID: `{unit_id}`",
                    f"- Depth: {depth_used}",
                ]
            )
        lines.append("")

        for unit in units:
            unit_id_value = unit["id"]
            lines.extend(
                [
                    f"## {_markdown_heading(unit['title'])}",
                    "",
                    f"- Unit ID: `{unit_id_value}`",
                    f"- Source: {unit['source_project']}/{unit['source_entity_type']}",
                    "",
                    "### Outgoing",
                    "",
                ]
            )
            if outgoing_edges[unit_id_value]:
                lines.extend(
                    edge_line(edge, edge["to_unit_id"], "to")
                    for edge in outgoing_edges[unit_id_value]
                )
            else:
                lines.append("- _None._")
            lines.extend(["", "### Incoming", ""])
            if incoming_edges[unit_id_value]:
                lines.extend(
                    edge_line(edge, edge["from_unit_id"], "from")
                    for edge in incoming_edges[unit_id_value]
                )
            else:
                lines.append("- _None._")
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        stats = {
            "path": str(output_path),
            "units_exported": len(units),
            "edges_exported": len(edges),
        }
        if center_unit_id is not None:
            stats["depth"] = depth_used
            stats["center_unit_id"] = center_unit_id
        return stats

    def export_turtle(
        self, path: str | Path, base_uri: str = "https://graph.local/unit/"
    ) -> dict:
        """Write the current graph to RDF Turtle and return export stats."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        units = sorted(self.store.get_all_units(), key=lambda unit: unit.id)
        unit_ids = {unit.id for unit in units}
        edges = sorted(
            (
                edge
                for edge in self.store.get_all_edges()
                if edge.from_unit_id in unit_ids and edge.to_unit_id in unit_ids
            ),
            key=lambda edge: (edge.from_unit_id, str(edge.relation), edge.to_unit_id),
        )
        outgoing_edges: dict[str, list] = {}
        for edge in edges:
            outgoing_edges.setdefault(edge.from_unit_id, []).append(edge)

        lines = [
            "@prefix graph: <https://graph.local/schema#> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "",
        ]

        for unit in units:
            predicates = [
                "a graph:KnowledgeUnit",
                f"graph:title {_turtle_literal(unit.title)}",
                f"graph:sourceProject {_turtle_literal(unit.source_project)}",
                f"graph:sourceId {_turtle_literal(unit.source_id)}",
                f"graph:sourceEntityType {_turtle_literal(unit.source_entity_type)}",
                f"graph:contentType {_turtle_literal(unit.content_type)}",
                f"graph:contentSnippet {_turtle_literal(unit.content[:240])}",
                f"graph:createdAt {_turtle_literal(unit.created_at.isoformat())}^^xsd:dateTime",
            ]
            if unit.utility_score is not None:
                predicates.append(
                    f"graph:utilityScore {_turtle_literal(unit.utility_score)}^^xsd:double"
                )
            for tag in unit.tags:
                predicates.append(f"graph:tag {_turtle_literal(tag)}")
            for edge in outgoing_edges.get(unit.id, []):
                relation = _turtle_local_name(edge.relation)
                predicates.append(
                    f"graph:{relation} {_unit_uri(base_uri, edge.to_unit_id)}"
                )

            lines.append(_unit_uri(base_uri, unit.id))
            for index, predicate in enumerate(predicates):
                terminator = " ." if index == len(predicates) - 1 else " ;"
                lines.append(f"    {predicate}{terminator}")
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return {
            "path": str(output_path),
            "node_count": len(units),
            "edge_count": len(edges),
            "base_uri": base_uri,
        }

    def export_markdown_folder(
        self,
        path: str | Path,
        *,
        clean: bool = False,
        tag: str | None = None,
        source_project: str | None = None,
        content_type: str | None = None,
    ) -> dict:
        """Write one portable Markdown file per matching unit."""
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        if clean:
            for child in output_path.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()

        units = [
            unit
            for unit in self.store.get_all_units(limit=1000000000)
            if (source_project is None or str(unit.source_project) == source_project)
            and (content_type is None or str(unit.content_type) == content_type)
            and (tag is None or tag in unit.tags)
        ]
        units.sort(
            key=lambda unit: (
                str(unit.source_project),
                str(unit.content_type),
                unit.title.lower(),
                unit.id,
            )
        )

        files: list[str] = []
        used_names: set[str] = set()
        for unit in units:
            source_key = ":".join(
                [
                    str(unit.source_project),
                    unit.source_entity_type,
                    unit.source_id,
                    unit.id,
                ]
            )
            digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:10]
            base_name = f"{_markdown_filename_stem(unit.title)}--{digest}"
            filename = f"{base_name}.md"
            if filename in used_names:
                collision_index = 2
                while f"{base_name}-{collision_index}.md" in used_names:
                    collision_index += 1
                filename = f"{base_name}-{collision_index}.md"
            used_names.add(filename)

            front_matter = {
                "id": unit.id,
                "source_project": str(unit.source_project),
                "source_id": unit.source_id,
                "source_entity_type": unit.source_entity_type,
                "content_type": str(unit.content_type),
                "tags": list(unit.tags),
                "confidence": unit.confidence,
                "utility_score": unit.utility_score,
                "created_at": _json_value(unit.created_at),
                "updated_at": _json_value(unit.updated_at),
                "metadata": unit.metadata,
            }
            text = "\n".join(
                [
                    "---",
                    yaml.safe_dump(
                        front_matter,
                        sort_keys=False,
                        allow_unicode=True,
                    ).rstrip(),
                    "---",
                    "",
                    f"# {unit.title}",
                    "",
                    unit.content.rstrip(),
                    "",
                ]
            )
            unit_path = output_path / filename
            unit_path.write_text(text, encoding="utf-8")
            files.append(filename)

        return {
            "path": str(output_path),
            "units_exported": len(units),
            "files_written": len(files),
            "filters": {
                "tag": tag,
                "source_project": source_project,
                "content_type": content_type,
            },
            "clean": clean,
            "files": files,
        }

    def _unit_export_data(self, unit) -> dict:
        return {
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

    def _edge_export_data(self, edge) -> dict:
        return {
            "id": edge.id,
            "from_unit_id": edge.from_unit_id,
            "to_unit_id": edge.to_unit_id,
            "relation": str(edge.relation),
            "weight": edge.weight,
            "source": str(edge.source),
            "metadata": edge.metadata,
            "created_at": _json_value(edge.created_at),
        }

    def _unit_summary_data(self, unit) -> dict | None:
        if unit is None:
            return None
        return {
            "id": unit.id,
            "source_project": str(unit.source_project),
            "source_id": unit.source_id,
            "source_entity_type": unit.source_entity_type,
            "title": unit.title,
            "content_type": str(unit.content_type),
        }

    def _edge_with_endpoint_summaries(self, edge) -> dict:
        return {
            **self._edge_export_data(edge),
            "from_unit": self._unit_summary_data(self.store.get_unit(edge.from_unit_id)),
            "to_unit": self._unit_summary_data(self.store.get_unit(edge.to_unit_id)),
        }

    def delete_edges_bulk(
        self,
        *,
        relation: str | None = None,
        source: str | None = None,
        from_unit_id: str | None = None,
        to_unit_id: str | None = None,
        source_project: str | None = None,
        limit: int | None = None,
        dry_run: bool = True,
        confirm: bool = False,
    ) -> dict:
        if not dry_run and not confirm:
            return {
                "dry_run": dry_run,
                "confirmed": confirm,
                "matched_count": 0,
                "deleted_count": 0,
                "edges": [],
                "error": "confirmation_required",
                "message": "Bulk edge deletion requires confirm=true when dry_run=false.",
                "filters": {
                    "relation": relation,
                    "source": source,
                    "from_unit_id": from_unit_id,
                    "to_unit_id": to_unit_id,
                    "source_project": source_project,
                    "limit": limit,
                },
            }

        filters = {
            "relation": relation,
            "source": source,
            "from_unit_id": from_unit_id,
            "to_unit_id": to_unit_id,
            "source_project": source_project,
            "limit": limit,
        }
        if dry_run:
            edges = self.store.find_edges(**filters)
            deleted_count = 0
        else:
            edges = self.store.delete_edges(**filters)
            deleted_count = len(edges)

        return {
            "dry_run": dry_run,
            "confirmed": confirm,
            "matched_count": len(edges),
            "deleted_count": deleted_count,
            "edges": [self._edge_with_endpoint_summaries(edge) for edge in edges],
            "filters": filters,
        }

    def build_neighborhood_export(self, unit_id: str, depth: int = 1) -> dict:
        """Build a portable JSON payload for one unit's local neighborhood."""
        capped_depth = max(1, min(depth, 3))
        if not self.G:
            self.rebuild()

        result = self.get_neighbors(unit_id, depth=capped_depth)
        if result["center"] is None:
            raise ValueError(
                json.dumps(
                    {
                        "error": "unit_not_found",
                        "message": f"Unit not found: {unit_id}",
                        "unit_id": unit_id,
                    }
                )
            )

        unit_ids = {unit_id, *result["neighbors"]}
        units = [
            unit
            for unit in (self.store.get_unit(found_id) for found_id in sorted(unit_ids))
            if unit is not None
        ]
        edges = sorted(
            (
                edge
                for edge in self.store.get_all_edges()
                if edge.from_unit_id in unit_ids and edge.to_unit_id in unit_ids
            ),
            key=lambda edge: (
                edge.from_unit_id,
                edge.to_unit_id,
                str(edge.relation),
                edge.id,
            ),
        )
        center = self.store.get_unit(unit_id)

        return {
            "schema_version": 1,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "center": self._unit_export_data(center) if center else None,
            "units": [self._unit_export_data(unit) for unit in units],
            "edges": [self._edge_export_data(edge) for edge in edges],
            "depth": capped_depth,
        }

    def get_backlinks(
        self,
        unit_id: str,
        *,
        relation: str | None = None,
        source_project: str | None = None,
        content_type: str | None = None,
        tag: str | None = None,
        limit: int = 20,
    ) -> dict:
        """Return incoming references to a unit with source unit summaries."""
        result = self.store.get_backlinks(
            unit_id,
            direction="incoming",
            relation=relation,
            source_project=source_project,
            content_type=content_type,
            tag=tag,
            limit=limit,
        )
        if result["center"] is None:
            return {
                "unit_id": unit_id,
                "center": None,
                "links": [],
                "error": "unit_not_found",
                "message": f"Unit not found: {unit_id}",
            }

        links = []
        for link in result["links"]:
            edge = link["edge"]
            source_unit = link["unit"]
            links.append(
                {
                    "relation": link["relation"],
                    "edge": self._edge_export_data(edge),
                    "source_unit": self._unit_export_data(source_unit),
                }
            )
        return {
            "unit_id": unit_id,
            "center": self._unit_export_data(result["center"]),
            "links": links,
            "filters": {
                "relation": relation,
                "source_project": source_project,
                "content_type": content_type,
                "tag": tag,
                "limit": max(0, limit),
            },
        }

    def export_neighborhood(
        self, unit_id: str, path: str | Path, depth: int = 1
    ) -> dict:
        """Write one unit's local subgraph to JSON and return export stats."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.build_neighborhood_export(unit_id, depth=depth)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "path": str(output_path),
            "unit_count": len(payload["units"]),
            "edge_count": len(payload["edges"]),
            "depth": payload["depth"],
            "center_unit_id": unit_id,
        }

    def get_neighbors(self, unit_id: str, depth: int = 1) -> dict:
        """Get unit and neighbors up to depth hops."""
        if unit_id not in self.G:
            return {"center": None, "neighbors": [], "edges": []}

        if depth == 1:
            neighbor_ids = set(self.G.predecessors(unit_id)) | set(
                self.G.successors(unit_id)
            )
        else:
            neighbor_ids = (
                set(
                    nx.single_source_shortest_path_length(
                        self.G.to_undirected(), unit_id, cutoff=depth
                    ).keys()
                )
                - {unit_id}
            )

        all_ids = neighbor_ids | {unit_id}
        edge_list = [
            {"from": u, "to": v, **d}
            for u, v, d in self.G.edges(data=True)
            if u in all_ids and v in all_ids
        ]

        return {
            "center": unit_id,
            "neighbors": list(neighbor_ids),
            "edges": edge_list,
        }

    def get_ego_metrics(self, unit_id: str, depth: int = 1) -> dict:
        """Compute stable metrics for one unit's ego network."""
        if not self.G:
            self.rebuild()

        capped_depth = max(1, min(int(depth), 3))
        if unit_id not in self.G:
            return {
                "unit_id": unit_id,
                "center": None,
                "metrics": {},
                "relation_counts": {},
                "depth": capped_depth,
                "error": "unit_not_found",
                "message": f"Unit not found: {unit_id}",
            }

        undirected = self.G.to_undirected()
        reachable = (
            set(
                nx.single_source_shortest_path_length(
                    undirected,
                    unit_id,
                    cutoff=capped_depth,
                ).keys()
            )
            - {unit_id}
        )
        relation_counts = Counter()
        for edge in self.store.get_all_edges():
            if edge.from_unit_id == unit_id or edge.to_unit_id == unit_id:
                relation_counts[str(edge.relation)] += 1

        bridge_score = 0.0
        if self.G.nodes:
            bridge_score = nx.betweenness_centrality(undirected).get(unit_id, 0.0)

        center = self.store.get_unit(unit_id)
        return {
            "unit_id": unit_id,
            "center": self._unit_summary_data(center),
            "metrics": {
                "degree": int(self.G.degree(unit_id)),
                "in_degree": int(self.G.in_degree(unit_id)),
                "out_degree": int(self.G.out_degree(unit_id)),
                "reachable_neighbor_count": len(reachable),
                "local_clustering_coefficient": nx.clustering(undirected, unit_id),
                "bridge_score": bridge_score,
            },
            "relation_counts": dict(sorted(relation_counts.items())),
            "depth": capped_depth,
        }

    def shortest_path(self, from_id: str, to_id: str) -> list[str] | None:
        """Find shortest path between two units."""
        try:
            return nx.shortest_path(self.G.to_undirected(), from_id, to_id)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return None

    def shortest_path_between(
        self,
        source_unit_id: str,
        target_unit_id: str,
        *,
        relation: str | None = None,
        max_paths: int = 1,
    ) -> list[dict]:
        """Explain deterministic shortest paths between two units."""
        if (
            not isinstance(max_paths, int)
            or isinstance(max_paths, bool)
            or max_paths < 0
        ):
            raise ValueError("max_paths must be a non-negative integer.")

        units_by_id = {
            unit.id: unit for unit in self.store.get_all_units(limit=1000000000)
        }
        missing_messages = []
        if source_unit_id not in units_by_id:
            missing_messages.append(f"source_unit_id not found: {source_unit_id}")
        if target_unit_id not in units_by_id:
            missing_messages.append(f"target_unit_id not found: {target_unit_id}")
        if missing_messages:
            raise ValueError("; ".join(missing_messages))
        if max_paths == 0:
            return []

        relation_filter = str(relation) if relation is not None else None
        projection = nx.Graph()
        projection.add_nodes_from(units_by_id)
        for edge in sorted(
            self.store.get_all_edges(),
            key=lambda item: (
                min(item.from_unit_id, item.to_unit_id),
                max(item.from_unit_id, item.to_unit_id),
                str(item.relation),
                item.id,
            ),
        ):
            if edge.from_unit_id not in units_by_id or edge.to_unit_id not in units_by_id:
                continue
            if relation_filter is not None and str(edge.relation) != relation_filter:
                continue
            if edge.from_unit_id == edge.to_unit_id:
                continue

            left_id, right_id = sorted((edge.from_unit_id, edge.to_unit_id))
            edge_payload = {
                "id": edge.id,
                "from_unit_id": edge.from_unit_id,
                "to_unit_id": edge.to_unit_id,
                "relation": str(edge.relation),
                "weight": float(edge.weight or 0.0),
                "source": str(edge.source),
            }
            if projection.has_edge(left_id, right_id):
                existing = projection[left_id][right_id]["edge"]
                existing_key = (
                    existing["relation"],
                    existing["id"],
                    existing["from_unit_id"],
                    existing["to_unit_id"],
                )
                new_key = (
                    edge_payload["relation"],
                    edge_payload["id"],
                    edge_payload["from_unit_id"],
                    edge_payload["to_unit_id"],
                )
                if new_key >= existing_key:
                    continue
            projection.add_edge(left_id, right_id, edge=edge_payload)

        try:
            raw_paths = nx.all_shortest_paths(
                projection,
                source_unit_id,
                target_unit_id,
            )
            ordered_paths = sorted(raw_paths)[:max_paths]
        except nx.NetworkXNoPath:
            return []

        explanations = []
        for path in ordered_paths:
            traversed_edges = []
            total_weight = 0.0
            for left_id, right_id in zip(path, path[1:], strict=False):
                edge = projection[left_id][right_id]["edge"]
                traversal_direction = (
                    "forward"
                    if edge["from_unit_id"] == left_id and edge["to_unit_id"] == right_id
                    else "reverse"
                )
                total_weight += edge["weight"]
                traversed_edges.append(
                    {
                        **edge,
                        "traversal_from_unit_id": left_id,
                        "traversal_to_unit_id": right_id,
                        "traversal_direction": traversal_direction,
                    }
                )

            explanations.append(
                {
                    "unit_ids": path,
                    "edge_ids": [edge["id"] for edge in traversed_edges],
                    "relations": [edge["relation"] for edge in traversed_edges],
                    "edges": traversed_edges,
                    "hop_count": len(path) - 1,
                    "total_weight": round(total_weight, 6),
                }
            )

        return explanations

    def build_shortest_path_payload(self, from_unit_id: str, to_unit_id: str) -> dict:
        """Build a structured shortest-path payload for API/MCP callers."""
        if not self.G:
            self.rebuild()

        missing_unit_ids = [
            unit_id
            for unit_id in (from_unit_id, to_unit_id)
            if unit_id not in self.G
        ]
        if missing_unit_ids:
            return {
                "from_unit_id": from_unit_id,
                "to_unit_id": to_unit_id,
                "path": [],
                "edges": [],
                "error": "unit_not_found",
                "missing_unit_ids": missing_unit_ids,
                "message": "One or more units were not found.",
            }

        path = self.shortest_path(from_unit_id, to_unit_id)
        if path is None:
            return {
                "from_unit_id": from_unit_id,
                "to_unit_id": to_unit_id,
                "path": [],
                "edges": [],
                "error": "not_connected",
                "message": "No path found between the selected units.",
            }

        path_units = [
            unit
            for unit in (self.store.get_unit(unit_id) for unit_id in path)
            if unit is not None
        ]
        edges = []
        for left_id, right_id in zip(path, path[1:], strict=False):
            edge = self.G.get_edge_data(left_id, right_id)
            traversal_direction = "forward"
            if edge is None:
                edge = self.G.get_edge_data(right_id, left_id)
                traversal_direction = "reverse"
            if edge is None:
                continue
            edge_payload = {
                "id": edge.get("id"),
                "from_unit_id": (
                    left_id if traversal_direction == "forward" else right_id
                ),
                "to_unit_id": (
                    right_id if traversal_direction == "forward" else left_id
                ),
                "relation": str(edge.get("relation", "")),
                "weight": edge.get("weight"),
                "source": str(edge.get("source", "")),
                "traversal_from_unit_id": left_id,
                "traversal_to_unit_id": right_id,
                "traversal_direction": traversal_direction,
            }
            edges.append(edge_payload)

        return {
            "from_unit_id": from_unit_id,
            "to_unit_id": to_unit_id,
            "path": [self._unit_export_data(unit) for unit in path_units],
            "edges": edges,
        }

    def find_cycles(
        self,
        *,
        relation: str | None = None,
        max_length: int | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Find bounded directed simple cycles with deterministic ordering."""
        capped_limit = max(0, limit)
        if capped_limit == 0:
            return []

        if max_length is not None:
            max_length = max(1, max_length)

        units_by_id = {unit.id: unit for unit in self.store.get_all_units()}
        cycle_graph = nx.DiGraph()
        cycle_graph.add_nodes_from(units_by_id)

        edges_by_pair: dict[tuple[str, str], KnowledgeEdge] = {}
        for edge in sorted(
            self.store.get_all_edges(),
            key=lambda item: (
                item.from_unit_id,
                item.to_unit_id,
                str(item.relation),
                str(item.source),
                item.id,
            ),
        ):
            if edge.from_unit_id not in units_by_id or edge.to_unit_id not in units_by_id:
                continue
            if relation is not None and str(edge.relation) != relation:
                continue
            pair = (edge.from_unit_id, edge.to_unit_id)
            if pair in edges_by_pair:
                continue
            edges_by_pair[pair] = edge
            cycle_graph.add_edge(edge.from_unit_id, edge.to_unit_id)

        def unit_sort_key(unit_id: str) -> tuple[str, str]:
            unit = units_by_id[unit_id]
            return (unit.title.lower(), unit_id)

        def canonical_cycle(cycle: list[str]) -> list[str]:
            rotations = [
                cycle[index:] + cycle[:index]
                for index in range(len(cycle))
            ]
            return min(
                rotations,
                key=lambda rotation: tuple(
                    unit_sort_key(unit_id) for unit_id in rotation
                ),
            )

        unique_cycles: dict[tuple[str, ...], list[str]] = {}
        for cycle in nx.simple_cycles(cycle_graph):
            if max_length is not None and len(cycle) > max_length:
                continue
            canonical = canonical_cycle(list(cycle))
            unique_cycles[tuple(canonical)] = canonical

        ordered_cycles = sorted(
            unique_cycles.values(),
            key=lambda cycle: (len(cycle), tuple(unit_sort_key(unit_id) for unit_id in cycle)),
        )

        results = []
        for cycle in ordered_cycles[:capped_limit]:
            cycle_edges = []
            for from_unit_id, to_unit_id in zip(cycle, cycle[1:] + cycle[:1], strict=False):
                edge = edges_by_pair[(from_unit_id, to_unit_id)]
                cycle_edges.append(self._edge_export_data(edge))
            results.append(
                {
                    "length": len(cycle),
                    "unit_ids": cycle,
                    "units": [
                        self._unit_export_data(units_by_id[unit_id])
                        for unit_id in cycle
                    ],
                    "edges": cycle_edges,
                    "relations": [edge["relation"] for edge in cycle_edges],
                }
            )

        return results

    def get_clusters(self, min_size: int = 3) -> list[list[str]]:
        """Find connected components / clusters."""
        if not self.G.nodes:
            return []
        undirected = self.G.to_undirected()
        components = [
            list(c)
            for c in nx.connected_components(undirected)
            if len(c) >= min_size
        ]
        components.sort(key=len, reverse=True)
        return components

    def detect_communities(
        self,
        *,
        min_size: int = 2,
        limit: int | None = None,
    ) -> list[dict]:
        """Detect deterministic communities in the undirected knowledge graph."""
        if not isinstance(min_size, int) or isinstance(min_size, bool) or min_size < 1:
            raise ValueError("min_size must be a positive integer.")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
        ):
            raise ValueError("limit must be a positive integer or None.")

        if not self.G:
            self.rebuild()
        if not self.G.nodes:
            return []

        undirected = self.G.to_undirected()
        raw_communities: list[set[str]] = []
        for component_ids in nx.connected_components(undirected):
            component = undirected.subgraph(component_ids)
            if component.number_of_edges() == 0:
                raw_communities.extend({str(node_id)} for node_id in component.nodes)
                continue

            detected = nx.algorithms.community.greedy_modularity_communities(
                component,
                weight="weight",
            )
            for community in detected:
                community_graph = component.subgraph(community)
                raw_communities.extend(
                    {str(node_id) for node_id in connected_ids}
                    for connected_ids in nx.connected_components(community_graph)
                )

        records = []
        for unit_ids_set in raw_communities:
            if len(unit_ids_set) < min_size:
                continue

            unit_ids = sorted(unit_ids_set)
            community_graph = undirected.subgraph(unit_ids)
            internal_edge_count = int(community_graph.number_of_edges())
            density = round(float(nx.density(community_graph)), 6)
            ranked_representatives = sorted(
                unit_ids,
                key=lambda unit_id: (
                    -int(community_graph.degree(unit_id)),
                    str(self.G.nodes[unit_id].get("title", "")).lower(),
                    unit_id,
                ),
            )
            digest = hashlib.sha1("\n".join(unit_ids).encode("utf-8")).hexdigest()[:12]
            records.append(
                {
                    "community_id": f"community-{digest}",
                    "size": len(unit_ids),
                    "unit_ids": unit_ids,
                    "representative_titles": [
                        str(self.G.nodes[unit_id].get("title", ""))
                        for unit_id in ranked_representatives[:3]
                    ],
                    "internal_edge_count": internal_edge_count,
                    "density": density,
                }
            )

        records.sort(
            key=lambda record: (
                -record["size"],
                -record["density"],
                -record["internal_edge_count"],
                record["unit_ids"],
            )
        )
        return records[:limit] if limit is not None else records

    def analyze_strongly_connected_components(
        self,
        min_size: int = 2,
        limit: int = 20,
    ) -> list[dict]:
        """Identify directed strongly connected component clusters."""
        if not isinstance(min_size, int) or isinstance(min_size, bool) or min_size < 1:
            raise ValueError("min_size must be a positive integer.")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit must be a non-negative integer.")
        if limit == 0:
            return []

        units_by_id = {unit.id: unit for unit in self.store.get_all_units(limit=1000000000)}
        if len(units_by_id) < min_size:
            return []

        edges = self.store.get_all_edges()
        graph = nx.DiGraph()
        graph.add_nodes_from(units_by_id)
        for edge in edges:
            if edge.from_unit_id in units_by_id and edge.to_unit_id in units_by_id:
                graph.add_edge(edge.from_unit_id, edge.to_unit_id)

        records = []
        for component_ids in nx.strongly_connected_components(graph):
            if len(component_ids) < min_size:
                continue

            unit_ids = sorted(str(unit_id) for unit_id in component_ids)
            component_set = set(unit_ids)
            units = [units_by_id[unit_id] for unit_id in unit_ids]

            source_project_counts = Counter(str(unit.source_project) for unit in units)
            tag_counts = Counter(
                str(tag)
                for unit in units
                for tag in (unit.tags or [])
                if str(tag)
            )
            relation_counts = Counter(
                str(edge.relation)
                for edge in edges
                if edge.from_unit_id in component_set
                and edge.to_unit_id in component_set
            )

            records.append(
                {
                    "size": len(unit_ids),
                    "unit_ids": unit_ids,
                    "titles": [unit.title for unit in units],
                    "source_project_counts": dict(sorted(source_project_counts.items())),
                    "relation_counts": dict(sorted(relation_counts.items())),
                    "representative_tags": [
                        tag
                        for tag, _count in sorted(
                            tag_counts.items(),
                            key=lambda item: (-item[1], item[0]),
                        )[:5]
                    ],
                }
            )

        records.sort(key=lambda record: (-record["size"], record["unit_ids"]))
        return records[:limit]

    def analyze_condensation_dag(self, limit: int = 20) -> dict:
        """Collapse strongly connected components into a directed acyclic graph."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer.")

        units_by_id = {unit.id: unit for unit in self.store.get_all_units(limit=1000000000)}
        graph = nx.DiGraph()
        graph.add_nodes_from(sorted(units_by_id))
        graph.add_edges_from(
            sorted(
                (edge.from_unit_id, edge.to_unit_id)
                for edge in self.store.get_all_edges()
                if edge.from_unit_id in units_by_id and edge.to_unit_id in units_by_id
            )
        )

        if not graph.nodes:
            return {
                "component_count": 0,
                "cyclic_component_count": 0,
                "source_component_count": 0,
                "sink_component_count": 0,
                "topological_order": [],
                "components": [],
            }

        strongly_connected_components = sorted(
            (set(component_ids) for component_ids in nx.strongly_connected_components(graph)),
            key=lambda component_ids: sorted(str(unit_id) for unit_id in component_ids),
        )
        condensation = nx.condensation(graph, strongly_connected_components)
        component_unit_ids = {
            component_id: sorted(str(unit_id) for unit_id in data["members"])
            for component_id, data in condensation.nodes(data=True)
        }
        topological_component_ids = list(
            nx.lexicographical_topological_sort(
                condensation,
                key=lambda component_id: component_unit_ids[component_id],
            )
        )
        public_component_ids = {
            component_id: f"component-{index:03d}"
            for index, component_id in enumerate(topological_component_ids, start=1)
        }

        components = []
        for component_id in topological_component_ids:
            unit_ids = component_unit_ids[component_id]
            units = [units_by_id[unit_id] for unit_id in unit_ids]
            is_cyclic = len(unit_ids) > 1 or any(
                graph.has_edge(unit_id, unit_id) for unit_id in unit_ids
            )
            components.append(
                {
                    "component_id": public_component_ids[component_id],
                    "size": len(unit_ids),
                    "unit_ids": unit_ids,
                    "representative_titles": [unit.title for unit in units[:3]],
                    "incoming_component_count": condensation.in_degree(component_id),
                    "outgoing_component_count": condensation.out_degree(component_id),
                    "cyclic": is_cyclic,
                }
            )

        return {
            "component_count": len(components),
            "cyclic_component_count": sum(
                1 for component in components if component["cyclic"]
            ),
            "source_component_count": sum(
                1
                for component_id in topological_component_ids
                if condensation.in_degree(component_id) == 0
            ),
            "sink_component_count": sum(
                1
                for component_id in topological_component_ids
                if condensation.out_degree(component_id) == 0
            ),
            "topological_order": [
                public_component_ids[component_id]
                for component_id in topological_component_ids
            ],
            "components": components[:limit],
        }

    def get_central_nodes(self, limit: int = 10) -> list[tuple[str, float]]:
        """Top nodes by PageRank."""
        if not self.G.nodes:
            return []

        # Keep this dependency-free so the CLI/test environment does not require
        # NumPy/SciPy C extensions just to rank a small graph.
        nodes = list(self.G.nodes)
        n = len(nodes)
        if n == 1:
            return [(nodes[0], 1.0)]

        damping = 0.85
        ranks = {node: 1.0 / n for node in nodes}

        for _ in range(100):
            sink_rank = sum(ranks[node] for node in nodes if self.G.out_degree(node) == 0)
            base_rank = (1.0 - damping) / n
            sink_share = damping * sink_rank / n
            new_ranks = {node: base_rank + sink_share for node in nodes}

            for node in nodes:
                out_edges = list(self.G.out_edges(node, data=True))
                if not out_edges:
                    continue
                total_weight = sum(float(data.get("weight", 1.0)) for _, _, data in out_edges)
                if total_weight <= 0:
                    continue
                share = damping * ranks[node]
                for _, target, data in out_edges:
                    weight = float(data.get("weight", 1.0))
                    new_ranks[target] += share * (weight / total_weight)

            delta = sum(abs(new_ranks[node] - ranks[node]) for node in nodes)
            ranks = new_ranks
            if delta < 1e-12:
                break

        sorted_pr = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
        return sorted_pr[:limit]

    def pagerank_centrality(
        self,
        top_n: int | None = None,
        relation_filter: str | EdgeRelation | list[str | EdgeRelation] | tuple[str | EdgeRelation, ...] | set[str | EdgeRelation] | None = None,
        weight: str | bool | None = "weight",
    ) -> list[dict]:
        """Return PageRank centrality summaries for knowledge units.

        When ``relation_filter`` is provided, all units remain in the graph but
        only matching edge relations are considered for scores and degrees. Set
        ``weight`` to ``None`` or ``False`` for unweighted PageRank.
        """
        if top_n is not None:
            capped_limit = max(0, int(top_n))
            if capped_limit == 0:
                return []
        else:
            capped_limit = None

        if not self.G:
            self.rebuild()
        if not self.G.nodes:
            return []

        graph = nx.DiGraph()
        graph.add_nodes_from(self.G.nodes(data=True))
        relation_values: set[str] | None = None
        if relation_filter is not None:
            if isinstance(relation_filter, (str, EdgeRelation)):
                relation_values = {str(relation_filter)}
            else:
                relation_values = {str(relation) for relation in relation_filter}

        for from_id, to_id, data in self.G.edges(data=True):
            if relation_values is not None and str(data.get("relation", "")) not in relation_values:
                continue
            graph.add_edge(from_id, to_id, **data)

        nodes = list(graph.nodes)
        node_count = len(nodes)
        weight_key = "weight" if weight is True else weight
        use_weights = isinstance(weight_key, str) and bool(weight_key)

        if node_count == 1:
            scores = {nodes[0]: 1.0}
        else:
            damping = 0.85
            scores = {node: 1.0 / node_count for node in nodes}

            for _ in range(100):
                sink_rank = 0.0
                outgoing: dict[str, list[tuple[str, float]]] = {}

                for node in nodes:
                    weighted_targets = []
                    for _, target, data in graph.out_edges(node, data=True):
                        if use_weights:
                            edge_weight = max(0.0, float(data.get(weight_key, 1.0)))
                        else:
                            edge_weight = 1.0
                        weighted_targets.append((target, edge_weight))

                    total_weight = sum(edge_weight for _, edge_weight in weighted_targets)
                    if total_weight <= 0.0:
                        sink_rank += scores[node]
                    else:
                        outgoing[node] = [
                            (target, edge_weight / total_weight)
                            for target, edge_weight in weighted_targets
                        ]

                base_rank = (1.0 - damping) / node_count
                sink_share = damping * sink_rank / node_count
                next_scores = {node: base_rank + sink_share for node in nodes}

                for node, weighted_targets in outgoing.items():
                    share = damping * scores[node]
                    for target, normalized_weight in weighted_targets:
                        next_scores[target] += share * normalized_weight

                delta = sum(abs(next_scores[node] - scores[node]) for node in nodes)
                scores = next_scores
                if delta < 1e-12:
                    break

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if capped_limit is not None:
            ranked = ranked[:capped_limit]

        results = []
        for unit_id, score in ranked:
            summary = self._unit_summary_data(self.store.get_unit(unit_id))
            if summary is None:
                node_data = graph.nodes[unit_id]
                summary = {
                    "id": unit_id,
                    "source_project": str(node_data.get("source_project", "")),
                    "source_id": "",
                    "source_entity_type": str(node_data.get("source_entity_type", "")),
                    "title": str(node_data.get("title", "")),
                    "content_type": str(node_data.get("content_type", "")),
                }
            results.append(
                {
                    **summary,
                    "score": float(score),
                    "in_degree": int(graph.in_degree(unit_id)),
                    "out_degree": int(graph.out_degree(unit_id)),
                }
            )
        return results

    def get_pagerank(self, limit: int = 10, weight: str | bool | None = "weight") -> list[dict]:
        """Return ranked unit summaries using weighted PageRank."""
        capped_limit = max(0, int(limit))
        if capped_limit == 0:
            return []

        ranked = sorted(
            self.pagerank_centrality(weight=weight),
            key=lambda item: (-item["score"], str(item["title"]).lower(), item["id"]),
        )
        results = []
        for item in ranked[:capped_limit]:
            summary = {
                "id": item["id"],
                "source_project": item["source_project"],
                "source_id": item["source_id"],
                "source_entity_type": item["source_entity_type"],
                "title": item["title"],
                "content_type": item["content_type"],
            }
            results.append({"unit": summary, "score": item["score"]})
        return results

    def eigenvector_centrality(
        self,
        *,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        limit: int | None = None,
    ) -> dict:
        """Return Eigenvector centrality rankings for the current graph."""
        if isinstance(max_iter, bool):
            raise ValueError("max_iter must be a positive integer.")
        try:
            capped_max_iter = int(max_iter)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_iter must be a positive integer.") from exc
        if capped_max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")

        if isinstance(tolerance, bool):
            raise ValueError("tolerance must be a positive finite number.")
        try:
            capped_tolerance = float(tolerance)
        except (TypeError, ValueError) as exc:
            raise ValueError("tolerance must be a positive finite number.") from exc
        if not math.isfinite(capped_tolerance) or capped_tolerance <= 0.0:
            raise ValueError("tolerance must be a positive finite number.")

        if limit is None:
            capped_limit = None
        else:
            if isinstance(limit, bool):
                raise ValueError("limit must be a non-negative integer.")
            try:
                capped_limit = int(limit)
            except (TypeError, ValueError) as exc:
                raise ValueError("limit must be a non-negative integer.") from exc
            if capped_limit < 0:
                raise ValueError("limit must be a non-negative integer.")

        if not self.G:
            self.rebuild()

        stats = {
            "node_count": self.G.number_of_nodes(),
            "edge_count": self.G.number_of_edges(),
            "max_iter": capped_max_iter,
            "tolerance": capped_tolerance,
            "limit": capped_limit,
            "converged": True,
            "error": None,
        }

        if not self.G.nodes:
            return {"stats": stats, "nodes": []}

        try:
            scores = nx.eigenvector_centrality(
                self.G,
                max_iter=capped_max_iter,
                tol=capped_tolerance,
                weight="weight",
            )
        except nx.PowerIterationFailedConvergence as exc:
            stats["converged"] = False
            stats["error"] = str(exc)
            return {"stats": stats, "nodes": []}

        def _score(value: float) -> float:
            score = float(value)
            return 0.0 if abs(score) < 1e-15 else score

        ranked_ids = sorted(
            self.G.nodes,
            key=lambda unit_id: (-_score(scores.get(unit_id, 0.0)), str(unit_id)),
        )
        if capped_limit is not None:
            ranked_ids = ranked_ids[:capped_limit]

        nodes = []
        for unit_id in ranked_ids:
            node_data = self.G.nodes[unit_id]
            nodes.append(
                {
                    "unit_id": unit_id,
                    "title": str(node_data.get("title", "")),
                    "score": _score(scores.get(unit_id, 0.0)),
                }
            )

        return {"stats": stats, "nodes": nodes}

    def betweenness_centrality(
        self,
        limit: int | None = 10,
        *,
        normalized: bool = True,
        weight: str | bool | None = None,
    ) -> dict:
        """Return betweenness centrality rankings for the undirected graph projection."""
        if limit is None:
            capped_limit = None
        else:
            if isinstance(limit, bool):
                raise ValueError("limit must be a non-negative integer or None.")
            try:
                capped_limit = int(limit)
            except (TypeError, ValueError) as exc:
                raise ValueError("limit must be a non-negative integer or None.") from exc
            if capped_limit < 0:
                raise ValueError("limit must be a non-negative integer or None.")

        if not isinstance(normalized, bool):
            raise ValueError("normalized must be a boolean.")

        if weight is True:
            weight_key = "weight"
        elif weight is False:
            weight_key = None
        elif weight is None or isinstance(weight, str):
            weight_key = weight
        else:
            raise ValueError("weight must be a string, True, False, or None.")

        if not self.G:
            self.rebuild()

        projection = nx.Graph()
        projection.add_nodes_from(self.G.nodes(data=True))
        for from_id, to_id, data in self.G.edges(data=True):
            if from_id == to_id:
                continue
            if projection.has_edge(from_id, to_id):
                if weight_key is not None:
                    existing_weight = float(projection[from_id][to_id].get(weight_key, 1.0))
                    edge_weight = float(data.get(weight_key, 1.0))
                    projection[from_id][to_id][weight_key] = min(existing_weight, edge_weight)
                continue
            projection.add_edge(from_id, to_id, **data)

        stats = {
            "node_count": projection.number_of_nodes(),
            "edge_count": projection.number_of_edges(),
            "limit": capped_limit,
            "normalized": normalized,
            "weight": weight_key,
        }

        if not projection.nodes:
            return {"stats": stats, "nodes": []}

        scores = nx.betweenness_centrality(
            projection,
            normalized=normalized,
            weight=weight_key,
        )

        def _score(value: float) -> float:
            score = float(value)
            return 0.0 if abs(score) < 1e-15 else score

        ranked_ids = sorted(
            projection.nodes,
            key=lambda unit_id: (-_score(scores.get(unit_id, 0.0)), str(unit_id)),
        )
        if capped_limit is not None:
            ranked_ids = ranked_ids[:capped_limit]

        nodes = []
        for unit_id in ranked_ids:
            node_data = projection.nodes[unit_id]
            nodes.append(
                {
                    "unit_id": unit_id,
                    "title": str(node_data.get("title", "")),
                    "score": _score(scores.get(unit_id, 0.0)),
                    "source_project": str(node_data.get("source_project", "")),
                    "degree": int(projection.degree(unit_id)),
                    "in_degree": int(self.G.in_degree(unit_id)),
                    "out_degree": int(self.G.out_degree(unit_id)),
                }
            )

        return {"stats": stats, "nodes": nodes}

    def analyze_hits(self, limit: int = 20, normalized: bool = True) -> dict:
        """Return HITS authority and hub rankings for the directed graph."""
        if isinstance(limit, bool):
            raise ValueError("limit must be a non-negative integer.")
        try:
            capped_limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit must be a non-negative integer.") from exc
        if capped_limit < 0:
            raise ValueError("limit must be a non-negative integer.")

        if not self.G:
            self.rebuild()

        stats = {
            "node_count": len(self.G.nodes),
            "edge_count": len(self.G.edges),
            "normalized": bool(normalized),
            "converged": True,
            "max_iter": 0,
            "error": None,
        }

        if not self.G.nodes:
            return {
                "stats": stats,
                "authorities": [],
                "hubs": [],
            }

        try:
            hubs, authorities = nx.hits(
                self.G,
                max_iter=100,
                normalized=bool(normalized),
            )
            stats["max_iter"] = 100
        except nx.PowerIterationFailedConvergence:
            try:
                hubs, authorities = nx.hits(
                    self.G,
                    max_iter=1000,
                    normalized=bool(normalized),
                )
                stats["max_iter"] = 1000
            except nx.PowerIterationFailedConvergence as exc:
                stats["converged"] = False
                stats["max_iter"] = 1000
                stats["error"] = str(exc)
                return {
                    "stats": stats,
                    "authorities": [],
                    "hubs": [],
                }

        def _score(value: float) -> float:
            score = float(value)
            return 0.0 if abs(score) < 1e-15 else score

        def _record(unit_id: str) -> dict:
            node_data = self.G.nodes[unit_id]
            return {
                "unit_id": unit_id,
                "title": str(node_data.get("title", "")),
                "authority_score": _score(authorities.get(unit_id, 0.0)),
                "hub_score": _score(hubs.get(unit_id, 0.0)),
            }

        authority_ids = sorted(
            self.G.nodes,
            key=lambda unit_id: (-_score(authorities.get(unit_id, 0.0)), str(unit_id)),
        )
        hub_ids = sorted(
            self.G.nodes,
            key=lambda unit_id: (-_score(hubs.get(unit_id, 0.0)), str(unit_id)),
        )

        return {
            "stats": stats,
            "authorities": [_record(unit_id) for unit_id in authority_ids[:capped_limit]],
            "hubs": [_record(unit_id) for unit_id in hub_ids[:capped_limit]],
        }

    def isolated_units(
        self,
        *,
        limit: int = 50,
        include_units: bool = True,
        source_project: str | None = None,
    ) -> dict:
        """Return units with total degree zero in the current graph."""
        if isinstance(limit, bool):
            raise ValueError("limit must be a non-negative integer.")
        try:
            capped_limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit must be a non-negative integer.") from exc
        if capped_limit < 0:
            raise ValueError("limit must be a non-negative integer.")

        if not self.G:
            self.rebuild()

        source_project_filter = None if source_project is None else str(source_project)
        total_units = self.G.number_of_nodes()
        isolated_ids = [
            node_id
            for node_id, data in self.G.nodes(data=True)
            if self.G.degree(node_id) == 0
            and (
                source_project_filter is None
                or str(data.get("source_project", "")) == source_project_filter
            )
        ]

        def _updated_at_timestamp(node_id: str) -> float:
            value = self.G.nodes[node_id].get("updated_at")
            if isinstance(value, datetime):
                parsed = value
            else:
                try:
                    parsed = datetime.fromisoformat(str(value))
                except (TypeError, ValueError):
                    return float("-inf")
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()

        isolated_ids.sort(
            key=lambda node_id: (
                -_updated_at_timestamp(node_id),
                str(self.G.nodes[node_id].get("title", "")).lower(),
                str(node_id),
            )
        )

        units = []
        if include_units:
            for node_id in isolated_ids[:capped_limit]:
                unit = self.store.get_unit(node_id)
                if unit is not None:
                    units.append(self._unit_export_data(unit))

        isolated_count = len(isolated_ids)
        return {
            "isolated_count": isolated_count,
            "total_units": total_units,
            "ratio": isolated_count / total_units if total_units else 0.0,
            "filters": {
                "source_project": source_project_filter,
                "limit": capped_limit,
                "include_units": bool(include_units),
            },
            "units": units,
        }

    def analyze_degree_distribution(
        self,
        *,
        direction: str = "total",
        top_n: int = 10,
    ) -> dict:
        """Summarize degree distribution for the current directed graph."""
        if direction not in {"total", "in", "out"}:
            raise ValueError(
                "Unsupported degree direction: "
                f"{direction!r}. Use 'total', 'in', or 'out'."
            )

        if isinstance(top_n, bool):
            raise ValueError("top_n must be a positive integer.")
        try:
            capped_limit = int(top_n)
        except (TypeError, ValueError) as exc:
            raise ValueError("top_n must be a positive integer.") from exc
        if capped_limit < 1:
            raise ValueError("top_n must be a positive integer.")

        if not self.G:
            self.rebuild()

        if not self.G.nodes:
            return {
                "direction": direction,
                "total_units": 0,
                "isolated_unit_count": 0,
                "histogram": [],
                "top_units": [],
            }

        degree_rows = []
        histogram_counts: Counter[int] = Counter()
        isolated_unit_count = 0
        for unit_id in self.G.nodes:
            in_degree = int(self.G.in_degree(unit_id))
            out_degree = int(self.G.out_degree(unit_id))
            if direction == "in":
                degree = in_degree
            elif direction == "out":
                degree = out_degree
            else:
                degree = in_degree + out_degree

            histogram_counts[degree] += 1
            if degree == 0:
                isolated_unit_count += 1
            degree_rows.append(
                {
                    "unit_id": unit_id,
                    "degree": degree,
                    "in_degree": in_degree,
                    "out_degree": out_degree,
                    "title": str(self.G.nodes[unit_id].get("title", "")),
                }
            )

        degree_rows.sort(
            key=lambda item: (
                -int(item["degree"]),
                -int(item["in_degree"]),
                -int(item["out_degree"]),
                str(item["title"]).lower(),
                str(item["unit_id"]),
            )
        )

        top_units = []
        for row in degree_rows[:capped_limit]:
            unit_id = row["unit_id"]
            summary = self._unit_summary_data(self.store.get_unit(unit_id))
            if summary is None:
                node_data = self.G.nodes[unit_id]
                summary = {
                    "id": unit_id,
                    "source_project": str(node_data.get("source_project", "")),
                    "source_id": "",
                    "source_entity_type": str(node_data.get("source_entity_type", "")),
                    "title": str(node_data.get("title", "")),
                    "content_type": str(node_data.get("content_type", "")),
                }
            top_units.append(
                {
                    **summary,
                    "degree": int(row["degree"]),
                    "in_degree": int(row["in_degree"]),
                    "out_degree": int(row["out_degree"]),
                }
            )

        return {
            "direction": direction,
            "total_units": len(self.G.nodes),
            "isolated_unit_count": isolated_unit_count,
            "histogram": [
                {"degree": degree, "unit_count": count}
                for degree, count in sorted(histogram_counts.items())
            ],
            "top_units": top_units,
        }

    def analyze_k_core(
        self,
        k: int | None = None,
        *,
        limit: int = 50,
        include_units: bool = True,
    ) -> dict:
        """Return ranked nodes in the selected undirected k-core."""
        if k is None:
            requested_k = None
        else:
            try:
                requested_k = int(k)
            except (TypeError, ValueError) as exc:
                raise ValueError("k must be a non-negative integer.") from exc
            if requested_k < 0:
                raise ValueError("k must be a non-negative integer.")

        try:
            capped_limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit must be a positive integer.") from exc
        if capped_limit < 1:
            raise ValueError("limit must be a positive integer.")

        if not self.G:
            self.rebuild()

        filters = {
            "k": requested_k,
            "limit": capped_limit,
            "include_units": bool(include_units),
        }
        if not self.G.nodes:
            return {
                "k": requested_k,
                "max_core": 0,
                "selected_core": 0 if requested_k is None else requested_k,
                "node_count": 0,
                "edge_count": 0,
                "filters": filters,
                "nodes": [],
                "edges": [],
            }

        undirected = self.G.to_undirected()
        core_numbers = nx.core_number(undirected)
        max_core = max(core_numbers.values(), default=0)
        selected_core = max_core if requested_k is None else requested_k
        selected_node_ids = {
            node_id
            for node_id, core_number in core_numbers.items()
            if int(core_number) >= selected_core
        }
        selected_graph = undirected.subgraph(selected_node_ids)

        ranked_node_ids = sorted(
            selected_node_ids,
            key=lambda node_id: (
                -int(core_numbers.get(node_id, 0)),
                -int(undirected.degree(node_id)),
                str(self.G.nodes[node_id].get("title", "")).lower(),
                str(node_id),
            ),
        )

        nodes = []
        for node_id in ranked_node_ids[:capped_limit]:
            node = {
                "id": node_id,
                "title": str(self.G.nodes[node_id].get("title", "")),
                "source_project": str(self.G.nodes[node_id].get("source_project", "")),
                "content_type": str(self.G.nodes[node_id].get("content_type", "")),
                "degree": int(undirected.degree(node_id)),
                "core_number": int(core_numbers.get(node_id, 0)),
            }
            if include_units:
                unit = self.store.get_unit(node_id)
                if unit is not None:
                    node["unit"] = self._unit_export_data(unit)
            nodes.append(node)

        edges = []
        for edge in self.store.get_all_edges():
            if edge.from_unit_id in selected_node_ids and edge.to_unit_id in selected_node_ids:
                edges.append(self._edge_export_data(edge))
        edges.sort(
            key=lambda edge: (
                edge["from_unit_id"],
                edge["to_unit_id"],
                edge["relation"],
                edge["id"] or "",
            )
        )

        return {
            "k": requested_k,
            "max_core": int(max_core),
            "selected_core": int(selected_core),
            "node_count": len(selected_node_ids),
            "edge_count": selected_graph.number_of_edges(),
            "filters": filters,
            "nodes": nodes,
            "edges": edges,
        }

    def get_k_core_decomposition(
        self,
        min_core: int = 1,
        limit: int | None = None,
    ) -> dict:
        """Return ranked unit core numbers from the undirected graph projection."""
        if (
            not isinstance(min_core, int)
            or isinstance(min_core, bool)
            or min_core < 1
        ):
            raise ValueError("min_core must be a positive integer.")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
        ):
            raise ValueError("limit must be a positive integer.")

        units_by_id = {unit.id: unit for unit in self.store.get_all_units(limit=1000000000)}
        projection = nx.Graph()
        projection.add_nodes_from(sorted(units_by_id))
        projection.add_edges_from(
            sorted(
                (edge.from_unit_id, edge.to_unit_id)
                for edge in self.store.get_all_edges()
                if edge.from_unit_id in units_by_id
                and edge.to_unit_id in units_by_id
                and edge.from_unit_id != edge.to_unit_id
            )
        )

        if not projection.nodes:
            return {
                "stats": {
                    "node_count": 0,
                    "edge_count": 0,
                    "max_core": 0,
                    "returned_count": 0,
                },
                "nodes": [],
            }

        core_numbers = nx.core_number(projection)
        max_core = max(core_numbers.values(), default=0)
        ranked_node_ids = sorted(
            (
                node_id
                for node_id, core_number in core_numbers.items()
                if int(core_number) >= min_core
            ),
            key=lambda node_id: (
                -int(core_numbers.get(node_id, 0)),
                -int(projection.degree(node_id)),
                str(units_by_id[node_id].title).lower(),
                str(node_id),
            ),
        )
        if limit is not None:
            ranked_node_ids = ranked_node_ids[:limit]

        nodes = [
            {
                "unit_id": node_id,
                "title": units_by_id[node_id].title,
                "source_project": str(units_by_id[node_id].source_project),
                "core_number": int(core_numbers[node_id]),
                "degree": int(projection.degree(node_id)),
                "neighbor_count": len(list(projection.neighbors(node_id))),
            }
            for node_id in ranked_node_ids
        ]

        return {
            "stats": {
                "node_count": projection.number_of_nodes(),
                "edge_count": projection.number_of_edges(),
                "max_core": int(max_core),
                "returned_count": len(nodes),
            },
            "nodes": nodes,
        }

    def analyze_triangles(
        self,
        limit: int = 20,
        *,
        min_weight: float = 0.0,
        tag: str | None = None,
    ) -> list[dict]:
        """Identify closed three-node motifs in the undirected knowledge graph."""
        try:
            capped_limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit must be a non-negative integer.") from exc
        if capped_limit < 0:
            raise ValueError("limit must be a non-negative integer.")
        if capped_limit == 0:
            return []

        try:
            minimum_weight = float(min_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("min_weight must be numeric.") from exc

        required_tag = tag.strip() if isinstance(tag, str) else tag
        if tag is not None and (not isinstance(tag, str) or not required_tag):
            raise ValueError("tag must be a non-empty string or None.")

        units_by_id = {unit.id: unit for unit in self.store.get_all_units(limit=1000000000)}
        if len(units_by_id) < 3:
            return []

        edge_groups: dict[tuple[str, str], dict] = {}
        for edge in sorted(
            self.store.get_all_edges(),
            key=lambda item: (
                min(item.from_unit_id, item.to_unit_id),
                max(item.from_unit_id, item.to_unit_id),
                str(item.relation),
                -float(item.weight or 0.0),
                item.id,
            ),
        ):
            if edge.from_unit_id not in units_by_id or edge.to_unit_id not in units_by_id:
                continue
            if edge.from_unit_id == edge.to_unit_id:
                continue

            pair = tuple(sorted((edge.from_unit_id, edge.to_unit_id)))
            group = edge_groups.setdefault(
                pair,
                {
                    "unit_ids": list(pair),
                    "labels": set(),
                    "weight": float(edge.weight or 0.0),
                },
            )
            group["labels"].add(str(edge.relation))
            group["weight"] = max(group["weight"], float(edge.weight or 0.0))

        eligible_pairs = {
            pair: group
            for pair, group in edge_groups.items()
            if float(group["weight"]) >= minimum_weight
        }
        if len(eligible_pairs) < 3:
            return []

        motif_graph = nx.Graph()
        motif_graph.add_nodes_from(units_by_id)
        motif_graph.add_edges_from(eligible_pairs)

        records = []
        for unit_ids_tuple in combinations(sorted(units_by_id), 3):
            a, b, c = unit_ids_tuple
            pairs = [(a, b), (a, c), (b, c)]
            if not all(motif_graph.has_edge(left, right) for left, right in pairs):
                continue

            unit_tags = [set(units_by_id[unit_id].tags or []) for unit_id in unit_ids_tuple]
            shared_tags = sorted(set.intersection(*unit_tags)) if unit_tags else []
            if required_tag is not None and required_tag not in shared_tags:
                continue

            tag_union = set.union(*unit_tags) if unit_tags else set()
            shared_tag_overlap = (
                len(shared_tags) / len(tag_union)
                if tag_union
                else 0.0
            )
            edge_weights = [float(eligible_pairs[pair]["weight"]) for pair in pairs]
            score = round((sum(edge_weights) / len(edge_weights)) + shared_tag_overlap, 6)

            relations = []
            for pair in pairs:
                edge_group = eligible_pairs[pair]
                relations.append(
                    {
                        "unit_ids": list(pair),
                        "labels": sorted(edge_group["labels"]),
                        "weight": round(float(edge_group["weight"]), 6),
                    }
                )

            records.append(
                {
                    "unit_ids": list(unit_ids_tuple),
                    "titles": [units_by_id[unit_id].title for unit_id in unit_ids_tuple],
                    "shared_tags": shared_tags,
                    "relations": relations,
                    "score": score,
                }
            )

        records.sort(key=lambda record: (-record["score"], record["unit_ids"]))
        return records[:capped_limit]

    def analyze_edge_bridges(self, limit: int = 20) -> list[dict]:
        """Identify edge bridges in the undirected knowledge graph projection."""
        try:
            capped_limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit must be a non-negative integer.") from exc
        if capped_limit < 0:
            raise ValueError("limit must be a non-negative integer.")
        if capped_limit == 0:
            return []

        units_by_id = {unit.id: unit for unit in self.store.get_all_units(limit=1000000000)}
        if len(units_by_id) < 2:
            return []

        edge_groups: dict[tuple[str, str], dict] = {}
        for edge in sorted(
            self.store.get_all_edges(),
            key=lambda item: (
                min(item.from_unit_id, item.to_unit_id),
                max(item.from_unit_id, item.to_unit_id),
                str(item.relation),
                str(item.source),
                -float(item.weight or 0.0),
                item.id,
            ),
        ):
            if edge.from_unit_id not in units_by_id or edge.to_unit_id not in units_by_id:
                continue
            if edge.from_unit_id == edge.to_unit_id:
                continue

            pair = tuple(sorted((edge.from_unit_id, edge.to_unit_id)))
            group = edge_groups.setdefault(
                pair,
                {
                    "unit_ids": list(pair),
                    "edges": [],
                    "relations": set(),
                    "sources": set(),
                    "weight": 0.0,
                    "total_weight": 0.0,
                },
            )
            edge_weight = float(edge.weight or 0.0)
            group["edges"].append(edge)
            group["relations"].add(str(edge.relation))
            group["sources"].add(str(edge.source))
            group["weight"] = max(float(group["weight"]), edge_weight)
            group["total_weight"] = float(group["total_weight"]) + edge_weight

        if not edge_groups:
            return []

        projection = nx.Graph()
        projection.add_nodes_from(units_by_id)
        projection.add_edges_from(edge_groups)
        component_count_before = nx.number_connected_components(projection)

        records = []
        for bridge_pair in sorted(tuple(sorted(pair)) for pair in nx.bridges(projection)):
            left_id, right_id = bridge_pair
            group = edge_groups[bridge_pair]
            component_before = nx.node_connected_component(projection, left_id)

            without_bridge = projection.copy()
            without_bridge.remove_edge(left_id, right_id)
            left_component_size = len(nx.node_connected_component(without_bridge, left_id))
            right_component_size = len(nx.node_connected_component(without_bridge, right_id))
            smaller_component_size = min(left_component_size, right_component_size)
            larger_component_size = max(left_component_size, right_component_size)

            records.append(
                {
                    "unit_ids": list(bridge_pair),
                    "endpoints": [
                        self._unit_summary_data(units_by_id[left_id]),
                        self._unit_summary_data(units_by_id[right_id]),
                    ],
                    "edges": [self._edge_export_data(edge) for edge in group["edges"]],
                    "relations": sorted(group["relations"]),
                    "sources": sorted(group["sources"]),
                    "weight": round(float(group["weight"]), 6),
                    "total_weight": round(float(group["total_weight"]), 6),
                    "impact": {
                        "component_count_before": component_count_before,
                        "component_count_after": component_count_before + 1,
                        "original_component_size": len(component_before),
                        "endpoint_component_sizes": {
                            left_id: left_component_size,
                            right_id: right_component_size,
                        },
                        "smaller_component_size": smaller_component_size,
                        "larger_component_size": larger_component_size,
                    },
                }
            )

        records.sort(
            key=lambda record: (
                -record["impact"]["smaller_component_size"],
                -record["impact"]["original_component_size"],
                record["unit_ids"],
            )
        )
        return records[:capped_limit]

    def analyze_articulation_points(self, limit: int = 20) -> list[dict]:
        """Identify articulation points in the undirected knowledge graph projection."""
        try:
            capped_limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit must be a non-negative integer.") from exc
        if capped_limit < 0:
            raise ValueError("limit must be a non-negative integer.")
        if capped_limit == 0:
            return []

        units_by_id = {unit.id: unit for unit in self.store.get_all_units(limit=1000000000)}
        if len(units_by_id) < 3:
            return []

        projection = nx.Graph()
        projection.add_nodes_from(units_by_id)
        projection.add_edges_from(
            tuple(sorted((edge.from_unit_id, edge.to_unit_id)))
            for edge in self.store.get_all_edges()
            if edge.from_unit_id in units_by_id
            and edge.to_unit_id in units_by_id
            and edge.from_unit_id != edge.to_unit_id
        )
        if projection.number_of_edges() == 0:
            return []

        component_count_before = nx.number_connected_components(projection)
        records = []
        for unit_id in sorted(nx.articulation_points(projection)):
            unit = units_by_id[unit_id]
            original_component = nx.node_connected_component(projection, unit_id)
            without_unit = projection.copy()
            without_unit.remove_node(unit_id)

            affected_component_sizes = sorted(
                (
                    len(component)
                    for component in nx.connected_components(
                        without_unit.subgraph(original_component - {unit_id})
                    )
                ),
                reverse=True,
            )
            largest_remaining_size = affected_component_sizes[0]
            component_size_impact = (
                len(original_component) - 1 - largest_remaining_size
            )

            records.append(
                {
                    "unit_id": unit_id,
                    "title": unit.title,
                    "source_project": str(unit.source_project),
                    "source_id": unit.source_id,
                    "source_entity_type": unit.source_entity_type,
                    "content_type": str(unit.content_type),
                    "component_size_impact": component_size_impact,
                    "neighbor_count": int(projection.degree[unit_id]),
                    "affected_component_sizes": affected_component_sizes[:5],
                    "impact": {
                        "component_count_before": component_count_before,
                        "component_count_after": component_count_before
                        + len(affected_component_sizes)
                        - 1,
                        "original_component_size": len(original_component),
                        "largest_remaining_component_size": largest_remaining_size,
                        "affected_component_sizes": affected_component_sizes,
                    },
                }
            )

        records.sort(
            key=lambda record: (
                -record["component_size_impact"],
                record["unit_id"],
            )
        )
        return records[:capped_limit]

    def analyze_brokers(self, limit: int = 20) -> list[dict]:
        """Identify high-betweenness broker nodes in the knowledge graph."""
        if isinstance(limit, bool):
            raise ValueError("limit must be a non-negative integer.")
        try:
            capped_limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit must be a non-negative integer.") from exc
        if capped_limit < 0:
            raise ValueError("limit must be a non-negative integer.")
        if capped_limit == 0:
            return []

        units_by_id = {unit.id: unit for unit in self.store.get_all_units(limit=1000000000)}
        if len(units_by_id) < 3:
            return []

        projection = nx.Graph()
        projection.add_nodes_from(sorted(units_by_id))
        projection.add_edges_from(
            tuple(sorted((edge.from_unit_id, edge.to_unit_id)))
            for edge in self.store.get_all_edges()
            if edge.from_unit_id in units_by_id
            and edge.to_unit_id in units_by_id
            and edge.from_unit_id != edge.to_unit_id
        )
        if projection.number_of_edges() == 0:
            return []

        centrality = nx.betweenness_centrality(projection)
        records = []
        for unit_id, score in centrality.items():
            if score <= 0:
                continue

            unit = units_by_id[unit_id]
            neighbor_ids = sorted(projection.neighbors(unit_id))
            neighbor_source_projects = sorted(
                {str(units_by_id[neighbor_id].source_project) for neighbor_id in neighbor_ids}
            )
            degree = int(projection.degree(unit_id))
            rounded_score = round(float(score), 6)
            records.append(
                {
                    "unit_id": unit_id,
                    "title": unit.title,
                    "source_project": str(unit.source_project),
                    "score": rounded_score,
                    "degree": degree,
                    "neighbor_source_project_diversity": len(neighbor_source_projects),
                    "explanation": (
                        f"Connects {degree} neighboring units across "
                        f"{len(neighbor_source_projects)} source projects."
                    ),
                }
            )

        records.sort(
            key=lambda record: (
                -float(centrality.get(record["unit_id"], 0.0)),
                record["unit_id"],
            )
        )
        return records[:capped_limit]

    def suggest_missing_links(
        self,
        limit: int = 20,
        min_score: float = 0.0,
    ) -> list[dict]:
        """Recommend missing unit links from common-neighbor topology signals."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit must be a non-negative integer.")
        if limit == 0:
            return []

        try:
            minimum_score = float(min_score)
        except (TypeError, ValueError) as exc:
            raise ValueError("min_score must be a non-negative number.") from exc
        if not math.isfinite(minimum_score) or minimum_score < 0:
            raise ValueError("min_score must be a non-negative number.")

        units_by_id = {unit.id: unit for unit in self.store.get_all_units(limit=1000000000)}
        if len(units_by_id) < 3:
            return []

        projection = nx.Graph()
        projection.add_nodes_from(units_by_id)
        existing_pairs: set[tuple[str, str]] = set()
        for edge in self.store.get_all_edges():
            if edge.from_unit_id not in units_by_id or edge.to_unit_id not in units_by_id:
                continue
            if edge.from_unit_id == edge.to_unit_id:
                continue

            pair = tuple(sorted((edge.from_unit_id, edge.to_unit_id)))
            existing_pairs.add(pair)
            projection.add_edge(*pair)

        candidates = []
        for left_id, right_id in combinations(sorted(units_by_id), 2):
            pair = (left_id, right_id)
            if pair in existing_pairs:
                continue

            common_neighbor_ids = sorted(nx.common_neighbors(projection, left_id, right_id))
            common_neighbor_count = len(common_neighbor_ids)
            if common_neighbor_count == 0:
                continue

            union_neighbors = set(projection.neighbors(left_id)) | set(
                projection.neighbors(right_id)
            )
            jaccard_bonus = (
                common_neighbor_count / len(union_neighbors) if union_neighbors else 0.0
            )
            score = round(float(common_neighbor_count) + jaccard_bonus, 6)
            if score < minimum_score:
                continue

            candidates.append(
                {
                    "unit_ids": [left_id, right_id],
                    "units": [
                        self._unit_export_data(units_by_id[left_id]),
                        self._unit_export_data(units_by_id[right_id]),
                    ],
                    "score": score,
                    "common_neighbor_count": common_neighbor_count,
                    "common_neighbors": [
                        self._unit_export_data(units_by_id[neighbor_id])
                        for neighbor_id in common_neighbor_ids
                    ],
                }
            )

        candidates.sort(
            key=lambda candidate: (
                -candidate["score"],
                -candidate["common_neighbor_count"],
                candidate["unit_ids"],
            )
        )
        return candidates[:limit]

    def get_bridges(self, limit: int = 10) -> list[tuple[str, float]]:
        """Find bridge nodes (betweenness centrality)."""
        if not self.G.nodes:
            return []
        bc = nx.betweenness_centrality(self.G.to_undirected())
        sorted_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)
        return sorted_bc[:limit]

    def find_gaps(self) -> list[dict]:
        """Identify under-connected areas."""
        gaps = []
        for node_id in self.G.nodes:
            degree = self.G.degree(node_id)
            data = self.G.nodes[node_id]
            utility = data.get("utility_score", 0) or 0
            if degree == 0:
                gaps.append(
                    {
                        "unit_id": node_id,
                        "gap_type": "isolated",
                        "score": (utility + 1) * 2.0,
                        "reason": "No connections to other knowledge",
                    }
                )
            elif degree == 1 and utility > 0.5:
                gaps.append(
                    {
                        "unit_id": node_id,
                        "gap_type": "leaf",
                        "score": utility * 1.5,
                        "reason": "High-value node with single connection",
                    }
                )
        gaps.sort(key=lambda g: g["score"], reverse=True)
        return gaps

    def find_orphan_units(
        self,
        *,
        source_project: str | None = None,
        content_type: str | None = None,
        tag: str | None = None,
        limit: int = 20,
    ) -> dict:
        """Return units with no incoming or outgoing edges."""
        units = self.store.get_all_units(limit=1000000000)
        connected_unit_ids: set[str] = set()
        for edge in self.store.get_all_edges():
            connected_unit_ids.add(edge.from_unit_id)
            connected_unit_ids.add(edge.to_unit_id)

        filters = {
            "source_project": source_project,
            "content_type": content_type,
            "tag": tag,
            "limit": max(0, int(limit)),
        }

        def _matches(unit) -> bool:
            return (
                unit.id not in connected_unit_ids
                and (source_project is None or str(unit.source_project) == source_project)
                and (content_type is None or str(unit.content_type) == content_type)
                and (tag is None or tag in unit.tags)
            )

        matching_units = [unit for unit in units if _matches(unit)]
        matching_units.sort(
            key=lambda unit: (
                str(unit.source_project),
                str(unit.content_type),
                unit.title.lower(),
                unit.id,
            )
        )
        returned_units = matching_units[: filters["limit"]]

        return {
            "total_count": len(matching_units),
            "returned_count": len(returned_units),
            "filters": filters,
            "units": [self._unit_export_data(unit) for unit in returned_units],
        }

    def cross_project_connections(self) -> list[dict]:
        """Analyze cross-project edge density."""
        project_pairs: dict[tuple[str, str], int] = {}
        for u, v in self.G.edges():
            p1 = self.G.nodes[u].get("source_project", "")
            p2 = self.G.nodes[v].get("source_project", "")
            if p1 != p2:
                pair = tuple(sorted([p1, p2]))
                project_pairs[pair] = project_pairs.get(pair, 0) + 1
        return [
            {"projects": list(k), "edge_count": v}
            for k, v in sorted(
                project_pairs.items(), key=lambda x: x[1], reverse=True
            )
        ]

    def analyze_assortativity(self, *, top_edge_limit: int = 10) -> dict:
        """Report whether graph edges connect similar units by source, type, and tags."""
        if not self.G:
            self.rebuild()

        edge_pairs = list(self.G.edges())
        edge_count = len(edge_pairs)
        unit_ids = sorted(self.G.nodes)
        node_count = len(unit_ids)

        def _safe_attribute_assortativity(attribute: str) -> float:
            if edge_count == 0:
                return 0.0

            values_on_edges = []
            all_same_endpoint_value = True
            for left_id, right_id in edge_pairs:
                left_value = str(self.G.nodes[left_id].get(attribute, ""))
                right_value = str(self.G.nodes[right_id].get(attribute, ""))
                values_on_edges.extend([left_value, right_value])
                if left_value != right_value:
                    all_same_endpoint_value = False

            if all_same_endpoint_value:
                return 1.0
            if len(set(values_on_edges)) < 2:
                return 0.0

            try:
                score = nx.attribute_assortativity_coefficient(self.G, attribute)
            except (ZeroDivisionError, nx.NetworkXException):
                return 0.0
            return 0.0 if math.isnan(score) else float(score)

        def _tag_set(unit_id: str) -> set[str]:
            tags = self.G.nodes[unit_id].get("tags") or []
            return {str(tag).strip().lower() for tag in tags if str(tag).strip()}

        def _jaccard(left_id: str, right_id: str) -> float:
            left_tags = _tag_set(left_id)
            right_tags = _tag_set(right_id)
            if not left_tags and not right_tags:
                return 0.0
            union = left_tags | right_tags
            return len(left_tags & right_tags) / len(union) if union else 0.0

        def _average_similarity(pairs: list[tuple[str, str]]) -> float:
            if not pairs:
                return 0.0
            return sum(_jaccard(left_id, right_id) for left_id, right_id in pairs) / len(
                pairs
            )

        connected_pairs = {
            tuple(sorted((left_id, right_id))) for left_id, right_id in edge_pairs
        }
        baseline_pairs = [
            (left_id, right_id)
            for left_id, right_id in combinations(unit_ids, 2)
            if tuple(sorted((left_id, right_id))) not in connected_pairs
        ]
        baseline_pairs.sort(
            key=lambda pair: (
                hashlib.sha256(f"{pair[0]}\0{pair[1]}".encode("utf-8")).hexdigest(),
                pair[0],
                pair[1],
            )
        )
        baseline_sample = baseline_pairs[:edge_count] if edge_count else []

        cross_source_edges = []
        for from_id, to_id, data in self.G.edges(data=True):
            from_node = self.G.nodes[from_id]
            to_node = self.G.nodes[to_id]
            from_source = str(from_node.get("source_project", ""))
            to_source = str(to_node.get("source_project", ""))
            if from_source == to_source:
                continue
            cross_source_edges.append(
                {
                    "from_unit_id": from_id,
                    "from_title": str(from_node.get("title", "")),
                    "from_source_project": from_source,
                    "to_unit_id": to_id,
                    "to_title": str(to_node.get("title", "")),
                    "to_source_project": to_source,
                    "relation": str(data.get("relation", "")),
                    "weight": float(data.get("weight", 1.0) or 0.0),
                }
            )

        cross_source_edges.sort(
            key=lambda item: (
                -item["weight"],
                item["from_title"].lower(),
                item["to_title"].lower(),
                item["from_unit_id"],
                item["to_unit_id"],
            )
        )

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "source_project_assortativity": _safe_attribute_assortativity("source_project"),
            "content_type_assortativity": _safe_attribute_assortativity("content_type"),
            "tag_similarity": _average_similarity(edge_pairs),
            "baseline_tag_similarity": _average_similarity(baseline_sample),
            "top_cross_source_edges": cross_source_edges[: max(0, int(top_edge_limit))],
        }

    def analyze_source_coverage(self) -> dict:
        """Summarize graph coverage by source project and entity type."""
        units = self.store.get_all_units(limit=1000000000)
        edges = self.store.get_all_edges()

        coverage: dict[tuple[str, str], dict] = {}

        def _entry(source_project: str, source_entity_type: str) -> dict:
            key = (source_project, source_entity_type)
            if key not in coverage:
                coverage[key] = {
                    "source_project": source_project,
                    "source_entity_type": source_entity_type,
                    "unit_count": 0,
                    "edge_count": 0,
                    "orphan_count": 0,
                    "oldest_created_at": None,
                    "newest_created_at": None,
                    "last_sync_at": None,
                    "last_source_id": None,
                    "items_synced": 0,
                    "has_sync_state": False,
                }
            return coverage[key]

        unit_source: dict[str, tuple[str, str]] = {}
        touched_unit_ids: set[str] = set()
        edge_ids_by_source: dict[tuple[str, str], set[str]] = {}

        for unit in units:
            source_project = str(unit.source_project)
            source_entity_type = unit.source_entity_type
            unit_source[unit.id] = (source_project, source_entity_type)
            entry = _entry(source_project, source_entity_type)
            entry["unit_count"] += 1
            created_at = (
                unit.created_at.isoformat()
                if hasattr(unit.created_at, "isoformat")
                else str(unit.created_at)
            )
            if entry["oldest_created_at"] is None or created_at < entry["oldest_created_at"]:
                entry["oldest_created_at"] = created_at
            if entry["newest_created_at"] is None or created_at > entry["newest_created_at"]:
                entry["newest_created_at"] = created_at

        for edge in edges:
            edge_id = edge.id or f"{edge.from_unit_id}:{edge.to_unit_id}:{edge.relation}"
            source_keys = set()
            for unit_id in (edge.from_unit_id, edge.to_unit_id):
                source_key = unit_source.get(unit_id)
                if source_key is None:
                    continue
                touched_unit_ids.add(unit_id)
                source_keys.add(source_key)
            for source_key in source_keys:
                edge_ids_by_source.setdefault(source_key, set()).add(edge_id)

        for source_key, edge_ids in edge_ids_by_source.items():
            _entry(*source_key)["edge_count"] = len(edge_ids)

        orphan_counts = Counter(
            unit_source[unit.id] for unit in units if unit.id not in touched_unit_ids
        )
        for source_key, count in orphan_counts.items():
            _entry(*source_key)["orphan_count"] = count

        rows = self.store.conn.execute(
            """SELECT source_project, source_entity_type, last_sync_at,
                      last_source_id, items_synced
               FROM sync_state"""
        ).fetchall()
        for row in rows:
            entry = _entry(str(row["source_project"]), str(row["source_entity_type"]))
            entry["has_sync_state"] = True
            entry["last_sync_at"] = row["last_sync_at"]
            entry["last_source_id"] = row["last_source_id"]
            entry["items_synced"] = row["items_synced"]

        sources = sorted(
            coverage.values(),
            key=lambda item: (item["source_project"], item["source_entity_type"]),
        )
        return {"sources": sources}

    def analyze_tags(
        self,
        *,
        tag: str | None = None,
        limit: int = 20,
        source_project: str | None = None,
        content_type: str | None = None,
    ) -> dict:
        """Analyze tag counts, filtered breakdowns, and co-occurrences."""
        units = [
            unit
            for unit in self.store.get_all_units(limit=1000000000)
            if (source_project is None or str(unit.source_project) == source_project)
            and (content_type is None or str(unit.content_type) == content_type)
        ]

        filters = {
            "source_project": source_project,
            "content_type": content_type,
        }

        def _breakdowns(matching_units) -> tuple[dict[str, int], dict[str, int]]:
            return (
                dict(Counter(str(unit.source_project) for unit in matching_units)),
                dict(Counter(str(unit.content_type) for unit in matching_units)),
            )

        def _unit_summary(unit) -> dict:
            return {
                "id": unit.id,
                "title": unit.title,
                "source_project": str(unit.source_project),
                "source_entity_type": unit.source_entity_type,
                "content_type": str(unit.content_type),
                "tags": unit.tags,
                "utility_score": unit.utility_score,
            }

        if tag:
            matching_units = [unit for unit in units if tag in unit.tags]
            source_projects, content_types = _breakdowns(matching_units)
            co_counts = Counter(
                other_tag
                for unit in matching_units
                for other_tag in unit.tags
                if other_tag != tag
            )
            co_occurring_tags = [
                {"tag": name, "count": count}
                for name, count in sorted(
                    co_counts.items(), key=lambda item: (-item[1], item[0])
                )[:limit]
            ]
            return {
                "tag": tag,
                "count": len(matching_units),
                "source_projects": source_projects,
                "content_types": content_types,
                "units": [_unit_summary(unit) for unit in matching_units[:limit]],
                "co_occurring_tags": co_occurring_tags,
                "filters": filters,
            }

        by_tag: dict[str, list] = {}
        for unit in units:
            for unit_tag in unit.tags:
                by_tag.setdefault(unit_tag, []).append(unit)

        tags = []
        for unit_tag, matching_units in by_tag.items():
            source_projects, content_types = _breakdowns(matching_units)
            tags.append(
                {
                    "tag": unit_tag,
                    "count": len(matching_units),
                    "source_projects": source_projects,
                    "content_types": content_types,
                }
            )

        tags.sort(key=lambda item: (-item["count"], item["tag"]))
        return {"tags": tags[:limit], "filters": filters}

    def tag_graph(
        self,
        *,
        source_project: str | None = None,
        content_type: str | None = None,
        min_count: int = 1,
        limit: int = 20,
    ) -> dict:
        """Build a tag co-occurrence graph from filtered knowledge units."""
        if min_count < 1:
            raise ValueError("min_count must be greater than or equal to 1.")
        if limit < 0:
            raise ValueError("limit must be greater than or equal to 0.")

        units = [
            unit
            for unit in self.store.get_all_units(limit=1000000000)
            if (source_project is None or str(unit.source_project) == source_project)
            and (content_type is None or str(unit.content_type) == content_type)
        ]

        node_counts: Counter[str] = Counter()
        unit_ids_by_pair: dict[tuple[str, str], set[str]] = {}
        for unit in units:
            unit_tags = sorted(
                {str(unit_tag).strip() for unit_tag in unit.tags if str(unit_tag).strip()}
            )
            node_counts.update(unit_tags)
            for left, right in combinations(unit_tags, 2):
                unit_ids_by_pair.setdefault((left, right), set()).add(unit.id)

        candidate_edges = [
            {
                "source": left,
                "target": right,
                "tags": [left, right],
                "co_occurrence_count": len(unit_ids),
                "representative_unit_ids": sorted(unit_ids),
            }
            for (left, right), unit_ids in unit_ids_by_pair.items()
            if len(unit_ids) >= min_count
        ]
        candidate_edges.sort(
            key=lambda edge: (
                -edge["co_occurrence_count"],
                edge["source"],
                edge["target"],
            )
        )
        edges = candidate_edges[:limit]

        graph_tags = {edge["source"] for edge in edges} | {edge["target"] for edge in edges}
        nodes = [
            {"id": tag, "tag": tag, "unit_count": node_counts[tag]}
            for tag in graph_tags
        ]
        nodes.sort(key=lambda node: (-node["unit_count"], node["tag"]))

        return {
            "nodes": nodes,
            "edges": edges,
            "filters": {
                "source_project": source_project,
                "content_type": content_type,
                "min_count": min_count,
                "limit": limit,
            },
        }

    def analyze_timeline(
        self,
        *,
        bucket: str = "month",
        field: str = "created_at",
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        limit: int | None = None,
        source_project: str | None = None,
        content_type: str | None = None,
        tag: str | None = None,
    ) -> dict:
        """Bucket knowledge units over time with per-bucket breakdowns."""
        if bucket not in _TIMELINE_BUCKETS:
            raise ValueError(
                f"Unsupported timeline bucket: {bucket}. Use day, week, month, or year."
            )
        if field not in _TIMELINE_FIELDS:
            raise ValueError(
                f"Unsupported timeline field: {field}. Use created_at, ingested_at, or updated_at."
            )
        if limit is not None and limit < 0:
            raise ValueError("limit must be greater than or equal to 0.")

        start_at = _parse_timeline_datetime(start, name="start")
        end_at = _parse_timeline_datetime(end, name="end")
        if start_at and end_at and start_at > end_at:
            raise ValueError("start must be before or equal to end.")

        buckets: dict[datetime, dict] = {}
        total = 0

        for unit in self.store.get_all_units(limit=1000000000):
            if source_project is not None and str(unit.source_project) != source_project:
                continue
            if content_type is not None and str(unit.content_type) != content_type:
                continue
            if tag is not None and tag not in unit.tags:
                continue

            raw_value = getattr(unit, field)
            if isinstance(raw_value, datetime):
                value = raw_value
            else:
                value = datetime.fromisoformat(str(raw_value))
            value = _ensure_aware(value)
            if start_at is not None and value < start_at:
                continue
            if end_at is not None and value > end_at:
                continue

            bucket_start = _timeline_bucket_start(value, bucket)
            entry = buckets.setdefault(
                bucket_start,
                {
                    "bucket": _timeline_bucket_label(bucket_start, bucket),
                    "start": bucket_start.isoformat(),
                    "end": _timeline_bucket_end(bucket_start, bucket).isoformat(),
                    "count": 0,
                    "source_projects": Counter(),
                    "content_types": Counter(),
                    "tags": Counter(),
                },
            )
            entry["count"] += 1
            entry["source_projects"][str(unit.source_project)] += 1
            entry["content_types"][str(unit.content_type)] += 1
            entry["tags"].update(str(unit_tag) for unit_tag in unit.tags)
            total += 1

        bucket_items = []
        for _, item in sorted(buckets.items(), key=lambda pair: pair[0]):
            tag_counts = item.pop("tags")
            item["source_projects"] = dict(item["source_projects"])
            item["content_types"] = dict(item["content_types"])
            item["top_tags"] = [
                {"tag": name, "count": count}
                for name, count in sorted(
                    tag_counts.items(), key=lambda tag_item: (-tag_item[1], tag_item[0])
                )[:10]
            ]
            bucket_items.append(item)

        if limit is not None:
            bucket_items = bucket_items[:limit]

        return {
            "bucket": bucket,
            "field": field,
            "total": total,
            "buckets": bucket_items,
            "filters": {
                "source_project": source_project,
                "content_type": content_type,
                "tag": tag,
                "start": start,
                "end": end,
                "limit": limit,
            },
        }

    def suggest_tag_synonyms(
        self, limit: int = 20, min_similarity: float = 0.8
    ) -> dict:
        """Suggest likely synonym or variant tags without modifying stored units."""
        tag_counts = Counter(
            str(unit_tag)
            for unit in self.store.get_all_units(limit=1000000000)
            for unit_tag in unit.tags
            if str(unit_tag).strip()
        )
        normalized_by_tag = {
            tag: _normalize_tag_variant(tag) for tag in sorted(tag_counts)
        }
        tags = [
            tag for tag, normalized in normalized_by_tag.items() if normalized
        ]

        parent = {tag: tag for tag in tags}

        def _find(tag: str) -> str:
            while parent[tag] != tag:
                parent[tag] = parent[parent[tag]]
                tag = parent[tag]
            return tag

        def _union(left: str, right: str) -> None:
            left_root = _find(left)
            right_root = _find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        for index, left in enumerate(tags):
            left_normalized = normalized_by_tag[left]
            for right in tags[index + 1 :]:
                right_normalized = normalized_by_tag[right]
                if _tag_similarity(left_normalized, right_normalized) >= min_similarity:
                    _union(left, right)

        groups: dict[str, list[str]] = {}
        for tag in tags:
            groups.setdefault(_find(tag), []).append(tag)

        suggestions = []
        for grouped_tags in groups.values():
            if len(grouped_tags) < 2:
                continue

            variants = [
                {
                    "tag": tag,
                    "count": tag_counts[tag],
                    "normalized": normalized_by_tag[tag],
                }
                for tag in sorted(grouped_tags, key=lambda item: (-tag_counts[item], item.lower(), item))
            ]
            normalized_values = [normalized_by_tag[tag] for tag in grouped_tags]
            canonical_normalized = Counter(normalized_values).most_common(1)[0][0]
            canonical_candidate = canonical_normalized.replace(" ", "-")
            similarities = [
                _tag_similarity(normalized_by_tag[left], normalized_by_tag[right])
                for index, left in enumerate(grouped_tags)
                for right in grouped_tags[index + 1 :]
            ]
            suggestions.append(
                {
                    "canonical_candidate": canonical_candidate,
                    "total_count": sum(tag_counts[tag] for tag in grouped_tags),
                    "variant_count": len(grouped_tags),
                    "similarity": round(min(similarities), 6) if similarities else 1.0,
                    "variants": variants,
                }
            )

        suggestions.sort(
            key=lambda item: (
                -item["total_count"],
                -item["variant_count"],
                item["canonical_candidate"],
            )
        )
        return {
            "suggestions": suggestions[:limit],
            "limit": limit,
            "min_similarity": min_similarity,
        }

    def suggest_tags(
        self,
        unit_id: str,
        limit: int = 10,
        min_score: float = 0.25,
    ) -> dict:
        """Suggest existing graph tags for one unit without modifying stored data."""
        if limit < 0:
            raise ValueError("limit must be greater than or equal to 0.")

        unit = self.store.get_unit(unit_id)
        if unit is None:
            raise ValueError(f"Unit not found: {unit_id}")

        assigned_tags = {str(tag) for tag in unit.tags}
        unit_tokens = _tag_suggestion_tokens(unit.title, unit.content)
        title_tokens = _tag_suggestion_tokens(unit.title)
        vocabulary = self.store.tag_vocabulary(exclude_unit_id=unit_id)

        suggestions = []
        for tag, count in sorted(vocabulary.items()):
            if tag in assigned_tags:
                continue
            tag_tokens = _tag_suggestion_tokens(tag)
            if not tag_tokens:
                continue

            matched_tokens = sorted(tag_tokens & unit_tokens)
            if not matched_tokens:
                continue

            score = len(matched_tokens) / len(tag_tokens)
            if tag_tokens & title_tokens:
                score += 0.1
            score = min(score, 1.0)
            score = round(score, 6)
            if score < min_score:
                continue

            reasons = [
                "matched terms: " + ", ".join(matched_tokens),
                f"tag used on {count} other unit{'s' if count != 1 else ''}",
            ]
            title_matches = sorted(tag_tokens & title_tokens)
            if title_matches:
                reasons.append("title matched terms: " + ", ".join(title_matches))

            suggestions.append(
                {
                    "tag": tag,
                    "score": score,
                    "matched_terms": matched_tokens,
                    "usage_count": count,
                    "reasons": reasons,
                }
            )

        suggestions.sort(
            key=lambda item: (
                -item["score"],
                -item["usage_count"],
                item["tag"].lower(),
                item["tag"],
            )
        )
        return {
            "unit_id": unit.id,
            "unit": {
                "id": unit.id,
                "title": unit.title,
                "source_project": str(unit.source_project),
                "source_entity_type": unit.source_entity_type,
                "content_type": str(unit.content_type),
                "tags": unit.tags,
            },
            "suggestions": suggestions[:limit],
            "limit": limit,
            "min_score": min_score,
        }

    def suggest_edges(
        self,
        limit: int = 20,
        min_score: float = 0.4,
        source_project: str | None = None,
    ) -> dict:
        """Suggest likely missing edges without modifying stored relationships."""
        units = [
            unit
            for unit in self.store.get_all_units(limit=1000000000)
            if source_project is None or str(unit.source_project) == source_project
        ]
        units.sort(key=lambda unit: (str(unit.source_project), unit.title, unit.id))

        existing_pairs = {
            tuple(sorted((edge.from_unit_id, edge.to_unit_id)))
            for edge in self.store.get_all_edges()
        }
        tag_sets = {
            unit.id: {str(tag).strip() for tag in unit.tags if str(tag).strip()}
            for unit in units
        }
        link_sets = {unit.id: _unit_external_urls(unit) for unit in units}
        token_sets = {
            unit.id: _edge_suggestion_tokens(unit.title, unit.content)
            for unit in units
        }

        def _unit_summary(unit) -> dict:
            return {
                "id": unit.id,
                "source_project": str(unit.source_project),
                "source_id": unit.source_id,
                "source_entity_type": unit.source_entity_type,
                "title": unit.title,
                "content_type": str(unit.content_type),
                "tags": unit.tags,
            }

        candidates = []
        for index, left in enumerate(units):
            for right in units[index + 1 :]:
                pair_key = tuple(sorted((left.id, right.id)))
                if pair_key in existing_pairs:
                    continue

                shared_tags = sorted(tag_sets[left.id] & tag_sets[right.id])
                shared_links = sorted(link_sets[left.id] & link_sets[right.id])
                shared_tokens = sorted(token_sets[left.id] & token_sets[right.id])

                tag_score = min(len(shared_tags) * 0.2, 0.5)
                link_score = min(len(shared_links) * 0.45, 0.7)
                token_score = 0.0
                if shared_tokens:
                    token_overlap = len(shared_tokens) / max(
                        len(token_sets[left.id]),
                        len(token_sets[right.id]),
                        1,
                    )
                    token_score = min(token_overlap * 0.35, 0.35)

                score = min(tag_score + link_score + token_score, 1.0)
                if score < min_score:
                    continue

                reasons = []
                if shared_tags:
                    reasons.append(f"shared tags: {', '.join(shared_tags[:5])}")
                if shared_links:
                    reasons.append(f"shared links: {', '.join(shared_links[:3])}")
                if shared_tokens:
                    reasons.append(f"title/content token overlap: {', '.join(shared_tokens[:8])}")

                candidates.append(
                    {
                        "from_id": left.id,
                        "to_id": right.id,
                        "score": round(score, 6),
                        "reasons": reasons,
                        "from_unit": _unit_summary(left),
                        "to_unit": _unit_summary(right),
                    }
                )

        candidates.sort(
            key=lambda item: (
                -item["score"],
                item["from_unit"]["title"],
                item["to_unit"]["title"],
                item["from_id"],
                item["to_id"],
            )
        )
        return {
            "candidates": candidates[:limit],
            "limit": limit,
            "min_score": min_score,
            "filters": {"source_project": source_project},
        }

    def infer_reference_edges(
        self,
        *,
        dry_run: bool = False,
        source_project: str | None = None,
        content_type: str | None = None,
        limit: int | None = None,
    ) -> dict:
        """Infer REFERENCES edges when a source unit mentions another unit's known URL."""
        source_units = self.store.get_units(
            source_project=source_project,
            content_type=content_type,
            limit=limit,
        )
        all_units = self.store.get_all_units(limit=1000000000)

        def _unit_summary(unit) -> dict:
            return {
                "id": unit.id,
                "source_project": str(unit.source_project),
                "source_id": unit.source_id,
                "source_entity_type": unit.source_entity_type,
                "title": unit.title,
                "content_type": str(unit.content_type),
            }

        url_targets: dict[str, dict[str, dict]] = {}
        for unit in all_units:
            known_fields = [("source_id", unit.source_id)]
            known_fields.extend(_metadata_url_field_values(unit.metadata))
            for field, value in known_fields:
                for url in _extract_urls_from_text(value):
                    target = url_targets.setdefault(url, {}).setdefault(
                        unit.id,
                        {"unit": unit, "fields": set()},
                    )
                    target["fields"].add(field)

        existing_references = {
            (edge.from_unit_id, edge.to_unit_id)
            for edge in self.store.get_all_edges()
            if str(edge.relation) == EdgeRelation.REFERENCES.value
        }
        planned_references: set[tuple[str, str]] = set()

        candidates = []
        inserted_edges = []
        inserted = 0
        would_insert = 0
        skipped_self = 0
        skipped_duplicates = 0
        skipped_ambiguous = 0

        for source_unit in source_units:
            mentioned_urls: dict[str, set[str]] = {}
            for url in _extract_urls_from_text(source_unit.content):
                mentioned_urls.setdefault(url, set()).add("content")
            for path, value in _metadata_strings(source_unit.metadata):
                for url in _extract_urls_from_text(value):
                    mentioned_urls.setdefault(url, set()).add(path)

            for url in sorted(mentioned_urls):
                targets_by_id = url_targets.get(url)
                if not targets_by_id:
                    continue

                base_candidate = {
                    "from_unit_id": source_unit.id,
                    "from_unit": _unit_summary(source_unit),
                    "url": url,
                    "source_fields": sorted(mentioned_urls[url]),
                }

                if len(targets_by_id) > 1:
                    skipped_ambiguous += 1
                    candidates.append(
                        {
                            **base_candidate,
                            "status": "skipped_ambiguous_match",
                            "target_units": [
                                {
                                    **_unit_summary(target["unit"]),
                                    "matched_fields": sorted(target["fields"]),
                                }
                                for target in sorted(
                                    targets_by_id.values(),
                                    key=lambda item: item["unit"].id,
                                )
                            ],
                        }
                    )
                    continue

                target = next(iter(targets_by_id.values()))
                target_unit = target["unit"]
                candidate = {
                    **base_candidate,
                    "to_unit_id": target_unit.id,
                    "to_unit": _unit_summary(target_unit),
                    "target_fields": sorted(target["fields"]),
                }

                if source_unit.id == target_unit.id:
                    skipped_self += 1
                    candidates.append({**candidate, "status": "skipped_self_reference"})
                    continue

                edge_key = (source_unit.id, target_unit.id)
                if edge_key in existing_references or edge_key in planned_references:
                    skipped_duplicates += 1
                    candidates.append({**candidate, "status": "skipped_duplicate"})
                    continue

                planned_references.add(edge_key)
                if dry_run:
                    would_insert += 1
                    candidates.append({**candidate, "status": "would_insert"})
                    continue

                edge = KnowledgeEdge(
                    from_unit_id=source_unit.id,
                    to_unit_id=target_unit.id,
                    relation=EdgeRelation.REFERENCES,
                    weight=1.0,
                    source=EdgeSource.INFERRED,
                    metadata={
                        "inference": "url_reference",
                        "url": url,
                        "source_fields": sorted(mentioned_urls[url]),
                        "target_fields": sorted(target["fields"]),
                        "source_project_filter": source_project,
                        "content_type_filter": content_type,
                    },
                )
                inserted_edge = self.store.insert_edge(edge)
                inserted += 1
                inserted_edges.append(self._edge_export_data(inserted_edge))
                candidates.append({**candidate, "status": "inserted", "edge_id": inserted_edge.id})

        return {
            "dry_run": dry_run,
            "inserted": inserted,
            "created": inserted,
            "would_insert": would_insert,
            "skipped_self": skipped_self,
            "skipped_duplicates": skipped_duplicates,
            "skipped_ambiguous": skipped_ambiguous,
            "skipped": skipped_self + skipped_duplicates + skipped_ambiguous,
            "source_units_scanned": len(source_units),
            "known_urls": len(url_targets),
            "limit": limit,
            "filters": {
                "source_project": source_project,
                "content_type": content_type,
            },
            "candidates": candidates,
            "inserted_edges": inserted_edges,
        }

    def extract_references(
        self,
        *,
        dry_run: bool = False,
        source_project: str | None = None,
        content_type: str | None = None,
        limit: int | None = None,
    ) -> dict:
        """Backward-compatible alias for URL-based REFERENCES edge inference."""
        return self.infer_reference_edges(
            dry_run=dry_run,
            source_project=source_project,
            content_type=content_type,
            limit=limit,
        )

    def rename_tag(
        self,
        old_tag: str,
        new_tag: str,
        *,
        dry_run: bool = False,
        source_project: str | None = None,
        content_type: str | None = None,
        sample_limit: int = 10,
    ) -> dict:
        """Rename or merge one exact tag across matching units."""
        result = self.store.rename_tag(
            old_tag,
            new_tag,
            dry_run=dry_run,
            source_project=source_project,
            content_type=content_type,
        )
        result["sample_units"] = result["changed_units"][:sample_limit]
        result["sample_limit"] = sample_limit
        return result

    def remove_tag(
        self,
        tag: str,
        *,
        dry_run: bool = False,
        source_project: str | None = None,
        content_type: str | None = None,
        limit: int | None = None,
        sample_limit: int = 10,
    ) -> dict:
        """Remove one exact tag from matching units."""
        result = self.store.remove_tag(
            tag,
            dry_run=dry_run,
            source_project=source_project,
            content_type=content_type,
            limit=limit,
        )
        result["sample_units"] = result["changed_units"][:sample_limit]
        result["sample_limit"] = sample_limit
        return result

    def analyze_duplicates(
        self,
        *,
        limit: int = 20,
        source_project: str | None = None,
        content_type: str | None = None,
        min_title_similarity: float = 0.92,
        content_similarity: float | None = None,
    ) -> dict:
        """Find likely duplicate units without modifying graph state."""
        units = [
            unit
            for unit in self.store.get_all_units(limit=1000000000)
            if (source_project is None or str(unit.source_project) == source_project)
            and (content_type is None or str(unit.content_type) == content_type)
        ]
        units.sort(key=lambda unit: (str(unit.source_project), unit.title, unit.id))

        filters = {
            "source_project": source_project,
            "content_type": content_type,
        }

        def _unit_summary(unit) -> dict:
            return {
                "id": unit.id,
                "source_project": str(unit.source_project),
                "source_id": unit.source_id,
                "source_entity_type": unit.source_entity_type,
                "title": unit.title,
                "content_type": str(unit.content_type),
                "tags": unit.tags,
                "utility_score": unit.utility_score,
            }

        unit_by_id = {unit.id: unit for unit in units}

        def _stable_group_id(reasons: list[str], unit_ids: list[str]) -> str:
            digest = hashlib.sha1(
                f"{'|'.join(sorted(reasons))}|{'|'.join(sorted(unit_ids))}".encode("utf-8")
            ).hexdigest()[:12]
            return f"dup_{digest}"

        def _group(reason: str, key: str, score: float, unit_ids: list[str], **extra) -> dict:
            ordered_ids = sorted(
                unit_ids,
                key=lambda unit_id: (
                    str(unit_by_id[unit_id].source_project),
                    unit_by_id[unit_id].title,
                    unit_by_id[unit_id].source_id,
                    unit_id,
                ),
            )
            return {
                "id": "",
                "reason": reason,
                "reasons": [reason],
                "score": round(score, 6),
                "units": [_unit_summary(unit_by_id[unit_id]) for unit_id in ordered_ids],
                "evidence": [
                    {
                        "reason": reason,
                        "value": key,
                        "score": round(score, 6),
                    }
                ],
                **extra,
            }

        groups = []

        url_groups: dict[tuple[str, str], list[str]] = {}
        for unit in units:
            for field, raw_value in _metadata_duplicate_url_values(unit.metadata):
                url = _normalize_external_url(raw_value)
                if url is not None:
                    url_groups.setdefault((field, url), []).append(unit.id)

        for (field, url), unit_ids in url_groups.items():
            unique_ids = sorted(set(unit_ids))
            if len(unique_ids) < 2:
                continue
            groups.append(_group(field, url, 1.0, unique_ids, value=url))

        identity_groups: dict[str, list[str]] = {}
        for unit in units:
            normalized_source_id = _normalize_text(unit.source_id)
            normalized_entity_type = _normalize_text(unit.source_entity_type)
            if normalized_source_id and normalized_entity_type:
                key = f"{str(unit.source_project)}:{normalized_entity_type}:{normalized_source_id}"
                identity_groups.setdefault(key, []).append(unit.id)

        for identity, unit_ids in identity_groups.items():
            unique_ids = sorted(set(unit_ids))
            if len(unique_ids) < 2:
                continue
            groups.append(_group("source_identity", identity, 1.0, unique_ids, value=identity))

        title_units = [
            unit
            for unit in units
            if _normalize_text(unit.title)
        ]
        by_project: dict[str, list] = {}
        for unit in title_units:
            by_project.setdefault(str(unit.source_project), []).append(unit)

        for project, project_units in by_project.items():
            adjacency: dict[str, set[str]] = {unit.id: set() for unit in project_units}
            pair_scores: dict[tuple[str, str], float] = {}
            normalized_titles = {
                unit.id: _normalize_text(unit.title)
                for unit in project_units
            }
            for index, left in enumerate(project_units):
                for right in project_units[index + 1 :]:
                    score = SequenceMatcher(
                        None,
                        normalized_titles[left.id],
                        normalized_titles[right.id],
                    ).ratio()
                    if score >= min_title_similarity:
                        adjacency[left.id].add(right.id)
                        adjacency[right.id].add(left.id)
                        pair_scores[tuple(sorted((left.id, right.id)))] = score

            seen: set[str] = set()
            for unit in project_units:
                if unit.id in seen or not adjacency[unit.id]:
                    continue
                stack = [unit.id]
                component = []
                seen.add(unit.id)
                while stack:
                    current = stack.pop()
                    component.append(current)
                    for neighbor in sorted(adjacency[current]):
                        if neighbor not in seen:
                            seen.add(neighbor)
                            stack.append(neighbor)

                if len(component) < 2:
                    continue
                component_pairs = [
                    score
                    for pair, score in pair_scores.items()
                    if pair[0] in component and pair[1] in component
                ]
                groups.append(
                    _group(
                        "title_similarity",
                        (
                            f"{project}:"
                            f"{'|'.join(sorted(normalized_titles[unit_id] for unit_id in component))}"
                        ),
                        min(component_pairs) if component_pairs else 1.0,
                        component,
                        source_project=project,
                        min_title_similarity=min_title_similarity,
                    )
                )

        merged_by_units: dict[tuple[str, ...], dict] = {}
        reason_rank = {
            "canonical_url": 0,
            "link": 1,
            "source_identity": 2,
            "title_similarity": 3,
        }
        for group in groups:
            unit_ids = tuple(sorted(unit["id"] for unit in group["units"]))
            existing = merged_by_units.get(unit_ids)
            if existing is None:
                merged_by_units[unit_ids] = dict(group)
                continue
            reasons = list(existing["reasons"])
            if group["reason"] not in reasons:
                reasons.append(group["reason"])
            reasons.sort(key=lambda reason: (reason_rank.get(reason, 99), reason))
            existing["reasons"] = reasons
            existing["reason"] = reasons[0]
            existing["score"] = round(max(existing["score"], group["score"]), 6)
            existing["evidence"].extend(group["evidence"])

        groups = list(merged_by_units.values())
        for group in groups:
            group["id"] = _stable_group_id(
                group["reasons"],
                [unit["id"] for unit in group["units"]],
            )
            group["evidence"].sort(
                key=lambda item: (reason_rank.get(item["reason"], 99), item["reason"])
            )

        groups.sort(
            key=lambda item: (
                -item["score"],
                item["reasons"],
                -len(item["units"]),
                item["units"][0]["title"],
                item["id"],
            )
        )
        limited = groups[:limit]
        return {
            "groups": limited,
            "results": limited,
            "limit": limit,
            "min_title_similarity": min_title_similarity,
            "filters": filters,
        }

    def build_review_queue(
        self,
        limit: int = 20,
        source_project: str | None = None,
        content_type: str | None = None,
    ) -> dict:
        """Rank knowledge units that are worth resurfacing for review."""
        units = [
            unit
            for unit in self.store.get_all_units(limit=1000000000)
            if (source_project is None or str(unit.source_project) == source_project)
            and (content_type is None or str(unit.content_type) == content_type)
        ]

        degree_by_unit = Counter()
        candidate_ids = {unit.id for unit in units}
        for edge in self.store.get_all_edges():
            if edge.from_unit_id in candidate_ids:
                degree_by_unit[edge.from_unit_id] += 1
            if edge.to_unit_id in candidate_ids:
                degree_by_unit[edge.to_unit_id] += 1

        now = datetime.now(timezone.utc)

        def _unit_summary(unit) -> dict:
            return {
                "id": unit.id,
                "source_project": str(unit.source_project),
                "source_id": unit.source_id,
                "source_entity_type": unit.source_entity_type,
                "title": unit.title,
                "content_type": str(unit.content_type),
                "tags": unit.tags,
                "utility_score": unit.utility_score,
            }

        queue = []
        for unit in units:
            created_at = _ensure_aware(unit.created_at)
            age_days = max(0, int((now - created_at).total_seconds() // 86400))
            age_score = min(age_days / 365, 1.0) * 35.0

            degree = int(degree_by_unit.get(unit.id, 0))
            if degree == 0:
                degree_score = 30.0
                degree_code = "isolated"
            elif degree == 1:
                degree_score = 20.0
                degree_code = "low_degree"
            elif degree == 2:
                degree_score = 10.0
                degree_code = "low_degree"
            else:
                degree_score = max(0.0, 8.0 - float(degree))
                degree_code = "connected"

            utility = max(0.0, min(float(unit.utility_score or 0.0), 1.0))
            utility_score = utility * 20.0

            reviewed_keys = [
                key
                for key in ("reviewed_at", "last_reviewed_at")
                if unit.metadata.get(key)
            ]
            review_score = 25.0 if not reviewed_keys else 0.0
            review_code = "unreviewed" if not reviewed_keys else "reviewed"

            reasons = [
                {
                    "code": "age",
                    "value": age_days,
                    "score": round(age_score, 6),
                    "max_score": 35.0,
                },
                {
                    "code": degree_code,
                    "value": degree,
                    "score": round(degree_score, 6),
                    "max_score": 30.0,
                },
                {
                    "code": "utility_score",
                    "value": utility,
                    "score": round(utility_score, 6),
                    "max_score": 20.0,
                },
                {
                    "code": review_code,
                    "value": reviewed_keys,
                    "score": round(review_score, 6),
                    "max_score": 25.0,
                },
            ]
            score = sum(reason["score"] for reason in reasons)
            queue.append(
                {
                    "unit": _unit_summary(unit),
                    "score": round(score, 6),
                    "reasons": reasons,
                    "degree": degree,
                    "age_days": age_days,
                }
            )

        queue.sort(
            key=lambda item: (
                -item["score"],
                -item["age_days"],
                item["degree"],
                item["unit"]["title"],
                item["unit"]["id"],
            )
        )
        return {
            "queue": queue[:limit],
            "filters": {
                "source_project": source_project,
                "content_type": content_type,
            },
        }

    def analyze_links(
        self,
        *,
        domain: str | None = None,
        limit: int = 20,
    ) -> dict:
        """Inventory external http/https links across unit content and metadata."""
        domain_filter = domain.lower().rstrip(".") if domain else None
        occurrences_by_url: dict[str, dict] = {}
        occurrences_by_domain: dict[str, list[dict]] = {}

        for unit in self.store.get_all_units(limit=1000000000):
            fields = [("content", unit.content)]
            fields.extend(_metadata_strings(unit.metadata))
            for field, text in fields:
                for match in _EXTERNAL_URL_RE.finditer(text):
                    url = _normalize_external_url(match.group(0))
                    if url is None:
                        continue
                    found_domain = _external_url_domain(url)
                    if found_domain is None:
                        continue
                    if domain_filter and found_domain != domain_filter:
                        continue

                    occurrence = {
                        "unit_id": unit.id,
                        "title": unit.title,
                        "source_project": str(unit.source_project),
                        "source_id": unit.source_id,
                        "source_entity_type": unit.source_entity_type,
                        "content_type": str(unit.content_type),
                        "field": field,
                    }
                    occurrences_by_domain.setdefault(found_domain, []).append(occurrence)
                    entry = occurrences_by_url.setdefault(
                        url,
                        {
                            "url": url,
                            "domain": found_domain,
                            "count": 0,
                            "occurrences": [],
                        },
                    )
                    entry["count"] += 1
                    entry["occurrences"].append(occurrence)

        links = sorted(
            occurrences_by_url.values(),
            key=lambda item: (-item["count"], item["domain"], item["url"]),
        )

        domains = []
        for found_domain, occurrences in occurrences_by_domain.items():
            domain_urls = [
                item
                for item in links
                if item["domain"] == found_domain
            ]
            representative_units = []
            seen_units = set()
            for occurrence in occurrences:
                unit_id = occurrence["unit_id"]
                if unit_id in seen_units:
                    continue
                seen_units.add(unit_id)
                representative_units.append(
                    {
                        "id": unit_id,
                        "title": occurrence["title"],
                        "source_project": occurrence["source_project"],
                        "source_id": occurrence["source_id"],
                    }
                )
                if len(representative_units) >= 5:
                    break
            domains.append(
                {
                    "domain": found_domain,
                    "count": len(occurrences),
                    "url_count": len(domain_urls),
                    "urls": [
                        {"url": item["url"], "count": item["count"]}
                        for item in domain_urls[:limit]
                    ],
                    "representative_units": representative_units,
                }
            )

        domains.sort(key=lambda item: (-item["count"], item["domain"]))
        return {
            "domains": domains[:limit],
            "links": links[:limit],
            "filters": {"domain": domain_filter},
            "limit": limit,
            "total_occurrences": sum(item["count"] for item in links),
            "total_urls": len(links),
            "total_domains": len(domains),
        }

    def stats(self) -> dict:
        """Graph summary statistics."""
        if not self.G.nodes:
            return {
                "nodes": 0,
                "edges": 0,
                "components": 0,
                "density": 0.0,
                "by_project": {},
                "by_content_type": {},
            }
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "components": nx.number_connected_components(self.G.to_undirected()),
            "density": round(nx.density(self.G), 6),
            "by_project": dict(
                Counter(
                    d.get("source_project", "unknown")
                    for _, d in self.G.nodes(data=True)
                )
            ),
            "by_content_type": dict(
                Counter(
                    d.get("content_type", "unknown")
                    for _, d in self.G.nodes(data=True)
                )
            ),
        }

    def stats_snapshot(self, *, top_degree_limit: int = 10) -> dict:
        """Build a machine-readable graph statistics snapshot."""
        self.rebuild()

        units = sorted(self.store.get_all_units(limit=1000000000), key=lambda unit: unit.id)
        valid_unit_ids = {unit.id for unit in units}
        edges = sorted(
            (
                edge
                for edge in self.store.get_all_edges()
                if edge.from_unit_id in valid_unit_ids and edge.to_unit_id in valid_unit_ids
            ),
            key=lambda edge: (
                str(edge.relation),
                str(edge.source),
                edge.from_unit_id,
                edge.to_unit_id,
            ),
        )

        source_project_counts: Counter[str] = Counter()
        content_type_counts: Counter[str] = Counter()
        tag_counts: Counter[str] = Counter()
        for unit in units:
            source_project_counts[str(unit.source_project)] += 1
            content_type_counts[str(unit.content_type)] += 1
            tag_counts.update(str(tag) for tag in unit.tags)

        relation_counts = Counter(str(edge.relation) for edge in edges)
        edge_source_counts = Counter(str(edge.source) for edge in edges)
        embedding_status = self.store.get_embedding_status()

        graph = self.G
        isolated_count = len(list(nx.isolates(graph.to_undirected()))) if graph.nodes else 0
        ranked_units = []
        for unit in units:
            if unit.id not in graph:
                continue
            in_degree = int(graph.in_degree(unit.id))
            out_degree = int(graph.out_degree(unit.id))
            ranked_units.append(
                {
                    "id": unit.id,
                    "title": unit.title,
                    "source_project": str(unit.source_project),
                    "content_type": str(unit.content_type),
                    "degree": in_degree + out_degree,
                    "in_degree": in_degree,
                    "out_degree": out_degree,
                }
            )
        ranked_units.sort(
            key=lambda item: (
                -int(item["degree"]),
                -int(item["in_degree"]),
                -int(item["out_degree"]),
                str(item["title"]).lower(),
                str(item["id"]),
            )
        )

        return {
            "unit_counts": {
                "total": len(units),
                "by_source_project": dict(sorted(source_project_counts.items())),
                "by_content_type": dict(sorted(content_type_counts.items())),
                "by_tag": dict(sorted(tag_counts.items())),
            },
            "edge_counts": {
                "total": len(edges),
                "by_relation": dict(sorted(relation_counts.items())),
                "by_source": dict(sorted(edge_source_counts.items())),
            },
            "embedding_counts": {
                "with_embeddings": int(embedding_status["total"]) - int(embedding_status["missing"]),
                "without_embeddings": int(embedding_status["missing"]),
            },
            "isolated_count": isolated_count,
            "top_degree_units": ranked_units[: max(0, top_degree_limit)],
        }

    def integrity_audit(self, *, repair_fts: bool = False, limit: int = 20) -> dict:
        """Audit persisted graph tables for consistency issues."""
        repair = None
        if repair_fts:
            repair = self.store.repair_fts_index_integrity()

        categories = {
            "dangling_edges": self.store.find_dangling_edges(limit=limit),
            "self_loop_edges": self.store.find_self_loop_edges(limit=limit),
            "duplicate_edge_triples": self.store.find_duplicate_edge_triples(limit=limit),
            "units_missing_fts_rows": self.store.find_units_missing_fts_rows(limit=limit),
            "stale_fts_rows": self.store.find_stale_fts_rows(limit=limit),
            "invalid_json_rows": self.store.find_invalid_json_rows(limit=limit),
            "blank_units": self.store.find_blank_units(limit=limit),
        }
        issue_count = sum(category["count"] for category in categories.values())
        payload = {
            "issue_count": issue_count,
            "has_issues": issue_count > 0,
            "categories": categories,
            "repair": repair
            or {
                "requested": False,
                "fts_rows_inserted": 0,
                "fts_rows_deleted": 0,
            },
        }
        if repair is not None:
            payload["repair"]["requested"] = True
        return payload
