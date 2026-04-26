"""Graph service using NetworkX for in-memory graph algorithms."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
import hashlib
from itertools import combinations
import html
import json
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
                utility_score=u.utility_score or 0.0,
                tags=u.tags,
                created_at=str(u.created_at),
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
                    created_at=str(e.created_at),
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
                utility_score=float(data.get("utility_score", 0.0) or 0.0),
                created_at=str(data.get("created_at", "")),
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

    def shortest_path(self, from_id: str, to_id: str) -> list[str] | None:
        """Find shortest path between two units."""
        try:
            return nx.shortest_path(self.G.to_undirected(), from_id, to_id)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return None

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

    def extract_references(
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
            "would_insert": would_insert,
            "skipped_self": skipped_self,
            "skipped_duplicates": skipped_duplicates,
            "skipped_ambiguous": skipped_ambiguous,
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
