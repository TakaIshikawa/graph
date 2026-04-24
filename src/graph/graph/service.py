"""Graph service using NetworkX for in-memory graph algorithms."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from urllib.parse import quote, urlsplit, urlunsplit

import networkx as nx

from graph.store.db import Store


_NORMALIZED_TEXT_RE = re.compile(r"[^a-z0-9]+")
_TTL_LOCAL_NAME_RE = re.compile(r"[^A-Za-z0-9_]")
_EXTERNAL_URL_RE = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)
_TRAILING_URL_PUNCTUATION = ".,;:!?)]}'\""


def _normalize_text(value: str) -> str:
    return _NORMALIZED_TEXT_RE.sub(" ", value.lower()).strip()


def _content_tokens(value: str) -> Counter[str]:
    return Counter(_normalize_text(value).split())


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

    def analyze_duplicates(
        self,
        *,
        limit: int = 20,
        source_project: str | None = None,
        content_type: str | None = None,
        content_similarity: float = 0.9,
    ) -> dict:
        """Find likely duplicate units by normalized title and content similarity."""
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

        results = []

        title_groups: dict[str, list] = {}
        for unit in units:
            normalized_title = _normalize_text(unit.title)
            if normalized_title:
                title_groups.setdefault(normalized_title, []).append(unit)

        for normalized_title, matching_units in title_groups.items():
            if len(matching_units) < 2:
                continue
            results.append(
                {
                    "reason": "same_title",
                    "score": 1.0,
                    "normalized_title": normalized_title,
                    "units": [_unit_summary(unit) for unit in matching_units],
                }
            )

        fingerprints = {
            unit.id: _content_tokens(unit.content)
            for unit in units
            if _normalize_text(unit.content)
        }
        adjacency: dict[str, set[str]] = {unit_id: set() for unit_id in fingerprints}
        pair_scores: dict[tuple[str, str], float] = {}
        unit_by_id = {unit.id: unit for unit in units}
        unit_ids = list(fingerprints)

        for index, left_id in enumerate(unit_ids):
            for right_id in unit_ids[index + 1 :]:
                score = _counter_similarity(fingerprints[left_id], fingerprints[right_id])
                if score >= content_similarity:
                    adjacency[left_id].add(right_id)
                    adjacency[right_id].add(left_id)
                    pair_scores[tuple(sorted((left_id, right_id)))] = score

        seen: set[str] = set()
        for unit_id in unit_ids:
            if unit_id in seen or not adjacency[unit_id]:
                continue
            stack = [unit_id]
            component = []
            seen.add(unit_id)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in sorted(adjacency[current]):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)

            if len(component) < 2:
                continue
            component.sort(key=lambda current_id: (unit_by_id[current_id].title, current_id))
            component_pair_scores = [
                score
                for pair, score in pair_scores.items()
                if pair[0] in component and pair[1] in component
            ]
            results.append(
                {
                    "reason": "similar_content",
                    "score": round(min(component_pair_scores), 6),
                    "units": [_unit_summary(unit_by_id[current_id]) for current_id in component],
                }
            )

        results.sort(
            key=lambda item: (
                -item["score"],
                item["reason"],
                -len(item["units"]),
                item["units"][0]["title"],
            )
        )
        return {"results": results[:limit], "filters": filters}

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
