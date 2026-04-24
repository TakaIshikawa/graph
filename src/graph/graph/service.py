"""Graph service using NetworkX for in-memory graph algorithms."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import networkx as nx

from graph.store.db import Store


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
