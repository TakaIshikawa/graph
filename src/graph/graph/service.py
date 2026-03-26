"""Graph service using NetworkX for in-memory graph algorithms."""

from __future__ import annotations

from collections import Counter

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
                content_type=u.content_type,
                utility_score=u.utility_score or 0.0,
                tags=u.tags,
            )
        edges = self.store.get_all_edges()
        for e in edges:
            if e.from_unit_id in self.G and e.to_unit_id in self.G:
                self.G.add_edge(
                    e.from_unit_id,
                    e.to_unit_id,
                    relation=e.relation,
                    weight=e.weight,
                    id=e.id,
                )
        return len(self.G.nodes)

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
        pr = nx.pagerank(self.G, weight="weight")
        sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
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
