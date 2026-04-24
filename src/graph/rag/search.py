"""RAG search service for semantic retrieval over knowledge units."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from graph.rag.embeddings import (
    EmbeddingProvider,
    cosine_similarity,
    deserialize_embedding,
    serialize_embedding,
)
from graph.store.db import Store
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge
from graph.types.models import KnowledgeUnit

SEARCH_SORTS = (
    "relevance",
    "created_at_desc",
    "created_at_asc",
    "updated_at_desc",
    "utility_desc",
    "confidence_desc",
)


def _iso(value) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value)


def _truncate_to_budget(text: str, budget: int) -> str:
    text = " ".join((text or "").split())
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    if budget <= 3:
        return text[:budget]
    return text[: budget - 3].rstrip() + "..."


def _consume_budget(text: str, remaining_budget: int) -> str:
    return _truncate_to_budget(text, remaining_budget)


def _content_excerpt(text: str, length: int = 500) -> str:
    return _truncate_to_budget(text, length)


def validate_search_sort(sort: str) -> str:
    if sort not in SEARCH_SORTS:
        valid = ", ".join(SEARCH_SORTS)
        raise ValueError(f"Unknown sort: {sort}. Use one of: {valid}.")
    return sort


def _sort_datetime(value) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def sort_search_results(results: list, sort: str) -> list:
    """Sort result tuples whose first item is a KnowledgeUnit."""
    validate_search_sort(sort)
    if sort == "relevance":
        return results

    def unit(item):
        return item[0]

    if sort == "created_at_desc":
        return sorted(results, key=lambda item: (_sort_datetime(unit(item).created_at), unit(item).id), reverse=True)
    if sort == "created_at_asc":
        return sorted(results, key=lambda item: (_sort_datetime(unit(item).created_at), unit(item).id))
    if sort == "updated_at_desc":
        return sorted(results, key=lambda item: (_sort_datetime(unit(item).updated_at), unit(item).id), reverse=True)
    if sort == "utility_desc":
        return sorted(
            results,
            key=lambda item: (
                unit(item).utility_score is not None,
                unit(item).utility_score if unit(item).utility_score is not None else float("-inf"),
                unit(item).id,
            ),
            reverse=True,
        )
    if sort == "confidence_desc":
        return sorted(
            results,
            key=lambda item: (
                unit(item).confidence is not None,
                unit(item).confidence if unit(item).confidence is not None else float("-inf"),
                unit(item).id,
            ),
            reverse=True,
        )
    return results


def _unit_matches_filters(
    unit: KnowledgeUnit,
    *,
    source_project: str | None = None,
    content_type: str | None = None,
    tag: str | None = None,
) -> bool:
    if source_project and str(unit.source_project) != source_project:
        return False
    if content_type and str(unit.content_type) != content_type:
        return False
    if tag and tag not in unit.tags:
        return False
    return True


def _similarity_seed_query(unit: KnowledgeUnit) -> str:
    parts = [unit.title, " ".join(unit.tags), _content_excerpt(unit.content)]
    return " ".join(part for part in parts if part).strip()


def _fts_or_query(text: str) -> str:
    terms = []
    seen = set()
    for term in re.findall(r"[\w-]+", text.lower()):
        if len(term) <= 1 or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return " OR ".join(terms) or text


def _context_unit_payload(
    unit: KnowledgeUnit,
    *,
    rank: int | None = None,
    score: float | None = None,
    snippet: str | None = None,
) -> dict:
    payload = {
        "id": unit.id,
        "source_project": str(unit.source_project),
        "source_id": unit.source_id,
        "source_entity_type": unit.source_entity_type,
        "title": unit.title,
        "content_type": str(unit.content_type),
        "tags": unit.tags,
        "metadata": unit.metadata,
        "created_at": _iso(unit.created_at),
        "updated_at": _iso(unit.updated_at),
    }
    if rank is not None:
        payload["rank"] = rank
    if score is not None:
        payload["score"] = score
    if snippet is not None:
        payload["snippet"] = snippet
    if unit.confidence is not None:
        payload["confidence"] = unit.confidence
    if unit.utility_score is not None:
        payload["utility_score"] = unit.utility_score
    return payload


def _context_edge_payload(edge: KnowledgeEdge) -> dict:
    return {
        "id": edge.id,
        "from_unit_id": edge.from_unit_id,
        "to_unit_id": edge.to_unit_id,
        "relation": str(edge.relation),
        "weight": edge.weight,
        "source": str(edge.source),
        "metadata": edge.metadata,
        "created_at": _iso(edge.created_at),
    }


class RAGService:
    """Semantic search over knowledge units."""

    def __init__(self, store: Store, provider: EmbeddingProvider | None) -> None:
        self.store = store
        self.provider = provider

    def embed_unit(self, unit: KnowledgeUnit) -> list[float]:
        """Generate embedding for a knowledge unit."""
        text = f"{unit.title}\n{unit.content}"
        if unit.tags:
            text += f"\n{' '.join(unit.tags)}"
        if self.provider is None:
            raise RuntimeError("Embedding provider is required to embed units")
        return self.provider.embed(text)

    def embed_and_store(self, unit_id: str) -> None:
        """Generate and persist embedding for a unit."""
        unit = self.store.get_unit(unit_id)
        if unit is None:
            return
        embedding = self.embed_unit(unit)
        self.store.update_embedding(unit_id, serialize_embedding(embedding))

    def embed_batch_and_store(self, unit_ids: list[str]) -> int:
        """Batch embed and persist. Returns count of embedded units."""
        units = [self.store.get_unit(uid) for uid in unit_ids]
        units = [u for u in units if u is not None]
        if not units:
            return 0

        texts = []
        for u in units:
            text = f"{u.title}\n{u.content}"
            if u.tags:
                text += f"\n{' '.join(u.tags)}"
            texts.append(text)

        if self.provider is None:
            raise RuntimeError("Embedding provider is required to embed units")
        embeddings = self.provider.embed_batch(texts)

        for unit, emb in zip(units, embeddings):
            self.store.update_embedding(unit.id, serialize_embedding(emb))
        return len(units)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        min_similarity: float = 0.5,
        source_project: str | None = None,
        content_type: str | None = None,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        updated_after: datetime | str | None = None,
        updated_before: datetime | str | None = None,
        sort: str = "relevance",
    ) -> list[tuple[KnowledgeUnit, float]]:
        """Semantic search. Returns (unit, similarity) pairs."""
        validate_search_sort(sort)
        if self.provider is None:
            raise RuntimeError("Embedding provider is required for semantic search")
        query_embedding = self.provider.embed(query)

        candidates = self.store.get_units_with_embeddings(
            source_project=source_project,
            content_type=content_type,
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
        )

        results = []
        for unit, emb_bytes in candidates:
            emb = deserialize_embedding(emb_bytes)
            sim = cosine_similarity(query_embedding, emb)
            if sim >= min_similarity:
                results.append((unit, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        results = sort_search_results(results, sort)
        return results[:limit]

    def hybrid_search(
        self,
        query: str,
        *,
        limit: int = 10,
        semantic_weight: float = 0.6,
        fts_weight: float = 0.4,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        updated_after: datetime | str | None = None,
        updated_before: datetime | str | None = None,
        sort: str = "relevance",
    ) -> list[tuple[KnowledgeUnit, float]]:
        """Combined semantic + full-text search."""
        validate_search_sort(sort)
        # Semantic results
        semantic_results = self.search(
            query,
            limit=limit * 2,
            min_similarity=0.3,
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
        )
        semantic_scores = {unit.id: sim for unit, sim in semantic_results}

        # FTS results
        fts_results = self.store.fts_search(
            query,
            limit=limit * 2,
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
        )
        fts_scores: dict[str, float] = {}
        if fts_results:
            max_rank = max(abs(r["rank"]) for r in fts_results) or 1.0
            fts_scores = {
                r["unit_id"]: abs(r["rank"]) / max_rank for r in fts_results
            }

        # Combine scores
        all_ids = set(semantic_scores) | set(fts_scores)
        combined = []
        for uid in all_ids:
            s_score = semantic_scores.get(uid, 0.0) * semantic_weight
            f_score = fts_scores.get(uid, 0.0) * fts_weight
            combined.append((uid, s_score + f_score))

        combined.sort(key=lambda x: x[1], reverse=True)

        results = []
        for uid, score in combined:
            unit = self.store.get_unit(uid)
            if unit:
                results.append((unit, score))
        results = sort_search_results(results, sort)
        return results[:limit]

    def similar_units(
        self,
        unit_id: str,
        *,
        limit: int = 10,
        source_project: str | None = None,
        content_type: str | None = None,
        tag: str | None = None,
    ) -> dict:
        """Find units similar to an existing unit without embedding the seed text."""
        seed = self.store.get_unit(unit_id)
        if seed is None:
            return {
                "seed_id": unit_id,
                "seed": None,
                "results": [],
                "source_mode": "missing",
                "filters": {
                    key: value
                    for key, value in {
                        "source_project": source_project,
                        "content_type": content_type,
                        "tag": tag,
                    }.items()
                    if value is not None
                },
                "error": "unit_not_found",
            }

        candidates = self.store.get_units_with_embeddings(
            source_project=source_project,
            content_type=content_type,
        )
        seed_embedding = None
        for unit, blob in self.store.get_units_with_embeddings():
            if unit.id == unit_id:
                seed_embedding = deserialize_embedding(blob)
                break

        filters = {
            key: value
            for key, value in {
                "source_project": source_project,
                "content_type": content_type,
                "tag": tag,
            }.items()
            if value is not None
        }

        if seed_embedding is not None:
            results = []
            for unit, blob in candidates:
                if unit.id == unit_id:
                    continue
                if not _unit_matches_filters(
                    unit,
                    source_project=source_project,
                    content_type=content_type,
                    tag=tag,
                ):
                    continue
                score = cosine_similarity(seed_embedding, deserialize_embedding(blob))
                results.append(
                    {
                        "unit": unit,
                        "score": score,
                        "reason": "embedding_similarity",
                        "source_mode": "embedding",
                        "snippet": _content_excerpt(unit.content, 160),
                    }
                )

            results.sort(key=lambda item: item["score"], reverse=True)
            return {
                "seed_id": seed.id,
                "seed": seed,
                "query": _similarity_seed_query(seed),
                "results": results[:limit],
                "source_mode": "embedding",
                "filters": filters,
            }

        query = _similarity_seed_query(seed)
        fts_results = self.store.fts_search(_fts_or_query(query), limit=max(limit * 4, 20))
        max_rank = max((abs(row["rank"]) for row in fts_results), default=1.0) or 1.0
        results = []
        seen: set[str] = set()
        for row in fts_results:
            candidate_id = row["unit_id"]
            if candidate_id == unit_id or candidate_id in seen:
                continue
            seen.add(candidate_id)
            unit = self.store.get_unit(candidate_id)
            if unit is None:
                continue
            if not _unit_matches_filters(
                unit,
                source_project=source_project,
                content_type=content_type,
                tag=tag,
            ):
                continue
            results.append(
                {
                    "unit": unit,
                    "score": abs(row["rank"]) / max_rank,
                    "reason": "seed_text_fulltext",
                    "source_mode": "local_search",
                    "snippet": row.get("snippet") or _content_excerpt(unit.content, 160),
                }
            )
            if len(results) >= limit:
                break

        return {
            "seed_id": seed.id,
            "seed": seed,
            "query": query,
            "results": results,
            "source_mode": "local_search",
            "filters": filters,
        }

    def context_pack(
        self,
        search_payload: dict,
        *,
        char_budget: int = 4000,
        neighbor_depth: int = 1,
    ) -> dict:
        """Build a compact LLM context pack from an existing search payload."""
        requested_depth = neighbor_depth
        capped_depth = min(max(neighbor_depth, 0), 2)
        char_budget = max(0, char_budget)
        remaining_budget = char_budget

        ranked_units = []
        selected_edges: dict[str, dict] = {}
        neighbor_units: dict[str, dict] = {}

        for rank, result in enumerate(search_payload.get("results", []), start=1):
            unit = self.store.get_unit(result["id"])
            if unit is None:
                continue

            unit_payload = _context_unit_payload(
                unit,
                rank=rank,
                score=result.get("score"),
                snippet=result.get("snippet"),
            )
            unit_payload["content_excerpt"] = _consume_budget(
                unit.content,
                remaining_budget,
            )
            remaining_budget -= len(unit_payload["content_excerpt"])

            context = self._neighbor_context(unit.id, capped_depth)
            unit_payload["neighbor_ids"] = context["neighbor_ids"]
            unit_payload["edge_ids"] = context["edge_ids"]
            ranked_units.append(unit_payload)

            for neighbor_id in context["neighbor_ids"]:
                if neighbor_id in neighbor_units:
                    continue
                neighbor = self.store.get_unit(neighbor_id)
                if neighbor is None:
                    continue
                neighbor_payload = _context_unit_payload(neighbor)
                neighbor_payload["content_excerpt"] = _consume_budget(
                    neighbor.content,
                    remaining_budget,
                )
                remaining_budget -= len(neighbor_payload["content_excerpt"])
                neighbor_units[neighbor_id] = neighbor_payload

            for edge_id in context["edge_ids"]:
                edge = self.store.get_edge(edge_id)
                if edge is not None:
                    selected_edges[edge.id] = _context_edge_payload(edge)

        return {
            "query": search_payload.get("query"),
            "mode": search_payload.get("mode"),
            "sort": search_payload.get("sort", "relevance"),
            "filters": search_payload.get("filters", {}),
            "ranked_units": ranked_units,
            "neighbors": list(neighbor_units.values()),
            "selected_edges": list(selected_edges.values()),
            "metadata": {
                "char_budget": char_budget,
                "content_chars_used": char_budget - remaining_budget,
                "neighbor_depth_requested": requested_depth,
                "neighbor_depth": capped_depth,
                "neighbor_depth_cap": 2,
                "result_count": len(ranked_units),
                "sort": search_payload.get("sort", "relevance"),
            },
        }

    def _neighbor_context(self, unit_id: str, depth: int) -> dict:
        if depth <= 0:
            return {"neighbor_ids": [], "edge_ids": []}

        visited = {unit_id}
        frontier = {unit_id}
        edge_ids: set[str] = set()
        neighbor_depths: dict[str, int] = {}

        for current_depth in range(1, depth + 1):
            next_frontier: set[str] = set()
            for current_id in sorted(frontier):
                edges = sorted(
                    self.store.get_edges_for_unit(current_id),
                    key=lambda edge: (
                        edge.from_unit_id,
                        edge.to_unit_id,
                        str(edge.relation),
                        edge.id,
                    ),
                )
                for edge in edges:
                    other_id = (
                        edge.to_unit_id
                        if edge.from_unit_id == current_id
                        else edge.from_unit_id
                    )
                    edge_ids.add(edge.id)
                    if other_id not in visited:
                        visited.add(other_id)
                        next_frontier.add(other_id)
                        neighbor_depths[other_id] = current_depth
            frontier = next_frontier

        return {
            "neighbor_ids": sorted(
                neighbor_depths,
                key=lambda uid: (neighbor_depths[uid], uid),
            ),
            "edge_ids": sorted(edge_ids),
        }

    def infer_similarity_edges(
        self,
        *,
        min_similarity: float = 0.75,
        limit: int = 100,
        source_project: str | None = None,
        content_type: str | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Infer RELATES_TO edges between embedded units above a similarity threshold."""
        candidates = self.store.get_units_with_embeddings(
            source_project=source_project,
            content_type=content_type,
        )

        similar_pairs = []
        for left_idx, (left_unit, left_blob) in enumerate(candidates):
            left_embedding = deserialize_embedding(left_blob)
            for right_unit, right_blob in candidates[left_idx + 1 :]:
                similarity = cosine_similarity(
                    left_embedding,
                    deserialize_embedding(right_blob),
                )
                if similarity >= min_similarity:
                    similar_pairs.append((left_unit, right_unit, similarity))

        similar_pairs.sort(key=lambda item: item[2], reverse=True)

        inserted = 0
        skipped = 0
        results = []
        for left_unit, right_unit, similarity in similar_pairs[:limit]:
            pair = {
                "from_unit_id": left_unit.id,
                "from_title": left_unit.title,
                "to_unit_id": right_unit.id,
                "to_title": right_unit.title,
                "similarity": similarity,
            }

            if self.store.edge_exists_between(left_unit.id, right_unit.id):
                skipped += 1
                results.append({**pair, "status": "skipped_existing_edge"})
                continue

            if dry_run:
                results.append({**pair, "status": "would_insert"})
                continue

            edge = KnowledgeEdge(
                from_unit_id=left_unit.id,
                to_unit_id=right_unit.id,
                relation=EdgeRelation.RELATES_TO,
                weight=similarity,
                source=EdgeSource.INFERRED,
                metadata={
                    "inference": "embedding_similarity",
                    "similarity": similarity,
                    "min_similarity": min_similarity,
                    "source_project_filter": source_project,
                    "content_type_filter": content_type,
                },
            )
            self.store.insert_edge(edge)
            inserted += 1
            results.append({**pair, "status": "inserted"})

        return {
            "inserted": inserted,
            "skipped": skipped,
            "dry_run": dry_run,
            "min_similarity": min_similarity,
            "limit": limit,
            "source_project": source_project,
            "content_type": content_type,
            "candidates": results,
        }
