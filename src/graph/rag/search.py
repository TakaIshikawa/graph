"""RAG search service for semantic retrieval over knowledge units."""

from __future__ import annotations

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
    ) -> list[tuple[KnowledgeUnit, float]]:
        """Semantic search. Returns (unit, similarity) pairs."""
        if self.provider is None:
            raise RuntimeError("Embedding provider is required for semantic search")
        query_embedding = self.provider.embed(query)

        candidates = self.store.get_units_with_embeddings(
            source_project=source_project,
            content_type=content_type,
        )

        results = []
        for unit, emb_bytes in candidates:
            emb = deserialize_embedding(emb_bytes)
            sim = cosine_similarity(query_embedding, emb)
            if sim >= min_similarity:
                results.append((unit, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def hybrid_search(
        self,
        query: str,
        *,
        limit: int = 10,
        semantic_weight: float = 0.6,
        fts_weight: float = 0.4,
    ) -> list[tuple[KnowledgeUnit, float]]:
        """Combined semantic + full-text search."""
        # Semantic results
        semantic_results = self.search(query, limit=limit * 2, min_similarity=0.3)
        semantic_scores = {unit.id: sim for unit, sim in semantic_results}

        # FTS results
        fts_results = self.store.fts_search(query, limit=limit * 2)
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
        for uid, score in combined[:limit]:
            unit = self.store.get_unit(uid)
            if unit:
                results.append((unit, score))
        return results

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
