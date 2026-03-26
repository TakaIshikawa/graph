"""Tests for the RAG service with a mock embedding provider."""

from __future__ import annotations

import os
import tempfile

import pytest

from graph.rag.embeddings import cosine_similarity, deserialize_embedding, serialize_embedding
from graph.rag.search import RAGService
from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


class MockEmbeddingProvider:
    """Mock provider that returns deterministic embeddings based on content."""

    def embed(self, text: str) -> list[float]:
        # Simple hash-based embedding for testing
        words = text.lower().split()
        vec = [0.0] * 8
        for w in words:
            h = hash(w) % 8
            vec[h] += 0.1
        # Normalize
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    yield s
    s.close()
    os.unlink(path)


@pytest.fixture
def rag_service(store: Store):
    provider = MockEmbeddingProvider()
    return RAGService(store, provider)


@pytest.fixture
def populated_store_with_embeddings(store: Store, rag_service: RAGService):
    units = [
        KnowledgeUnit(
            source_project=SourceProject.FORTY_TWO,
            source_id="n1",
            source_entity_type="knowledge_node",
            title="Solar energy efficiency",
            content="Monocrystalline solar panels achieve 22% efficiency under test conditions",
            content_type=ContentType.FINDING,
            tags=["energy", "solar"],
        ),
        KnowledgeUnit(
            source_project=SourceProject.MAX,
            source_id="i1",
            source_entity_type="insight",
            title="API monitoring demand",
            content="Developers need better API monitoring tools for microservices",
            content_type=ContentType.INSIGHT,
            tags=["devtools", "monitoring"],
        ),
        KnowledgeUnit(
            source_project=SourceProject.PRESENCE,
            source_id="k1",
            source_entity_type="knowledge_item",
            title="Async patterns in Python",
            content="Using asyncio and structured concurrency improves throughput",
            content_type=ContentType.INSIGHT,
            tags=["python", "async"],
        ),
    ]
    ids = []
    for u in units:
        inserted = store.insert_unit(u)
        store.fts_index_unit(inserted)
        ids.append(inserted.id)

    rag_service.embed_batch_and_store(ids)
    return store


class TestEmbeddingUtils:
    def test_serialize_deserialize(self):
        embedding = [0.1, 0.2, 0.3, 0.4]
        blob = serialize_embedding(embedding)
        restored = deserialize_embedding(blob)
        assert len(restored) == 4
        for a, b in zip(embedding, restored):
            assert abs(a - b) < 1e-6

    def test_cosine_similarity_identical(self):
        vec = [1.0, 0.0, 1.0]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 1.0]
        assert cosine_similarity(a, b) == 0.0


class TestRAGService:
    def test_embed_and_store(self, store: Store, rag_service: RAGService):
        unit = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="test",
                source_entity_type="knowledge_node",
                title="Test",
                content="Test content",
            )
        )
        rag_service.embed_and_store(unit.id)
        results = store.get_units_with_embeddings()
        assert len(results) == 1

    def test_embed_batch_and_store(self, store: Store, rag_service: RAGService):
        ids = []
        for i in range(3):
            unit = store.insert_unit(
                KnowledgeUnit(
                    source_project=SourceProject.MAX,
                    source_id=f"batch-{i}",
                    source_entity_type="insight",
                    title=f"Batch unit {i}",
                    content=f"Content for unit {i}",
                )
            )
            ids.append(unit.id)

        count = rag_service.embed_batch_and_store(ids)
        assert count == 3
        assert len(store.get_units_with_embeddings()) == 3

    def test_semantic_search(
        self, populated_store_with_embeddings: Store, rag_service: RAGService
    ):
        results = rag_service.search("solar energy panels", min_similarity=0.0)
        assert len(results) > 0
        # Results are (unit, similarity) tuples
        assert all(isinstance(sim, float) for _, sim in results)

    def test_hybrid_search(
        self, populated_store_with_embeddings: Store, rag_service: RAGService
    ):
        results = rag_service.hybrid_search("solar")
        assert len(results) > 0

    def test_search_no_embeddings(self, store: Store, rag_service: RAGService):
        # No units with embeddings
        results = rag_service.search("anything")
        assert len(results) == 0

    def test_embed_nonexistent_unit(self, store: Store, rag_service: RAGService):
        # Should not raise
        rag_service.embed_and_store("nonexistent-id")
