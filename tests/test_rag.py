"""Tests for the RAG service with a mock embedding provider."""

from __future__ import annotations

import os
import tempfile

import pytest

from graph.rag.embeddings import cosine_similarity, deserialize_embedding, serialize_embedding
from graph.rag.search import RAGService
from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


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
        assert store.get_embedding_status()["fresh"] == 1

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

    def test_embedded_unit_becomes_stale_after_content_update(
        self, store: Store, rag_service: RAGService
    ):
        unit = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="stale-after-update",
                source_entity_type="insight",
                title="Original",
                content="Original content",
            )
        )
        rag_service.embed_and_store(unit.id)

        assert store.get_embedding_status() == {
            "total": 1,
            "missing": 0,
            "fresh": 1,
            "stale": 0,
        }

        store.update_unit_fields(unit.id, content="Changed content")

        assert store.get_embedding_status() == {
            "total": 1,
            "missing": 0,
            "fresh": 0,
            "stale": 1,
        }

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
        assert all(isinstance(sim, float) for _, sim in results)

    def test_search_no_embeddings(self, store: Store, rag_service: RAGService):
        # No units with embeddings
        results = rag_service.search("anything")
        assert len(results) == 0

    def test_embed_nonexistent_unit(self, store: Store, rag_service: RAGService):
        # Should not raise
        rag_service.embed_and_store("nonexistent-id")

    def test_infer_similarity_edges_dry_run_does_not_write(self, store: Store, rag_service: RAGService):
        a = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="a",
                source_entity_type="insight",
                title="Alpha",
                content="Alpha content",
            )
        )
        b = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="b",
                source_entity_type="insight",
                title="Beta",
                content="Beta content",
            )
        )
        c = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="c",
                source_entity_type="insight",
                title="Gamma",
                content="Gamma content",
            )
        )
        store.update_embedding(a.id, serialize_embedding([1.0, 0.0]))
        store.update_embedding(b.id, serialize_embedding([0.8, 0.6]))
        store.update_embedding(c.id, serialize_embedding([0.0, 1.0]))

        result = rag_service.infer_similarity_edges(min_similarity=0.75, dry_run=True)

        assert result["inserted"] == 0
        assert result["skipped"] == 0
        assert len(result["candidates"]) == 1
        assert result["candidates"][0]["status"] == "would_insert"
        assert len(store.get_all_edges()) == 0

    def test_infer_similarity_edges_inserts_inferred_relates_to_edge(self, store: Store, rag_service: RAGService):
        a = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="a",
                source_entity_type="insight",
                title="Alpha",
                content="Alpha content",
            )
        )
        b = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="b",
                source_entity_type="insight",
                title="Beta",
                content="Beta content",
            )
        )
        store.update_embedding(a.id, serialize_embedding([1.0, 0.0]))
        store.update_embedding(b.id, serialize_embedding([0.8, 0.6]))

        result = rag_service.infer_similarity_edges(min_similarity=0.75)

        assert result["inserted"] == 1
        assert result["skipped"] == 0
        edges = store.get_all_edges()
        assert len(edges) == 1
        assert edges[0].from_unit_id == a.id
        assert edges[0].to_unit_id == b.id
        assert edges[0].relation == EdgeRelation.RELATES_TO
        assert edges[0].source == EdgeSource.INFERRED
        assert edges[0].weight == pytest.approx(0.8)
        assert edges[0].metadata["inference"] == "embedding_similarity"

    def test_infer_similarity_edges_skips_existing_direct_edge(self, store: Store, rag_service: RAGService):
        a = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="a",
                source_entity_type="insight",
                title="Alpha",
                content="Alpha content",
            )
        )
        b = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="b",
                source_entity_type="insight",
                title="Beta",
                content="Beta content",
            )
        )
        store.update_embedding(a.id, serialize_embedding([1.0, 0.0]))
        store.update_embedding(b.id, serialize_embedding([0.8, 0.6]))
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=b.id,
                to_unit_id=a.id,
                relation=EdgeRelation.BUILDS_ON,
            )
        )

        result = rag_service.infer_similarity_edges(min_similarity=0.75)

        assert result["inserted"] == 0
        assert result["skipped"] == 1
        assert result["candidates"][0]["status"] == "skipped_existing_edge"
        assert len(store.get_all_edges()) == 1

    def test_infer_similarity_edges_applies_source_and_content_filters(
        self, store: Store, rag_service: RAGService
    ):
        units = [
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="max-insight-a",
                source_entity_type="insight",
                title="Max insight A",
                content="A",
                content_type=ContentType.INSIGHT,
            ),
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="max-finding",
                source_entity_type="finding",
                title="Max finding",
                content="B",
                content_type=ContentType.FINDING,
            ),
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="forty-two-insight",
                source_entity_type="knowledge_node",
                title="Forty two insight",
                content="C",
                content_type=ContentType.INSIGHT,
            ),
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="max-insight-d",
                source_entity_type="insight",
                title="Max insight D",
                content="D",
                content_type=ContentType.INSIGHT,
            ),
        ]
        inserted = [store.insert_unit(unit) for unit in units]
        for unit in inserted:
            store.update_embedding(unit.id, serialize_embedding([1.0, 0.0]))

        result = rag_service.infer_similarity_edges(
            min_similarity=0.99,
            source_project="max",
            content_type="insight",
        )

        assert result["inserted"] == 1
        edges = store.get_all_edges()
        assert len(edges) == 1
        assert {edges[0].from_unit_id, edges[0].to_unit_id} == {
            inserted[0].id,
            inserted[3].id,
        }
