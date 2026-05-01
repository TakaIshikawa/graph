"""Tests for the RAG service with a mock embedding provider."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime

import pytest

from graph.rag.embeddings import cosine_similarity, deserialize_embedding, serialize_embedding
from graph.rag.search import (
    RAGService,
    build_search_snippet,
    validate_mmr_lambda,
    validate_snippet_length,
)
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


class FixedQueryEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def test_search_snippet_prefers_query_terms_and_respects_length():
    content = "Intro text that is not relevant. " * 6 + "Solar storage economics improve fast."

    snippet = build_search_snippet(content, "storage economics", length=60)

    assert len(snippet) <= 60
    assert "storage economics" in snippet
    assert snippet.startswith("...")


def test_search_snippet_falls_back_when_terms_do_not_match():
    content = "Battery market context without the requested literal words."

    snippet = build_search_snippet(content, "solar", length=24)

    assert len(snippet) <= 24
    assert snippet.startswith("Battery")


def test_validate_snippet_length_rejects_invalid_values():
    with pytest.raises(ValueError, match="snippet_length must be between 1 and 2000"):
        validate_snippet_length(0)


def test_validate_mmr_lambda_rejects_invalid_values():
    with pytest.raises(ValueError, match="lambda_mult must be between 0 and 1"):
        validate_mmr_lambda(1.01)


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

    def test_semantic_search_mmr_prefers_diverse_embeddings(self, store: Store):
        duplicate_a = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="duplicate-a",
                source_entity_type="insight",
                title="Solar duplicate A",
                content="Solar storage duplicate A",
            )
        )
        duplicate_b = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="duplicate-b",
                source_entity_type="insight",
                title="Solar duplicate B",
                content="Solar storage duplicate B",
            )
        )
        diverse = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="diverse",
                source_entity_type="insight",
                title="Solar diverse",
                content="Solar storage diverse",
            )
        )
        store.update_embedding(duplicate_a.id, serialize_embedding([0.96, 0.28]))
        store.update_embedding(duplicate_b.id, serialize_embedding([0.95, 0.31]))
        store.update_embedding(diverse.id, serialize_embedding([0.7, 0.714]))
        service = RAGService(store, FixedQueryEmbeddingProvider())

        baseline = service.search("solar", limit=3, min_similarity=0.0)
        reranked = service.search(
            "solar",
            limit=3,
            min_similarity=0.0,
            rerank_mmr=True,
            lambda_mult=0.3,
        )

        assert [unit.source_id for unit, _score in baseline[:2]] == [
            "duplicate-a",
            "duplicate-b",
        ]
        assert [unit.source_id for unit, _score in reranked[:2]] == [
            "duplicate-a",
            "diverse",
        ]

    def test_hybrid_search(
        self, populated_store_with_embeddings: Store, rag_service: RAGService
    ):
        results = rag_service.hybrid_search("solar")
        assert len(results) > 0
        assert all(isinstance(sim, float) for _, sim in results)

    def test_hybrid_search_mmr_prefers_diverse_embeddings(self, store: Store):
        duplicate_a = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="hybrid-duplicate-a",
                source_entity_type="insight",
                title="Solar duplicate A",
                content="Solar storage duplicate A",
            )
        )
        duplicate_b = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="hybrid-duplicate-b",
                source_entity_type="insight",
                title="Solar duplicate B",
                content="Solar storage duplicate B",
            )
        )
        diverse = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="hybrid-diverse",
                source_entity_type="insight",
                title="Solar diverse",
                content="Solar storage diverse",
            )
        )
        for unit in [duplicate_a, duplicate_b, diverse]:
            store.fts_index_unit(unit)
        store.update_embedding(duplicate_a.id, serialize_embedding([0.96, 0.28]))
        store.update_embedding(duplicate_b.id, serialize_embedding([0.95, 0.31]))
        store.update_embedding(diverse.id, serialize_embedding([0.7, 0.714]))
        service = RAGService(store, FixedQueryEmbeddingProvider())

        baseline = service.hybrid_search("solar", limit=3)
        reranked = service.hybrid_search(
            "solar",
            limit=3,
            rerank_mmr=True,
            lambda_mult=0.3,
        )

        assert [unit.source_id for unit, _score in baseline[:2]] == [
            "hybrid-duplicate-a",
            "hybrid-duplicate-b",
        ]
        assert [unit.source_id for unit, _score in reranked[:2]] == [
            "hybrid-duplicate-a",
            "hybrid-diverse",
        ]

    def test_search_results_default_payload_omits_explanations(self, store: Store):
        unit = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="payload-default",
                source_entity_type="insight",
                title="Solar default payload",
                content="Solar content",
                content_type=ContentType.INSIGHT,
            )
        )
        store.fts_index_unit(unit)

        payload = RAGService(store, provider=None).search_results("solar")

        assert payload["results"][0]["id"] == unit.id
        assert "explanation" not in payload["results"][0]

    def test_fulltext_search_results_include_explanations_and_filters(self, store: Store):
        keep = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="fulltext-explain-keep",
                source_entity_type="insight",
                title="Solar grid planning",
                content="Battery storage supports solar grid operations",
                content_type=ContentType.INSIGHT,
                tags=["energy", "solar"],
                metadata={"project": {"area": "grid"}},
            )
        )
        skip = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="fulltext-explain-skip",
                source_entity_type="insight",
                title="Solar archive",
                content="Solar grid archive",
                content_type=ContentType.INSIGHT,
                tags=["archive", "solar"],
                metadata={"project": {"area": "grid"}},
            )
        )
        for unit in [keep, skip]:
            store.fts_index_unit(unit)

        payload = RAGService(store, provider=None).search_results(
            "solar grid",
            source_project="max",
            content_type="insight",
            tag="energy",
            exclude_tag="archive",
            metadata_key="project.area",
            metadata_value="grid",
            include_explanations=True,
        )

        assert [result["id"] for result in payload["results"]] == [keep.id]
        explanation = payload["results"][0]["explanation"]
        assert explanation["retrieval_mode"] == "fulltext"
        assert explanation["scores"]["fulltext_score"] == payload["results"][0]["score"]
        assert explanation["matched_terms"] == {
            "title": ["solar", "grid"],
            "content": ["solar", "grid"],
            "tags": ["solar"],
        }
        assert explanation["filters"] == {
            "source_project": "max",
            "content_type": "insight",
            "tag": "energy",
            "exclude_tag": "archive",
            "metadata_key": "project.area",
            "metadata_value": "grid",
        }

    def test_semantic_search_results_include_explanations_with_fake_provider(self, store: Store):
        unit = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="semantic-explain",
                source_entity_type="insight",
                title="Solar semantic match",
                content="Panels and storage",
                content_type=ContentType.INSIGHT,
                tags=["solar"],
            )
        )
        store.update_embedding(unit.id, serialize_embedding([1.0, 0.0]))

        payload = RAGService(store, FixedQueryEmbeddingProvider()).search_results(
            "solar panels",
            mode="semantic",
            min_similarity=0.0,
            include_explanations=True,
        )

        result = payload["results"][0]
        explanation = result["explanation"]
        assert result["id"] == unit.id
        assert explanation["retrieval_mode"] == "semantic"
        assert explanation["scores"]["similarity"] == result["score"]
        assert explanation["scores"]["final_score"] == result["score"]
        assert explanation["matched_terms"]["title"] == ["solar"]
        assert explanation["matched_terms"]["content"] == ["panels"]
        assert explanation["matched_terms"]["tags"] == ["solar"]

    def test_hybrid_search_results_explain_score_components(self, store: Store):
        unit = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="hybrid-explain",
                source_entity_type="insight",
                title="Solar hybrid explanation",
                content="Solar storage content",
                content_type=ContentType.INSIGHT,
                tags=["solar"],
            )
        )
        store.fts_index_unit(unit)
        store.update_embedding(unit.id, serialize_embedding([1.0, 0.0]))

        payload = RAGService(store, FixedQueryEmbeddingProvider()).search_results(
            "solar storage",
            mode="hybrid",
            include_explanations=True,
        )

        explanation = payload["results"][0]["explanation"]
        assert explanation["retrieval_mode"] == "hybrid"
        assert explanation["scores"]["matched_modes"] == ["semantic", "fulltext"]
        assert explanation["scores"]["semantic_score"] == 1.0
        assert explanation["scores"]["fulltext_score"] > 0.0
        assert explanation["scores"]["hybrid_score"] == payload["results"][0]["score"]

    def test_hybrid_mmr_explanation_reports_position_changes(self, store: Store):
        duplicate_a = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="mmr-explain-duplicate-a",
                source_entity_type="insight",
                title="Solar duplicate A",
                content="Solar storage duplicate A",
            )
        )
        duplicate_b = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="mmr-explain-duplicate-b",
                source_entity_type="insight",
                title="Solar duplicate B",
                content="Solar storage duplicate B",
            )
        )
        diverse = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="mmr-explain-diverse",
                source_entity_type="insight",
                title="Solar diverse",
                content="Solar storage diverse",
            )
        )
        for unit in [duplicate_a, duplicate_b, diverse]:
            store.fts_index_unit(unit)
        store.update_embedding(duplicate_a.id, serialize_embedding([0.96, 0.28]))
        store.update_embedding(duplicate_b.id, serialize_embedding([0.95, 0.31]))
        store.update_embedding(diverse.id, serialize_embedding([0.7, 0.714]))

        payload = RAGService(store, FixedQueryEmbeddingProvider()).search_results(
            "solar",
            mode="hybrid",
            limit=3,
            rerank_mmr=True,
            lambda_mult=0.3,
            include_explanations=True,
        )

        assert [result["source_id"] for result in payload["results"][:2]] == [
            "mmr-explain-duplicate-a",
            "mmr-explain-diverse",
        ]
        diverse_explanation = payload["results"][1]["explanation"]
        assert diverse_explanation["mmr"] == {
            "applied": True,
            "original_rank": 3,
            "final_rank": 2,
            "position_changed": True,
        }

    def test_semantic_and_hybrid_search_exclude_exact_tag(
        self, populated_store_with_embeddings: Store, rag_service: RAGService
    ):
        semantic = rag_service.search("solar energy panels", min_similarity=0.0, exclude_tag="solar")
        hybrid = rag_service.hybrid_search("solar", exclude_tag="solar")

        assert semantic
        assert all("solar" not in unit.tags for unit, _score in semantic)
        assert all("solar" not in unit.tags for unit, _score in hybrid)

    def test_semantic_and_hybrid_search_filter_by_nested_metadata(
        self, store: Store, rag_service: RAGService
    ):
        keep = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="metadata-keep",
                source_entity_type="insight",
                title="Solar metadata keep",
                content="Solar energy storage",
                content_type=ContentType.INSIGHT,
                metadata={"project": {"area": "grid"}},
            )
        )
        skip = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="metadata-skip",
                source_entity_type="insight",
                title="Solar metadata skip",
                content="Solar energy storage",
                content_type=ContentType.INSIGHT,
                metadata={"project": {"area": "storage"}},
            )
        )
        for unit in [keep, skip]:
            store.fts_index_unit(unit)
        rag_service.embed_batch_and_store([keep.id, skip.id])

        semantic = rag_service.search(
            "solar energy",
            min_similarity=0.0,
            metadata_key="project.area",
            metadata_value="grid",
        )
        hybrid = rag_service.hybrid_search(
            "solar",
            metadata_key="project.area",
            metadata_value="grid",
        )

        assert [unit.id for unit, _score in semantic] == [keep.id]
        assert [unit.id for unit, _score in hybrid] == [keep.id]

    def test_semantic_search_can_sort_by_created_at_desc(
        self, store: Store, rag_service: RAGService
    ):
        older = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="older",
                source_entity_type="insight",
                title="Solar older",
                content="Solar storage",
                content_type=ContentType.INSIGHT,
                created_at="2026-04-20T00:00:00+00:00",
            )
        )
        newer = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="newer",
                source_entity_type="insight",
                title="Solar newer",
                content="Solar storage",
                content_type=ContentType.INSIGHT,
                created_at="2026-04-24T00:00:00+00:00",
            )
        )
        rag_service.embed_batch_and_store([older.id, newer.id])

        results = rag_service.search(
            "solar storage",
            min_similarity=0.0,
            sort="created_at_desc",
        )

        assert [unit.source_id for unit, _score in results[:2]] == ["newer", "older"]

    def test_hybrid_search_utility_sort_puts_missing_values_last(
        self, store: Store, rag_service: RAGService
    ):
        high = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="high",
                source_entity_type="insight",
                title="Solar high utility",
                content="Solar utility",
                content_type=ContentType.INSIGHT,
                utility_score=0.9,
            )
        )
        missing = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="missing",
                source_entity_type="insight",
                title="Solar missing utility",
                content="Solar utility",
                content_type=ContentType.INSIGHT,
            )
        )
        low = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="low",
                source_entity_type="insight",
                title="Solar low utility",
                content="Solar utility",
                content_type=ContentType.INSIGHT,
                utility_score=0.2,
            )
        )
        for unit in [high, missing, low]:
            store.fts_index_unit(unit)
        rag_service.embed_batch_and_store([high.id, missing.id, low.id])

        results = rag_service.hybrid_search("solar", sort="utility_desc")

        assert [unit.source_id for unit, _score in results] == ["high", "low", "missing"]

    @pytest.mark.parametrize(
        ("method_name", "kwargs", "error"),
        [
            (
                "search",
                {"created_after": "not-a-date"},
                "created_after must be an ISO-8601 date or datetime.",
            ),
            (
                "search",
                {"created_after": "2026-04-25", "created_before": "2026-04-24"},
                "created_after must be on or before created_before.",
            ),
            (
                "search",
                {"updated_after": "2026-04-25", "updated_before": "2026-04-24"},
                "updated_after must be on or before updated_before.",
            ),
            (
                "hybrid_search",
                {"created_after": "not-a-date"},
                "created_after must be an ISO-8601 date or datetime.",
            ),
            (
                "hybrid_search",
                {"created_after": "2026-04-25", "created_before": "2026-04-24"},
                "created_after must be on or before created_before.",
            ),
            (
                "hybrid_search",
                {"updated_after": "2026-04-25", "updated_before": "2026-04-24"},
                "updated_after must be on or before updated_before.",
            ),
        ],
    )
    def test_search_rejects_invalid_date_filters_before_execution(
        self,
        store: Store,
        method_name: str,
        kwargs: dict,
        error: str,
    ):
        service = RAGService(store, provider=None)

        with pytest.raises(ValueError, match=error):
            getattr(service, method_name)("solar", **kwargs)

    def test_search_accepts_valid_boundary_date_ranges(
        self, store: Store, rag_service: RAGService
    ):
        unit = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="boundary",
                source_entity_type="insight",
                title="Solar boundary",
                content="Solar boundary content",
                content_type=ContentType.INSIGHT,
                created_at=datetime.fromisoformat("2026-04-24T00:00:00+00:00"),
                updated_at=datetime.fromisoformat("2026-04-24T12:00:00+00:00"),
            )
        )
        store.fts_index_unit(unit)
        rag_service.embed_batch_and_store([unit.id])

        semantic = rag_service.search(
            "solar",
            min_similarity=0.0,
            created_after="2026-04-24T00:00:00+00:00",
            created_before="2026-04-24T00:00:00+00:00",
            updated_after="2026-04-24T12:00:00+00:00",
            updated_before="2026-04-24T12:00:00+00:00",
        )
        hybrid = rag_service.hybrid_search(
            "solar",
            created_after="2026-04-24T00:00:00+00:00",
            created_before="2026-04-24T00:00:00+00:00",
            updated_after="2026-04-24T12:00:00+00:00",
            updated_before="2026-04-24T12:00:00+00:00",
        )

        assert [result_unit.id for result_unit, _score in semantic] == [unit.id]
        assert [result_unit.id for result_unit, _score in hybrid] == [unit.id]

    def test_similar_units_uses_stored_seed_embedding_without_provider(self, store: Store):
        seed = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="seed",
                source_entity_type="insight",
                title="Solar storage seed",
                content="Solar storage adoption",
                content_type=ContentType.INSIGHT,
                tags=["energy"],
            )
        )
        close = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="close",
                source_entity_type="insight",
                title="Solar storage match",
                content="Solar storage growth",
                content_type=ContentType.INSIGHT,
                tags=["energy"],
            )
        )
        far = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="far",
                source_entity_type="knowledge_node",
                title="Python asyncio",
                content="Concurrency primitives",
                content_type=ContentType.FINDING,
                tags=["python"],
            )
        )
        store.update_embedding(seed.id, serialize_embedding([1.0, 0.0]))
        store.update_embedding(close.id, serialize_embedding([0.9, 0.1]))
        store.update_embedding(far.id, serialize_embedding([0.0, 1.0]))

        result = RAGService(store, provider=None).similar_units(seed.id, limit=5)

        assert result["source_mode"] == "embedding"
        assert [item["unit"].id for item in result["results"]] == [close.id, far.id]
        assert seed.id not in [item["unit"].id for item in result["results"]]
        assert result["results"][0]["reason"] == "embedding_similarity"

    def test_similar_units_falls_back_to_local_search_without_embeddings(self, store: Store):
        seed = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="seed",
                source_entity_type="insight",
                title="Solar storage seed",
                content="Solar battery storage market adoption",
                content_type=ContentType.INSIGHT,
                tags=["energy", "solar"],
            )
        )
        match = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="match",
                source_entity_type="insight",
                title="Battery storage market",
                content="Solar storage adoption is increasing",
                content_type=ContentType.INSIGHT,
                tags=["energy", "storage"],
            )
        )
        skip = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="skip",
                source_entity_type="knowledge_node",
                title="Solar grid finding",
                content="Solar storage grid research",
                content_type=ContentType.FINDING,
                tags=["energy"],
            )
        )
        for unit in [seed, match, skip]:
            store.fts_index_unit(unit)

        result = RAGService(store, provider=None).similar_units(
            seed.id,
            limit=5,
            source_project="max",
            content_type="insight",
            tag="storage",
        )

        assert result["source_mode"] == "local_search"
        assert [item["unit"].id for item in result["results"]] == [match.id]
        assert result["results"][0]["reason"] == "seed_text_fulltext"

    def test_similar_units_excludes_exact_tag(self, store: Store):
        seed = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="seed",
                source_entity_type="insight",
                title="Solar storage seed",
                content="Solar storage adoption",
                content_type=ContentType.INSIGHT,
                tags=["energy"],
            )
        )
        keep = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="keep",
                source_entity_type="insight",
                title="Solar storage match",
                content="Solar storage growth",
                content_type=ContentType.INSIGHT,
                tags=["energy"],
            )
        )
        skip = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="skip",
                source_entity_type="insight",
                title="Solar archived match",
                content="Solar storage archive",
                content_type=ContentType.INSIGHT,
                tags=["energy", "archive"],
            )
        )
        store.update_embedding(seed.id, serialize_embedding([1.0, 0.0]))
        store.update_embedding(keep.id, serialize_embedding([0.9, 0.1]))
        store.update_embedding(skip.id, serialize_embedding([0.8, 0.2]))

        result = RAGService(store, provider=None).similar_units(
            seed.id,
            limit=5,
            exclude_tag="archive",
        )

        assert result["filters"] == {"exclude_tag": "archive"}
        assert [item["unit"].id for item in result["results"]] == [keep.id]

    def test_search_facets_fulltext_counts_without_embedding_provider(self, store: Store):
        units = [
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="facet-max-1",
                source_entity_type="insight",
                title="Solar storage",
                content="Solar storage adoption",
                content_type=ContentType.INSIGHT,
                tags=["solar", "energy"],
            ),
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="facet-max-2",
                source_entity_type="finding",
                title="Solar financing",
                content="Solar finance programs",
                content_type=ContentType.FINDING,
                tags=["solar", "finance"],
            ),
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="facet-42",
                source_entity_type="knowledge_node",
                title="Solar grid",
                content="Solar grid constraints",
                content_type=ContentType.FINDING,
                tags=["solar", "energy"],
            ),
            KnowledgeUnit(
                source_project=SourceProject.PRESENCE,
                source_id="facet-presence",
                source_entity_type="knowledge_item",
                title="Solar artifact",
                content="Solar artifact notes",
                content_type=ContentType.ARTIFACT,
                tags=["artifact"],
            ),
        ]
        for unit in units:
            store.fts_index_unit(store.insert_unit(unit))

        result = RAGService(store, provider=None).search_facets("solar", mode="fulltext")

        assert result["total_matches"] == 4
        assert result["facets"]["source_project"] == {
            "max": 2,
            "forty_two": 1,
            "presence": 1,
        }
        assert result["facets"]["content_type"] == {
            "finding": 2,
            "artifact": 1,
            "insight": 1,
        }
        assert result["facets"]["source_entity_type"] == {
            "finding": 1,
            "insight": 1,
            "knowledge_item": 1,
            "knowledge_node": 1,
        }
        assert list(result["facets"]["tags"].items()) == [
            ("solar", 3),
            ("energy", 2),
            ("artifact", 1),
            ("finance", 1),
        ]

    def test_search_facets_honors_fulltext_filters_without_embedding_provider(
        self, store: Store
    ):
        keep = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="facet-keep",
                source_entity_type="insight",
                title="Solar grid keep",
                content="Solar grid planning",
                content_type=ContentType.INSIGHT,
                tags=["energy", "solar"],
                metadata={"project": {"area": "grid"}},
            )
        )
        skip_archived = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="facet-skip-archived",
                source_entity_type="insight",
                title="Solar grid archived",
                content="Solar grid archive",
                content_type=ContentType.INSIGHT,
                tags=["archive", "energy", "solar"],
                metadata={"project": {"area": "grid"}},
            )
        )
        skip_project = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="facet-skip-project",
                source_entity_type="knowledge_node",
                title="Solar grid other project",
                content="Solar grid research",
                content_type=ContentType.FINDING,
                tags=["energy", "solar"],
                metadata={"project": {"area": "grid"}},
            )
        )
        skip_metadata = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="facet-skip-metadata",
                source_entity_type="insight",
                title="Solar storage",
                content="Solar storage planning",
                content_type=ContentType.INSIGHT,
                tags=["energy", "solar"],
                metadata={"project": {"area": "storage"}},
            )
        )
        for unit in [keep, skip_archived, skip_project, skip_metadata]:
            store.fts_index_unit(unit)

        result = RAGService(store, provider=None).search_facets(
            "solar",
            mode="fulltext",
            source_project="max",
            content_type="insight",
            tag="energy",
            exclude_tag="archive",
            metadata_key="project.area",
            metadata_value="grid",
        )

        assert result["total_matches"] == 1
        assert result["filters"] == {
            "source_project": "max",
            "content_type": "insight",
            "tag": "energy",
            "exclude_tag": "archive",
            "metadata_key": "project.area",
            "metadata_value": "grid",
        }
        assert result["facets"]["source_project"] == {"max": 1}
        assert result["facets"]["content_type"] == {"insight": 1}
        assert result["facets"]["source_entity_type"] == {"insight": 1}
        assert result["facets"]["tags"] == {"energy": 1, "solar": 1}

    def test_search_facets_collects_hybrid_results(self, store: Store):
        keep = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="facet-hybrid-keep",
                source_entity_type="insight",
                title="Solar hybrid keep",
                content="Solar hybrid content",
                content_type=ContentType.INSIGHT,
                tags=["energy", "solar"],
            )
        )
        finding = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="facet-hybrid-finding",
                source_entity_type="finding",
                title="Solar hybrid finding",
                content="Solar hybrid finding content",
                content_type=ContentType.FINDING,
                tags=["solar"],
            )
        )
        other = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.FORTY_TWO,
                source_id="facet-hybrid-other",
                source_entity_type="knowledge_node",
                title="Solar hybrid other",
                content="Solar hybrid other content",
                content_type=ContentType.INSIGHT,
                tags=["solar"],
            )
        )
        for unit in [keep, finding, other]:
            store.fts_index_unit(unit)
        store.update_embedding(keep.id, serialize_embedding([1.0, 0.0]))
        store.update_embedding(finding.id, serialize_embedding([1.0, 0.0]))
        store.update_embedding(other.id, serialize_embedding([1.0, 0.0]))

        result = RAGService(store, FixedQueryEmbeddingProvider()).search_facets(
            "solar",
            mode="hybrid",
            source_project="max",
        )

        assert result["total_matches"] == 2
        assert result["facets"]["source_project"] == {"max": 2}
        assert result["facets"]["content_type"] == {"finding": 1, "insight": 1}
        assert result["facets"]["source_entity_type"] == {"finding": 1, "insight": 1}

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

    def test_context_pack_clamps_depth_and_respects_content_budget(
        self, store: Store, rag_service: RAGService
    ):
        center = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="center",
                source_entity_type="insight",
                title="Solar center",
                content="Solar content " * 20,
                content_type=ContentType.INSIGHT,
            )
        )
        first_hop = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="first-hop",
                source_entity_type="insight",
                title="First hop",
                content="First hop content " * 20,
                content_type=ContentType.INSIGHT,
            )
        )
        second_hop = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="second-hop",
                source_entity_type="insight",
                title="Second hop",
                content="Second hop content " * 20,
                content_type=ContentType.INSIGHT,
            )
        )
        third_hop = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="third-hop",
                source_entity_type="insight",
                title="Third hop",
                content="Third hop content",
                content_type=ContentType.INSIGHT,
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=center.id,
                to_unit_id=first_hop.id,
                relation=EdgeRelation.RELATES_TO,
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=first_hop.id,
                to_unit_id=second_hop.id,
                relation=EdgeRelation.RELATES_TO,
            )
        )
        store.insert_edge(
            KnowledgeEdge(
                from_unit_id=second_hop.id,
                to_unit_id=third_hop.id,
                relation=EdgeRelation.RELATES_TO,
            )
        )

        payload = rag_service.context_pack(
            {
                "query": "solar",
                "mode": "fulltext",
                "results": [{"id": center.id, "snippet": "Solar content"}],
            },
            char_budget=50,
            neighbor_depth=9,
        )

        excerpt_chars = sum(
            len(unit["content_excerpt"])
            for unit in [*payload["ranked_units"], *payload["neighbors"]]
        )
        assert excerpt_chars <= 50
        assert payload["metadata"]["neighbor_depth"] == 2
        assert payload["metadata"]["neighbor_depth_cap"] == 2
        assert {unit["id"] for unit in payload["neighbors"]} == {
            first_hop.id,
            second_hop.id,
        }
        assert third_hop.id not in {unit["id"] for unit in payload["neighbors"]}

    def test_context_pack_excludes_tagged_neighbors_and_edges(
        self, store: Store, rag_service: RAGService
    ):
        center = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="center",
                source_entity_type="insight",
                title="Solar center",
                content="Solar content",
                content_type=ContentType.INSIGHT,
            )
        )
        keep = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="keep",
                source_entity_type="insight",
                title="Keep neighbor",
                content="Keep content",
                content_type=ContentType.INSIGHT,
            )
        )
        archived = store.insert_unit(
            KnowledgeUnit(
                source_project=SourceProject.MAX,
                source_id="archived",
                source_entity_type="insight",
                title="Archived neighbor",
                content="Archived content",
                content_type=ContentType.INSIGHT,
                tags=["archive"],
            )
        )
        keep_edge = store.insert_edge(
            KnowledgeEdge(
                from_unit_id=center.id,
                to_unit_id=keep.id,
                relation=EdgeRelation.RELATES_TO,
            )
        )
        archived_edge = store.insert_edge(
            KnowledgeEdge(
                from_unit_id=center.id,
                to_unit_id=archived.id,
                relation=EdgeRelation.RELATES_TO,
            )
        )

        payload = rag_service.context_pack(
            {
                "query": "solar",
                "mode": "fulltext",
                "filters": {"exclude_tag": "archive"},
                "results": [{"id": center.id}],
            },
            neighbor_depth=1,
        )

        assert [unit["id"] for unit in payload["neighbors"]] == [keep.id]
        assert {edge["id"] for edge in payload["selected_edges"]} == {keep_edge.id}
        assert archived.id not in {unit["id"] for unit in payload["neighbors"]}
        assert archived_edge.id not in {edge["id"] for edge in payload["selected_edges"]}
