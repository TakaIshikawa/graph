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
from graph.store.db import Store, metadata_path_matches
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
DEFAULT_SEARCH_SNIPPET_LENGTH = 160
MIN_SEARCH_SNIPPET_LENGTH = 1
MAX_SEARCH_SNIPPET_LENGTH = 2000
DEFAULT_MMR_LAMBDA = 0.5
SEARCH_FACET_MODES = ("fulltext", "semantic", "hybrid")


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


def validate_snippet_length(length: int) -> int:
    if isinstance(length, bool):
        raise ValueError("snippet_length must be an integer.")
    try:
        value = int(length)
    except (TypeError, ValueError) as exc:
        raise ValueError("snippet_length must be an integer.") from exc
    if value < MIN_SEARCH_SNIPPET_LENGTH or value > MAX_SEARCH_SNIPPET_LENGTH:
        raise ValueError(
            "snippet_length must be between "
            f"{MIN_SEARCH_SNIPPET_LENGTH} and {MAX_SEARCH_SNIPPET_LENGTH}."
        )
    return value


def _query_terms(query: str) -> list[str]:
    terms = []
    seen = set()
    for term in re.findall(r"[\w-]+", query.lower()):
        if term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _snippet_window(text: str, start: int, length: int) -> str:
    if len(text) <= length:
        return text
    if length <= 3:
        return text[start : start + length]

    start = max(0, min(start, len(text) - 1))
    prefix = start > 0
    core_budget = length - (3 if prefix else 0) - 3
    if core_budget <= 0:
        return _truncate_to_budget(text[start:], length)

    end = min(len(text), start + core_budget)
    if end == len(text):
        core_budget = length - (3 if prefix else 0)
        start = max(0, len(text) - core_budget)
        prefix = start > 0
        end = len(text)
    suffix = end < len(text)

    snippet = text[start:end].strip()
    if prefix:
        snippet = "..." + snippet
    if suffix:
        snippet = snippet.rstrip() + "..."
    return _truncate_to_budget(snippet, length)


def build_search_snippet(
    content: str,
    query: str,
    *,
    length: int = DEFAULT_SEARCH_SNIPPET_LENGTH,
) -> str:
    """Return a bounded content snippet, preferring text around query terms."""
    length = validate_snippet_length(length)
    text = " ".join((content or "").split())
    if not text or len(text) <= length:
        return text

    terms = _query_terms(query)
    matches: list[tuple[int, int, int]] = []
    lowered = text.lower()
    for term in terms:
        for match in re.finditer(re.escape(term), lowered):
            window_start = max(0, match.start() - length // 3)
            window_end = min(len(text), window_start + length)
            term_hits = sum(1 for candidate in terms if candidate in lowered[window_start:window_end])
            matches.append((term_hits, -match.start(), window_start))

    if not matches:
        return _content_excerpt(text, length)

    _hits, _neg_position, window_start = max(matches)
    return _snippet_window(text, window_start, length)


def validate_search_sort(sort: str) -> str:
    if sort not in SEARCH_SORTS:
        valid = ", ".join(SEARCH_SORTS)
        raise ValueError(f"Unknown sort: {sort}. Use one of: {valid}.")
    return sort


def validate_mmr_lambda(lambda_mult: float) -> float:
    if isinstance(lambda_mult, bool):
        raise ValueError("lambda_mult must be a number between 0 and 1.")
    try:
        value = float(lambda_mult)
    except (TypeError, ValueError) as exc:
        raise ValueError("lambda_mult must be a number between 0 and 1.") from exc
    if value < 0.0 or value > 1.0:
        raise ValueError("lambda_mult must be between 0 and 1.")
    return value


def _mmr_rerank(
    results: list[tuple[KnowledgeUnit, float, list[float] | None]],
    *,
    limit: int,
    lambda_mult: float,
) -> list[tuple[KnowledgeUnit, float]]:
    """Diversify ranked results with maximal marginal relevance."""
    return [
        (unit, score)
        for _original_rank, unit, score in _mmr_rerank_with_positions(
            results,
            limit=limit,
            lambda_mult=lambda_mult,
        )
    ]


def _mmr_rerank_with_positions(
    results: list[tuple[KnowledgeUnit, float, list[float] | None]],
    *,
    limit: int,
    lambda_mult: float,
) -> list[tuple[int, KnowledgeUnit, float]]:
    """Diversify ranked results and keep each result's pre-rerank position."""
    if limit <= 0 or not results:
        return []

    remaining = list(enumerate(results))
    selected: list[tuple[int, tuple[KnowledgeUnit, float, list[float] | None]]] = []

    while remaining and len(selected) < limit:
        if not selected:
            best = max(remaining, key=lambda item: (item[1][1], -item[0]))
        else:
            selected_embeddings = [item[1][2] for item in selected if item[1][2] is not None]

            def mmr_score(item):
                index, (_unit, relevance, embedding) = item
                diversity_penalty = 0.0
                if embedding is not None and selected_embeddings:
                    diversity_penalty = max(
                        cosine_similarity(embedding, selected_embedding)
                        for selected_embedding in selected_embeddings
                    )
                return (
                    lambda_mult * relevance - (1.0 - lambda_mult) * diversity_penalty,
                    relevance,
                    -index,
                )

            best = max(remaining, key=mmr_score)

        selected.append(best)
        remaining.remove(best)

    return [(index + 1, unit, score) for index, (unit, score, _embedding) in selected]


def parse_search_datetime_filter(
    value: datetime | str | None,
    *,
    name: str,
) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO-8601 date or datetime.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def validate_search_date_range(
    after: datetime | str | None,
    before: datetime | str | None,
    *,
    after_name: str,
    before_name: str,
) -> None:
    parsed_after = parse_search_datetime_filter(after, name=after_name)
    parsed_before = parse_search_datetime_filter(before, name=before_name)
    if parsed_after and parsed_before and parsed_after > parsed_before:
        raise ValueError(f"{after_name} must be on or before {before_name}.")


def validate_search_date_filters(
    *,
    created_after: datetime | str | None = None,
    created_before: datetime | str | None = None,
    updated_after: datetime | str | None = None,
    updated_before: datetime | str | None = None,
) -> None:
    validate_search_date_range(
        created_after,
        created_before,
        after_name="created_after",
        before_name="created_before",
    )
    validate_search_date_range(
        updated_after,
        updated_before,
        after_name="updated_after",
        before_name="updated_before",
    )


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
    exclude_tag: str | None = None,
    metadata_key: str | None = None,
    metadata_value: object | None = None,
) -> bool:
    if source_project and str(unit.source_project) != source_project:
        return False
    if content_type and str(unit.content_type) != content_type:
        return False
    if tag and tag not in unit.tags:
        return False
    if exclude_tag and exclude_tag in unit.tags:
        return False
    if metadata_key is not None and metadata_value is not None:
        if not metadata_path_matches(unit.metadata, metadata_key, metadata_value):
            return False
    return True


def _sorted_count_dict(counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _facet_counts(units: list[KnowledgeUnit]) -> dict[str, dict[str, int]]:
    facets: dict[str, dict[str, int]] = {
        "source_project": {},
        "content_type": {},
        "tags": {},
        "source_entity_type": {},
    }

    for unit in units:
        values = {
            "source_project": str(unit.source_project),
            "content_type": str(unit.content_type),
            "source_entity_type": str(unit.source_entity_type),
        }
        for facet_name, value in values.items():
            counts = facets[facet_name]
            counts[value] = counts.get(value, 0) + 1
        for tag in unit.tags:
            counts = facets["tags"]
            counts[tag] = counts.get(tag, 0) + 1

    return {
        facet_name: _sorted_count_dict(counts)
        for facet_name, counts in facets.items()
    }


def _search_filter_payload(
    *,
    source_project: str | None = None,
    content_type: str | None = None,
    tag: str | None = None,
    exclude_tag: str | None = None,
    created_after: datetime | str | None = None,
    created_before: datetime | str | None = None,
    updated_after: datetime | str | None = None,
    updated_before: datetime | str | None = None,
    metadata_key: str | None = None,
    metadata_value: object | None = None,
) -> dict:
    return {
        key: value
        for key, value in {
            "source_project": source_project,
            "content_type": content_type,
            "tag": tag,
            "exclude_tag": exclude_tag,
            "created_after": created_after,
            "created_before": created_before,
            "updated_after": updated_after,
            "updated_before": updated_before,
            "metadata_key": metadata_key,
            "metadata_value": metadata_value,
        }.items()
        if value is not None
    }


def _matched_query_terms(unit: KnowledgeUnit, query: str) -> dict[str, list[str]]:
    terms = _query_terms(query)
    title = (unit.title or "").lower()
    content = (unit.content or "").lower()
    tags = [tag.lower() for tag in unit.tags]
    return {
        "title": [term for term in terms if term in title],
        "content": [term for term in terms if term in content],
        "tags": [term for term in terms if any(term in tag for tag in tags)],
    }


def _result_explanation(
    unit: KnowledgeUnit,
    query: str,
    *,
    mode: str,
    score_fields: dict,
    filters: dict,
    original_rank: int,
    final_rank: int,
    mmr_applied: bool = False,
) -> dict:
    return {
        "retrieval_mode": mode,
        "scores": score_fields,
        "matched_terms": _matched_query_terms(unit, query),
        "filters": filters,
        "mmr": {
            "applied": mmr_applied,
            "original_rank": original_rank,
            "final_rank": final_rank,
            "position_changed": original_rank != final_rank,
        },
    }


def _search_result_payload(
    unit: KnowledgeUnit,
    query: str,
    *,
    rank: int,
    score: float,
    mode: str,
    score_fields: dict,
    filters: dict,
    snippet: str | None = None,
    include_explanations: bool = False,
    original_rank: int | None = None,
    mmr_applied: bool = False,
) -> dict:
    payload = _context_unit_payload(
        unit,
        rank=rank,
        score=score,
        snippet=snippet,
    )
    if include_explanations:
        payload["explanation"] = _result_explanation(
            unit,
            query,
            mode=mode,
            score_fields=score_fields,
            filters=filters,
            original_rank=original_rank or rank,
            final_rank=rank,
            mmr_applied=mmr_applied,
        )
    return payload


def _validate_search_facet_mode(mode: str) -> str:
    if mode not in SEARCH_FACET_MODES:
        valid = ", ".join(SEARCH_FACET_MODES)
        raise ValueError(f"Unknown mode: {mode}. Use one of: {valid}.")
    return mode


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
        tag: str | None = None,
        exclude_tag: str | None = None,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        updated_after: datetime | str | None = None,
        updated_before: datetime | str | None = None,
        metadata_key: str | None = None,
        metadata_value: object | None = None,
        sort: str = "relevance",
        rerank_mmr: bool = False,
        lambda_mult: float = DEFAULT_MMR_LAMBDA,
        include_explanations: bool = False,
    ) -> list[tuple[KnowledgeUnit, float]] | list[dict]:
        """Semantic search. Returns (unit, similarity) pairs."""
        validate_search_sort(sort)
        lambda_mult = validate_mmr_lambda(lambda_mult)
        validate_search_date_filters(
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
        )
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
            metadata_key=metadata_key,
            metadata_value=metadata_value,
        )

        results = []
        for unit, emb_bytes in candidates:
            if not _unit_matches_filters(
                unit,
                source_project=source_project,
                content_type=content_type,
                tag=tag,
                exclude_tag=exclude_tag,
                metadata_key=metadata_key,
                metadata_value=metadata_value,
            ):
                continue
            emb = deserialize_embedding(emb_bytes)
            sim = cosine_similarity(query_embedding, emb)
            if sim >= min_similarity:
                results.append((unit, sim, emb))

        results.sort(key=lambda x: x[1], reverse=True)
        filters = _search_filter_payload(
            source_project=source_project,
            content_type=content_type,
            tag=tag,
            exclude_tag=exclude_tag,
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
            metadata_key=metadata_key,
            metadata_value=metadata_value,
        )
        if rerank_mmr and sort == "relevance":
            if not include_explanations:
                return _mmr_rerank(results, limit=limit, lambda_mult=lambda_mult)
            reranked = _mmr_rerank_with_positions(
                results,
                limit=limit,
                lambda_mult=lambda_mult,
            )
            return [
                _search_result_payload(
                    unit,
                    query,
                    rank=rank,
                    score=score,
                    mode="semantic",
                    score_fields={
                        "similarity": score,
                        "final_score": score,
                    },
                    filters=filters,
                    snippet=build_search_snippet(unit.content, query),
                    include_explanations=True,
                    original_rank=original_rank,
                    mmr_applied=True,
                )
                for rank, (original_rank, unit, score) in enumerate(reranked, start=1)
            ]

        pairs = [(unit, score) for unit, score, _embedding in results]
        pairs = sort_search_results(pairs, sort)
        pairs = pairs[:limit]
        if not include_explanations:
            return pairs
        return [
            _search_result_payload(
                unit,
                query,
                rank=rank,
                score=score,
                mode="semantic",
                score_fields={
                    "similarity": score,
                    "final_score": score,
                },
                filters=filters,
                snippet=build_search_snippet(unit.content, query),
                include_explanations=True,
            )
            for rank, (unit, score) in enumerate(pairs, start=1)
        ]

    def hybrid_search(
        self,
        query: str,
        *,
        limit: int = 10,
        semantic_weight: float = 0.6,
        fts_weight: float = 0.4,
        source_project: str | None = None,
        content_type: str | None = None,
        tag: str | None = None,
        exclude_tag: str | None = None,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        updated_after: datetime | str | None = None,
        updated_before: datetime | str | None = None,
        metadata_key: str | None = None,
        metadata_value: object | None = None,
        sort: str = "relevance",
        rerank_mmr: bool = False,
        lambda_mult: float = DEFAULT_MMR_LAMBDA,
        include_explanations: bool = False,
    ) -> list[tuple[KnowledgeUnit, float]] | list[dict]:
        """Combined semantic + full-text search."""
        validate_search_sort(sort)
        lambda_mult = validate_mmr_lambda(lambda_mult)
        validate_search_date_filters(
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
        )
        # Semantic results
        semantic_results = self.search(
            query,
            limit=limit * 2,
            min_similarity=0.3,
            source_project=source_project,
            content_type=content_type,
            tag=tag,
            exclude_tag=exclude_tag,
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
            metadata_key=metadata_key,
            metadata_value=metadata_value,
            rerank_mmr=False,
        )
        semantic_scores = {unit.id: sim for unit, sim in semantic_results}
        semantic_ranks = {
            unit.id: rank for rank, (unit, _sim) in enumerate(semantic_results, start=1)
        }

        # FTS results
        fts_results = self.store.fts_search(
            query,
            limit=limit * 2,
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
            metadata_key=metadata_key,
            metadata_value=metadata_value,
        )
        fts_scores: dict[str, float] = {}
        if fts_results:
            max_rank = max(abs(r["rank"]) for r in fts_results) or 1.0
            fts_scores = {
                r["unit_id"]: abs(r["rank"]) / max_rank for r in fts_results
            }
        fts_ranks = {
            row["unit_id"]: rank for rank, row in enumerate(fts_results, start=1)
        }

        # Combine scores
        all_ids = set(semantic_scores) | set(fts_scores)
        combined = []
        for uid in all_ids:
            s_score = semantic_scores.get(uid, 0.0) * semantic_weight
            f_score = fts_scores.get(uid, 0.0) * fts_weight
            combined.append((uid, s_score + f_score))

        combined.sort(key=lambda x: x[1], reverse=True)
        combined_ranks = {
            uid: rank for rank, (uid, _score) in enumerate(combined, start=1)
        }
        filters = _search_filter_payload(
            source_project=source_project,
            content_type=content_type,
            tag=tag,
            exclude_tag=exclude_tag,
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
            metadata_key=metadata_key,
            metadata_value=metadata_value,
        )

        embedding_by_id: dict[str, list[float]] = {}
        if rerank_mmr and sort == "relevance":
            for embedded_unit, blob in self.store.get_units_with_embeddings(
                source_project=source_project,
                content_type=content_type,
                created_after=created_after,
                created_before=created_before,
                updated_after=updated_after,
                updated_before=updated_before,
                metadata_key=metadata_key,
                metadata_value=metadata_value,
            ):
                embedding_by_id[embedded_unit.id] = deserialize_embedding(blob)

        results: list[tuple[KnowledgeUnit, float]] = []
        for uid, score in combined:
            unit = self.store.get_unit(uid)
            if unit and _unit_matches_filters(
                unit,
                source_project=source_project,
                content_type=content_type,
                tag=tag,
                exclude_tag=exclude_tag,
                metadata_key=metadata_key,
                metadata_value=metadata_value,
            ):
                results.append((unit, score))
        if rerank_mmr and sort == "relevance":
            rerank_input = [
                (unit, score, embedding_by_id.get(unit.id)) for unit, score in results
            ]
            if not include_explanations:
                return _mmr_rerank(
                    rerank_input,
                    limit=limit,
                    lambda_mult=lambda_mult,
                )
            reranked = _mmr_rerank_with_positions(
                rerank_input,
                limit=limit,
                lambda_mult=lambda_mult,
            )
            return [
                _search_result_payload(
                    unit,
                    query,
                    rank=rank,
                    score=score,
                    mode="hybrid",
                    score_fields={
                        "semantic_score": semantic_scores.get(unit.id, 0.0),
                        "fulltext_score": fts_scores.get(unit.id, 0.0),
                        "semantic_weighted_score": semantic_scores.get(unit.id, 0.0)
                        * semantic_weight,
                        "fulltext_weighted_score": fts_scores.get(unit.id, 0.0)
                        * fts_weight,
                        "hybrid_score": score,
                        "final_score": score,
                        "semantic_rank": semantic_ranks.get(unit.id),
                        "fulltext_rank": fts_ranks.get(unit.id),
                        "matched_modes": [
                            mode_name
                            for mode_name, present in {
                                "semantic": unit.id in semantic_scores,
                                "fulltext": unit.id in fts_scores,
                            }.items()
                            if present
                        ],
                    },
                    filters=filters,
                    snippet=build_search_snippet(unit.content, query),
                    include_explanations=True,
                    original_rank=original_rank,
                    mmr_applied=True,
                )
                for rank, (original_rank, unit, score) in enumerate(reranked, start=1)
            ]

        results = sort_search_results(results, sort)
        results = results[:limit]
        if not include_explanations:
            return results
        return [
            _search_result_payload(
                unit,
                query,
                rank=rank,
                score=score,
                mode="hybrid",
                score_fields={
                    "semantic_score": semantic_scores.get(unit.id, 0.0),
                    "fulltext_score": fts_scores.get(unit.id, 0.0),
                    "semantic_weighted_score": semantic_scores.get(unit.id, 0.0)
                    * semantic_weight,
                    "fulltext_weighted_score": fts_scores.get(unit.id, 0.0)
                    * fts_weight,
                    "hybrid_score": score,
                    "final_score": score,
                    "combined_rank": combined_ranks.get(unit.id),
                    "semantic_rank": semantic_ranks.get(unit.id),
                    "fulltext_rank": fts_ranks.get(unit.id),
                    "matched_modes": [
                        mode_name
                        for mode_name, present in {
                            "semantic": unit.id in semantic_scores,
                            "fulltext": unit.id in fts_scores,
                        }.items()
                        if present
                    ],
                },
                filters=filters,
                snippet=build_search_snippet(unit.content, query),
                include_explanations=True,
                original_rank=combined_ranks.get(unit.id),
            )
            for rank, (unit, score) in enumerate(results, start=1)
        ]

    def fulltext_search(
        self,
        query: str,
        *,
        limit: int = 10,
        source_project: str | None = None,
        content_type: str | None = None,
        tag: str | None = None,
        exclude_tag: str | None = None,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        updated_after: datetime | str | None = None,
        updated_before: datetime | str | None = None,
        metadata_key: str | None = None,
        metadata_value: object | None = None,
        sort: str = "relevance",
        snippet_length: int = DEFAULT_SEARCH_SNIPPET_LENGTH,
        include_explanations: bool = False,
    ) -> list[dict]:
        """Full-text search result payloads."""
        validate_search_sort(sort)
        snippet_length = validate_snippet_length(snippet_length)
        validate_search_date_filters(
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
        )
        filters = _search_filter_payload(
            source_project=source_project,
            content_type=content_type,
            tag=tag,
            exclude_tag=exclude_tag,
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
            metadata_key=metadata_key,
            metadata_value=metadata_value,
        )
        rows = self.store.fts_search(
            query,
            limit=max(limit * 4, limit),
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
            metadata_key=metadata_key,
            metadata_value=metadata_value,
        )
        max_rank = max((abs(row["rank"]) for row in rows), default=1.0) or 1.0
        ranked: list[tuple[KnowledgeUnit, float, dict, int]] = []
        seen: set[str] = set()
        for row_rank, row in enumerate(rows, start=1):
            unit_id = row["unit_id"]
            if unit_id in seen:
                continue
            seen.add(unit_id)
            unit = self.store.get_unit(unit_id)
            if unit is None:
                continue
            if not _unit_matches_filters(
                unit,
                source_project=source_project,
                content_type=content_type,
                tag=tag,
                exclude_tag=exclude_tag,
                metadata_key=metadata_key,
                metadata_value=metadata_value,
            ):
                continue
            score = abs(row["rank"]) / max_rank
            ranked.append((unit, score, row, row_rank))

        if sort != "relevance":
            rank_by_id = {unit.id: row_rank for unit, _score, _row, row_rank in ranked}
            row_by_id = {unit.id: row for unit, _score, row, _row_rank in ranked}
            pairs = sort_search_results(
                [(unit, score) for unit, score, _row, _row_rank in ranked],
                sort,
            )
            ranked = [
                (unit, score, row_by_id[unit.id], rank_by_id[unit.id])
                for unit, score in pairs
            ]

        payloads = []
        for final_rank, (unit, score, row, row_rank) in enumerate(
            ranked[:limit],
            start=1,
        ):
            payloads.append(
                _search_result_payload(
                    unit,
                    query,
                    rank=final_rank,
                    score=score,
                    mode="fulltext",
                    score_fields={
                        "fulltext_score": score,
                        "fulltext_rank": row_rank,
                        "raw_rank": row["rank"],
                        "final_score": score,
                    },
                    filters=filters,
                    snippet=row.get("snippet")
                    or build_search_snippet(unit.content, query, length=snippet_length),
                    include_explanations=include_explanations,
                    original_rank=row_rank,
                )
            )
        return payloads

    def search_results(
        self,
        query: str,
        *,
        mode: str = "fulltext",
        limit: int = 10,
        min_similarity: float = 0.5,
        source_project: str | None = None,
        content_type: str | None = None,
        tag: str | None = None,
        exclude_tag: str | None = None,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        updated_after: datetime | str | None = None,
        updated_before: datetime | str | None = None,
        metadata_key: str | None = None,
        metadata_value: object | None = None,
        sort: str = "relevance",
        rerank_mmr: bool = False,
        lambda_mult: float = DEFAULT_MMR_LAMBDA,
        snippet_length: int = DEFAULT_SEARCH_SNIPPET_LENGTH,
        include_explanations: bool = False,
    ) -> dict:
        """Return search results as a payload with optional per-result explanations."""
        mode = _validate_search_facet_mode(mode)
        filters = _search_filter_payload(
            source_project=source_project,
            content_type=content_type,
            tag=tag,
            exclude_tag=exclude_tag,
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
            metadata_key=metadata_key,
            metadata_value=metadata_value,
        )
        if mode == "fulltext":
            results = self.fulltext_search(
                query,
                limit=limit,
                source_project=source_project,
                content_type=content_type,
                tag=tag,
                exclude_tag=exclude_tag,
                created_after=created_after,
                created_before=created_before,
                updated_after=updated_after,
                updated_before=updated_before,
                metadata_key=metadata_key,
                metadata_value=metadata_value,
                sort=sort,
                snippet_length=snippet_length,
                include_explanations=include_explanations,
            )
        elif mode == "semantic":
            results = self.search(
                query,
                limit=limit,
                min_similarity=min_similarity,
                source_project=source_project,
                content_type=content_type,
                tag=tag,
                exclude_tag=exclude_tag,
                created_after=created_after,
                created_before=created_before,
                updated_after=updated_after,
                updated_before=updated_before,
                metadata_key=metadata_key,
                metadata_value=metadata_value,
                sort=sort,
                rerank_mmr=rerank_mmr,
                lambda_mult=lambda_mult,
                include_explanations=include_explanations,
            )
        else:
            results = self.hybrid_search(
                query,
                limit=limit,
                source_project=source_project,
                content_type=content_type,
                tag=tag,
                exclude_tag=exclude_tag,
                created_after=created_after,
                created_before=created_before,
                updated_after=updated_after,
                updated_before=updated_before,
                metadata_key=metadata_key,
                metadata_value=metadata_value,
                sort=sort,
                rerank_mmr=rerank_mmr,
                lambda_mult=lambda_mult,
                include_explanations=include_explanations,
            )

        if not include_explanations and mode in {"semantic", "hybrid"}:
            results = [
                _search_result_payload(
                    unit,
                    query,
                    rank=rank,
                    score=score,
                    mode=mode,
                    score_fields={"final_score": score},
                    filters=filters,
                    snippet=build_search_snippet(
                        unit.content,
                        query,
                        length=snippet_length,
                    ),
                )
                for rank, (unit, score) in enumerate(results, start=1)
            ]

        payload = {
            "query": query,
            "mode": mode,
            "sort": sort,
            "results": results,
            "metadata": {"sort": sort},
        }
        if filters:
            payload["filters"] = filters
        if rerank_mmr:
            payload["metadata"]["rerank_mmr"] = True
            payload["metadata"]["lambda_mult"] = lambda_mult
        return payload

    def search_facets(
        self,
        query: str,
        *,
        mode: str = "fulltext",
        limit: int | None = None,
        min_similarity: float = 0.5,
        source_project: str | None = None,
        content_type: str | None = None,
        tag: str | None = None,
        exclude_tag: str | None = None,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        updated_after: datetime | str | None = None,
        updated_before: datetime | str | None = None,
        metadata_key: str | None = None,
        metadata_value: object | None = None,
    ) -> dict:
        """Return deterministic facet counts for units matched by a search query."""
        mode = _validate_search_facet_mode(mode)
        validate_search_date_filters(
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
        )
        if tag and tag == exclude_tag:
            raise ValueError("tag and exclude_tag cannot be identical.")
        if (metadata_key is None) != (metadata_value is None):
            raise ValueError("metadata_key and metadata_value must be supplied together.")

        units = self._search_facet_units(
            query,
            mode=mode,
            limit=limit,
            min_similarity=min_similarity,
            source_project=source_project,
            content_type=content_type,
            tag=tag,
            exclude_tag=exclude_tag,
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
            metadata_key=metadata_key,
            metadata_value=metadata_value,
        )
        filters = _search_filter_payload(
            source_project=source_project,
            content_type=content_type,
            tag=tag,
            exclude_tag=exclude_tag,
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
            metadata_key=metadata_key,
            metadata_value=metadata_value,
        )
        payload = {
            "query": query,
            "mode": mode,
            "total_matches": len(units),
            "facets": _facet_counts(units),
        }
        if filters:
            payload["filters"] = filters
        return payload

    def _search_facet_units(
        self,
        query: str,
        *,
        mode: str,
        limit: int | None,
        min_similarity: float,
        source_project: str | None,
        content_type: str | None,
        tag: str | None,
        exclude_tag: str | None,
        created_after: datetime | str | None,
        created_before: datetime | str | None,
        updated_after: datetime | str | None,
        updated_before: datetime | str | None,
        metadata_key: str | None,
        metadata_value: object | None,
    ) -> list[KnowledgeUnit]:
        if mode == "fulltext":
            return self._fulltext_facet_units(
                query,
                limit=limit,
                source_project=source_project,
                content_type=content_type,
                tag=tag,
                exclude_tag=exclude_tag,
                created_after=created_after,
                created_before=created_before,
                updated_after=updated_after,
                updated_before=updated_before,
                metadata_key=metadata_key,
                metadata_value=metadata_value,
            )

        def fetch_results(fetch_limit: int) -> list[tuple[KnowledgeUnit, float]]:
            if mode == "semantic":
                return self.search(
                    query,
                    limit=fetch_limit,
                    min_similarity=min_similarity,
                    source_project=source_project,
                    content_type=content_type,
                    tag=tag,
                    exclude_tag=exclude_tag,
                    created_after=created_after,
                    created_before=created_before,
                    updated_after=updated_after,
                    updated_before=updated_before,
                    metadata_key=metadata_key,
                    metadata_value=metadata_value,
                )
            return self.hybrid_search(
                query,
                limit=fetch_limit,
                source_project=source_project,
                content_type=content_type,
                tag=tag,
                exclude_tag=exclude_tag,
                created_after=created_after,
                created_before=created_before,
                updated_after=updated_after,
                updated_before=updated_before,
                metadata_key=metadata_key,
                metadata_value=metadata_value,
            )

        fetch_limit = 100 if limit is None else max(limit, 0)
        if fetch_limit == 0:
            return []
        while True:
            pairs = fetch_results(fetch_limit)
            units = [unit for unit, _score in pairs]
            if limit is not None:
                return units[:limit]
            if len(pairs) < fetch_limit:
                return units
            fetch_limit *= 2

    def _fulltext_facet_units(
        self,
        query: str,
        *,
        limit: int | None,
        source_project: str | None,
        content_type: str | None,
        tag: str | None,
        exclude_tag: str | None,
        created_after: datetime | str | None,
        created_before: datetime | str | None,
        updated_after: datetime | str | None,
        updated_before: datetime | str | None,
        metadata_key: str | None,
        metadata_value: object | None,
    ) -> list[KnowledgeUnit]:
        fetch_limit = 100 if limit is None else max(limit, 0)
        if fetch_limit == 0:
            return []

        while True:
            rows = self.store.fts_search(
                query,
                limit=fetch_limit,
                created_after=created_after,
                created_before=created_before,
                updated_after=updated_after,
                updated_before=updated_before,
                metadata_key=metadata_key,
                metadata_value=metadata_value,
            )
            units = []
            seen: set[str] = set()
            for row in rows:
                unit = self.store.get_unit(row["unit_id"])
                if unit is None or unit.id in seen:
                    continue
                seen.add(unit.id)
                if _unit_matches_filters(
                    unit,
                    source_project=source_project,
                    content_type=content_type,
                    tag=tag,
                    exclude_tag=exclude_tag,
                    metadata_key=metadata_key,
                    metadata_value=metadata_value,
                ):
                    units.append(unit)
                    if limit is not None and len(units) >= limit:
                        return units
            if len(rows) < fetch_limit:
                return units
            fetch_limit *= 2

    def similar_units(
        self,
        unit_id: str,
        *,
        limit: int = 10,
        source_project: str | None = None,
        content_type: str | None = None,
        tag: str | None = None,
        exclude_tag: str | None = None,
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
                        "exclude_tag": exclude_tag,
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
                "exclude_tag": exclude_tag,
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
                    exclude_tag=exclude_tag,
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
                exclude_tag=exclude_tag,
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
        exclude_tag = search_payload.get("filters", {}).get("exclude_tag")

        for rank, result in enumerate(search_payload.get("results", []), start=1):
            unit = self.store.get_unit(result["id"])
            if unit is None:
                continue
            if exclude_tag and exclude_tag in unit.tags:
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
                if exclude_tag and exclude_tag in neighbor.tags:
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
                if edge is None:
                    continue
                if exclude_tag and (
                    self._unit_has_tag(edge.from_unit_id, exclude_tag)
                    or self._unit_has_tag(edge.to_unit_id, exclude_tag)
                ):
                    continue
                selected_edges[edge.id] = _context_edge_payload(edge)

        metadata = {
            **search_payload.get("metadata", {}),
            "char_budget": char_budget,
            "content_chars_used": char_budget - remaining_budget,
            "neighbor_depth_requested": requested_depth,
            "neighbor_depth": capped_depth,
            "neighbor_depth_cap": 2,
            "result_count": len(ranked_units),
            "sort": search_payload.get("sort", "relevance"),
        }

        return {
            "query": search_payload.get("query"),
            "mode": search_payload.get("mode"),
            "sort": search_payload.get("sort", "relevance"),
            "filters": search_payload.get("filters", {}),
            "ranked_units": ranked_units,
            "neighbors": list(neighbor_units.values()),
            "selected_edges": list(selected_edges.values()),
            "metadata": metadata,
        }

    def _unit_has_tag(self, unit_id: str, tag: str) -> bool:
        unit = self.store.get_unit(unit_id)
        return unit is not None and tag in unit.tags

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
