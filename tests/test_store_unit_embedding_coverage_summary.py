from __future__ import annotations

from graph.store.unit_embedding_coverage_summary import unit_embedding_coverage_summary
from graph.types.models import KnowledgeUnit


def _unit(unit_id: str, metadata: dict, *, updated_at: str = "2026-05-02T00:00:00+00:00"):
    return KnowledgeUnit(
        id=unit_id,
        source_project="notes",
        source_id=unit_id,
        source_entity_type="note",
        title=unit_id,
        content="",
        metadata=metadata,
        updated_at=updated_at,
    )


def test_unit_embedding_coverage_summary_counts_current_stale_missing_and_malformed():
    rows = unit_embedding_coverage_summary(
        [
            _unit(
                "current",
                {"embedding": {"provider": "openai", "model": "text-a", "dimension": 3, "updated_at": "2026-05-03T00:00:00+00:00"}},
            ),
            _unit(
                "stale",
                {"embedding": {"provider": "openai", "model": "text-a", "vector": [0.1, 0.2], "updated_at": "2026-05-01T00:00:00+00:00"}},
            ),
            _unit("missing", {}),
            _unit("bad", {"embedding": {"provider": "local", "model": "bad", "dimension": "nope"}}),
        ]
    )

    assert rows == [
        {"provider": "openai", "model": "text-a", "dimension": 3, "status": "current", "count": 1},
        {"provider": "local", "model": "bad", "dimension": None, "status": "malformed", "count": 1},
        {"provider": None, "model": None, "dimension": None, "status": "missing", "count": 1},
        {"provider": "openai", "model": "text-a", "dimension": 2, "status": "stale", "count": 1},
    ]
