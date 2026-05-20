from __future__ import annotations

from types import SimpleNamespace

import pytest

from graph.rag.context_source_limits import plan_context_source_limits


def test_context_source_limits_preserves_score_order_with_source_caps():
    report = plan_context_source_limits(
        [
            ({"id": "a", "source_project": "docs"}, 0.9),
            {"id": "b", "metadata": {"source_project": "docs", "score": 0.8}},
            SimpleNamespace(id="c", source_project="docs", score=0.7),
            {"id": "d", "source_project": "web", "score": 0.1},
        ],
        max_results=3,
        max_per_source=2,
    )

    assert report["kept_ids"] == ["a", "b", "d"]
    assert report["deferred_ids"] == ["c"]
    assert report["source_counts"] == {"docs": 2, "web": 1}
    assert report["warnings"] == ["heavy_source_concentration"]


def test_context_source_limits_validates_limits():
    with pytest.raises(ValueError, match="max_per_source"):
        plan_context_source_limits([], max_per_source=0)
