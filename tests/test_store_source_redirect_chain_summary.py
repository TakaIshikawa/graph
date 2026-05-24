from __future__ import annotations

from graph.store.source_redirect_chain_summary import source_redirect_chain_summary
from graph.types.models import KnowledgeUnit


def _unit(source_id: str, metadata: dict):
    return KnowledgeUnit(
        id=f"unit-{source_id}",
        source_project="web",
        source_id=source_id,
        source_entity_type="page",
        title=source_id,
        content="",
        metadata=metadata,
    )


def test_source_redirect_chain_summary_follows_targets_and_flags_cross_host_and_loops():
    rows = source_redirect_chain_summary(
        [
            _unit("a", {"url": "https://a.test/start", "redirect_url": "https://b.test/next"}),
            _unit("b", {"url": "https://b.test/next", "redirect_target": "https://b.test/final"}),
            _unit("loop-a", {"url": "https://loop.test/a", "redirect_url": "https://loop.test/b"}),
            _unit("loop-b", {"url": "https://loop.test/b", "redirect_url": "https://loop.test/a"}),
            _unit("plain", {"url": "https://plain.test/x"}),
        ]
    )

    assert rows == [
        {
            "source_id": "a",
            "source_project": "web",
            "start_url": "https://a.test/start",
            "final_url": "https://b.test/final",
            "hop_count": 2,
            "loop": False,
            "cross_host": True,
        },
        {
            "source_id": "b",
            "source_project": "web",
            "start_url": "https://b.test/next",
            "final_url": "https://b.test/final",
            "hop_count": 1,
            "loop": False,
            "cross_host": False,
        },
        {
            "source_id": "loop-a",
            "source_project": "web",
            "start_url": "https://loop.test/a",
            "final_url": "https://loop.test/a",
            "hop_count": 2,
            "loop": True,
            "cross_host": False,
        },
        {
            "source_id": "loop-b",
            "source_project": "web",
            "start_url": "https://loop.test/b",
            "final_url": "https://loop.test/b",
            "hop_count": 2,
            "loop": True,
            "cross_host": False,
        },
        {
            "source_id": "plain",
            "source_project": "web",
            "start_url": "https://plain.test/x",
            "final_url": "https://plain.test/x",
            "hop_count": 0,
            "loop": False,
            "cross_host": False,
        },
    ]
