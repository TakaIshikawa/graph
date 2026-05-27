from __future__ import annotations

from graph.store.unit_embedding_model_mix_summary import unit_embedding_model_mix_summary


class Unit:
    def __init__(self, unit_id: str, metadata: dict):
        self.id = unit_id
        self.metadata = metadata


def test_unit_embedding_model_mix_summary_groups_provider_model_and_dimensions():
    rows = unit_embedding_model_mix_summary(
        [
            {"id": "a", "metadata": {"embedding": {"provider": "openai", "model": "text-small", "dimensions": 1536}}},
            {"id": "b", "metadata": {"embedding": {"provider": "openai", "model": "text-small", "vector": [0.1, 0.2]}}},
            {"id": "c", "metadata": {"embedding": {"provider": "local", "model": "mini", "dimension": "384"}}},
            Unit("d", {"embedding_provider": "openai", "embedding_model": "text-small", "embedding_dimensions": 1536}),
        ]
    )

    assert rows == [
        {
            "provider": "openai",
            "model": "text-small",
            "dimensions": 1536,
            "count": 2,
            "share": 0.5,
            "sample_unit_ids": ["a", "d"],
        },
        {
            "provider": "local",
            "model": "mini",
            "dimensions": 384,
            "count": 1,
            "share": 0.25,
            "sample_unit_ids": ["c"],
        },
        {
            "provider": "openai",
            "model": "text-small",
            "dimensions": 2,
            "count": 1,
            "share": 0.25,
            "sample_unit_ids": ["b"],
        },
    ]


def test_unit_embedding_model_mix_summary_includes_missing_unknown_bucket():
    rows = unit_embedding_model_mix_summary([{"id": "missing", "metadata": {}}, Unit("partial", {"embedding": {"provider": "openai"}})])

    assert rows == [
        {
            "provider": "openai",
            "model": "unknown",
            "dimensions": None,
            "count": 1,
            "share": 0.5,
            "sample_unit_ids": ["partial"],
        },
        {
            "provider": "unknown",
            "model": "unknown",
            "dimensions": None,
            "count": 1,
            "share": 0.5,
            "sample_unit_ids": ["missing"],
        },
    ]
