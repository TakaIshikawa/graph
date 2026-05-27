from __future__ import annotations

from types import SimpleNamespace

from graph.store.relation_label_pair_summary import summarize_relation_label_pairs


def test_relation_label_pair_summary_counts_pairs_from_mappings_and_metadata():
    summary = summarize_relation_label_pairs(
        [
            {"source_label": "Claim", "target_label": "Evidence"},
            {"metadata": {"source_label": "claim", "target_label": "Evidence"}},
            {"from_label": "Question"},
        ]
    )

    assert summary == {
        "total_relations": 3,
        "pair_counts": [
            {"source_label": "claim", "target_label": "evidence", "count": 2},
            {"source_label": "question", "target_label": "unknown", "count": 1},
        ],
    }


def test_relation_label_pair_summary_supports_objects_and_unknowns():
    assert summarize_relation_label_pairs([SimpleNamespace(target_label="Answer")])["pair_counts"] == [
        {"source_label": "unknown", "target_label": "answer", "count": 1}
    ]
