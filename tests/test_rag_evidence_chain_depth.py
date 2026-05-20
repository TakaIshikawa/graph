from __future__ import annotations

from graph.rag.evidence_chain_depth import analyze_evidence_chain_depth


def test_evidence_chain_depth_reports_single_hop_evidence():
    summary = analyze_evidence_chain_depth(
        [
            {"id": "source", "title": "Primary source migration details"},
            {"id": "claim", "title": "Claim summary", "depends_on": "source"},
        ]
    )

    assert summary["max_depth"] == 2
    assert summary["chains"][0] == {"root_id": "claim", "result_ids": ["claim", "source"], "depth": 2}
    assert summary["warnings"] == []


def test_evidence_chain_depth_reports_multi_hop_chains():
    summary = analyze_evidence_chain_depth(
        [
            {"id": "a", "title": "API release notes"},
            {"id": "b", "title": "API release analysis", "references": ["a"]},
            {"id": "c", "title": "API release recommendation", "cites": ["b"]},
        ]
    )

    assert summary["max_depth"] == 3
    assert summary["chains"][0]["result_ids"] == ["c", "b", "a"]
    assert summary["edge_count"] == 2


def test_evidence_chain_depth_detects_circular_references():
    summary = analyze_evidence_chain_depth(
        [
            {"id": "a", "depends_on": "b"},
            {"id": "b", "depends_on": "a"},
        ]
    )

    assert summary["circular_chains"] == [["a", "b", "a"]]
    assert "circular_evidence_chain" in summary["warnings"]


def test_evidence_chain_depth_uses_text_overlap_when_link_metadata_is_missing():
    summary = analyze_evidence_chain_depth(
        [
            {"id": "base", "title": "OAuth token rotation policy", "snippet": "access refresh token"},
            {"id": "derived", "title": "OAuth token rotation guidance", "snippet": "refresh token access"},
            {"id": "other", "title": "Billing invoice export"},
        ],
        min_overlap_terms=3,
    )

    assert summary["max_depth"] == 2
    assert summary["chains"][0]["result_ids"] == ["derived", "base"]
    assert summary["orphan_result_ids"] == ["other"]
    assert "missing_link_metadata" in summary["warnings"]


def test_evidence_chain_depth_uses_stable_chain_ordering_and_relation_labels():
    summary = analyze_evidence_chain_depth(
        [
            {"id": "a", "title": "Alpha source"},
            {"id": "b", "title": "Beta source"},
            {"id": "c", "relations": [{"relation_label": "supports", "target_id": "b"}]},
            {"id": "d", "relations": [{"relation_label": "supports", "target_id": "a"}]},
        ]
    )

    assert [chain["result_ids"] for chain in summary["chains"][:2]] == [["c", "b"], ["d", "a"]]
    assert summary["max_depth"] == 2
