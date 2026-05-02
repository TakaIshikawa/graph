from __future__ import annotations

import networkx as nx
import pytest

from graph.graph.service import analyze_hub_authority


def test_analyze_hub_authority_returns_ranked_hub_and_authority_payloads():
    units = [
        {"id": "hub-a", "title": "Primary Hub"},
        {"id": "hub-b", "title": "Secondary Hub"},
        {"id": "authority-a", "title": "Primary Authority"},
        {"id": "authority-b", "title": "Secondary Authority"},
        {"id": "authority-c", "title": "Tertiary Authority"},
    ]
    edges = [
        {"from_unit_id": "hub-a", "to_unit_id": "authority-a"},
        {"from_unit_id": "hub-a", "to_unit_id": "authority-b"},
        {"from_unit_id": "hub-a", "to_unit_id": "authority-c"},
        {"from_unit_id": "hub-b", "to_unit_id": "authority-a"},
    ]

    result = analyze_hub_authority(units, edges, top_n=2)

    assert result["node_count"] == 5
    assert result["edge_count"] == 4
    assert result["convergence"] == {"converged": True, "max_iter": 100, "error": None}
    assert [item["unit_id"] for item in result["top_hubs"]] == ["hub-a", "hub-b"]
    assert [item["unit_id"] for item in result["top_authorities"]] == [
        "authority-a",
        "authority-b",
    ]
    assert result["top_hubs"][0]["hub_score"] == result["top_hubs"][0]["score"]
    assert result["top_authorities"][0]["authority_score"] == result["top_authorities"][0]["score"]


def test_analyze_hub_authority_empty_and_single_node_graphs_are_sensible():
    assert analyze_hub_authority([], []) == {
        "top_hubs": [],
        "top_authorities": [],
        "node_count": 0,
        "edge_count": 0,
        "convergence": {"converged": True, "max_iter": 0, "error": None},
    }

    result = analyze_hub_authority([{"id": "solo", "title": "Solo"}], [])

    assert result["node_count"] == 1
    assert result["edge_count"] == 0
    assert result["top_hubs"] == [
        {"unit_id": "solo", "title": "Solo", "score": 0.0, "hub_score": 0.0}
    ]
    assert result["top_authorities"] == [
        {"unit_id": "solo", "title": "Solo", "score": 0.0, "authority_score": 0.0}
    ]


@pytest.mark.parametrize("top_n", [-1, None, "many", True])
def test_analyze_hub_authority_rejects_invalid_top_n(top_n):
    with pytest.raises(ValueError, match="top_n must be a non-negative integer"):
        analyze_hub_authority([], [], top_n=top_n)


def test_analyze_hub_authority_tie_breaks_by_unit_id():
    units = [{"id": "hub-b"}, {"id": "hub-a"}, {"id": "authority"}]
    edges = [
        {"from_unit_id": "hub-b", "to_unit_id": "authority"},
        {"from_unit_id": "hub-a", "to_unit_id": "authority"},
    ]

    result = analyze_hub_authority(units, edges)

    assert [item["unit_id"] for item in result["top_hubs"][:2]] == ["hub-a", "hub-b"]
    assert result["top_hubs"][0]["score"] == pytest.approx(result["top_hubs"][1]["score"])


def test_analyze_hub_authority_handles_self_loops_disconnected_nodes_and_weights():
    units = [{"id": "hub"}, {"id": "heavy"}, {"id": "light"}, {"id": "isolated"}]
    edges = [
        {"from_unit_id": "hub", "to_unit_id": "hub", "strength": 100.0},
        {"from_unit_id": "hub", "to_unit_id": "heavy", "strength": 10.0},
        {"from_unit_id": "hub", "to_unit_id": "light", "strength": 1.0},
    ]

    result = analyze_hub_authority(units, edges, weight_key="strength")

    assert result["edge_count"] == 2
    assert result["top_hubs"][0]["unit_id"] == "hub"
    assert result["top_authorities"][0]["unit_id"] == "heavy"
    assert result["top_authorities"][-1]["unit_id"] == "isolated"
    assert result["top_authorities"][-1]["score"] == 0.0


def test_analyze_hub_authority_falls_back_deterministically_when_hits_does_not_converge(
    monkeypatch: pytest.MonkeyPatch,
):
    units = [{"id": "b"}, {"id": "a"}]
    edges = [{"from_unit_id": "b", "to_unit_id": "a"}]
    calls: list[int] = []

    def fake_hits(graph, *, max_iter, normalized):
        calls.append(max_iter)
        raise nx.PowerIterationFailedConvergence(max_iter)

    monkeypatch.setattr(nx, "hits", fake_hits)

    result = analyze_hub_authority(units, edges)

    assert calls == [100, 1000]
    assert result["convergence"]["converged"] is False
    assert result["convergence"]["max_iter"] == 1000
    assert [item["unit_id"] for item in result["top_hubs"]] == ["a", "b"]
    assert [item["score"] for item in result["top_hubs"]] == [0.0, 0.0]
