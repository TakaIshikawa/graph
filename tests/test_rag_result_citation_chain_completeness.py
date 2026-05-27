from __future__ import annotations

from graph.rag.result_citation_chain_completeness import analyze_result_citation_chain_completeness


class Citation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_result_citation_chain_completeness_detects_complete_chains():
    summary = analyze_result_citation_chain_completeness([{"citation_id": "root"}, {"citation_id": "child", "parent_citation_id": "root"}])

    assert summary["roots"] == ["root"]
    assert summary["linked_children"] == ["child"]
    assert summary["orphan_children"] == 0


def test_result_citation_chain_completeness_reports_orphans_and_missing_ids():
    summary = analyze_result_citation_chain_completeness([{"metadata": {"citation_id": "child", "parent_citation_id": "missing"}}, {"text": "no id"}])

    assert summary["missing_citation_id_count"] == 1
    assert summary["orphan_children"] == 1
    assert summary["orphan_ids"] == ["child"]


def test_result_citation_chain_completeness_supports_object_inputs_and_stable_order():
    summary = analyze_result_citation_chain_completeness([Citation(citation_id="b", parent_citation_id="a"), Citation(citation_id="a")])

    assert summary["roots"] == ["a"]
    assert summary["linked_children"] == ["b"]
