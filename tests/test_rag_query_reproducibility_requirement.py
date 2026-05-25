from __future__ import annotations

import pytest

from graph.rag.query_reproducibility_requirement import detect_query_reproducibility_requirement


def test_reproducibility_detects_artifacts_and_method_cues():
    result = detect_query_reproducibility_requirement(
        "Find reproducible studies with code available, dataset available, preregistered protocol, and random seed."
    )

    assert result["requires_reproducibility_evidence"] is True
    assert result["reproducibility_cues"] == ["reproducible", "preregistered"]
    assert result["artifact_requirements"] == ["code", "dataset", "protocol"]
    assert result["method_transparency_cues"] == ["random_seed"]
    assert result["confidence"] == 0.85


def test_reproducibility_separates_replication_from_method_transparency():
    result = detect_query_reproducibility_requirement("Prefer replication papers with a methods appendix.")

    assert result["reproducibility_cues"] == ["replication"]
    assert result["artifact_requirements"] == []
    assert result["method_transparency_cues"] == ["methods_appendix"]


def test_reproducibility_no_cues_is_false():
    result = detect_query_reproducibility_requirement("What did the study conclude?")

    assert result["requires_reproducibility_evidence"] is False
    assert result["reproducibility_cues"] == []
    assert result["confidence"] == 0.0


@pytest.mark.parametrize("query", ["", None])
def test_reproducibility_validates_query(query):
    with pytest.raises(ValueError):
        detect_query_reproducibility_requirement(query)  # type: ignore[arg-type]
