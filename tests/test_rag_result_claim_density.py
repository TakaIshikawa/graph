from __future__ import annotations

from types import SimpleNamespace

import pytest

from graph.rag.result_claim_density import score_result_claim_density
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, content: str) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project="notes",
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content=content,
        metadata={},
        tags=[],
    )


def test_claim_density_scores_numeric_date_comparison_and_citation_cues_higher():
    rows = score_result_claim_density(
        [
            {
                "id": "dense",
                "title": "Dense",
                "content": "Revenue increased 24% in 2024 versus 2023 because retention improved [1].",
            },
            {"id": "plain", "title": "Plain", "content": "This is a general overview of the project."},
        ]
    )

    assert rows[0]["result_id"] == "dense"
    assert rows[0]["claim_count"] > rows[1]["claim_count"]
    assert rows[0]["claim_density"] > rows[1]["claim_density"]
    assert {cue["cue_type"] for cue in rows[0]["top_cue_snippets"]} == {
        "causal",
        "citation",
        "comparison",
        "date",
        "numeric",
    }


def test_claim_density_supports_objects_tuples_and_nested_units():
    wrapped = SimpleNamespace(id="wrapper", unit=unit("nested", "Churn fell to 5% in 2025."))
    rows = score_result_claim_density(
        [
            unit("object", "The rate increased from 10 to 20."),
            ({"unit_id": "tuple", "snippet": "Published 2024-01-01 with doi:10.1/example."}, 0.9),
            wrapped,
        ]
    )

    assert [row["result_id"] for row in rows] == ["object", "tuple", "wrapper"]
    assert rows[0]["title"] == "Title object"
    assert rows[1]["claim_count"] == 2
    assert rows[2]["claim_count"] == 3


def test_claim_density_max_results_limits_rows_without_reordering():
    rows = score_result_claim_density(
        [
            {"id": "first", "content": "Plain text."},
            {"id": "second", "content": "Value rose 30% in 2024."},
            {"id": "third", "content": "Another 2025 value."},
        ],
        max_results=2,
    )

    assert [row["result_id"] for row in rows] == ["first", "second"]


@pytest.mark.parametrize("max_results", [-1, True])
def test_claim_density_validates_max_results(max_results):
    with pytest.raises(ValueError, match="max_results must be a non-negative integer or None"):
        score_result_claim_density([], max_results=max_results)
