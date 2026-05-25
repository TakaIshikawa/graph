from __future__ import annotations

import pytest

from graph.rag.result_dataset_artifact_plan import plan_result_dataset_artifacts


def test_dataset_artifact_plan_detects_required_and_present_artifacts():
    plan = plan_result_dataset_artifacts(
        "Find the raw data, CSV, codebook, and replication package for this empirical study.",
        [
            {
                "id": "a",
                "metadata": {
                    "attachments": [
                        {"name": "study.csv"},
                        {"name": "codebook.pdf"},
                    ]
                },
            },
            {"id": "b", "content": "The authors mention a GitHub repository with replication materials."},
        ],
    )

    assert plan["required_artifacts"] == [
        "codebook",
        "csv",
        "data_dictionary",
        "raw_data",
        "replication_package",
    ]
    assert plan["present_artifacts"] == ["codebook", "csv", "replication_package", "repository"]
    assert plan["missing_artifacts"] == ["data_dictionary", "raw_data"]
    assert plan["warnings"] == ["missing_data_dictionary", "missing_raw_data"]


def test_dataset_artifact_plan_detects_metadata_links_and_result_level_artifacts():
    plan = plan_result_dataset_artifacts(
        "Collect the schema and notebook for the published dataset.",
        [
            {
                "id": "schema",
                "metadata": {"links": [{"title": "Table schema"}]},
            },
            {
                "id": "notebook",
                "artifacts": [{"type": "Jupyter notebook"}],
            },
        ],
    )

    assert plan["required_artifacts"] == ["data_dictionary", "notebook", "raw_data", "schema"]
    assert plan["present_artifacts"] == ["notebook", "schema"]
    assert plan["missing_artifacts"] == ["data_dictionary", "raw_data"]
    assert [row["result_id"] for row in plan["result_artifacts"]] == ["schema", "notebook"]


def test_dataset_artifact_plan_is_neutral_for_non_dataset_query_without_artifacts():
    plan = plan_result_dataset_artifacts("Summarize the policy argument.", [{"id": "a", "content": "Editorial"}])

    assert plan["required_artifacts"] == []
    assert plan["present_artifacts"] == []
    assert plan["missing_artifacts"] == []
    assert plan["retrieval_hints"] == []
    assert plan["warnings"] == []


@pytest.mark.parametrize("query", ["", "  ", None])
def test_dataset_artifact_plan_validates_query(query):
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        plan_result_dataset_artifacts(query)  # type: ignore[arg-type]
