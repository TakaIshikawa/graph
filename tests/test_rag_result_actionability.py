from __future__ import annotations

from graph.rag.result_actionability import classify_result_actionability


def test_result_actionability_labels_procedural_snippets_as_actionable():
    report = classify_result_actionability(
        [{"id": "proc", "snippet": "Step 1: install the package. Then run pytest and verify the output."}]
    )

    assert report["results"][0]["label"] == "actionable"
    assert set(report["results"][0]["matched_cues"]) >= {"imperative_verb", "procedure", "command"}
    assert report["results"][0]["confidence"] > 0.7


def test_result_actionability_labels_explanatory_and_reference_only_snippets():
    report = classify_result_actionability(
        [
            {"id": "explain", "text": "A vector index refers to a structure that describes nearby embeddings."},
            {"id": "ref", "text": "API reference: parameter top_k is an integer field in the schema."},
            {"id": "low", "text": "Short note."},
        ]
    )

    assert [row["label"] for row in report["results"]] == ["explanatory", "reference-only", "low-action"]
    assert report["label_counts"] == {"actionable": 0, "explanatory": 1, "reference-only": 1, "low-action": 1}
