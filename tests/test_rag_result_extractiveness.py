from __future__ import annotations

from types import SimpleNamespace

from graph.rag.result_extractiveness import score_result_extractiveness


def test_scores_result_extractiveness_and_reason_counts():
    report = score_result_extractiveness(
        [
            {
                "id": "strong",
                "source": "Journal",
                "content": 'The trial included a quoted passage: "Participants improved by a measured amount over eight weeks." It reported outcomes clearly.',
            },
            {"id": "weak", "snippet": "Too short"},
        ],
        min_text_chars=80,
    )

    assert report["total_results"] == 2
    assert report["results"][0]["extractiveness_score"] > report["results"][1]["extractiveness_score"]
    assert report["reason_counts"]["short_text"] == 1
    assert report["reason_counts"]["missing_source_metadata"] == 1
    assert "weak_extractiveness" in report["warnings"]


def test_reads_object_and_tuple_wrapped_results():
    result = (SimpleNamespace(id="obj", text="This is a complete sentence with enough detail to inspect.", metadata={"source": "docs"}), 0.9)

    report = score_result_extractiveness([result], min_text_chars=20)

    assert report["results"][0]["result_id"] == "obj"
    assert report["results"][0]["has_source_metadata"] is True
