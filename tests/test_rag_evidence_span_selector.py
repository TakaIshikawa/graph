from __future__ import annotations

from dataclasses import dataclass

from graph.rag.evidence_span_selector import select_evidence_spans


@dataclass
class Result:
    id: str
    content: str = ""
    metadata: dict | None = None


def test_select_evidence_spans_returns_compact_matches_without_duplicate_terms():
    spans = select_evidence_spans(
        [
            {
                "id": "a",
                "content": "Solar storage improves the grid. Storage planning needs solar forecasts.",
            }
        ],
        "storage solar storage",
        window=80,
    )

    assert spans == [
        {
            "result_id": "a",
            "span": "Solar storage improves the grid. Storage planning needs solar forecasts.",
            "matched_terms": ["storage", "solar"],
            "start": 0,
            "end": 72,
        }
    ]


def test_select_evidence_spans_supports_objects_tuple_wrappers_and_metadata_fallback():
    spans = select_evidence_spans(
        [
            (Result("object", content="Object text mentions ALPHA."), 0.9),
            {"metadata": {"unit_id": "meta", "text": "Metadata text mentions alpha twice: alpha."}},
        ],
        "alpha",
        window=80,
    )

    assert [span["result_id"] for span in spans] == ["meta", "object"]
    assert spans[0]["matched_terms"] == ["alpha"]
    assert spans[1]["span"] == "Object text mentions ALPHA."


def test_select_evidence_spans_clips_long_spans_and_normalizes_whitespace():
    text = "Intro\n\npadding " + ("filler " * 20) + "Target evidence phrase with context " + ("tail " * 20)

    span = select_evidence_spans([{"id": "long", "content": text}], "target evidence", window=60)[0]

    assert span["span"] == "filler filler filler Target evidence phrase with context"
    assert span["matched_terms"] == ["target", "evidence"]
    assert span["end"] - span["start"] <= 60


def test_select_evidence_spans_orders_by_match_density_then_result_order():
    spans = select_evidence_spans(
        [
            {"id": "sparse", "content": "alpha " + ("padding " * 20)},
            {"id": "dense", "content": "alpha beta alpha"},
            {"id": "dense-second", "content": "alpha beta alpha"},
        ],
        "alpha beta",
        window=80,
    )

    assert [span["result_id"] for span in spans] == ["dense", "dense-second", "sparse"]


def test_select_evidence_spans_empty_query_or_no_matches_return_empty_list_without_mutation():
    original = {"id": "a", "content": "Solar storage", "metadata": {"tag": "energy"}}
    before = dict(original)

    assert select_evidence_spans([original], "") == []
    assert select_evidence_spans([original], "wind") == []
    assert original == before
