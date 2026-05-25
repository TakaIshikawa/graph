from __future__ import annotations

from graph.rag.query_output_constraints import detect_query_output_constraints


def test_query_output_constraints_detects_length_format_sections_tone_and_reading_level():
    result = detect_query_output_constraints(
        "Answer in 100 words with three bullets as a table, include risks and next steps, formal tone, plain English."
    )

    assert result["word_count"]["value"] == 100
    assert result["bullet_count"]["value"] == 3
    assert result["table_format"]["confidence"] == "high"
    assert result["include_sections"]["sections"] == ["risks", "next steps"]
    assert result["tone"]["raw_phrase"] == "formal tone"
    assert result["reading_level"]["raw_phrase"] == "plain English"


def test_query_output_constraints_detects_json_exclusions_and_deadline():
    result = detect_query_output_constraints("Return JSON without sources by tomorrow.")

    assert "json_format" in result
    assert result["exclude_sections"]["sections"] == ["sources"]
    assert result["deadline"]["raw_phrase"] == "by tomorrow"


def test_query_output_constraints_returns_empty_map_for_unconstrained_queries():
    assert detect_query_output_constraints("What does the policy say?") == {}
