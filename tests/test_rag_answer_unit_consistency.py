from __future__ import annotations

from graph.rag.answer_unit_consistency import audit_answer_unit_consistency


def test_flags_mixed_units_without_conversion():
    result = audit_answer_unit_consistency("The route is 10 km in one source and 8 miles in another.")

    assert result["issues"][0]["unit_family"] == "distance"
    assert {"km", "mi"}.issubset(set(result["issues"][0]["units"]))
    assert result["issues"][0]["snippets"]


def test_allows_repeated_compatible_units():
    assert audit_answer_unit_consistency("Costs were $10 then $12.")["issues"] == []


def test_allows_conversion_explanations():
    assert audit_answer_unit_consistency("The route is 10 km, roughly equivalent to 6.2 miles.")["issues"] == []
