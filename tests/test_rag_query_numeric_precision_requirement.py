from graph.rag.query_numeric_precision_requirement import detect_query_numeric_precision_requirements


def test_pricing_query_requires_exact_numeric_evidence():
    result = detect_query_numeric_precision_requirements("What is the exact USD price for the pro plan?")

    assert "pricing" in result["numeric_intents"]
    assert "exact" in result["requested_precision_words"]
    assert "usd" in result["unit_hints"]
    assert result["approximate_answer_risky"] is True


def test_dosage_query_detects_unit_hints():
    result = detect_query_numeric_precision_requirements("Recommended dosage in mg for adults")

    assert "dosage" in result["numeric_intents"]
    assert "mg" in result["unit_hints"]
    assert result["requires_exact_numeric_evidence"] is True


def test_benchmark_and_ranking_queries_are_risky():
    benchmark = detect_query_numeric_precision_requirements("Compare benchmark latency in ms")
    ranking = detect_query_numeric_precision_requirements("Top 10 highest ranked models")

    assert "benchmark" in benchmark["numeric_intents"]
    assert "ranking" in ranking["numeric_intents"]
    assert benchmark["approximate_answer_risky"] is True
    assert ranking["approximate_answer_risky"] is True


def test_approximate_exploratory_query_is_not_risky():
    result = detect_query_numeric_precision_requirements("Give me a rough ballpark estimate of storage cost")

    assert result["approximate_exploratory"] is True
    assert result["approximate_answer_risky"] is False


def test_non_numeric_query_has_no_requirements():
    result = detect_query_numeric_precision_requirements("Explain the onboarding process")

    assert result["numeric_intents"] == []
    assert result["requires_exact_numeric_evidence"] is False
