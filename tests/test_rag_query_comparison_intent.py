from graph.rag.query_comparison_intent import detect_query_comparison_intent


def test_detects_explicit_a_vs_b_comparison():
    assert detect_query_comparison_intent("Postgres vs MySQL for analytics") == [
        {
            "comparison_type": "versus",
            "matched_phrases": ["vs"],
            "candidate_entities": ["Postgres", "MySQL"],
            "severity": "medium",
        }
    ]


def test_detects_open_ended_best_choice_and_tradeoff_queries():
    rows = detect_query_comparison_intent("Which is the best option, Redis or Memcached? Include tradeoffs.")
    assert rows[0]["comparison_type"] in {"preference", "tradeoff"}
    assert "tradeoffs" in rows[0]["matched_phrases"]
