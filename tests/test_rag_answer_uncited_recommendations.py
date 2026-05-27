from graph.rag.answer_uncited_recommendations import audit_answer_uncited_recommendations


def test_uncited_recommendations_flags_only_recommendation_sentences_without_citations():
    report = audit_answer_uncited_recommendations(
        "Latency fell by 20%. You should add retries [1]. Consider a queue. Avoid polling https://example.com."
    )

    assert report == {
        "recommendation_count": 3,
        "cited_recommendation_count": 2,
        "uncited_recommendation_count": 1,
        "uncited_ratio": 0.3333,
        "findings": [{"type": "uncited_recommendation", "snippet": "Consider a queue."}],
    }


def test_uncited_recommendations_zero_safe_for_factual_answers():
    assert audit_answer_uncited_recommendations("The report lists three options.") == {
        "recommendation_count": 0,
        "cited_recommendation_count": 0,
        "uncited_recommendation_count": 0,
        "uncited_ratio": 0.0,
        "findings": [],
    }
