from graph.rag.answer_counterevidence_handling import audit_answer_counterevidence_handling


def test_omission_reports_missing_counterevidence_handling():
    result = audit_answer_counterevidence_handling(
        "The rollout should proceed.",
        [{"id": "a", "snippet": "However, the pilot contradicts the forecast."}],
    )

    assert result["counterevidence_count"] == 1
    assert result["answer_acknowledges_counterevidence"] is False
    assert result["missing_counterevidence_handling"] is True
    assert result["cue_counts"]["however"] == 1
    assert result["cue_counts"]["contradicts"] == 1
    assert result["samples"] == [{"result_id": "a", "cue": "however"}]


def test_acknowledgement_clears_missing_flag():
    result = audit_answer_counterevidence_handling(
        "However, there is a caveat around pilot size.",
        [{"id": "a", "content": "One exception applies to small customers."}],
    )

    assert result["counterevidence_count"] == 1
    assert result["answer_acknowledges_counterevidence"] is True
    assert result["missing_counterevidence_handling"] is False
    assert result["cue_counts"]["exception"] == 1


def test_case_insensitive_mapping_records_and_sample_limit():
    evidence = [
        {"id": "a", "metadata": {"note": "CONTRARY result."}},
        {"id": "b", "text": "On the other hand, churn improved."},
    ]

    assert audit_answer_counterevidence_handling("", evidence, sample_limit=1)["samples"] == [
        {"result_id": "a", "cue": "contrary"}
    ]
    assert audit_answer_counterevidence_handling("", evidence, sample_limit=0)["samples"] == []
