from graph.rag import detect_query_dpa_requirement


def test_dpa_and_processing_addendum_queries_are_detected():
    assert detect_query_dpa_requirement("Need the DPA and data processing addendum.")["requires_dpa"] is True


def test_dpa_cue_categories_are_identified():
    report = detect_query_dpa_requirement("Data processing agreement with subprocessors, controller obligations, processor obligations, and SCCs.")

    assert report["cue_categories"] == ["subprocessor", "processor", "controller", "scc"]


def test_implementation_data_processing_without_agreement_does_not_match():
    assert detect_query_dpa_requirement("How does the batch job process data?")["requires_dpa"] is False
