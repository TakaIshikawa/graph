from graph.rag.query_data_processing_location_requirement import detect_query_data_processing_location_requirements


def test_detects_processing_storage_transfer_and_subprocessor_location_categories():
    rows = detect_query_data_processing_location_requirements(
        "Ask where data is processed, storage region, cross-border transfer, and subprocessors by location."
    )

    assert rows == [
        {"matched_text": "cross-border transfer", "category": "cross_border_transfer", "requirement_strength": "high"},
        {"matched_text": "where data is processed", "category": "processing_region", "requirement_strength": "high"},
        {"matched_text": "storage region", "category": "storage_region", "requirement_strength": "high"},
        {"matched_text": "subprocessors by location", "category": "subprocessor_location", "requirement_strength": "medium"},
    ]


def test_detects_customer_selectable_region_and_failover_location():
    rows = detect_query_data_processing_location_requirements("Need a customer-selectable region and regional failover.")

    assert rows == [
        {"matched_text": "customer-selectable region", "category": "customer_selectable_region", "requirement_strength": "high"},
        {"matched_text": "regional failover", "category": "regional_failover_location", "requirement_strength": "medium"},
    ]


def test_excludes_data_residency_only_wording_and_unrelated_queries():
    assert detect_query_data_processing_location_requirements("data residency requirements") == []
    assert detect_query_data_processing_location_requirements("Compare vendor pricing by region.") == []
