from graph.rag.query_latency_sla_requirement import detect_query_latency_sla_requirement


def test_extracts_millisecond_and_second_targets():
    report = detect_query_latency_sla_requirement("Need response time under 200ms and timeout within 2 seconds.")
    assert report["requires_latency_sla"] is True
    assert report["latency_targets"] == ["under 200ms", "within 2 seconds"]
    assert report["signals"] == ["response_time", "timeout"]


def test_preserves_percentile_latency_target():
    report = detect_query_latency_sla_requirement("The SLA requires p95 latency under 1.5s for real-time search.")
    assert report["latency_targets"] == ["p95 latency under 1.5s"]
    assert report["signals"] == ["latency", "p95", "real_time", "sla"]


def test_no_latency_constraint_is_neutral():
    assert detect_query_latency_sla_requirement("Summarize retrieval quality.") == {
        "requires_latency_sla": False,
        "latency_targets": [],
        "signals": [],
    }
