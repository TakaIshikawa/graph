from graph.rag.answer_missing_definition import audit_answer_missing_definitions


def test_defined_terms_are_not_flagged():
    assert audit_answer_missing_definitions("Latency is a delay before response.", ["Latency"]) == []


def test_undefined_terms_are_flagged_once():
    findings = audit_answer_missing_definitions("Latency matters. Latency can vary.", ["Latency", "latency"])

    assert findings == [
        {
            "term": "Latency",
            "first_occurrence_sentence": "Latency matters.",
            "severity": "medium",
            "recommendation": "Define 'Latency' near its first use.",
        }
    ]


def test_unused_terms_are_not_flagged():
    assert audit_answer_missing_definitions("The answer is short.", ["throughput"]) == []


def test_empty_required_terms_returns_empty_list():
    assert audit_answer_missing_definitions("Latency matters.", []) == []
