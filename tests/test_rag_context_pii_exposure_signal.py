from graph.rag.context_pii_exposure_signal import analyze_context_pii_exposure_signals


class Context:
    def __init__(self, content):
        self.content = content


def test_counts_mixed_pii_types_from_content_and_metadata():
    summary = analyze_context_pii_exposure_signals(
        [
            {"text": "Email jane@example.com or call 415-555-1212."},
            Context("SSN 123-45-6789 and card 4111 1111 1111 1111."),
            {"metadata": {"addr": "Ship to 123 Main St"}},
        ]
    )

    assert summary["pii_item_count"] == 3
    assert summary["pii_type_counts"] == {"address": 1, "credit_card": 1, "email": 1, "phone": 1, "ssn": 1}
    assert summary["redaction_recommended"] is True


def test_clean_context_returns_zero_counts():
    assert analyze_context_pii_exposure_signals([{"text": "No private data here."}]) == {
        "pii_item_count": 0,
        "pii_type_counts": {"address": 0, "credit_card": 0, "email": 0, "phone": 0, "ssn": 0},
        "redaction_recommended": False,
    }
