from graph.rag.query_webhook_requirement import detect_query_webhook_requirement


def test_detects_combined_webhook_requirements():
    result = detect_query_webhook_requirement(
        "For the webhook API, document signing secrets, retries with backoff, idempotency, event types, delivery logs, replay, timeouts, and subscription verification."
    )

    assert result["has_webhook_requirement"] is True
    assert [row["category"] for row in result["requirements"]] == [
        "delivery_logs",
        "event_types",
        "idempotency",
        "replay",
        "retries",
        "signing_secrets",
        "subscription_verification",
        "timeout",
    ]


def test_api_event_context_allows_event_type_detection():
    result = detect_query_webhook_requirement("Which API event subscriptions support replay and delivery history?")

    assert [row["category"] for row in result["requirements"]] == ["delivery_logs", "replay"]


def test_unrelated_event_wording_is_not_flagged():
    assert detect_query_webhook_requirement("Summarize company events and retry the search if sources are sparse.") == {
        "has_webhook_requirement": False,
        "requirements": [],
    }
