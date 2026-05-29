from graph.rag.answer_accessibility_disclosure import audit_answer_accessibility_disclosure


def test_accessibility_related_query_requires_disclosure():
    report = audit_answer_accessibility_disclosure("The flow is fast.", query="Does this meet WCAG?")
    assert report["required"] is True
    assert report["disclosed"] is False
    assert report["recommendation"] == "add_accessibility_considerations"


def test_answer_with_accessibility_language_is_disclosed():
    report = audit_answer_accessibility_disclosure("It documents WCAG, screen reader, and keyboard behavior.", query="Accessibility?")
    assert report["signals"] == ["WCAG", "screen_readers", "keyboard"]
    assert report["disclosed"] is True
    assert report["recommendation"] == ""
