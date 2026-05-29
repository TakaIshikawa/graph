from graph.rag.answer_jargon import audit_answer_jargon


def test_flags_repeated_unexplained_acronyms_and_terms():
    report = audit_answer_jargon("RAG improves RAG with embedding search. Embedding quality affects retrieval and retrieval latency.")
    assert report["passes"] is False
    assert report["flagged_terms"] == ["embedding", "RAG", "retrieval"]


def test_allowed_terms_are_not_flagged():
    report = audit_answer_jargon("RAG uses RAG pipelines and embedding embedding.", allowed_terms=["RAG", "embedding"])
    assert report["flagged_terms"] == []
    assert report["passes"] is True


def test_parenthetical_acronym_definitions_are_explained():
    report = audit_answer_jargon("Retrieval augmented generation (RAG) improves RAG grounding.")
    assert "RAG" not in report["flagged_terms"]
