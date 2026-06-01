from graph.rag.answer_acronym_definitions import audit_answer_acronym_definitions


def test_treats_long_name_parenthetical_as_definition():
    assert audit_answer_acronym_definitions("Use Retrieval Augmented Generation (RAG) with GPUs.") == [
        {"acronym": "GPU", "defined_on_first_use": False, "severity": "medium"},
        {"acronym": "RAG", "defined_on_first_use": True, "severity": "none"},
    ]


def test_treats_acronym_parenthetical_expansion_as_definition():
    assert audit_answer_acronym_definitions("RAG (Retrieval Augmented Generation) improves grounding.") == [
        {"acronym": "RAG", "defined_on_first_use": True, "severity": "none"}
    ]
