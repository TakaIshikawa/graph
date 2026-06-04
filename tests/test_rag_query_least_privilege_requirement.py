from graph.rag.query_least_privilege_requirement import detect_query_least_privilege_requirement


def test_detects_least_privilege_synonyms_and_structured_evidence_terms():
    report = detect_query_least_privilege_requirement(
        "Need least privilege evidence, minimum required permissions, and need-to-know access."
    )

    assert report["requires_least_privilege"] is True
    assert report["classification"] == "least_privilege_requirement"
    assert report["requirements"] == [
        {
            "category": "least_privilege",
            "matched_text": "least privilege",
            "severity": "high",
            "evidence_terms": ["least", "privilege"],
        },
        {
            "category": "minimum_permissions",
            "matched_text": "minimum required permissions",
            "severity": "high",
            "evidence_terms": ["minimum", "required", "permissions"],
        },
        {
            "category": "need_to_know_access",
            "matched_text": "need-to-know access",
            "severity": "medium",
            "evidence_terms": ["need", "to", "know", "access"],
        },
    ]
    assert {"access", "least", "minimum", "permissions", "privilege"} <= set(report["evidence_terms"])


def test_detects_privilege_reduction_and_excessive_access_review():
    report = detect_query_least_privilege_requirement(
        "Require controls to reduce account privileges and detect excessive permissions."
    )

    assert report["requires_least_privilege"] is True
    assert [row["category"] for row in report["requirements"]] == [
        "excessive_access_review",
        "privilege_reduction",
    ]
    assert "reduce" in report["evidence_terms"]
    assert "excessive" in report["evidence_terms"]
    assert "permissions" in report["evidence_terms"]


def test_broad_access_control_or_rbac_queries_do_not_trigger_least_privilege():
    assert detect_query_least_privilege_requirement("Summarize RBAC roles and access control options.") == {
        "requires_least_privilege": False,
        "classification": "unrelated",
        "requirements": [],
        "evidence_terms": [],
    }
    assert detect_query_least_privilege_requirement("List administrator roles and role assignments.") == {
        "requires_least_privilege": False,
        "classification": "unrelated",
        "requirements": [],
        "evidence_terms": [],
    }


def test_unrelated_security_queries_do_not_trigger_least_privilege():
    assert detect_query_least_privilege_requirement("Need MFA and audit logging evidence for all users.") == {
        "requires_least_privilege": False,
        "classification": "unrelated",
        "requirements": [],
        "evidence_terms": [],
    }
