from graph.rag.query_rbac_requirement import detect_query_rbac_requirements


def test_detects_rbac_acronym_and_role_permission_matrix():
    result = detect_query_rbac_requirements("RBAC requirements need a role permission matrix.")

    assert result["has_rbac_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == ["role_permissions"]


def test_detects_admin_role_constraints_and_least_privilege():
    result = detect_query_rbac_requirements("Access control docs must cover admin roles and least privilege.")

    assert [row["category"] for row in result["requirements"]] == ["admin_roles", "least_privilege"]


def test_detects_role_assignment_and_separation_of_duties():
    result = detect_query_rbac_requirements("RBAC should define role assignment and separation of duties.")

    assert [row["category"] for row in result["requirements"]] == ["role_assignment", "separation_of_duties"]


def test_ignores_theatrical_or_job_role_wording_without_access_control_context():
    assert detect_query_rbac_requirements("Which actor has the lead role and which job role owns the meeting notes?") == {
        "has_rbac_requirements": False,
        "requirements": [],
    }
