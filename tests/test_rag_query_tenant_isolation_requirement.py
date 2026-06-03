from graph.rag.query_tenant_isolation_requirement import detect_query_tenant_isolation_requirements


def test_detects_saas_tenant_isolation_categories():
    result = detect_query_tenant_isolation_requirements(
        "For the multi-tenant SaaS app, require tenant data isolation, per-tenant VPCs, and isolated workers."
    )

    assert result["has_tenant_isolation_requirements"] is True
    assert result["requirements"] == [
        {"category": "compute_isolation", "matched_text": "isolated workers", "severity": "medium"},
        {"category": "data_isolation", "matched_text": "tenant data isolation", "severity": "high"},
        {"category": "network_isolation", "matched_text": "per-tenant VPCs", "severity": "medium"},
    ]


def test_detects_cross_tenant_access_and_noisy_neighbor_concerns():
    result = detect_query_tenant_isolation_requirements(
        "Prevent cross-tenant access at the tenant boundary and address noisy neighbor risk with per-tenant quotas."
    )

    assert result["has_tenant_isolation_requirements"] is True
    assert result["requirements"] == [
        {"category": "cross_tenant_access", "matched_text": "cross-tenant access", "severity": "high"},
        {"category": "noisy_neighbor", "matched_text": "noisy neighbor", "severity": "medium"},
        {"category": "tenant_boundary", "matched_text": "tenant boundary", "severity": "high"},
    ]


def test_ignores_generic_legal_tenant_language():
    assert detect_query_tenant_isolation_requirements("Summarize tenant lease isolation rules for an apartment.") == {
        "has_tenant_isolation_requirements": False,
        "requirements": [],
    }
