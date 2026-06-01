from __future__ import annotations

from graph.rag.query_ip_allowlist_requirement import detect_query_ip_allowlist_requirements


def test_detects_ip_allowlist_categories():
    rows = detect_query_ip_allowlist_requirements(
        "Use an IP allowlist, IP blocklist, CIDR range, trusted network, "
        "and admin console source IP restriction."
    )

    assert [row["category"] for row in rows] == [
        "admin_source_ip",
        "allowlist",
        "blocklist",
        "cidr_range",
        "trusted_network",
    ]


def test_supports_whitelist_and_blacklist_variants():
    rows = detect_query_ip_allowlist_requirements("Whitelist IPs from VPN only and blacklist IPs in the subnet range.")

    assert [row["category"] for row in rows] == ["allowlist", "blocklist", "cidr_range", "trusted_network"]


def test_returns_deterministic_rows_for_multiple_categories():
    rows = detect_query_ip_allowlist_requirements("CIDR and IP allow list for a corporate network.")

    assert [row["category"] for row in rows] == ["allowlist", "cidr_range", "trusted_network"]
