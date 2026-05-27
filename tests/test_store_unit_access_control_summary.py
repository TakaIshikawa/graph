from __future__ import annotations

from graph.store.unit_access_control_summary import summarize_unit_access_control


def test_access_control_summary_groups_sources_and_mutually_exclusive_access_buckets():
    summary = summarize_unit_access_control(
        [
            {"id": "pub", "source_project": "docs", "metadata": {"visibility": "public"}},
            {"id": "priv", "source_project": "docs", "metadata": {"access": "private", "sensitive": True}},
            {"id": "rest", "source_project": "docs", "metadata": {"privacy": "internal", "contains_pii": "yes"}},
            {"id": "missing", "metadata": {"source": "inbox", "sensitivity": "high"}},
        ]
    )

    assert summary == {
        "total_units": 4,
        "rows": [
            {
                "source": "docs",
                "unit_count": 3,
                "public_count": 1,
                "private_count": 1,
                "restricted_count": 1,
                "missing_access_count": 0,
                "sensitive_count": 2,
            },
            {
                "source": "inbox",
                "unit_count": 1,
                "public_count": 0,
                "private_count": 0,
                "restricted_count": 0,
                "missing_access_count": 1,
                "sensitive_count": 1,
            },
        ],
        "source_summaries": [
            {
                "source": "docs",
                "unit_count": 3,
                "public_count": 1,
                "private_count": 1,
                "restricted_count": 1,
                "missing_access_count": 0,
                "sensitive_count": 2,
            },
            {
                "source": "inbox",
                "unit_count": 1,
                "public_count": 0,
                "private_count": 0,
                "restricted_count": 0,
                "missing_access_count": 1,
                "sensitive_count": 1,
            },
        ],
    }


def test_access_control_summary_interprets_boolean_and_string_sensitivity_consistently():
    summary = summarize_unit_access_control(
        [
            {"metadata": {"source": "s", "sharing": "anyone-with-link", "contains_pii": True}},
            {"metadata": {"source": "s", "sharing": "invite_only", "sensitive": "false"}},
            {"metadata": {"source": "s", "sensitive": "PII"}},
        ]
    )

    row = summary["rows"][0]
    assert row["public_count"] == 1
    assert row["restricted_count"] == 1
    assert row["missing_access_count"] == 1
    assert row["sensitive_count"] == 2
