from __future__ import annotations

from graph.store import summarize_source_oauth_scopes


class Source:
    def __init__(self, source_id: str, metadata: dict[str, object] | None = None):
        self.id = source_id
        self.metadata = metadata or {}


def test_source_oauth_scope_summary_normalizes_strings_lists_and_metadata():
    summary = summarize_source_oauth_scopes(
        [
            {"id": "s1", "scope": "read write,profile"},
            {"id": "s2", "scopes": ["READ", "delete:item", ""]},
            Source("s3", metadata={"oauth_scopes": "admin full_access"}),
            {"id": "s4"},
        ]
    )

    assert summary["scope_counts"] == {"admin": 1, "delete:item": 1, "full_access": 1, "profile": 1, "read": 2, "write": 1}
    assert summary["sources_with_scopes"] == 3
    assert summary["sources_without_scopes"] == 1
    assert summary["broad_scope_samples"] == [
        {"source_id": "s1", "scope": "write"},
        {"source_id": "s2", "scope": "delete:item"},
        {"source_id": "s3", "scope": "admin"},
        {"source_id": "s3", "scope": "full_access"},
    ]


def test_source_oauth_scope_summary_limits_samples_deterministically():
    summary = summarize_source_oauth_scopes(
        [
            {"id": "s2", "scope": "read write"},
            {"id": "s1", "scope": "admin"},
        ],
        sample_limit=1,
    )

    assert summary["source_samples"] == [{"source_id": "s2", "scopes": ["read", "write"]}]
    assert summary["broad_scope_samples"] == [{"source_id": "s2", "scope": "write"}]
