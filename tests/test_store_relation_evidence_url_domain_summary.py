from __future__ import annotations

from graph.store import summarize_relation_evidence_url_domains


class Relation:
    def __init__(self, relation_id: str = "", metadata: dict[str, object] | None = None, **values: object):
        self.id = relation_id
        self.metadata = metadata or {}
        for key, value in values.items():
            setattr(self, key, value)


def test_relation_evidence_url_domain_summary_extracts_top_level_and_metadata_urls():
    summary = summarize_relation_evidence_url_domains(
        [
            {"id": "r1", "evidence": [{"url": "https://WWW.Example.com/a"}, {"href": "https://docs.example.org/x"}]},
            {"id": "r2", "metadata": {"evidence_urls": ["example.com/b", "https://other.test"]}},
            {"id": "r3", "metadata": {"note": "none"}},
        ]
    )

    assert summary["domain_counts"] == {"docs.example.org": 1, "example.com": 2, "other.test": 1}
    assert summary["relations_with_evidence_urls"] == 2
    assert summary["relations_without_evidence_urls"] == 1
    assert summary["external_domain_count"] == 3
    assert summary["samples"][0] == {"relation_id": "r1", "domains": ["docs.example.org", "example.com"]}


def test_relation_evidence_url_domain_summary_supports_objects_and_relation_id_fallback():
    summary = summarize_relation_evidence_url_domains(
        [
            Relation("", source="u1", target="u2", relation_type="cites", metadata={"source_url": "HTTP://www.Example.COM/path"}),
            Relation("r2", evidence=[{"link": "https://sub.example.com"}]),
        ]
    )

    assert summary["domain_counts"] == {"example.com": 1, "sub.example.com": 1}
    assert summary["samples"] == [
        {"relation_id": "r2", "domains": ["sub.example.com"]},
        {"relation_id": "u1|u2|cites", "domains": ["example.com"]},
    ]
