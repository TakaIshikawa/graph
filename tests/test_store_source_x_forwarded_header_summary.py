from graph.store import summarize_source_x_forwarded_headers


def test_x_forwarded_header_summary_counts_lookup_hops_proto_and_private_hints():
    summary = summarize_source_x_forwarded_headers(
        [
            {
                "source_id": "a",
                "X-Forwarded-For": "198.51.100.10, 10.0.0.5",
                "X-Forwarded-Proto": "HTTPS",
                "X-Forwarded-Host": "example.test",
                "X-Forwarded-Port": "443",
            },
            {
                "source_id": "b",
                "metadata": {
                    "headers": {
                        "x-forwarded-for": "203.0.113.7",
                        "X-Forwarded-Proto": "http",
                    }
                },
            },
            {
                "source_id": "c",
                "response_headers": {
                    "X_FORWARDED_HOST": "internal.test",
                    "X-Forwarded-Port": "8080",
                },
            },
            {"source_id": "d"},
        ],
        sample_limit=2,
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_x_forwarded"] == 3
    assert summary["header_presence_counts"] == {"for": 2, "proto": 2, "host": 2, "port": 2}
    assert summary["proto_counts"] == {"http": 1, "https": 1}
    assert summary["max_hop_count"] == 2
    assert summary["private_address_hint_count"] == 2
    assert summary["missing_x_forwarded_count"] == 1
    assert [sample["source_id"] for sample in summary["samples"]] == ["a", "b"]
