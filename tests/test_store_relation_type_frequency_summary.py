from graph.store import summarize_relation_type_frequency


def test_relation_type_frequency_empty_store():
    assert summarize_relation_type_frequency([]) == {
        "total_relations": 0,
        "missing_type_count": 0,
        "relation_type_counts": [],
    }


def test_relation_type_frequency_counts_repeated_types_sorted_by_frequency():
    report = summarize_relation_type_frequency(
        [
            {"id": "r1", "relation_type": "references"},
            {"id": "r2", "relation": "supports"},
            {"id": "r3", "relation_type": "references"},
        ]
    )

    assert report["relation_type_counts"][0] == {
        "relation_type": "references",
        "count": 2,
        "sample_relation_ids": ["r1", "r3"],
    }


def test_relation_type_frequency_buckets_missing_and_blank_types():
    report = summarize_relation_type_frequency([{"id": "blank", "type": " "}, {"id": "missing"}])

    assert report["missing_type_count"] == 2
    assert report["relation_type_counts"] == [
        {"relation_type": "missing", "count": 2, "sample_relation_ids": ["blank", "missing"]}
    ]
