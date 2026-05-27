from graph.rag.result_media_type_mix import analyze_result_media_type_mix


def test_normalizes_top_level_and_metadata_types():
    report = analyze_result_media_type_mix(
        [{"media_type": "article"}, {"metadata": {"mime_type": "application/pdf"}}, {"content_type": "video/mp4"}]
    )

    assert report["media_type_counts"] == {"article": 1, "pdf": 1, "video": 1}
    assert report["diverse_media_types"] is True


def test_counts_unknowns():
    report = analyze_result_media_type_mix([{}, {"type": "mystery"}])

    assert report["unknown_count"] == 2
    assert report["dominant_media_type"] == "unknown"


def test_dominant_ties_are_deterministic():
    report = analyze_result_media_type_mix([{"type": "video"}, {"type": "article"}])

    assert report["dominant_media_type"] == "article"
