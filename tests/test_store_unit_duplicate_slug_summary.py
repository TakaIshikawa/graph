from graph.store import summarize_unit_duplicate_slugs


def test_explicit_slug_duplicates():
    summary = summarize_unit_duplicate_slugs([{"id": "1", "slug": "Same"}, {"id": "2", "metadata": {"slug": "same"}}])
    assert summary["duplicate_slug_count"] == 1
    assert summary["affected_unit_count"] == 2
    assert "same" in summary["groups"]


def test_title_derived_slug_duplicates_and_normalization():
    summary = summarize_unit_duplicate_slugs([{"id": "1", "title": "Hello, World!"}, {"id": "2", "title": "hello world"}])
    assert "hello-world" in summary["groups"]


def test_unique_and_missing_titles():
    summary = summarize_unit_duplicate_slugs([{"id": "1", "title": "A"}, {"id": "2"}])
    assert summary == {"duplicate_slug_count": 0, "affected_unit_count": 0, "groups": {}}
