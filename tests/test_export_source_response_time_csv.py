from types import SimpleNamespace

from graph.export import export_source_response_time_csv


def test_exports_numeric_and_string_timings_with_documented_buckets():
    text = export_source_response_time_csv(
        [
            {"source_id": "b", "name": "Beta", "response_time_ms": "99"},
            {"id": "a", "name": "Alpha", "metadata": {"elapsed_ms": 250}},
            SimpleNamespace(id="c", name="Gamma", metadata={"duration_ms": "1000"}),
            {"source_id": "d", "name": "Delta", "latency_ms": "5,000"},
        ]
    )

    assert text.splitlines() == [
        "source_id,name,timing_key,response_time_ms,bucket",
        "a,Alpha,elapsed_ms,250,moderate",
        "b,Beta,response_time_ms,99,fast",
        "c,Gamma,duration_ms,1000,slow",
        "d,Delta,latency_ms,5000,very_slow",
    ]


def test_omits_sources_without_valid_recognized_timing_metadata(tmp_path):
    sources = [{"id": "bad", "response_time_ms": "n/a"}, {"id": "missing", "metadata": {"other": 1}}, {"id": "ok", "fetch_duration_ms": 12.5}]
    expected = "source_id,name,timing_key,response_time_ms,bucket\nok,,fetch_duration_ms,12.5,fast\n"

    assert export_source_response_time_csv(sources) == expected
    stats = export_source_response_time_csv(sources, tmp_path / "timings.csv")
    assert stats["source_count"] == 3
    assert stats["rows_exported"] == 1
    assert (tmp_path / "timings.csv").read_text(encoding="utf-8") == expected
