from __future__ import annotations

import pytest

from graph.rag.query_update_frequency_requirement import detect_query_update_frequency_requirement


def test_update_frequency_detects_cadence_and_staleness_hint():
    result = detect_query_update_frequency_requirement("Use the weekly update frequency for release cadence.")

    assert result["requires_cadence_awareness"] is True
    assert result["cadence_terms"] == ["weekly", "release_cadence"]
    assert result["realtime_cues"] == []
    assert result["stale_if_older_than"] == "P7D"


def test_update_frequency_detects_realtime_cues():
    result = detect_query_update_frequency_requirement("Need live continuously updated real-time status.")

    assert result["realtime_cues"] == ["real_time", "live", "continuously_updated"]
    assert result["stale_if_older_than"] == "PT1H"


def test_update_frequency_does_not_flag_generic_dates():
    result = detect_query_update_frequency_requirement("What happened on 2024-02-01?")

    assert result["requires_cadence_awareness"] is False
    assert result["confidence"] == 0.0


@pytest.mark.parametrize("query", ["", None])
def test_update_frequency_validates_query(query):
    with pytest.raises(ValueError):
        detect_query_update_frequency_requirement(query)  # type: ignore[arg-type]
