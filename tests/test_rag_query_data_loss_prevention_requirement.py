import pytest

from graph.rag.query_data_loss_prevention_requirement import detect_query_data_loss_prevention_requirement


def test_detects_dlp_policy_and_channels():
    result = detect_query_data_loss_prevention_requirement(
        "Need DLP policy evidence for email and endpoint uploads that block sensitive data, inspect content, and quarantine violations."
    )

    assert result["requires_data_loss_prevention"] is True
    assert result["cue_categories"] == ["dlp_policy", "sensitive_data_blocking", "content_inspection", "quarantine_workflow"]
    assert result["channels"] == ["email", "endpoint", "upload"]


def test_detects_exfiltration_prevention_for_cloud_storage():
    result = detect_query_data_loss_prevention_requirement("How do we prevent data exfiltration from cloud storage?")

    assert result["requires_data_loss_prevention"] is True
    assert result["cue_categories"] == ["exfiltration_prevention"]
    assert result["channels"] == ["cloud_storage"]


def test_generic_privacy_or_classification_does_not_trigger_dlp():
    assert detect_query_data_loss_prevention_requirement("Explain privacy classification for customer records.") == {
        "requires_data_loss_prevention": False,
        "cue_categories": [],
        "channels": [],
    }


@pytest.mark.parametrize("query", ["", None])
def test_invalid_query_raises_value_error(query):
    with pytest.raises(ValueError):
        detect_query_data_loss_prevention_requirement(query)  # type: ignore[arg-type]
