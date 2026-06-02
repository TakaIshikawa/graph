from __future__ import annotations

import pytest

from graph.rag.query_hardware_requirement import detect_query_hardware_requirement


def test_detects_hardware_cues_and_values():
    result = detect_query_hardware_requirement(
        "Need minimum hardware with 16 GB RAM, NVIDIA A100 GPU, Apple Silicon CPU, x86_64, ARM64, SSD, and TPM 2.0."
    )

    assert result["requires_hardware_requirement"] is True
    assert result["cue_categories"] == ["minimum_hardware", "ram", "gpu", "cpu", "storage"]
    assert result["hardware_values"] == ["16 GB RAM", "NVIDIA A100", "Apple Silicon", "x86_64", "ARM64", "SSD", "TPM 2.0"]


def test_detects_device_accelerator_and_on_prem_appliance_cues():
    result = detect_query_hardware_requirement(
        "List device model support, accelerator requirements, and on-prem appliance prerequisites."
    )

    assert result["requires_hardware_requirement"] is True
    assert result["cue_categories"] == ["device_model", "accelerator", "on_prem_appliance"]


def test_pure_software_version_query_does_not_match():
    result = detect_query_hardware_requirement("Which Python and PostgreSQL versions are supported?")

    assert result["requires_hardware_requirement"] is False
    assert result["cue_categories"] == []


def test_empty_query_raises_value_error():
    with pytest.raises(ValueError):
        detect_query_hardware_requirement("")
