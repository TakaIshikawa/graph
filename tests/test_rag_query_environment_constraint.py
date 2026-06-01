from graph.rag.query_environment_constraint import detect_query_environment_constraints


def test_detects_multiple_environments_in_deterministic_order():
    report = detect_query_environment_constraints(
        "Compare prod, production, staging, UAT, dev, sandbox, local env, "
        "air-gapped lab, and the EU region environment."
    )

    assert report == {
        "has_environment_constraints": True,
        "environments": [
            "dev",
            "staging",
            "production",
            "sandbox",
            "air_gapped_lab",
            "region_specific",
            "local",
        ],
        "production_sensitive": True,
    }


def test_detects_production_aliases_as_sensitive():
    report = detect_query_environment_constraints("Can this handle live traffic in pre-prod first?")

    assert report["environments"] == ["staging", "production"]
    assert report["production_sensitive"] is True


def test_detects_local_and_region_specific_aliases():
    report = detect_query_environment_constraints("Run it on my machine for the JP-region env before the offline lab.")

    assert report == {
        "has_environment_constraints": True,
        "environments": ["air_gapped_lab", "region_specific", "local"],
        "production_sensitive": False,
    }


def test_unrelated_query_has_no_environment_constraints():
    assert detect_query_environment_constraints("Summarize the retrieved evidence by source quality.") == {
        "has_environment_constraints": False,
        "environments": [],
        "production_sensitive": False,
    }
