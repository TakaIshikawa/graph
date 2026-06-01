from graph.rag.query_training_material_requirement import detect_query_training_material_requirement


def test_employee_onboarding_training_materials():
    report = detect_query_training_material_requirement("Create employee onboarding internal docs and a playbook.")

    assert report["requires_training_material"] is True
    assert report["training_terms"] == ["onboarding", "playbook", "internal_docs"]
    assert report["audience_terms"] == ["employee"]
    assert "prepare docs" in report["recommendations"]


def test_customer_training_and_certification_assets():
    report = detect_query_training_material_requirement("Need a customer workshop, training deck, hands-on lab, and certification.")

    assert report["training_terms"] == ["training_deck", "workshop", "certification", "hands_on_lab"]
    assert report["audience_terms"] == ["customer"]
    assert "prepare labs" in report["recommendations"]
    assert "prepare enablement assets" in report["recommendations"]


def test_model_training_mentions_do_not_trigger_human_materials():
    assert detect_query_training_material_requirement("Train the model on new examples.") == {
        "requires_training_material": False,
        "training_terms": [],
        "audience_terms": [],
        "matched_phrases": [],
        "recommendations": [],
        "confidence": "none",
    }
