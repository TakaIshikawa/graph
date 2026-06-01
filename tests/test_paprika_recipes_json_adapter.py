import json

from graph.adapters.paprika_recipes_json import PaprikaRecipesJsonAdapter
from graph.adapters.registry import get_adapter, list_adapters


def test_paprika_recipes_json_ingests_wrapper_records(tmp_path):
    export = tmp_path / "recipes.json"
    export.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "uid": "abc",
                        "name": "Pancakes",
                        "source_url": "https://example.com/pancakes",
                        "categories": ["Breakfast", "Weekend"],
                        "rating": 5,
                        "prep_time": "10 min",
                        "cook_time": "15 min",
                        "ingredients": ["Flour", "Eggs"],
                        "directions": "Mix\nCook",
                        "notes": "Use butter",
                        "created": "2026-05-01",
                        "modified": "2026-05-02",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    unit = PaprikaRecipesJsonAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "paprika_recipes_json"
    assert unit.source_entity_type == "recipe"
    assert unit.metadata["source_url"] == "https://example.com/pancakes"
    assert unit.metadata["categories"] == ["Breakfast", "Weekend"]
    assert unit.metadata["rating"] == 5
    assert unit.metadata["prep_time"] == "10 min"
    assert unit.metadata["cook_time"] == "15 min"
    assert "Ingredients:" in unit.content
    assert "Directions:" in unit.content
    assert "Source URL: https://example.com/pancakes" in unit.content


def test_paprika_recipes_json_supports_single_list_directory_and_filtering(tmp_path):
    (tmp_path / "one.json").write_text(json.dumps({"name": "Soup", "categories": "Dinner, Easy"}), encoding="utf-8")
    (tmp_path / "many.json").write_text(json.dumps([{"name": "Salad"}]), encoding="utf-8")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")

    result = PaprikaRecipesJsonAdapter(path=str(tmp_path)).ingest()

    assert sorted(unit.title for unit in result.units) == ["Salad", "Soup"]
    soup = next(unit for unit in result.units if unit.title == "Soup")
    assert soup.metadata["categories"] == ["Dinner", "Easy"]
    assert PaprikaRecipesJsonAdapter(path=str(tmp_path)).ingest(entity_types=["meal"]).units == []


def test_paprika_recipes_json_is_registered():
    assert "paprika_recipes_json" in list_adapters()
    assert isinstance(get_adapter("paprika-recipes-json"), PaprikaRecipesJsonAdapter)
