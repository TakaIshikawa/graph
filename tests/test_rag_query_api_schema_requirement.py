from graph.rag.query_api_schema_requirement import detect_query_api_schema_requirements


def test_detects_multiple_api_schema_categories_sorted():
    result = detect_query_api_schema_requirements("Need OpenAPI docs, request/response schema, and schema versioning.")

    assert result["has_api_schema_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == ["openapi", "request_response_schema", "schema_versioning"]


def test_handles_spacing_and_schema_variants():
    result = detect_query_api_schema_requirements("Compare Swagger, JSON Schema, GraphQL schema, and protocol buffer payloads.")

    assert [row["category"] for row in result["requirements"]] == ["graphql_schema", "json_schema", "openapi", "protobuf"]


def test_avoids_unrelated_database_schema_wording():
    assert detect_query_api_schema_requirements("Show the database schema for analytics tables.") == {
        "has_api_schema_requirements": False,
        "requirements": [],
    }
