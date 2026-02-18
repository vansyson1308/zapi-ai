from src.api.models import FunctionParameters


def test_function_parameters_allows_extra_fields():
    params = FunctionParameters(custom_schema_hint={"x": 1})

    assert params.custom_schema_hint == {"x": 1}
    assert params.type == "object"
    assert params.additionalProperties is False
