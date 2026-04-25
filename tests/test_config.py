import pytest

from mdrouter.config import AppConfig


def test_provider_prefix_validation():
    payload = {
        "providers": {"novita": {"base_url": "https://example.com/v1"}},
        "models": {
            "badmodel": {
                "provider": "novita",
                "upstream_model": "x",
            }
        },
        "routing": {"strict_provider_prefix": True, "unknown_model_behavior": "error"},
    }
    with pytest.raises(ValueError):
        AppConfig.model_validate(payload)

