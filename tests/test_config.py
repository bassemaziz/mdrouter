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


def test_from_file_merges_provider_files_and_filters_enabled(tmp_path):
        novita_file = tmp_path / "novita.json"
        novita_file.write_text(
                """
{
    "providers": {
        "novita": {
            "base_url": "https://api.novita.ai/v3/openai"
        }
    },
    "models": {
        "novita/deepseek-r1": {
            "provider": "novita",
            "upstream_model": "deepseek/deepseek-r1"
        }
    }
}
""".strip(),
                encoding="utf-8",
        )
        go_file = tmp_path / "go.json"
        go_file.write_text(
                """
{
    "providers": {
        "go": {
            "base_url": "https://opencode.ai/zen/go/v1"
        }
    },
    "models": {
        "go/glm-5": {
            "provider": "go",
            "upstream_model": "glm-5"
        }
    }
}
""".strip(),
                encoding="utf-8",
        )
        root_file = tmp_path / "providers.json"
        root_file.write_text(
                """
{
    "provider_files": ["novita.json", "go.json"],
    "enabled_providers": ["go"],
    "routing": {
        "strict_provider_prefix": true,
        "unknown_model_behavior": "error"
    }
}
""".strip(),
                encoding="utf-8",
        )

        cfg = AppConfig.from_file(root_file)
        assert set(cfg.providers.keys()) == {"go"}
        assert set(cfg.models.keys()) == {"go/glm-5"}


def test_from_file_rejects_duplicate_provider_from_includes(tmp_path):
        first_file = tmp_path / "one.json"
        first_file.write_text(
                """
{
    "providers": {
        "go": {
            "base_url": "https://a.example/v1"
        }
    },
    "models": {
        "go/model-a": {
            "provider": "go",
            "upstream_model": "model-a"
        }
    }
}
""".strip(),
                encoding="utf-8",
        )
        second_file = tmp_path / "two.json"
        second_file.write_text(
                """
{
    "providers": {
        "go": {
            "base_url": "https://b.example/v1"
        }
    },
    "models": {
        "go/model-b": {
            "provider": "go",
            "upstream_model": "model-b"
        }
    }
}
""".strip(),
                encoding="utf-8",
        )
        root_file = tmp_path / "providers.json"
        root_file.write_text(
                """
{
    "provider_files": ["one.json", "two.json"]
}
""".strip(),
                encoding="utf-8",
        )

        with pytest.raises(ValueError):
                AppConfig.from_file(root_file)

