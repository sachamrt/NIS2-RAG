"""The factories are the whole point of the provider abstraction: every
provider must be reachable through the same call, and selecting one must never
require the others to be installed or configured.
"""

import pytest

from app.core.config import Settings
from app.core.embeddings_factory import build_embeddings
from app.core.llm_factory import build_llm


def _settings(**overrides) -> Settings:
    base = {
        "mistral_api_key": "test-key",
        "openai_api_key": "test-key",
        "anthropic_api_key": "test-key",
        "_env_file": None,  # ignore the developer's real .env
    }
    return Settings(**{**base, **overrides})


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("mistral", "ChatMistralAI"),
        ("openai", "ChatOpenAI"),
        ("local", "ChatOllama"),
    ],
)
def test_build_llm_returns_provider_class(provider, expected):
    llm = build_llm(provider, _settings())
    assert type(llm).__name__ == expected


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("mistral", "MistralAIEmbeddings"),
        ("openai", "OpenAIEmbeddings"),
        ("local", "OllamaEmbeddings"),
    ],
)
def test_build_embeddings_returns_provider_class(provider, expected):
    assert type(build_embeddings(provider, _settings())).__name__ == expected


def test_configured_provider_is_used_by_default():
    assert type(build_llm(settings=_settings(llm_provider="mistral"))).__name__ == "ChatMistralAI"
    assert type(build_llm(settings=_settings(llm_provider="openai"))).__name__ == "ChatOpenAI"


def test_model_name_comes_from_settings():
    llm = build_llm("mistral", _settings(mistral_llm_model="mistral-small-latest"))
    assert llm.model == "mistral-small-latest"


def test_missing_api_key_is_a_clear_error():
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        build_llm("mistral", _settings(mistral_api_key=None))


def test_unknown_provider_lists_the_known_ones():
    with pytest.raises(ValueError, match="mistrl"):
        build_llm("mistrl", _settings())


def test_generation_params_are_provider_independent():
    settings = _settings(llm_temperature=0.7, llm_max_tokens=512)
    for provider in ("mistral", "openai"):
        llm = build_llm(provider, settings)
        assert llm.temperature == 0.7
        assert llm.max_tokens == 512
