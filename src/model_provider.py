from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    aliases = {
        "chatgpt": "openai", "openai-compatible": "custom",
        "google": "gemini", "google-genai": "gemini",
        "claude": "anthropic", "anthorpic": "anthropic",
        "local": "ollama", "open-router": "openrouter",
    }
    normalized = value.strip().lower()
    return aliases.get(normalized, normalized)


def build_chat_model(config: ProviderConfig):
    """Student TODO: instantiate the real chat model for the selected provider.

    Pseudocode:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`
    """

    provider = normalize_provider(config.provider)
    try:
        if provider in {"openai", "custom", "openrouter"}:
            from langchain_openai import ChatOpenAI
            kwargs = {"model": config.model_name, "temperature": config.temperature,
                      "api_key": config.api_key}
            if provider in {"custom", "openrouter"}:
                kwargs["base_url"] = config.base_url
            return ChatOpenAI(**kwargs)
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=config.model_name, temperature=config.temperature,
                                          google_api_key=config.api_key)
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=config.model_name, temperature=config.temperature,
                                 api_key=config.api_key)
        if provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=config.model_name, temperature=config.temperature,
                              base_url=config.base_url)
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency for provider '{provider}'") from exc
    raise ValueError(f"Unsupported provider: {config.provider}")
