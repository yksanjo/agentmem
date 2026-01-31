"""
Kimi (Moonshot AI) Integration

OpenAI-compatible API for cost-effective LLM operations.
https://platform.moonshot.cn/docs/api
"""

import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class KimiConfig:
    """Kimi API configuration."""

    api_key: str
    base_url: str = "https://api.moonshot.cn/v1"
    model: str = "moonshot-v1-8k"  # 8k, 32k, 128k available
    timeout: float = 60.0


class KimiClient:
    """
    Kimi (Moonshot AI) client for LLM operations.

    Cost comparison:
    - Kimi: ~$0.12/1M tokens (moonshot-v1-8k)
    - OpenAI: ~$2.50/1M tokens (gpt-4o-mini)

    ~20x cheaper for similar quality.
    """

    def __init__(self, config: KimiConfig | None = None):
        """
        Initialize Kimi client.

        Args:
            config: Kimi configuration. If None, reads from env.
        """
        if config is None:
            config = KimiConfig(
                api_key=os.environ.get("KIMI_API_KEY", ""),
                model=os.environ.get("KIMI_MODEL", "moonshot-v1-8k"),
            )

        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.timeout,
        )

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The prompt to complete
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        response = await self._client.post(
            "/chat/completions",
            json={
                "model": self.config.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            },
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """
        Generate a chat completion.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            **kwargs: Additional parameters

        Returns:
            Generated response
        """
        response = await self._client.post(
            "/chat/completions",
            json={
                "model": self.config.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            },
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def get_embeddings(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Get embeddings for texts (if Kimi supports it).

        Note: Kimi may not support embeddings directly.
        Use OpenAI or local models for embeddings.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        # Kimi may not have embedding endpoint
        # Fall back to error or use alternative
        raise NotImplementedError(
            "Kimi does not support embeddings. "
            "Use OpenAI or local embeddings instead."
        )

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        await self.close()


def create_kimi_client(
    api_key: str | None = None,
    model: str = "moonshot-v1-8k",
) -> KimiClient:
    """
    Factory function to create Kimi client.

    Args:
        api_key: API key (defaults to KIMI_API_KEY env var)
        model: Model to use (8k, 32k, 128k)

    Returns:
        Configured KimiClient
    """
    config = KimiConfig(
        api_key=api_key or os.environ.get("KIMI_API_KEY", ""),
        model=model,
    )
    return KimiClient(config)
