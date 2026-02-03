"""Multi-provider abstraction layer for VLM and LLM backends."""

from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from .config import Config, ModelParams, Provider

logger = logging.getLogger(__name__)

# Ollama API timeout (vision models can be slow)
OLLAMA_TIMEOUT = 120.0


class VLMProvider(ABC):
    """Abstract base class for vision-language model providers."""

    @abstractmethod
    def analyze_patch(self, image_path: Path, prompt: str) -> str:
        """Analyze a tissue patch image.

        Args:
            image_path: Path to the PNG patch.
            prompt: Analysis prompt for the model.

        Returns:
            Model's textual description of the patch.
        """
        ...

    @abstractmethod
    async def analyze_patch_async(self, image_path: Path, prompt: str) -> str:
        """Async version of analyze_patch for concurrent processing.

        Args:
            image_path: Path to the PNG patch.
            prompt: Analysis prompt for the model.

        Returns:
            Model's textual description of the patch.
        """
        ...


class LLMProvider(ABC):
    """Abstract base class for large language model providers."""

    @abstractmethod
    def generate_diagnosis(self, context: str, findings: list[str]) -> str:
        """Generate a diagnostic report from findings.

        Args:
            context: Background context (WSI metadata, anatomical region).
            findings: List of VLM observations from analyzed patches.

        Returns:
            Structured diagnostic report.
        """
        ...


class OllamaVLM(VLMProvider):
    """Ollama-based vision-language model provider."""

    def __init__(self, host: str, model: str, params: ModelParams) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._params = params

    def analyze_patch(self, image_path: Path, prompt: str) -> str:
        """Analyze patch using Ollama VLM via /api/chat.

        Args:
            image_path: Path to the PNG patch.
            prompt: Analysis prompt for the model.

        Returns:
            Model's textual response.

        Raises:
            RuntimeError: If Ollama server is unreachable or returns error.
        """
        # Encode image as base64
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "options": {
                "temperature": self._params.temperature,
                "top_p": self._params.top_p,
                "num_predict": self._params.max_tokens,
            },
        }

        url = f"{self._host}/api/chat"
        logger.debug(f"OllamaVLM request to {url} with model {self._model}")

        try:
            with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(f"Ollama server not reachable at {self._host}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e.response.status_code} - {e.response.text}")

        data = response.json()
        content = data.get("message", {}).get("content", "")

        if not content:
            logger.warning(f"Empty response from Ollama VLM: {data}")
            return ""

        logger.debug(f"OllamaVLM response: {content[:100]}...")
        return content

    async def analyze_patch_async(self, image_path: Path, prompt: str) -> str:
        """Async version of analyze_patch for concurrent processing.

        Args:
            image_path: Path to the PNG patch.
            prompt: Analysis prompt for the model.

        Returns:
            Model's textual response.

        Raises:
            RuntimeError: If Ollama server is unreachable or returns error.
        """
        # Encode image as base64
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "options": {
                "temperature": self._params.temperature,
                "top_p": self._params.top_p,
                "num_predict": self._params.max_tokens,
            },
        }

        url = f"{self._host}/api/chat"
        logger.debug(f"OllamaVLM async request to {url} with model {self._model}")

        try:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(f"Ollama server not reachable at {self._host}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e.response.status_code} - {e.response.text}")

        data = response.json()
        content = data.get("message", {}).get("content", "")

        if not content:
            logger.warning(f"Empty response from Ollama VLM: {data}")
            return ""

        logger.debug(f"OllamaVLM async response: {content[:100]}...")
        return content


class OllamaLLM(LLMProvider):
    """Ollama-based large language model provider."""

    def __init__(self, host: str, model: str, params: ModelParams) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._params = params

    def generate_diagnosis(self, context: str, findings: list[str]) -> str:
        """Generate diagnosis using Ollama LLM via /api/generate.

        Args:
            context: Background context (prompt with case evidence).
            findings: List of VLM observations (appended to prompt).

        Returns:
            Model's diagnostic response.

        Raises:
            RuntimeError: If Ollama server is unreachable or returns error.
        """
        # Build full prompt with findings
        findings_text = "\n".join(f"- {f}" for f in findings) if findings else ""
        full_prompt = f"{context}\n\nKey Findings:\n{findings_text}" if findings_text else context

        payload = {
            "model": self._model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self._params.temperature,
                "top_p": self._params.top_p,
                "num_predict": self._params.max_tokens,
            },
        }

        url = f"{self._host}/api/generate"
        logger.debug(f"OllamaLLM request to {url} with model {self._model}")

        try:
            with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(f"Ollama server not reachable at {self._host}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e.response.status_code} - {e.response.text}")

        data = response.json()
        content = data.get("response", "")

        if not content:
            logger.warning(f"Empty response from Ollama LLM: {data}")
            return ""

        logger.debug(f"OllamaLLM response: {content[:100]}...")
        return content


def get_vlm() -> VLMProvider:
    """Factory function returning the active VLM provider.

    Returns:
        VLMProvider instance based on config.yaml active_provider.

    Raises:
        NotImplementedError: If provider is not yet implemented.
    """
    config = Config.get()

    if config.active_provider == Provider.OLLAMA:
        return OllamaVLM(
            host=config.ollama_host,
            model=config.ollama_vlm_model,
            params=config.model_params,
        )

    raise NotImplementedError(f"VLM provider not implemented: {config.active_provider.value}")


def get_llm() -> LLMProvider:
    """Factory function returning the active LLM provider.

    Returns:
        LLMProvider instance based on config.yaml active_provider.

    Raises:
        NotImplementedError: If provider is not yet implemented.
    """
    config = Config.get()

    if config.active_provider == Provider.OLLAMA:
        return OllamaLLM(
            host=config.ollama_host,
            model=config.ollama_llm_model,
            params=config.model_params,
        )

    raise NotImplementedError(f"LLM provider not implemented: {config.active_provider.value}")
