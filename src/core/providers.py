"""Multi-provider abstraction layer for VLM and LLM backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from .config import Config, ModelParams, Provider

logger = logging.getLogger(__name__)


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
        self._host = host
        self._model = model
        self._params = params

    def analyze_patch(self, image_path: Path, prompt: str) -> str:
        """Analyze patch using Ollama VLM.

        TODO: Implement actual Ollama API call with base64 image encoding.
        """
        logger.warning(f"OllamaVLM.analyze_patch is a stub. Model: {self._model}")
        return f"[STUB] Analysis of {image_path.name} with prompt: {prompt[:50]}..."


class OllamaLLM(LLMProvider):
    """Ollama-based large language model provider."""

    def __init__(self, host: str, model: str, params: ModelParams) -> None:
        self._host = host
        self._model = model
        self._params = params

    def generate_diagnosis(self, context: str, findings: list[str]) -> str:
        """Generate diagnosis using Ollama LLM.

        TODO: Implement actual Ollama API call.
        """
        logger.warning(f"OllamaLLM.generate_diagnosis is a stub. Model: {self._model}")
        return f"[STUB] Diagnosis from {len(findings)} findings."


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
