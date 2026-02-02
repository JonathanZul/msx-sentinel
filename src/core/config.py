"""Configuration singleton for MSX-Sentinel."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class Environment(Enum):
    """Execution environment."""
    LOCAL = "local"
    HPC = "hpc"


class Provider(Enum):
    """Supported LLM/VLM providers."""
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass(frozen=True)
class ModelParams:
    """Inference parameters for deterministic VLM/LLM output.

    Args:
        temperature: Sampling temperature. 0.0 for deterministic output.
        top_p: Nucleus sampling threshold.
        max_tokens: Maximum tokens in model response.
    """
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512


@dataclass(frozen=True)
class ScaleConfig:
    """Configuration for a single magnification scale.

    Args:
        magnification: Display magnification (e.g., 1.25, 10, 40).
        level: OpenSlide pyramid level index.
        downsample: Downsample factor relative to Level 0.
        tile_size: Default tile dimension in pixels at this scale.
    """
    magnification: float
    level: int
    downsample: float
    tile_size: int

    def to_level0_coords(self, x: int, y: int) -> tuple[int, int]:
        """Convert coordinates at this scale to Level 0 (40x)."""
        return int(x * self.downsample), int(y * self.downsample)

    def from_level0_coords(self, x: int, y: int) -> tuple[int, int]:
        """Convert Level 0 coordinates to this scale."""
        return int(x / self.downsample), int(y / self.downsample)


@dataclass
class Config:
    """Central configuration singleton.

    Loads from config.yaml if present, otherwise uses defaults.
    Environment is auto-detected based on SLURM_JOB_ID or platform.
    """
    environment: Environment = field(default_factory=lambda: Config._detect_environment())
    active_provider: Provider = Provider.OLLAMA

    # Multi-scale definitions (40x base)
    scales: dict[str, ScaleConfig] = field(default_factory=lambda: {
        "macro": ScaleConfig(magnification=1.25, level=7, downsample=32.0, tile_size=512),
        "mid": ScaleConfig(magnification=10.0, level=3, downsample=4.0, tile_size=512),
        "high": ScaleConfig(magnification=40.0, level=0, downsample=1.0, tile_size=256),
    })

    # Provider-specific settings
    ollama_host: str = "http://localhost:11434"
    ollama_vlm_model: str = "llava:13b"
    ollama_llm_model: str = "llama3:8b"

    # Model inference parameters (deterministic by default)
    model_params: ModelParams = field(default_factory=ModelParams)

    # Debug/thesis artifact settings
    debug_dpi: int = 300
    debug_enabled: bool = True

    _instance: Config | None = field(default=None, repr=False, init=False)
    _config_path: Path | None = field(default=None, repr=False)

    @staticmethod
    def _detect_environment() -> Environment:
        """Detect execution environment from system indicators."""
        if os.environ.get("SLURM_JOB_ID"):
            return Environment.HPC
        if platform.system() == "Darwin":
            return Environment.LOCAL
        return Environment.LOCAL

    @classmethod
    def get(cls) -> Config:
        """Return the singleton Config instance."""
        if cls._instance is None:
            cls._instance = cls._load()
        return cls._instance

    @classmethod
    def _load(cls) -> Config:
        """Load configuration from config.yaml or return defaults."""
        config = cls()

        # Search for config.yaml in project root
        search_paths = [
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent.parent / "config.yaml",
        ]

        for path in search_paths:
            if path.exists():
                config._config_path = path
                config._apply_yaml(path)
                break

        return config

    def _apply_yaml(self, path: Path) -> None:
        """Apply settings from a YAML config file."""
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        if provider := data.get("active_provider"):
            self.active_provider = Provider(provider.lower())

        if ollama := data.get("ollama"):
            self.ollama_host = ollama.get("host", self.ollama_host)
            self.ollama_vlm_model = ollama.get("vlm_model", self.ollama_vlm_model)
            self.ollama_llm_model = ollama.get("llm_model", self.ollama_llm_model)

        if debug := data.get("debug"):
            self.debug_dpi = debug.get("dpi", self.debug_dpi)
            self.debug_enabled = debug.get("enabled", self.debug_enabled)

        if model := data.get("model_params"):
            self.model_params = ModelParams(
                temperature=model.get("temperature", 0.0),
                top_p=model.get("top_p", 1.0),
                max_tokens=model.get("max_tokens", 512),
            )

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance. Primarily for testing."""
        cls._instance = None

    @property
    def is_hpc(self) -> bool:
        """Check if running on HPC cluster."""
        return self.environment == Environment.HPC

    @property
    def is_local(self) -> bool:
        """Check if running on local machine."""
        return self.environment == Environment.LOCAL

    def get_scale(self, name: str) -> ScaleConfig:
        """Retrieve scale configuration by name.

        Args:
            name: Scale identifier ("macro", "mid", or "high").

        Returns:
            ScaleConfig for the requested scale.

        Raises:
            KeyError: If scale name is not recognized.
        """
        return self.scales[name]
