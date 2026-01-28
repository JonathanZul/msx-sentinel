"""Path resolution singleton for MSX-Sentinel.

Resolves logical path aliases to physical locations based on the
detected execution environment (LOCAL vs HPC).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import Config, Environment

logger = logging.getLogger(__name__)


@dataclass
class Paths:
    """Path resolution singleton.

    Provides environment-aware path aliases. All paths are resolved
    lazily and cached. Paths can be overridden via config.yaml.
    """

    _instance: Paths | None = field(default=None, repr=False, init=False)
    _overrides: dict[str, Path] = field(default_factory=dict, repr=False)
    _project_root: Path | None = field(default=None, repr=False)

    # Default path templates per environment
    _LOCAL_DEFAULTS: dict[str, str] = field(default_factory=lambda: {
        "raw_wsi": "/Volumes/One Touch/MSX Project/data/exports/wsis",
        "masks": "/Volumes/One Touch/MSX Project/data/interim/oyster_masks",
        "processed": "/Volumes/One Touch/MSX Project/data/processed",
        "manifest": "{project}/data/manifests",
        "models": "{project}/models",
        "debug": "{project}/data/debug",
    }, repr=False)

    _HPC_DEFAULTS: dict[str, str] = field(default_factory=lambda: {
        "raw_wsi": "/home/jezul/scratch/oyster_data/raw",
        "masks": "/home/jezul/scratch/oyster_data/interim/oyster_masks",
        "processed": "/home/jezul/scratch/oyster_data/processed",
        "manifest": "/project/def-agodbout/jezul/msx_sentinel/manifests",
        "models": "/project/def-agodbout/jezul/msx_sentinel/models",
        "debug": "/project/def-agodbout/jezul/msx_sentinel/debug",
    }, repr=False)

    @classmethod
    def get(cls) -> Paths:
        """Return the singleton Paths instance."""
        if cls._instance is None:
            cls._instance = cls._load()
        return cls._instance

    @classmethod
    def _load(cls) -> Paths:
        """Load path configuration."""
        paths = cls()
        paths._project_root = cls._find_project_root()
        paths._load_overrides()
        return paths

    @staticmethod
    def _find_project_root() -> Path:
        """Locate project root by searching for config.yaml or .git."""
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "config.yaml").exists() or (parent / ".git").exists():
                return parent
        return Path.cwd()

    def _load_overrides(self) -> None:
        """Load path overrides from config.yaml."""
        config_path = self._project_root / "config.yaml"
        if not config_path.exists():
            return

        with open(config_path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        if path_data := data.get("paths"):
            for key, value in path_data.items():
                if value:
                    self._overrides[key] = Path(value).expanduser()

    def _resolve(self, key: str) -> Path:
        """Resolve a path alias to its physical location."""
        if key in self._overrides:
            return self._overrides[key]

        config = Config.get()
        defaults = (
            self._HPC_DEFAULTS if config.environment == Environment.HPC
            else self._LOCAL_DEFAULTS
        )

        template = defaults.get(key)
        if template is None:
            raise KeyError(f"Unknown path alias: {key}")

        resolved = template.format(project=self._project_root)
        return Path(resolved)

    @property
    def project_root(self) -> Path:
        """Project root directory."""
        return self._project_root or Path.cwd()

    @property
    def raw_wsi_dir(self) -> Path:
        """Directory containing raw WSI files (OME-TIFF)."""
        return self._resolve("raw_wsi")

    @property
    def masks_dir(self) -> Path:
        """Directory containing tissue segmentation masks."""
        return self._resolve("masks")

    @property
    def processed_dir(self) -> Path:
        """Directory for extracted tiles and CROI patches."""
        return self._resolve("processed")

    @property
    def manifest_dir(self) -> Path:
        """Directory containing SQLite manifest databases."""
        return self._resolve("manifest")

    @property
    def models_dir(self) -> Path:
        """Directory containing model weights."""
        return self._resolve("models")

    @property
    def debug_dir(self) -> Path:
        """Directory for debug artifacts and thesis figures.

        Creates the directory if it does not exist and debug is enabled.
        """
        path = self._resolve("debug")
        config = Config.get()
        if config.debug_enabled and not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created debug directory: {path}")
        return path

    @property
    def manifest_db(self) -> Path:
        """Path to the tiles.db SQLite manifest."""
        return self.manifest_dir / "tiles.db"

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist.

        Only creates directories that are within the project root
        or on scratch space (HPC). External drive paths are not created.
        """
        config = Config.get()

        # Directories safe to create
        safe_dirs = [self.manifest_dir, self.debug_dir]
        if config.is_hpc:
            safe_dirs.append(self.processed_dir)

        for directory in safe_dirs:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance. Primarily for testing."""
        cls._instance = None
