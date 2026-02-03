"""VLM Eye: Stage 2 morphological verification of YOLO candidates."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
import zarr
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.bridge.manifest import ManifestManager, TileRecord
from src.core.config import Config
from src.core.paths import Paths
from src.core import providers

logger = logging.getLogger(__name__)

# JSON ontology system prompt for deterministic morphological analysis
VLM_SYSTEM_PROMPT = """You are a veterinary histopathologist analyzing 40x H&E oyster tissue for Haplosporidium nelsoni (MSX).

Analyze the image and return ONLY a JSON object with these exact fields:
{
  "organism_present": <bool>,
  "multinucleated": <bool>,
  "nucleus_count": <int>,
  "cytoplasm_texture": "<granular|smooth|vacuolated|artifact>",
  "parasite_diameter_um": <float>,
  "confidence": <float 0.0-1.0>
}

DIAGNOSTIC CRITERIA:
- MSX plasmodia are 5-50um, multinucleated (2-20+ nuclei), granular eosinophilic cytoplasm
- Hemocytes are smaller (<10um), mononuclear — do NOT flag as MSX
- If no organism visible, set organism_present=false and other fields to defaults (0, "artifact", 0.0)

Return ONLY valid JSON. No explanation."""

# Default response when VLM fails or no organism detected
DEFAULT_VLM_RESPONSE = {
    "organism_present": False,
    "multinucleated": False,
    "nucleus_count": 0,
    "cytoplasm_texture": "artifact",
    "parasite_diameter_um": 0.0,
    "confidence": 0.0,
}


@dataclass
class Candidate:
    """YOLO detection candidate pending VLM verification.

    Args:
        tile_id: Manifest tile ID containing the detection.
        wsi_name: Source WSI filename.
        x_level0: Detection center X in Level 0 coords.
        y_level0: Detection center Y in Level 0 coords.
        w_level0: Detection width in Level 0 pixels.
        h_level0: Detection height in Level 0 pixels.
        confidence: YOLO detection confidence.
        organ_type: Anatomical region if available.
    """
    tile_id: int
    wsi_name: str
    x_level0: int
    y_level0: int
    w_level0: int
    h_level0: int
    confidence: float
    organ_type: str | None = None


class VLMEye:
    """Stage 2 VLM-based morphological verification.

    Extracts 40x patches centered on YOLO detections and uses
    a VLM to verify MSX plasmodia presence via strict JSON ontology.
    """

    PATCH_SIZE = 640  # 40x patch dimension

    def __init__(self, manifest: ManifestManager | None = None) -> None:
        """Initialize the VLM Eye.

        Args:
            manifest: ManifestManager instance. Creates default if None.
        """
        self._manifest = manifest or ManifestManager()
        self._config = Config.get()
        self._paths = Paths.get()
        self._vlm = providers.get_vlm()

        # Reference exemplars for few-shot prompting
        self._reference_dir = self._paths.project_root / "data" / "reference"
        self._positive_exemplar = self._reference_dir / "positive_exemplar.png"
        self._negative_exemplar = self._reference_dir / "negative_exemplar.png"

    def get_unverified_candidates(
        self,
        wsi_name: str,
        min_confidence: float = 0.0,
        force: bool = False,
    ) -> list[Candidate]:
        """Query manifest for YOLO detections pending VLM verification.

        Args:
            wsi_name: WSI filename to query.
            min_confidence: Minimum YOLO confidence threshold (0.0-1.0).
                           Detections below this threshold are skipped.
            force: If True, include already-verified candidates for re-verification.

        Returns:
            List of Candidate objects where yolo_box is set but vlm_description is NULL
            (or all candidates if force=True).
        """
        tiles = self._manifest.get_tiles_by_wsi(wsi_name)
        candidates: list[Candidate] = []
        skipped_low_conf = 0

        for tile in tiles:
            if tile.yolo_box is None:
                continue
            if not force and tile.vlm_description is not None:
                continue

            detections = tile.yolo_box.get("detections", [])
            for det in detections:
                conf = det.get("confidence", 0.0)
                if conf < min_confidence:
                    skipped_low_conf += 1
                    continue

                bbox = det.get("bbox_level0", {})
                candidates.append(Candidate(
                    tile_id=tile.id,
                    wsi_name=tile.wsi_name,
                    x_level0=bbox.get("x", 0),
                    y_level0=bbox.get("y", 0),
                    w_level0=bbox.get("w", 0),
                    h_level0=bbox.get("h", 0),
                    confidence=conf,
                    organ_type=tile.organ_type,
                ))

        force_label = " (force=True, re-verifying all)" if force else ""
        logger.info(
            f"Found {len(candidates)} candidates for {wsi_name}{force_label} "
            f"(skipped {skipped_low_conf} below {min_confidence:.2f} confidence)"
        )
        return candidates

    def verify_candidates(
        self,
        wsi_name: str,
        min_confidence: float = 0.0,
        max_concurrent: int = 1,
        force: bool = False,
    ) -> dict[str, Any]:
        """Run VLM verification on all unverified candidates for a WSI.

        Args:
            wsi_name: WSI filename to process.
            min_confidence: Minimum YOLO confidence to verify (0.0-1.0).
            max_concurrent: Maximum concurrent VLM requests (1=sequential).
            force: If True, re-verify all candidates regardless of existing results.

        Returns:
            Summary dict with candidate_count, verified_count, and positive_count.
        """
        candidates = self.get_unverified_candidates(wsi_name, min_confidence, force)
        if not candidates:
            logger.info(f"No unverified candidates for {wsi_name}")
            return {"candidate_count": 0, "verified_count": 0, "positive_count": 0}

        # Resolve WSI path
        wsi_path = self._resolve_wsi_path(wsi_name)
        if wsi_path is None:
            logger.error(f"WSI not found: {wsi_name}")
            return {"candidate_count": len(candidates), "verified_count": 0, "positive_count": 0}

        # Debug output directory
        debug_dir = None
        if self._config.debug_enabled:
            debug_dir = self._paths.debug_dir / "vlm" / wsi_name
            debug_dir.mkdir(parents=True, exist_ok=True)

        # Use concurrent processing if max_concurrent > 1
        if max_concurrent > 1:
            return asyncio.run(
                self._verify_candidates_concurrent(
                    candidates, wsi_path, debug_dir, max_concurrent
                )
            )

        # Sequential processing (original behavior)
        verified = 0
        positives = 0

        with self._progress_bar() as progress:
            task = progress.add_task("VLM Verify", total=len(candidates))

            for candidate in candidates:
                progress.update(task, advance=1)

                result = self._verify_single(candidate, wsi_path, debug_dir)
                if result is not None:
                    verified += 1
                    if result.get("organism_present", False):
                        positives += 1

        logger.info(
            f"VLM verification complete: {verified}/{len(candidates)} verified, "
            f"{positives} positives"
        )

        return {
            "wsi_name": wsi_name,
            "candidate_count": len(candidates),
            "verified_count": verified,
            "positive_count": positives,
        }

    async def _verify_candidates_concurrent(
        self,
        candidates: list[Candidate],
        wsi_path: Path,
        debug_dir: Path | None,
        max_concurrent: int,
    ) -> dict[str, Any]:
        """Process candidates concurrently with semaphore-limited parallelism.

        Args:
            candidates: List of candidates to verify.
            wsi_path: Path to source WSI.
            debug_dir: Debug output directory, or None.
            max_concurrent: Maximum concurrent VLM requests.

        Returns:
            Summary dict with verification results.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        verified = 0
        positives = 0
        results: list[dict[str, Any] | None] = []

        # Pre-extract all patches (I/O bound, done sequentially to avoid WSI contention)
        logger.info(f"Pre-extracting {len(candidates)} patches...")
        patch_data: list[tuple[Candidate, np.ndarray | None, Path | None]] = []

        for i, candidate in enumerate(candidates):
            patch = self._extract_patch(candidate, wsi_path)
            if patch is not None:
                # Save to unique temp file
                patch_path = self._paths.debug_dir / "vlm" / f"_temp_patch_{i}.png"
                patch_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(patch_path), patch)
                patch_data.append((candidate, patch, patch_path))
            else:
                patch_data.append((candidate, None, None))

        logger.info(f"Launching {max_concurrent} concurrent VLM workers...")

        async def verify_one(
            candidate: Candidate,
            patch: np.ndarray | None,
            patch_path: Path | None,
        ) -> dict[str, Any] | None:
            """Verify a single candidate with semaphore control."""
            if patch is None or patch_path is None:
                return None

            async with semaphore:
                prompt = self._build_prompt()
                try:
                    response = await self._vlm.analyze_patch_async(patch_path, prompt)
                    result = self._parse_vlm_response(response)
                except Exception as e:
                    logger.warning(f"VLM call failed for candidate {candidate.tile_id}: {e}")
                    result = DEFAULT_VLM_RESPONSE.copy()

                # Persist to manifest (thread-safe for SQLite)
                result_json = json.dumps(result)
                self._manifest.update_diagnostic(candidate.tile_id, vlm_description=result_json)

                # Save debug artifacts
                if debug_dir:
                    self._save_debug_artifacts(candidate, patch, result, debug_dir)

                return result

        # Launch all tasks
        tasks = [
            verify_one(candidate, patch, patch_path)
            for candidate, patch, patch_path in patch_data
        ]

        # Process with progress bar
        with self._progress_bar() as progress:
            task_id = progress.add_task("VLM Verify (concurrent)", total=len(tasks))

            for coro in asyncio.as_completed(tasks):
                result = await coro
                progress.update(task_id, advance=1)
                results.append(result)

        # Clean up temp files
        for _, _, patch_path in patch_data:
            if patch_path and patch_path.exists():
                patch_path.unlink()

        # Tally results
        for result in results:
            if result is not None:
                verified += 1
                if result.get("organism_present", False):
                    positives += 1

        logger.info(
            f"VLM verification complete: {verified}/{len(candidates)} verified, "
            f"{positives} positives"
        )

        return {
            "wsi_name": candidates[0].wsi_name if candidates else "",
            "candidate_count": len(candidates),
            "verified_count": verified,
            "positive_count": positives,
        }

    def _verify_single(
        self,
        candidate: Candidate,
        wsi_path: Path,
        debug_dir: Path | None,
    ) -> dict[str, Any] | None:
        """Verify a single candidate via VLM.

        Args:
            candidate: Detection candidate to verify.
            wsi_path: Path to source WSI.
            debug_dir: Debug output directory, or None.

        Returns:
            Parsed JSON response dict, or None on failure.
        """
        # Extract 40x patch centered on detection
        patch = self._extract_patch(candidate, wsi_path)
        if patch is None:
            return None

        # Save patch to temp file for VLM
        patch_path = self._paths.debug_dir / "vlm" / "_temp_patch.png"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(patch_path), patch)

        # Build prompt with few-shot exemplars if available
        prompt = self._build_prompt()

        # Call VLM
        try:
            response = self._vlm.analyze_patch(patch_path, prompt)
            result = self._parse_vlm_response(response)
        except Exception as e:
            logger.warning(f"VLM call failed for candidate {candidate.tile_id}: {e}")
            result = DEFAULT_VLM_RESPONSE.copy()

        # Persist to manifest
        result_json = json.dumps(result)
        self._manifest.update_diagnostic(candidate.tile_id, vlm_description=result_json)

        # Save debug artifacts
        if debug_dir:
            self._save_debug_artifacts(candidate, patch, result, debug_dir)

        return result

    def _extract_patch(self, candidate: Candidate, wsi_path: Path) -> np.ndarray | None:
        """Extract 640x640 patch from Level 0 WSI centered on detection.

        Uses tifffile+zarr for memory-efficient region access.

        Args:
            candidate: Detection candidate with Level 0 coordinates.
            wsi_path: Path to WSI file.

        Returns:
            BGR image array, or None on failure.
        """
        try:
            with tifffile.TiffFile(wsi_path) as tif:
                if not tif.pages:
                    logger.error(f"No pages in WSI: {wsi_path}")
                    return None

                # Get Level 0 dimensions
                level0 = tif.pages[0]
                wsi_width = level0.shape[1]
                wsi_height = level0.shape[0]

                # Calculate top-left corner for centered patch
                half_size = self.PATCH_SIZE // 2
                x = candidate.x_level0 - half_size
                y = candidate.y_level0 - half_size

                # Clamp to slide bounds
                x = max(0, min(x, wsi_width - self.PATCH_SIZE))
                y = max(0, min(y, wsi_height - self.PATCH_SIZE))

                # Use zarr for memory-efficient region access
                store = tif.aszarr()
                zarr_array = zarr.open(store, mode="r")

                # Determine if Group (multi-level) or Array (single level)
                is_group = hasattr(zarr_array, "keys") and callable(zarr_array.keys)

                if is_group:
                    level0_arr = zarr_array["0"]
                    region = np.array(level0_arr[y:y + self.PATCH_SIZE, x:x + self.PATCH_SIZE])
                else:
                    region = np.array(zarr_array[y:y + self.PATCH_SIZE, x:x + self.PATCH_SIZE])

                store.close()

                # Convert to BGR for OpenCV compatibility
                if region.ndim == 2:
                    # Grayscale
                    patch = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
                elif region.shape[2] == 3:
                    # RGB -> BGR
                    patch = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
                elif region.shape[2] == 4:
                    # RGBA -> BGR
                    patch = cv2.cvtColor(region, cv2.COLOR_RGBA2BGR)
                else:
                    patch = region

                return patch

        except Exception as e:
            logger.warning(f"Failed to extract patch at ({candidate.x_level0}, {candidate.y_level0}): {e}")
            return None

    def _build_prompt(self) -> str:
        """Build VLM prompt with optional few-shot exemplars.

        Returns:
            Complete prompt string including system instructions and exemplar references.
        """
        prompt_parts = [VLM_SYSTEM_PROMPT]

        # Check for reference exemplars
        has_positive = self._positive_exemplar.exists()
        has_negative = self._negative_exemplar.exists()

        if has_positive or has_negative:
            prompt_parts.append("\nREFERENCE EXAMPLES:")
            if has_positive:
                prompt_parts.append(
                    f"- POSITIVE: See {self._positive_exemplar.name} for confirmed MSX plasmodium"
                )
            if has_negative:
                prompt_parts.append(
                    f"- NEGATIVE: See {self._negative_exemplar.name} for normal hemocytes/tissue"
                )

        prompt_parts.append("\nAnalyze the provided image now.")
        return "\n".join(prompt_parts)

    def _parse_vlm_response(self, response: str) -> dict[str, Any]:
        """Parse VLM response into structured JSON.

        Args:
            response: Raw VLM text response.

        Returns:
            Validated JSON dict, or defaults on parse failure.
        """
        # Strip markdown code blocks if present
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse VLM response as JSON: {text[:100]}...")
            return DEFAULT_VLM_RESPONSE.copy()

        # Validate and normalize fields
        result = DEFAULT_VLM_RESPONSE.copy()

        if isinstance(data.get("organism_present"), bool):
            result["organism_present"] = data["organism_present"]

        if isinstance(data.get("multinucleated"), bool):
            result["multinucleated"] = data["multinucleated"]

        if isinstance(data.get("nucleus_count"), (int, float)):
            result["nucleus_count"] = int(data["nucleus_count"])

        valid_textures = {"granular", "smooth", "vacuolated", "artifact"}
        if data.get("cytoplasm_texture") in valid_textures:
            result["cytoplasm_texture"] = data["cytoplasm_texture"]

        if isinstance(data.get("parasite_diameter_um"), (int, float)):
            result["parasite_diameter_um"] = float(data["parasite_diameter_um"])

        if isinstance(data.get("confidence"), (int, float)):
            result["confidence"] = max(0.0, min(1.0, float(data["confidence"])))

        return result

    def _save_debug_artifacts(
        self,
        candidate: Candidate,
        patch: np.ndarray,
        result: dict[str, Any],
        debug_dir: Path,
    ) -> None:
        """Save diagnostic patch and sidecar JSON for thesis/audit.

        Args:
            candidate: Source candidate.
            patch: Extracted 40x patch image.
            result: VLM analysis result.
            debug_dir: Output directory.
        """
        # Filename based on tile ID and coordinates
        base_name = f"tile{candidate.tile_id}_{candidate.x_level0}_{candidate.y_level0}"

        # Save patch image
        patch_path = debug_dir / f"{base_name}.png"
        cv2.imwrite(str(patch_path), patch)

        # Save sidecar JSON with VLM result
        sidecar_path = debug_dir / f"{base_name}.json"
        sidecar_data = {
            "tile_id": candidate.tile_id,
            "wsi_name": candidate.wsi_name,
            "coordinates_level0": {
                "x": candidate.x_level0,
                "y": candidate.y_level0,
                "w": candidate.w_level0,
                "h": candidate.h_level0,
            },
            "yolo_confidence": candidate.confidence,
            "organ_type": candidate.organ_type,
            "vlm_result": result,
        }
        with open(sidecar_path, "w") as f:
            json.dump(sidecar_data, f, indent=2)

        logger.debug(f"Saved debug artifacts: {patch_path}")

    def _resolve_wsi_path(self, wsi_name: str) -> Path | None:
        """Resolve WSI filename to full path.

        Args:
            wsi_name: WSI filename (with or without extension).

        Returns:
            Full path to WSI, or None if not found.
        """
        wsi_dir = self._paths.raw_wsi_dir

        # Try exact match first
        exact = wsi_dir / wsi_name
        if exact.exists():
            return exact

        # Try common extensions
        for ext in [".ome.tiff", ".ome.tif", ".svs", ".ndpi", ".tiff", ".tif"]:
            candidate = wsi_dir / f"{wsi_name}{ext}"
            if candidate.exists():
                return candidate

        # Search for partial match
        for path in wsi_dir.iterdir():
            if path.stem == wsi_name or path.name.startswith(wsi_name):
                return path

        return None

    def _progress_bar(self) -> Progress:
        """Create a rich progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[vlm]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
