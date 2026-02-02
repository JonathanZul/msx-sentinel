"""LLM Brain: Stage 3 diagnostic synthesis from aggregated evidence."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.bridge.manifest import ManifestManager, TileRecord
from src.core.config import Config
from src.core.paths import Paths
from src.core import providers

logger = logging.getLogger(__name__)

# RAG knowledge anchor - loaded from diagnostic_criteria.md
DIAGNOSTIC_CRITERIA_PATH = ".claude/knowledge/diagnostic_criteria.md"

PATHOLOGIST_PROMPT = """You are a Senior Clinical Pathologist specializing in shellfish diseases. You are diagnosing a whole slide image (WSI) for Haplosporidium nelsoni (MSX) infection in Eastern oysters.

## KNOWLEDGE ANCHOR (Diagnostic Criteria)
{diagnostic_criteria}

## CASE EVIDENCE
WSI: {wsi_name}

### YOLO Scout Findings
- Total candidates detected: {yolo_count}
- Tiles with detections: {tiles_with_detections}

### VLM Eye Verification
- Candidates analyzed: {vlm_analyzed}
- Confirmed organism present: {vlm_confirmed}
- Multinucleated structures: {multinucleated_count}
- Mean confidence: {vlm_mean_confidence:.2f}

### Hemocyte Biomarker
- Mean density: {mean_density:.1f}/mm²
- Peak density: {peak_density:.1f}/mm²
- Elevated regions (>500/mm²): {elevated_regions}

### Anatomical Distribution
{organ_summary}

## TASK
Based on the evidence above and the diagnostic criteria, provide your assessment.

Return ONLY a JSON object:
{{
  "final_severity_score": <int 0-3>,
  "confidence_score": <float 0.0-1.0>,
  "justification": "<2-3 sentences citing VLM and biomarker findings>",
  "suggested_next_steps": "<recommendation for pathologist>"
}}

Apply the severity scale strictly:
- 0: No plasmodia observed
- 1: 1-5 plasmodia, localized
- 2: Multiple organs affected
- 3: Systemic, heavy infiltration"""


@dataclass
class CaseEvidence:
    """Aggregated evidence for a single WSI case.

    Args:
        wsi_name: Source WSI filename.
        yolo_count: Total YOLO detections.
        tiles_with_detections: Number of tiles containing detections.
        vlm_analyzed: Number of candidates sent to VLM.
        vlm_confirmed: Number confirmed as organism present.
        multinucleated_count: Number with multinucleated morphology.
        vlm_mean_confidence: Mean VLM confidence score.
        mean_density: Mean hemocyte density across tiles.
        peak_density: Maximum hemocyte density observed.
        elevated_regions: Count of tiles with density >500/mm².
        organ_counts: Dict mapping organ_type to detection count.
    """
    wsi_name: str
    yolo_count: int = 0
    tiles_with_detections: int = 0
    vlm_analyzed: int = 0
    vlm_confirmed: int = 0
    multinucleated_count: int = 0
    vlm_mean_confidence: float = 0.0
    mean_density: float = 0.0
    peak_density: float = 0.0
    elevated_regions: int = 0
    organ_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class DiagnosticReport:
    """Final diagnostic report for a WSI.

    Args:
        wsi_name: Source WSI filename.
        final_severity_score: MSX severity level (0-3).
        confidence_score: Model confidence (0.0-1.0).
        justification: Clinical justification text.
        suggested_next_steps: Recommendations for pathologist.
        evidence: Aggregated case evidence.
    """
    wsi_name: str
    final_severity_score: int
    confidence_score: float
    justification: str
    suggested_next_steps: str
    evidence: CaseEvidence

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "wsi_name": self.wsi_name,
            "final_severity_score": self.final_severity_score,
            "confidence_score": self.confidence_score,
            "justification": self.justification,
            "suggested_next_steps": self.suggested_next_steps,
            "evidence_summary": {
                "yolo_count": self.evidence.yolo_count,
                "vlm_confirmed": self.evidence.vlm_confirmed,
                "mean_hemocyte_density": self.evidence.mean_density,
                "peak_hemocyte_density": self.evidence.peak_density,
                "organ_distribution": self.evidence.organ_counts,
            },
        }


class DiagnosticBrain:
    """Stage 3 LLM-based diagnostic synthesis.

    Aggregates YOLO, VLM, and biomarker evidence from the manifest
    and uses an LLM to generate a final severity assessment.
    """

    ELEVATED_DENSITY_THRESHOLD = 500.0  # hemocytes/mm²

    def __init__(self, manifest: ManifestManager | None = None) -> None:
        """Initialize the diagnostic brain.

        Args:
            manifest: ManifestManager instance. Creates default if None.
        """
        self._manifest = manifest or ManifestManager()
        self._config = Config.get()
        self._paths = Paths.get()
        self._llm = providers.get_llm()
        self._console = Console()

        # Load RAG knowledge anchor
        self._diagnostic_criteria = self._load_diagnostic_criteria()

    def _load_diagnostic_criteria(self) -> str:
        """Load diagnostic criteria markdown for RAG injection."""
        criteria_path = self._paths.project_root / DIAGNOSTIC_CRITERIA_PATH
        if not criteria_path.exists():
            logger.warning(f"Diagnostic criteria not found: {criteria_path}")
            return "[Diagnostic criteria not available]"

        return criteria_path.read_text()

    def summarize_case(self, wsi_name: str, client_id: str | None = None) -> DiagnosticReport:
        """Aggregate evidence and generate diagnostic report for a WSI.

        Args:
            wsi_name: WSI filename to analyze.
            client_id: Optional filter by client.

        Returns:
            DiagnosticReport with severity assessment and justification.
        """
        # Aggregate evidence from manifest
        evidence = self._aggregate_evidence(wsi_name, client_id)

        # Build prompt with RAG-injected knowledge
        prompt = self._build_prompt(evidence)

        # Call LLM for diagnosis
        findings = [
            f"YOLO detected {evidence.yolo_count} candidates",
            f"VLM confirmed {evidence.vlm_confirmed}/{evidence.vlm_analyzed} as organism present",
            f"Peak hemocyte density: {evidence.peak_density:.1f}/mm²",
        ]

        try:
            response = self._llm.generate_diagnosis(prompt, findings)
            result = self._parse_llm_response(response)
        except Exception as e:
            logger.error(f"LLM diagnosis failed: {e}")
            result = self._default_result(evidence)

        report = DiagnosticReport(
            wsi_name=wsi_name,
            final_severity_score=result["final_severity_score"],
            confidence_score=result["confidence_score"],
            justification=result["justification"],
            suggested_next_steps=result["suggested_next_steps"],
            evidence=evidence,
        )

        # Display rich panel
        self._display_report(report)

        return report

    def _aggregate_evidence(
        self,
        wsi_name: str,
        client_id: str | None,
    ) -> CaseEvidence:
        """Aggregate all diagnostic evidence from manifest.

        Args:
            wsi_name: WSI filename.
            client_id: Optional client filter.

        Returns:
            CaseEvidence dataclass with aggregated metrics.
        """
        tiles = self._manifest.get_tiles_by_wsi(wsi_name, client_id)

        evidence = CaseEvidence(wsi_name=wsi_name)
        organ_detections: dict[str, int] = defaultdict(int)

        vlm_confidences: list[float] = []
        densities: list[float] = []

        for tile in tiles:
            # YOLO evidence
            if tile.yolo_box:
                detections = tile.yolo_box.get("detections", [])
                if detections:
                    evidence.tiles_with_detections += 1
                    evidence.yolo_count += len(detections)

                    # Track by organ
                    organ = tile.organ_type or "Unknown"
                    organ_detections[organ] += len(detections)

            # VLM evidence
            if tile.vlm_description:
                evidence.vlm_analyzed += 1
                try:
                    vlm_data = json.loads(tile.vlm_description)
                    if vlm_data.get("organism_present", False):
                        evidence.vlm_confirmed += 1
                    if vlm_data.get("multinucleated", False):
                        evidence.multinucleated_count += 1
                    if "confidence" in vlm_data:
                        vlm_confidences.append(vlm_data["confidence"])
                except json.JSONDecodeError:
                    pass

            # Hemocyte evidence
            if tile.hemocyte_density is not None:
                densities.append(tile.hemocyte_density)
                if tile.hemocyte_density > self.ELEVATED_DENSITY_THRESHOLD:
                    evidence.elevated_regions += 1

        # Compute aggregates
        if vlm_confidences:
            evidence.vlm_mean_confidence = sum(vlm_confidences) / len(vlm_confidences)

        if densities:
            evidence.mean_density = sum(densities) / len(densities)
            evidence.peak_density = max(densities)

        evidence.organ_counts = dict(organ_detections)

        logger.info(
            f"Aggregated evidence for {wsi_name}: "
            f"{evidence.yolo_count} YOLO, {evidence.vlm_confirmed} VLM confirmed"
        )

        return evidence

    def _build_prompt(self, evidence: CaseEvidence) -> str:
        """Build the pathologist reasoning prompt with evidence.

        Args:
            evidence: Aggregated case evidence.

        Returns:
            Formatted prompt string.
        """
        # Format organ summary
        if evidence.organ_counts:
            organ_lines = [
                f"- {organ}: {count} detections"
                for organ, count in sorted(
                    evidence.organ_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]
            organ_summary = "\n".join(organ_lines)
        else:
            organ_summary = "- No anatomical annotations available"

        return PATHOLOGIST_PROMPT.format(
            diagnostic_criteria=self._diagnostic_criteria,
            wsi_name=evidence.wsi_name,
            yolo_count=evidence.yolo_count,
            tiles_with_detections=evidence.tiles_with_detections,
            vlm_analyzed=evidence.vlm_analyzed,
            vlm_confirmed=evidence.vlm_confirmed,
            multinucleated_count=evidence.multinucleated_count,
            vlm_mean_confidence=evidence.vlm_mean_confidence,
            mean_density=evidence.mean_density,
            peak_density=evidence.peak_density,
            elevated_regions=evidence.elevated_regions,
            organ_summary=organ_summary,
        )

    def _parse_llm_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response into structured diagnostic result.

        Args:
            response: Raw LLM text response.

        Returns:
            Dict with severity, confidence, justification, next_steps.
        """
        # Strip markdown code blocks if present
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response as JSON: {text[:100]}...")
            return self._default_result(None)

        # Validate and normalize
        result = {
            "final_severity_score": 0,
            "confidence_score": 0.0,
            "justification": "Unable to parse LLM response.",
            "suggested_next_steps": "Manual review recommended.",
        }

        if isinstance(data.get("final_severity_score"), int):
            result["final_severity_score"] = max(0, min(3, data["final_severity_score"]))

        if isinstance(data.get("confidence_score"), (int, float)):
            result["confidence_score"] = max(0.0, min(1.0, float(data["confidence_score"])))

        if isinstance(data.get("justification"), str):
            result["justification"] = data["justification"]

        if isinstance(data.get("suggested_next_steps"), str):
            result["suggested_next_steps"] = data["suggested_next_steps"]

        return result

    def _default_result(self, evidence: CaseEvidence | None) -> dict[str, Any]:
        """Generate default result when LLM fails.

        Args:
            evidence: Case evidence for heuristic fallback, or None.

        Returns:
            Default diagnostic result dict.
        """
        # Simple heuristic fallback
        severity = 0
        if evidence and evidence.vlm_confirmed > 0:
            if evidence.vlm_confirmed <= 5:
                severity = 1
            elif len(evidence.organ_counts) > 1:
                severity = 2
            else:
                severity = 2
            if evidence.vlm_confirmed > 10 and len(evidence.organ_counts) >= 3:
                severity = 3

        return {
            "final_severity_score": severity,
            "confidence_score": 0.5,
            "justification": "LLM unavailable. Severity estimated from VLM confirmation count.",
            "suggested_next_steps": "Manual pathologist review required.",
        }

    def _display_report(self, report: DiagnosticReport) -> None:
        """Display diagnostic report as rich panel.

        Args:
            report: Completed diagnostic report.
        """
        # Severity color mapping
        severity_colors = {0: "green", 1: "yellow", 2: "orange1", 3: "red"}
        severity_labels = {0: "NEGATIVE", 1: "LIGHT", 2: "MODERATE", 3: "HEAVY"}

        color = severity_colors.get(report.final_severity_score, "white")
        label = severity_labels.get(report.final_severity_score, "UNKNOWN")

        # Build evidence table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="dim")
        table.add_column("Value")

        table.add_row("YOLO Candidates", str(report.evidence.yolo_count))
        table.add_row("VLM Confirmed", f"{report.evidence.vlm_confirmed}/{report.evidence.vlm_analyzed}")
        table.add_row("Multinucleated", str(report.evidence.multinucleated_count))
        table.add_row("Mean Hemocyte Density", f"{report.evidence.mean_density:.1f}/mm²")
        table.add_row("Peak Hemocyte Density", f"{report.evidence.peak_density:.1f}/mm²")
        table.add_row("Elevated Regions", str(report.evidence.elevated_regions))

        if report.evidence.organ_counts:
            organs = ", ".join(
                f"{k}({v})" for k, v in sorted(
                    report.evidence.organ_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
            table.add_row("Organ Distribution", organs)

        # Build panel content
        content = f"""[bold {color}]SEVERITY: {report.final_severity_score} ({label})[/]
[dim]Confidence: {report.confidence_score:.0%}[/]

[bold]Evidence Summary[/]
{table}

[bold]Justification[/]
{report.justification}

[bold]Next Steps[/]
{report.suggested_next_steps}"""

        panel = Panel(
            content,
            title=f"[bold]MSX Diagnosis: {report.wsi_name}[/]",
            border_style=color,
            padding=(1, 2),
        )

        self._console.print(panel)
