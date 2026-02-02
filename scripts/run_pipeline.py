#!/usr/bin/env python3
"""MSX-Sentinel end-to-end diagnostic pipeline orchestrator.

Usage:
    python scripts/run_pipeline.py /path/to/wsi.ome.tiff /path/to/mask.png
    python scripts/run_pipeline.py wsi.ome.tiff mask.png --client-id lab_001 --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

# Ensure src is importable when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bridge.manifest import ManifestManager
from src.hpc import BiomarkerAnalyzer, TilingEngine, YOLODetector
from src.local import DiagnosticBrain, VLMEye

console = Console()


def configure_logging(verbose: bool = False) -> None:
    """Configure rich logging handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MSX-Sentinel diagnostic pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "wsi_path",
        type=Path,
        help="Path to OME-TIFF whole slide image",
    )
    parser.add_argument(
        "mask_path",
        type=Path,
        help="Path to tissue mask PNG",
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default="default",
        help="Multi-tenant client identifier (default: 'default')",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report output path (default: data/reports/{wsi_name}_report.json)",
    )
    parser.add_argument(
        "--skip-tiling",
        action="store_true",
        help="Skip tiling phase (use existing tiles in manifest)",
    )
    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="Skip YOLO detection phase",
    )
    parser.add_argument(
        "--skip-biomarkers",
        action="store_true",
        help="Skip hemocyte biomarker analysis",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip VLM verification phase",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def validate_inputs(wsi_path: Path, mask_path: Path) -> None:
    """Validate input files exist."""
    if not wsi_path.exists():
        console.print(f"[red]Error:[/] WSI not found: {wsi_path}")
        sys.exit(1)

    if not mask_path.exists():
        console.print(f"[red]Error:[/] Mask not found: {mask_path}")
        sys.exit(1)


def print_phase_header(phase: int, name: str, description: str) -> None:
    """Print styled phase header."""
    console.print()
    console.rule(f"[bold cyan]Phase {phase}: {name}[/]")
    console.print(f"[dim]{description}[/]")


def print_summary_table(results: dict[str, Any]) -> None:
    """Print pipeline execution summary."""
    table = Table(title="Pipeline Summary", show_header=True, header_style="bold")
    table.add_column("Phase", style="cyan")
    table.add_column("Metric", style="white")
    table.add_column("Value", style="green")

    if "tiling" in results:
        for scale, count in results["tiling"].items():
            table.add_row("Tiling", f"{scale} tiles", str(count))

    if "detection" in results:
        table.add_row("Detection", "Candidates", str(results["detection"]["detection_count"]))
        table.add_row("Detection", "Tiles w/ detections", str(results["detection"]["tiles_with_detections"]))

    if "biomarkers" in results:
        table.add_row("Biomarkers", "Mean density", f"{results['biomarkers']['mean_density']:.1f}/mm²")
        table.add_row("Biomarkers", "Tiles analyzed", str(results["biomarkers"]["tile_count"]))

    if "vlm" in results:
        table.add_row("VLM", "Verified", str(results["vlm"]["verified_count"]))
        table.add_row("VLM", "Confirmed positive", str(results["vlm"]["positive_count"]))

    console.print()
    console.print(table)


def main() -> int:
    """Execute the MSX-Sentinel diagnostic pipeline."""
    args = parse_args()
    configure_logging(args.verbose)

    wsi_path = args.wsi_path.resolve()
    mask_path = args.mask_path.resolve()

    validate_inputs(wsi_path, mask_path)

    wsi_name = wsi_path.stem
    manifest = ManifestManager()
    results: dict[str, Any] = {}

    # Header
    console.print()
    console.print(Panel.fit(
        f"[bold white]WSI:[/] {wsi_name}\n"
        f"[bold white]Client:[/] {args.client_id}",
        title="[bold blue]MSX-Sentinel Control Room[/]",
        border_style="blue",
    ))

    # ─── Phase 1: Tiling ───────────────────────────────────────────────
    if not args.skip_tiling:
        print_phase_header(1, "Tiling", "Extracting multi-scale tiles from WSI")

        engine = TilingEngine(manifest=manifest, client_id=args.client_id)
        tile_counts = engine.process_wsi(
            wsi_path,
            mask_path,
            scales=["macro", "mid", "high"],
        )

        results["tiling"] = tile_counts
        total_tiles = sum(tile_counts.values())
        console.print(f"[green]✓[/] Extracted {total_tiles} tiles across {len(tile_counts)} scales")
    else:
        console.print("[yellow]⏭[/] Skipping tiling phase")

    # ─── Phase 2: YOLO Detection ───────────────────────────────────────
    if not args.skip_detection:
        print_phase_header(2, "YOLO Scout", "Detecting MSX candidates on mid-resolution tiles")

        try:
            detector = YOLODetector(manifest=manifest)
            yolo_summary = detector.detect_tiles(wsi_name, client_id=args.client_id)
            results["detection"] = yolo_summary
            console.print(
                f"[green]✓[/] Found {yolo_summary['detection_count']} candidates "
                f"in {yolo_summary['tiles_with_detections']} tiles"
            )
        except FileNotFoundError as e:
            console.print(f"[yellow]⚠[/] YOLO model not found: {e}")
            console.print("[dim]Run training first or provide model path[/]")
    else:
        console.print("[yellow]⏭[/] Skipping detection phase")

    # ─── Phase 3: Biomarker Analysis ───────────────────────────────────
    if not args.skip_biomarkers:
        print_phase_header(3, "Biomarkers", "Analyzing hemocyte density on mid tiles")

        analyzer = BiomarkerAnalyzer(manifest=manifest)
        bio_summary = analyzer.analyze_wsi(wsi_name, client_id=args.client_id)

        results["biomarkers"] = bio_summary
        console.print(
            f"[green]✓[/] Mean density: {bio_summary['mean_density']:.1f}/mm² "
            f"across {bio_summary['tile_count']} tiles"
        )
    else:
        console.print("[yellow]⏭[/] Skipping biomarker phase")

    # ─── Phase 4: VLM Verification ─────────────────────────────────────
    if not args.skip_vlm:
        print_phase_header(4, "VLM Eye", "Morphological verification of candidates")

        eye = VLMEye(manifest=manifest)
        vlm_summary = eye.verify_candidates(wsi_name)

        results["vlm"] = vlm_summary
        if vlm_summary["verified_count"] > 0:
            console.print(
                f"[green]✓[/] Verified {vlm_summary['verified_count']} candidates, "
                f"{vlm_summary['positive_count']} confirmed positive"
            )
        else:
            console.print("[dim]No candidates to verify[/]")
    else:
        console.print("[yellow]⏭[/] Skipping VLM phase")

    # ─── Phase 5: Diagnostic Synthesis ─────────────────────────────────
    print_phase_header(5, "Diagnostic Brain", "Synthesizing final diagnosis")

    brain = DiagnosticBrain(manifest=manifest)
    report = brain.summarize_case(wsi_name, client_id=args.client_id)

    results["diagnosis"] = report.to_dict()

    # ─── Summary Table ─────────────────────────────────────────────────
    print_summary_table(results)

    # ─── Export JSON Report ────────────────────────────────────────────
    output_path = args.output
    if output_path is None:
        output_path = Path("data/reports") / f"{wsi_name}_report.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_data = {
        "wsi_name": wsi_name,
        "client_id": args.client_id,
        "pipeline_results": results,
    }

    output_path.write_text(json.dumps(report_data, indent=2))
    console.print(f"\n[dim]Report saved: {output_path}[/]")

    console.print()
    console.rule("[bold green]Pipeline Complete[/]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
