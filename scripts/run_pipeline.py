#!/usr/bin/env python3
"""MSX-Sentinel end-to-end diagnostic pipeline orchestrator.

Usage:
    # Standalone (full pipeline locally)
    python scripts/run_pipeline.py /path/to/wsi.ome.tiff /path/to/mask.png

    # HPC mode (Phases 1-3, generates bridge package)
    python scripts/run_pipeline.py wsi.ome.tiff mask.png --mode hpc

    # Local mode (Phases 4-5, consumes bridge package)
    python scripts/run_pipeline.py --mode local --package /path/to/bridge_package/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime
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
        nargs="?",
        default=None,
        help="Path to OME-TIFF whole slide image (required for standalone/hpc modes)",
    )
    parser.add_argument(
        "mask_path",
        type=Path,
        nargs="?",
        default=None,
        help="Path to tissue mask PNG (required for standalone/hpc modes)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standalone", "hpc", "local"],
        default="standalone",
        help="Execution mode: standalone (full local), hpc (phases 1-3), local (phases 4-5 from package)",
    )
    parser.add_argument(
        "--package",
        type=Path,
        default=None,
        help="Path to bridge package directory (required for --mode local)",
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


def validate_inputs(wsi_path: Path | None, mask_path: Path | None, mode: str, package: Path | None) -> None:
    """Validate input files exist based on mode."""
    if mode in ("standalone", "hpc"):
        if wsi_path is None or mask_path is None:
            console.print(f"[red]Error:[/] wsi_path and mask_path required for --mode {mode}")
            sys.exit(1)
        if not wsi_path.exists():
            console.print(f"[red]Error:[/] WSI not found: {wsi_path}")
            sys.exit(1)
        if not mask_path.exists():
            console.print(f"[red]Error:[/] Mask not found: {mask_path}")
            sys.exit(1)
    elif mode == "local":
        if package is None:
            console.print("[red]Error:[/] --package required for --mode local")
            sys.exit(1)
        if not package.exists():
            console.print(f"[red]Error:[/] Bridge package not found: {package}")
            sys.exit(1)
        manifest_path = package / "tiles.db"
        if not manifest_path.exists():
            console.print(f"[red]Error:[/] Manifest not found in package: {manifest_path}")
            sys.exit(1)


def prepare_bridge_package(
    wsi_name: str,
    client_id: str,
    manifest: "ManifestManager",
    results: dict[str, Any],
) -> Path:
    """Create a self-contained package for Globus transfer to local machine.

    Args:
        wsi_name: Name of the processed WSI.
        client_id: Multi-tenant client identifier.
        manifest: ManifestManager instance.
        results: Pipeline results dict from HPC phases.

    Returns:
        Path to the created bridge package directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_dir = Path("data/bridge_packages") / f"{wsi_name}_{timestamp}"
    package_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[cyan]Preparing bridge package:[/] {package_dir}")

    # Copy manifest database
    manifest_src = manifest.db_path
    manifest_dst = package_dir / "tiles.db"
    shutil.copy2(manifest_src, manifest_dst)
    console.print(f"  [dim]Copied manifest:[/] {manifest_dst.name}")

    # Copy processed tiles for this WSI
    processed_src = Path("data/processed") / wsi_name
    if processed_src.exists():
        processed_dst = package_dir / "processed" / wsi_name
        shutil.copytree(processed_src, processed_dst)
        tile_count = sum(1 for _ in processed_dst.rglob("*.png"))
        console.print(f"  [dim]Copied tiles:[/] {tile_count} files")
    else:
        console.print(f"  [yellow]Warning:[/] No processed tiles found at {processed_src}")

    # Copy debug artifacts if present
    debug_src = Path("data/debug")
    if debug_src.exists():
        debug_dst = package_dir / "debug"
        shutil.copytree(debug_src, debug_dst)
        console.print(f"  [dim]Copied debug artifacts[/]")

    # Write pipeline results metadata
    meta_path = package_dir / "metadata.json"
    metadata = {
        "wsi_name": wsi_name,
        "client_id": client_id,
        "created_at": timestamp,
        "hpc_results": results,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    console.print(f"  [dim]Wrote metadata:[/] {meta_path.name}")

    # Generate checksums for transfer integrity verification
    checksum_path = package_dir / "checksum.txt"
    checksums = []
    for file_path in sorted(package_dir.rglob("*")):
        if file_path.is_file() and file_path.name != "checksum.txt":
            file_hash = _compute_sha256(file_path)
            rel_path = file_path.relative_to(package_dir)
            checksums.append(f"{file_hash}  {rel_path}")

    checksum_path.write_text("\n".join(checksums) + "\n")
    console.print(f"  [dim]Wrote checksums:[/] {len(checksums)} files")

    return package_dir


def _compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


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

    validate_inputs(args.wsi_path, args.mask_path, args.mode, args.package)

    # Mode-specific initialization
    if args.mode == "local":
        # Load from bridge package
        package_path = args.package.resolve()
        meta_path = package_path / "metadata.json"
        metadata = json.loads(meta_path.read_text())
        wsi_name = metadata["wsi_name"]
        client_id = metadata.get("client_id", args.client_id)
        results = metadata.get("hpc_results", {})
        manifest = ManifestManager(db_path=package_path / "tiles.db")
    else:
        wsi_path = args.wsi_path.resolve()
        mask_path = args.mask_path.resolve()
        wsi_name = wsi_path.stem
        client_id = args.client_id
        results: dict[str, Any] = {}
        manifest = ManifestManager()

    # Header
    console.print()
    mode_label = {"standalone": "Full Pipeline", "hpc": "HPC Mode (Phases 1-3)", "local": "Local Mode (Phases 4-5)"}
    console.print(Panel.fit(
        f"[bold white]WSI:[/] {wsi_name}\n"
        f"[bold white]Client:[/] {client_id}\n"
        f"[bold white]Mode:[/] {mode_label[args.mode]}",
        title="[bold blue]MSX-Sentinel Control Room[/]",
        border_style="blue",
    ))

    # ─── Phase 1: Tiling ───────────────────────────────────────────────
    run_tiling = args.mode in ("standalone", "hpc") and not args.skip_tiling
    if run_tiling:
        print_phase_header(1, "Tiling", "Extracting multi-scale tiles from WSI")

        engine = TilingEngine(manifest=manifest, client_id=client_id)
        tile_counts = engine.process_wsi(
            wsi_path,
            mask_path,
            scales=["macro", "mid", "high"],
        )

        results["tiling"] = tile_counts
        total_tiles = sum(tile_counts.values())
        console.print(f"[green]✓[/] Extracted {total_tiles} tiles across {len(tile_counts)} scales")
    elif args.mode == "local":
        console.print("[dim]⏭ Tiling skipped (loaded from package)[/]")
    else:
        console.print("[yellow]⏭[/] Skipping tiling phase")

    # ─── Phase 2: YOLO Detection ───────────────────────────────────────
    run_detection = args.mode in ("standalone", "hpc") and not args.skip_detection
    if run_detection:
        print_phase_header(2, "YOLO Scout", "Detecting MSX candidates on mid-resolution tiles")

        try:
            detector = YOLODetector(manifest=manifest)
            yolo_summary = detector.detect_tiles(wsi_name, client_id=client_id)
            results["detection"] = yolo_summary
            console.print(
                f"[green]✓[/] Found {yolo_summary['detection_count']} candidates "
                f"in {yolo_summary['tiles_with_detections']} tiles"
            )
        except FileNotFoundError as e:
            console.print(f"[yellow]⚠[/] YOLO model not found: {e}")
            console.print("[dim]Run training first or provide model path[/]")
    elif args.mode == "local":
        console.print("[dim]⏭ Detection skipped (loaded from package)[/]")
    else:
        console.print("[yellow]⏭[/] Skipping detection phase")

    # ─── Phase 3: Biomarker Analysis ───────────────────────────────────
    run_biomarkers = args.mode in ("standalone", "hpc") and not args.skip_biomarkers
    if run_biomarkers:
        print_phase_header(3, "Biomarkers", "Analyzing hemocyte density on mid tiles")

        analyzer = BiomarkerAnalyzer(manifest=manifest)
        bio_summary = analyzer.analyze_wsi(wsi_name, client_id=client_id)

        results["biomarkers"] = bio_summary
        console.print(
            f"[green]✓[/] Mean density: {bio_summary['mean_density']:.1f}/mm² "
            f"across {bio_summary['tile_count']} tiles"
        )
    elif args.mode == "local":
        console.print("[dim]⏭ Biomarkers skipped (loaded from package)[/]")
    else:
        console.print("[yellow]⏭[/] Skipping biomarker phase")

    # ─── HPC Mode: Package for Globus transfer ─────────────────────────
    if args.mode == "hpc":
        package_dir = prepare_bridge_package(wsi_name, client_id, manifest, results)
        console.print()
        console.rule("[bold green]HPC Pipeline Complete[/]")
        console.print(f"\n[bold]Next step:[/] Transfer [cyan]{package_dir}[/] to local machine via Globus")
        console.print(f"Then run: [dim]python scripts/run_pipeline.py --mode local --package {package_dir.name}[/]")
        return 0

    # ─── Phase 4: VLM Verification ─────────────────────────────────────
    run_vlm = args.mode in ("standalone", "local") and not args.skip_vlm
    if run_vlm:
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
    run_diagnosis = args.mode in ("standalone", "local")
    if run_diagnosis:
        print_phase_header(5, "Diagnostic Brain", "Synthesizing final diagnosis")

        brain = DiagnosticBrain(manifest=manifest)
        report = brain.summarize_case(wsi_name, client_id=client_id)

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
        "client_id": client_id,
        "pipeline_results": results,
    }

    output_path.write_text(json.dumps(report_data, indent=2))
    console.print(f"\n[dim]Report saved: {output_path}[/]")

    console.print()
    console.rule("[bold green]Pipeline Complete[/]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
