"""Integration test for TilingEngine with mock WSI and mask."""

from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from rich.console import Console
from rich.panel import Panel

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bridge.manifest import ManifestManager
from src.core.config import Config
from src.core.paths import Paths
from src.hpc.tiling import TilingEngine

console = Console()

# Test data paths (relative to project root)
TEST_DATA_DIR = PROJECT_ROOT / "data"
TEST_RAW_DIR = TEST_DATA_DIR / "raw"
TEST_MASKS_DIR = TEST_DATA_DIR / "masks"
TEST_PROCESSED_DIR = TEST_DATA_DIR / "processed"
TEST_MANIFEST_DIR = TEST_DATA_DIR / "manifests"

WSI_NAME = "test_wsi"
WSI_PATH = TEST_RAW_DIR / f"{WSI_NAME}.tif"
MASK_PATH = TEST_MASKS_DIR / f"{WSI_NAME}_mask.png"
DB_PATH = TEST_MANIFEST_DIR / "tiles.db"


def create_mock_wsi(path: Path, width: int = 1024, height: int = 1024) -> None:
    """Create a mock RGB TIFF for testing.

    Generates a simple gradient pattern to verify tile extraction.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # RGB gradient: red increases left-to-right, green increases top-to-bottom
    r = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    g = np.tile(np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1), (1, width))
    b = np.full((height, width), 128, dtype=np.uint8)

    image = np.stack([r, g, b], axis=-1)

    tifffile.imwrite(path, image, photometric="rgb")
    console.print(f"[dim]Created mock WSI: {path} ({width}x{height})[/dim]")


def create_mock_mask(path: Path, width: int = 128, height: int = 128) -> None:
    """Create a mock tissue mask with 50% background, 50% tissue.

    Left half (x < width/2) = background (0)
    Right half (x >= width/2) = tissue (255)

    At MASK_SCALING_FACTOR=8, this 128x128 mask corresponds to a 1024x1024 WSI.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, width // 2:] = 255  # Right half is tissue

    Image.fromarray(mask, mode="L").save(path)
    console.print(f"[dim]Created mock mask: {path} ({width}x{height}, 50% tissue)[/dim]")


def cleanup_test_data() -> None:
    """Remove test artifacts from previous runs."""
    for path in [WSI_PATH, MASK_PATH, DB_PATH]:
        if path.exists():
            path.unlink()

    processed_dir = TEST_PROCESSED_DIR / WSI_NAME
    if processed_dir.exists():
        shutil.rmtree(processed_dir)


def count_tiles_in_directory(directory: Path) -> int:
    """Count PNG files in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob("*.png")))


def count_tiles_in_db(db_path: Path, wsi_name: str) -> int:
    """Count tile rows in SQLite manifest."""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT COUNT(*) FROM tiles WHERE wsi_name = ?", (wsi_name,)
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_tile_coordinates(db_path: Path, wsi_name: str) -> list[tuple[int, int]]:
    """Retrieve all tile coordinates from manifest."""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT x_level0, y_level0 FROM tiles WHERE wsi_name = ?", (wsi_name,)
    )
    coords = [(row[0], row[1]) for row in cursor.fetchall()]
    conn.close()
    return coords


def run_test() -> bool:
    """Execute integration test for TilingEngine.

    Returns:
        True if all assertions pass, False otherwise.
    """
    console.print("\n[bold]Integration Test: TilingEngine[/bold]\n")

    # Reset singletons for clean test state
    Config.reset()
    Paths.reset()

    # Setup
    console.print("[cyan]1. Setup[/cyan]")
    cleanup_test_data()
    create_mock_wsi(WSI_PATH)
    create_mock_mask(MASK_PATH)

    # Initialize components
    console.print("\n[cyan]2. Initialize Components[/cyan]")
    config = Config.get()
    paths = Paths.get()

    # Override paths for test isolation
    paths._overrides["processed"] = TEST_PROCESSED_DIR
    paths._overrides["manifest"] = TEST_MANIFEST_DIR

    manifest = ManifestManager(db_path=DB_PATH)
    engine = TilingEngine(manifest=manifest, client_id="test_client")

    console.print(f"[dim]  Config environment: {config.environment.value}[/dim]")
    console.print(f"[dim]  Manifest DB: {DB_PATH}[/dim]")

    # Run tiling (only "high" scale fits in 1024x1024 WSI)
    console.print("\n[cyan]3. Run TilingEngine[/cyan]")
    tile_counts = engine.process_wsi(
        wsi_path=WSI_PATH,
        mask_path=MASK_PATH,
        scales=["high"],
        tissue_threshold=0.1,
    )

    console.print(f"[dim]  Tiles extracted: {tile_counts}[/dim]")

    # Assertions
    console.print("\n[cyan]4. Assertions[/cyan]")
    errors: list[str] = []

    # 4.1: Verify tiles saved to processed directory
    output_dir = TEST_PROCESSED_DIR / WSI_NAME / "high"
    file_count = count_tiles_in_directory(output_dir)

    if not output_dir.exists():
        errors.append(f"Output directory not created: {output_dir}")
    elif file_count == 0:
        errors.append(f"No tiles saved to {output_dir}")
    else:
        console.print(f"  [green]\u2713[/green] Tiles saved to {output_dir} ({file_count} files)")

    # 4.2: Verify file count matches DB count
    db_count = count_tiles_in_db(DB_PATH, WSI_NAME)

    if file_count != db_count:
        errors.append(f"File count ({file_count}) != DB count ({db_count})")
    else:
        console.print(f"  [green]\u2713[/green] File count matches DB count ({db_count})")

    # 4.3: Verify no tiles entirely in background region
    # Tiles straddling the boundary (x0 < 512 but x0 + tile_size > 512) are valid
    # because they contain tissue. Only tiles 100% in background should be skipped.
    coords = get_tile_coordinates(DB_PATH, WSI_NAME)
    tile_size_l0 = config.get_scale("high").tile_size  # 256 for high scale
    background_tiles = [c for c in coords if (c[0] + tile_size_l0) <= 512]

    if background_tiles:
        errors.append(
            f"Found {len(background_tiles)} tiles entirely in background (x + {tile_size_l0} <= 512): "
            f"{background_tiles[:3]}..."
        )
    else:
        console.print(f"  [green]\u2713[/green] No tiles extracted from pure background region")

    # 4.4: Verify at least some tiles were extracted (from tissue or boundary)
    tissue_tiles = [c for c in coords if (c[0] + tile_size_l0) > 512]

    if not tissue_tiles:
        errors.append("No tiles extracted from tissue region")
    else:
        console.print(f"  [green]\u2713[/green] {len(tissue_tiles)} tiles contain tissue (boundary + interior)")

    # Result
    console.print()
    if errors:
        console.print(Panel(
            "\n".join(f"[red]\u2717[/red] {e}" for e in errors),
            title="[bold red]Test Failed[/bold red]",
            border_style="red",
        ))
        return False
    else:
        console.print(Panel(
            f"All assertions passed.\n\n"
            f"[dim]Tiles: {tile_counts.get('high', 0)} | "
            f"DB rows: {db_count} | "
            f"Tissue region only: {len(tissue_tiles)}[/dim]",
            title="[bold green]Test Passed[/bold green]",
            border_style="green",
        ))
        return True


def main() -> None:
    """Entry point for integration test."""
    try:
        success = run_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        console.print(f"\n[bold red]Test Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup is optional; comment out to inspect test artifacts
        # cleanup_test_data()
        pass


if __name__ == "__main__":
    main()
