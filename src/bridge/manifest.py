"""SQLite manifest manager for tile metadata and diagnostic results."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from src.core.paths import Paths

logger = logging.getLogger(__name__)


@dataclass
class TileRecord:
    """Representation of a single tile entry.

    Args:
        id: Primary key (auto-assigned on insert).
        client_id: Multi-tenant client identifier.
        wsi_name: Source WSI filename (without path).
        scale: Magnification scale name ("macro", "mid", "high").
        x_level0: X coordinate normalized to Level 0 (40x).
        y_level0: Y coordinate normalized to Level 0 (40x).
        width: Tile width in pixels at extraction scale.
        height: Tile height in pixels at extraction scale.
        organ_type: Anatomical region (Gills, Mantle, Digestive Gland, Gonads).
        hemocyte_density: Hemocytes per mm² (nullable).
        yolo_box: YOLO detection bbox as dict (nullable).
        vlm_description: VLM morphological analysis text (nullable).
        final_severity: MSX severity level 0-3 (nullable).
    """
    id: int | None
    client_id: str
    wsi_name: str
    scale: str
    x_level0: int
    y_level0: int
    width: int
    height: int
    organ_type: str | None = None
    hemocyte_density: float | None = None
    yolo_box: dict[str, Any] | None = None
    vlm_description: str | None = None
    final_severity: int | None = None


class ManifestManager:
    """SQLite-backed manifest for tile metadata and diagnostic results.

    Provides CRUD operations for the tiles table. All coordinates are
    stored normalized to Level 0 (40x) resolution per architecture spec.
    """

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS tiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id TEXT NOT NULL,
            wsi_name TEXT NOT NULL,
            scale TEXT NOT NULL,
            x_level0 INTEGER NOT NULL,
            y_level0 INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            organ_type TEXT,
            hemocyte_density REAL,
            yolo_box TEXT,
            vlm_description TEXT,
            final_severity INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_tiles_client ON tiles(client_id);
        CREATE INDEX IF NOT EXISTS idx_tiles_wsi ON tiles(wsi_name);
        CREATE INDEX IF NOT EXISTS idx_tiles_coords ON tiles(x_level0, y_level0);
        CREATE INDEX IF NOT EXISTS idx_tiles_organ ON tiles(organ_type);
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the manifest manager.

        Args:
            db_path: Override path to SQLite database. If None, uses
                     Paths.get().manifest_db.
        """
        self._db_path = db_path or Paths.get().manifest_db
        self._ensure_schema()

    @property
    def db_path(self) -> Path:
        """Return the path to the SQLite database."""
        return self._db_path

    def _ensure_schema(self) -> None:
        """Create tables and indices if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.executescript(self.SCHEMA)
            logger.debug(f"Schema initialized: {self._db_path}")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def register_tile(
        self,
        client_id: str,
        wsi_name: str,
        scale: str,
        x_level0: int,
        y_level0: int,
        width: int,
        height: int,
        organ_type: str | None = None,
    ) -> int:
        """Register a new tile in the manifest.

        Args:
            client_id: Multi-tenant client identifier.
            wsi_name: Source WSI filename.
            scale: Magnification scale ("macro", "mid", "high").
            x_level0: X coordinate at Level 0 (40x).
            y_level0: Y coordinate at Level 0 (40x).
            width: Tile width in pixels.
            height: Tile height in pixels.
            organ_type: Optional anatomical region.

        Returns:
            The auto-generated tile ID.
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tiles (
                    client_id, wsi_name, scale, x_level0, y_level0,
                    width, height, organ_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (client_id, wsi_name, scale, x_level0, y_level0,
                 width, height, organ_type),
            )
            tile_id = cursor.lastrowid
            logger.debug(f"Registered tile {tile_id}: {wsi_name} @ ({x_level0}, {y_level0})")
            return tile_id

    def update_diagnostic(
        self,
        tile_id: int,
        *,
        hemocyte_density: float | None = None,
        yolo_box: dict[str, Any] | None = None,
        vlm_description: str | None = None,
        final_severity: int | None = None,
    ) -> None:
        """Update diagnostic results for a tile.

        Args:
            tile_id: Target tile ID.
            hemocyte_density: Hemocytes per mm².
            yolo_box: YOLO detection bounding box dict.
            vlm_description: VLM morphological analysis.
            final_severity: MSX severity level (0-3).

        Raises:
            ValueError: If tile_id does not exist.
        """
        updates: list[str] = ["updated_at = CURRENT_TIMESTAMP"]
        params: list[Any] = []

        if hemocyte_density is not None:
            updates.append("hemocyte_density = ?")
            params.append(hemocyte_density)

        if yolo_box is not None:
            updates.append("yolo_box = ?")
            params.append(json.dumps(yolo_box))

        if vlm_description is not None:
            updates.append("vlm_description = ?")
            params.append(vlm_description)

        if final_severity is not None:
            if not 0 <= final_severity <= 3:
                raise ValueError(f"Severity must be 0-3, got {final_severity}")
            updates.append("final_severity = ?")
            params.append(final_severity)

        params.append(tile_id)

        with self._connection() as conn:
            cursor = conn.execute(
                f"UPDATE tiles SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Tile {tile_id} not found")
            logger.debug(f"Updated diagnostics for tile {tile_id}")

    def get_tile(self, tile_id: int) -> TileRecord | None:
        """Retrieve a tile by ID.

        Args:
            tile_id: Target tile ID.

        Returns:
            TileRecord if found, None otherwise.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM tiles WHERE id = ?", (tile_id,)
            ).fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def get_tiles_by_wsi(
        self,
        wsi_name: str,
        client_id: str | None = None,
    ) -> list[TileRecord]:
        """Retrieve all tiles for a given WSI.

        Args:
            wsi_name: Source WSI filename.
            client_id: Optional filter by client.

        Returns:
            List of TileRecord objects.
        """
        query = "SELECT * FROM tiles WHERE wsi_name = ?"
        params: list[Any] = [wsi_name]

        if client_id is not None:
            query += " AND client_id = ?"
            params.append(client_id)

        query += " ORDER BY x_level0, y_level0"

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_record(row) for row in rows]

    def get_tiles_by_coords(
        self,
        x_level0: int,
        y_level0: int,
        radius: int = 0,
        client_id: str | None = None,
    ) -> list[TileRecord]:
        """Retrieve tiles near a Level 0 coordinate.

        Args:
            x_level0: X coordinate at Level 0.
            y_level0: Y coordinate at Level 0.
            radius: Search radius in Level 0 pixels.
            client_id: Optional filter by client.

        Returns:
            List of TileRecord objects within the search area.
        """
        query = """
            SELECT * FROM tiles
            WHERE x_level0 BETWEEN ? AND ?
            AND y_level0 BETWEEN ? AND ?
        """
        params: list[Any] = [
            x_level0 - radius, x_level0 + radius,
            y_level0 - radius, y_level0 + radius,
        ]

        if client_id is not None:
            query += " AND client_id = ?"
            params.append(client_id)

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_record(row) for row in rows]

    def _row_to_record(self, row: sqlite3.Row) -> TileRecord:
        """Convert a database row to a TileRecord."""
        yolo_box = None
        if row["yolo_box"]:
            yolo_box = json.loads(row["yolo_box"])

        return TileRecord(
            id=row["id"],
            client_id=row["client_id"],
            wsi_name=row["wsi_name"],
            scale=row["scale"],
            x_level0=row["x_level0"],
            y_level0=row["y_level0"],
            width=row["width"],
            height=row["height"],
            organ_type=row["organ_type"],
            hemocyte_density=row["hemocyte_density"],
            yolo_box=yolo_box,
            vlm_description=row["vlm_description"],
            final_severity=row["final_severity"],
        )
