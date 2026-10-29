"""Shared package-wide constants."""

from pathlib import Path

MM_TO_M = 1e-3
IRON_DENSITY_KG_M3 = 7870.0
MESHES_DIR = (
    Path(__file__).resolve().parents[1] / "bodies" / "assets" / "meshes"
)
DEFAULT_SIZE = (900, 700)
VIEW_NAMES = ("front", "back", "left", "right", "top", "bottom")

__all__ = [
    "MM_TO_M",
    "IRON_DENSITY_KG_M3",
    "MESHES_DIR",
    "DEFAULT_SIZE",
    "VIEW_NAMES",
]
