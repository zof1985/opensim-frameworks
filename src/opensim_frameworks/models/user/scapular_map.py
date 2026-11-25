"""Bilinear shoulder-girdle maps fitted to collision-free reference poses."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from .shoulder import SHOULDER_GIRDLE_COORDINATES

_ASSETS = Path(__file__).parent / "assets"
_COEFFICIENTS_PATH = _ASSETS / "scapulothoracic_map_coefficients.csv"
_COEFFICIENT_NAMES = ("b0", "b1", "b2", "b3")
_OUTPUT_NAMES = tuple(SHOULDER_GIRDLE_COORDINATES)


@dataclass(frozen=True)
class BilinearShoulderMap:
    """Predict all five girdle coordinates from shoulder angles in radians."""

    gender: str
    bounds_deg: tuple[tuple[float, float], tuple[float, float]]
    coefficients_deg: np.ndarray

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients_deg, dtype=float)
        if coefficients.shape != (4, len(_OUTPUT_NAMES)) or not np.all(
            np.isfinite(coefficients)
        ):
            raise ValueError("shoulder-map coefficients must be a finite 4-by-5 array")
        if self.gender not in ("M", "F"):
            raise ValueError("gender must be 'M' or 'F'")
        if any(
            len(bounds) != 2
            or not np.all(np.isfinite(bounds))
            or bounds[0] >= bounds[1]
            for bounds in self.bounds_deg
        ):
            raise ValueError("shoulder-map bounds must be finite increasing pairs")
        object.__setattr__(self, "coefficients_deg", coefficients)

    def predict(self, protraction: float, elevation: float) -> np.ndarray:
        """Return the five predicted coordinates, in radians.

        Parameters
        ----------
        protraction, elevation : float
            Sternoclavicular angles in radians.

        Raises
        ------
        ValueError
            If either angle is non-finite.
        """
        angles_deg = np.degrees(np.asarray([protraction, elevation], dtype=float))
        if not np.all(np.isfinite(angles_deg)):
            raise ValueError("shoulder angles must be finite")
        p, e = angles_deg
        features = np.array([1.0, p, e, p * e])
        predicted_deg = features @ self.coefficients_deg
        predicted_deg[:2] = angles_deg
        return np.radians(predicted_deg)

    def contains(self, protraction: float, elevation: float) -> bool:
        """Return whether the input angles lie inside the fitted domain."""
        angles_deg = np.degrees(np.asarray([protraction, elevation], dtype=float))
        if not np.all(np.isfinite(angles_deg)):
            raise ValueError("shoulder angles must be finite")
        return all(
            lower - 1e-10 <= value <= upper + 1e-10
            for value, (lower, upper) in zip(angles_deg, self.bounds_deg)
        )

@lru_cache(maxsize=2)
def load_scapular_map(gender: str) -> BilinearShoulderMap:
    """Load the saved bilinear coefficients for one gender."""
    normalized_gender = str(gender).strip().upper()
    if normalized_gender not in ("M", "F"):
        raise ValueError("gender must be 'M' or 'F'")

    rows: dict[str, dict[str, str]] = {}
    with _COEFFICIENTS_PATH.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["gender"] == normalized_gender:
                name = row["coefficient"]
                if name in rows:
                    raise ValueError(
                        f"duplicate {name!r} coefficient for {normalized_gender}"
                    )
                rows[name] = row

    if set(rows) != set(_COEFFICIENT_NAMES):
        raise ValueError(
            f"missing bilinear coefficients for gender {normalized_gender}"
        )

    first = rows[_COEFFICIENT_NAMES[0]]
    bounds = (
        (float(first["protraction_min_deg"]), float(first["protraction_max_deg"])),
        (float(first["elevation_min_deg"]), float(first["elevation_max_deg"])),
    )
    coefficients = np.array(
        [
            [float(rows[name][coordinate]) for coordinate in _OUTPUT_NAMES]
            for name in _COEFFICIENT_NAMES
        ],
        dtype=float,
    )
    for row in rows.values():
        row_bounds = (
            (float(row["protraction_min_deg"]), float(row["protraction_max_deg"])),
            (float(row["elevation_min_deg"]), float(row["elevation_max_deg"])),
        )
        if row_bounds != bounds:
            raise ValueError(
                f"inconsistent bilinear-map bounds for gender {normalized_gender}"
            )
    return BilinearShoulderMap(normalized_gender, bounds, coefficients)
