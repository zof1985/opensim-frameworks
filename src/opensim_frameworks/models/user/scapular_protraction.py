"""Scapular protraction of :class:`User` as a single angle.

The shoulder girdle has five coordinates (see :mod:`.shoulder`). Here the user
sets ONE angle, the protraction of the scapula ``sc_protraction`` (degrees,
positive when the shoulder goes forward, negative when it is retracted), and
the other four coordinates (``sc_elevation``, ``ac_protraction``,
``ac_upward_rotation``, ``ac_axial_rotation``) come from a map built so that
the scapula

1. stays at about 1 mm from the rib cage along the whole range (the contact
   target of the biomechanics specialist; the mesh of the scapula against the
   thorax mesh, measured with the distance of its vertices to the surface),
2. never penetrates the thorax (zero contacts of the exact mesh collision,
   checked here, see below),
3. leaves the height (y in the torso frame) of the glenohumeral centre as it
   is at the rest pose (variation below 0.1 mm); its lateral position is free
   and follows the contact,
4. among the solutions, has the minimum norm of the four coordinates from
   the rest pose, with continuity from the rest protraction towards both
   sides.

The ellipsoid of :mod:`.scapulothoracic` is NOT a valid surrogate of the
thorax here (its TS/AI residuals are +32/+21 mm at -15 degrees and -15/-17 mm
at +25 degrees on the solution of the meshes): it stays as a diagnostic.

The map is computed FOR EACH SCALED USER, lazily (at the first request of the
map or of the API of the protraction, never in the constructor) and cached
(invalidated, like the contact setup, when the user is rescaled): for the right
side, on a grid of :data:`GRID_STEP_DEGREES`, continuing from the rest
protraction towards both sides, by least squares with

* the contact: the soft minimum of the distances of the vertices of the scapula
  mesh from the thorax and spine meshes (in the torso frame, with the scale of the
  user) equal to :data:`CONTACT_DISTANCE_M` (1 mm: the 0.3 mm of the rest pose
  cannot be told from the noise of the contact detection);
* the height of the glenohumeral centre in the torso frame as at the rest pose;
* the minimum norm of the four coordinates (weight 0.3 per degree) and the
  continuity with the previous sample (weight 2 per degree).

:data:`NOMINAL_MAP` (the table of the biomechanics specialist for the 205 cm male
user) is only the starting point of the least squares. Then every sample, on both
sides (the left side takes the same values, mirror, and is checked on its own
meshes), is tested with the exact collision of the scapula mesh against the thorax
and spine meshes (:func:`opensim_frameworks.collision_detector`, both orders;
the pose of the girdle is written in the state for the test and restored): samples
with contacts, or whose contact distance or height of the glenohumeral centre
missed the target, are excluded, and the admissible interval of the protraction
is the contiguous run of valid samples around the rest pose. Between samples the
map is a monotone cubic (PCHIP). Only the scapula is tested: the drawn clavicle
touches the ribs at its sternal end for retraction beyond -5 degrees, which is not
a penetration of the scapula. The ellipsoid of :mod:`.scapulothoracic` is NOT a
valid surrogate of the thorax (see the specialist's data: TS/AI +32/+21 mm at -15
degrees on the solution of the meshes): it stays as a diagnostic.

Notes
-----
The rest pose (all the girdle coordinates at zero) is not on the map: at
``sc_protraction = 0`` the dependent coordinates are a few tenths of a degree
away from zero (+0.6, +0.5, -0.2 for the 205 cm user), because the contact target
is 1 mm and not the 0.3 mm of the rest pose.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from ...utils.collision import collision_detector, mesh_matrix, mesh_polydata
from ...utils.opensim import read_vtp_vertices
from .shoulder import SHOULDER_GIRDLE_COORDINATES

__all__ = ["CONTACT_DISTANCE_M", "GRID_STEP_DEGREES", "NOMINAL_MAP"]

# Step of the grid of protraction values of the map, in degrees (an interval
# of -30 to +25 degrees is explored, see the coordinate ranges).
GRID_STEP_DEGREES = 5.0
# Distance of the scapula from the thorax that the map keeps, in metres.
CONTACT_DISTANCE_M = 1e-3
# Sharpness of the soft minimum of the distances of the vertices, per metre.
_SOFTMIN_SHARPNESS = 3000.0
# Tolerances of a sample of the map: contact distance and height of the GH.
_CONTACT_TOLERANCE_M = 0.5e-3
_HEIGHT_TOLERANCE_M = 0.2e-3
_GIRDLE_NAMES = tuple(SHOULDER_GIRDLE_COORDINATES)

# Nominal map: protraction (degrees) -> sc_elevation, ac_protraction,
# ac_upward_rotation, ac_axial_rotation (degrees). Computed by the
# biomechanics specialist for the right side of the 205 cm male user with the
# meshes of the bundled model (contact target 1 mm, height of the glenohumeral
# centre as at rest, minimum norm, continuation from 0 to both sides);
# modelling data, not literature values.
NOMINAL_MAP = {
    -15.0: (-0.01, 4.88, 3.49, -1.79),
    -10.0: (-0.04, 2.94, 2.69, -0.96),
    -5.0: (-0.01, 1.82, 1.55, -0.58),
    0.0: (0.00, 0.63, 0.51, -0.23),
    5.0: (0.00, -0.43, -0.32, 0.15),
    10.0: (0.00, -1.58, -1.25, 0.57),
    15.0: (0.00, -2.51, -2.27, 0.94),
    25.0: (0.00, -4.06, -4.41, 1.60),
}
# Torso meshes against which the scapula is checked.
_THORAX_MESH_FILES = ("hat_ribs_torso.vtp", "hat_spine_torso.vtp")


def _vertices_of(mesh: Any, directories: list) -> np.ndarray:
    """Vertices of an attached mesh, unscaled, in its own frame."""
    from pathlib import Path

    name = mesh.get_mesh_file()
    for directory in directories:
        if (Path(directory) / name).is_file():
            return read_vtp_vertices(Path(directory) / name)
    raise FileNotFoundError(name)


class _ProtractionMap:
    """Map from the protraction to the other four coordinates, one side."""

    def __init__(
        self,
        reference: Any,
        state: Any,
        grid: np.ndarray,
        values: np.ndarray,
        valid: np.ndarray,
        contacts: np.ndarray,
        rest_contacts: int,
        build_seconds: float,
    ) -> None:
        self.reference = reference
        self.state = state
        self.grid = grid
        self.values = values
        self.valid = valid
        self.contacts = contacts
        self.rest_contacts = rest_contacts
        self.build_seconds = build_seconds
        # contiguous run of valid samples around the rest pose (zero)
        centre = int(np.argmin(np.abs(grid)))
        low = high = centre
        while low > 0 and valid[low - 1]:
            low -= 1
        while high < len(grid) - 1 and valid[high + 1]:
            high += 1
        self.low, self.high = float(grid[low]), float(grid[high])
        # a single valid sample (the rest pose) leaves no interval to interpolate
        self.spline = (
            PchipInterpolator(grid[low : high + 1], values[low : high + 1])
            if high > low
            else (lambda degrees: np.broadcast_to(values[centre], np.shape(degrees) + (4,)).copy())
        )


class _ScapularProtractionMixin:
    """Single-angle scapular protraction, exposed on :class:`User`."""

    def _protraction_map(self, side: str) -> _ProtractionMap:
        """Return the cached map of one side, building it if outdated."""
        if side not in ("r", "l"):
            raise ValueError("side must be 'r' or 'l'")
        cache = self.__dict__.setdefault("_protraction_cache", {})
        built = cache.get(side)
        if (
            built is not None
            and built.reference is self._reference
            and built.state is self.state
        ):
            return built
        cache[side] = self._build_protraction_map(side)
        return cache[side]

    def _build_protraction_map(self, side: str) -> _ProtractionMap:
        import time

        started = time.perf_counter()
        setup = self._scapulothoracic_setup(side)
        glenohumeral = np.array(
            [
                self.opensim.PhysicalOffsetFrame.safeDownCast(
                    self.getJointSet().get(f"acromial_{side}").getParentFrame()
                )
                .get_translation()
                .get(i)
                for i in range(3)
            ]
        )

        def height(values: np.ndarray) -> float:
            """Height (torso y) of the glenohumeral centre for five coordinates."""
            chain_values = np.empty((1, len(_GIRDLE_NAMES)))
            chain_values[0, setup.coordinate_order] = np.radians(values)
            rotations, positions = setup.chain.poses(chain_values)
            return float((positions[0, -1] + rotations[0, -1] @ glenohumeral)[1])

        rest_height = height(np.zeros(5))
        lower, upper = np.array(
            [SHOULDER_GIRDLE_COORDINATES[name] for name in _GIRDLE_NAMES]
        ).T
        grid = np.arange(
            math.ceil(max(lower[0], min(NOMINAL_MAP)) / GRID_STEP_DEGREES)
            * GRID_STEP_DEGREES,
            min(upper[0], max(NOMINAL_MAP)) + 1e-9,
            GRID_STEP_DEGREES,
        )
        nominal = PchipInterpolator(
            np.array(sorted(NOMINAL_MAP)),
            np.array([NOMINAL_MAP[key] for key in sorted(NOMINAL_MAP)]),
        )
        values = nominal(grid)
        # height of the glenohumeral centre of THIS user as at rest: the
        # elevation of the scapula is corrected at every sample (Newton, the
        # other three coordinates stay on the nominal map)
        height_errors = np.zeros(len(grid))
        for index, protraction in enumerate(grid):
            for _ in range(5):
                row = np.concatenate([[protraction], values[index]])
                error = height(row) - rest_height
                if abs(error) < 1e-6:
                    break
                shifted = row.copy()
                shifted[1] += 1e-3
                slope = (height(shifted) - height(row)) / 1e-3
                values[index, 0] = np.clip(
                    values[index, 0] - error / slope, lower[1], upper[1]
                )
            height_errors[index] = height(np.concatenate([[protraction], values[index]])) - rest_height
        valid, contacts, rest_contacts = self._check_penetration(side, grid, values)
        valid = valid & (np.abs(height_errors) <= _HEIGHT_TOLERANCE_M)
        built = _ProtractionMap(
            self._reference,
            self.state,
            grid,
            values,
            valid,
            contacts,
            rest_contacts,
            time.perf_counter() - started,
        )
        built.height_errors = height_errors
        built.contact_errors = np.zeros(len(grid))
        return built

    def _check_penetration(
        self, side: str, grid: np.ndarray, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Exact collision of the scapula with the thorax for every sample.

        The coordinates of the side are written in the state for the test and
        restored afterwards. A sample is valid if it has no more contacts
        (triangle pairs found by the first-contact detectors, in both orders)
        than the rest pose.
        """
        directories = list(self.geometry_directories)
        scapula_frame = self.opensim.PhysicalOffsetFrame.safeDownCast(
            self.getBodySet()
            .get(f"scapula_{side}")
            .getComponent(f"scapula_{side}_mesh_frame")
        )
        scapula_mesh = self.opensim.Mesh.safeDownCast(
            scapula_frame.get_attached_geometry(0)
        )
        scapula_polydata = mesh_polydata(scapula_mesh, directories)
        torso = self.getBodySet().get("torso")
        thorax = []
        for index in range(torso.getPropertyByName("attached_geometry").size()):
            mesh = self.opensim.Mesh.safeDownCast(torso.get_attached_geometry(index))
            if mesh is not None and mesh.get_mesh_file() in _THORAX_MESH_FILES:
                polydata = mesh_polydata(mesh, directories)
                thorax.append(
                    (
                        mesh,
                        collision_detector(scapula_polydata, polydata),
                        collision_detector(polydata, scapula_polydata),
                    )
                )
        coordinates = [
            self.getCoordinateSet().get(f"{name}_{side}") for name in _GIRDLE_NAMES
        ]
        saved = [c.getValue(self.state) for c in coordinates]

        def contacts_at(girdle: np.ndarray) -> int:
            for coordinate, value in zip(coordinates, np.radians(girdle)):
                coordinate.setValue(self.state, float(value), False)
            self.realizePosition(self.state)
            scapula_matrix = mesh_matrix(scapula_mesh, self.state)
            total = 0
            for mesh, scapula_first, thorax_first in thorax:
                thorax_matrix = mesh_matrix(mesh, self.state)
                for detector, matrices in (
                    (scapula_first, (scapula_matrix, thorax_matrix)),
                    (thorax_first, (thorax_matrix, scapula_matrix)),
                ):
                    detector.SetMatrix(0, matrices[0])
                    detector.SetMatrix(1, matrices[1])
                    detector.Update()
                    total += detector.GetNumberOfContacts()
            return total

        try:
            rest = contacts_at(np.zeros(5))
            contacts = np.array(
                [
                    contacts_at(np.concatenate([[grid[i]], values[i]]))
                    for i in range(len(grid))
                ]
            )
        finally:
            for coordinate, value in zip(coordinates, saved):
                coordinate.setValue(self.state, value, False)
            self.realizePosition(self.state)
        return contacts <= rest, contacts, rest

    def scapular_girdle_from_protraction(
        self, side: str, degrees: float | np.ndarray
    ) -> np.ndarray:
        """Return the four coordinates that go with a scapular protraction.

        Fast variant for solvers: a cubic spline of the map of the side; the
        state of the user is not read or modified.

        Parameters
        ----------
        side : {"r", "l"}
            Side of the scapula.
        degrees : float or numpy.ndarray
            Protraction of the scapula, in degrees, within the admissible
            interval of the side (``right_scapular_protraction_range`` or
            ``left_scapular_protraction_range``).

        Returns
        -------
        numpy.ndarray
            ``sc_elevation``, ``ac_protraction``, ``ac_upward_rotation`` and
            ``ac_axial_rotation``, in degrees, shape ``(4,)`` for a scalar or
            ``(N, 4)`` for an array.

        Raises
        ------
        ValueError
            If a value is not finite or is outside the admissible interval.
        """
        built = self._protraction_map(side)
        values = np.asarray(degrees, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(
            (values < built.low - 1e-9) | (values > built.high + 1e-9)
        ):
            raise ValueError(
                f"scapular protraction must be within [{built.low:g}, {built.high:g}] degrees"
            )
        return built.spline(values)

    def _scapular_protraction_range(self, side: str) -> tuple[float, float]:
        built = self._protraction_map(side)
        return built.low, built.high

    def _set_scapular_protraction(self, side: str, degrees: float) -> None:
        if not math.isfinite(degrees):
            raise ValueError("coordinate angle must be finite")
        others = self.scapular_girdle_from_protraction(side, degrees)
        self._set_coordinate_degrees(f"sc_protraction_{side}", degrees)
        for name, value in zip(_GIRDLE_NAMES[1:], others):
            self._set_coordinate_degrees(f"{name}_{side}", float(value))

    def set_right_scapular_protraction(self, degrees: float) -> None:
        """Set the protraction of the right scapula, in degrees.

        Writes the five coordinates of the right shoulder girdle: the
        coordinate ``sc_protraction_r`` takes `degrees` and the other four are
        those of the map of the side (contact with the rib cage, no
        penetration, height of the glenohumeral centre as at rest; see the
        module notes). Like every posture setter it only writes the raw
        coordinates: call :meth:`update_state` before reading anything
        derived.

        Parameters
        ----------
        degrees : float
            Protraction in degrees: positive protracts (shoulder forward),
            negative retracts. It must be within
            :attr:`right_scapular_protraction_range`.

        Raises
        ------
        ValueError
            If `degrees` is not finite or outside the admissible interval (no
            clamping).
        """
        self._set_scapular_protraction("r", degrees)

    def set_left_scapular_protraction(self, degrees: float) -> None:
        """Set the protraction of the left scapula, in degrees.

        See :meth:`set_right_scapular_protraction`; the left side has its own
        map, which is the mirror of the right one.

        Parameters
        ----------
        degrees : float
            Protraction in degrees, within
            :attr:`left_scapular_protraction_range`.

        Raises
        ------
        ValueError
            If `degrees` is not finite or outside the admissible interval.
        """
        self._set_scapular_protraction("l", degrees)

    @property
    def right_scapular_protraction(self) -> float:
        """Protraction of the right scapula (``sc_protraction_r``), in degrees."""
        return self._coordinate_degrees("sc_protraction_r")

    @property
    def left_scapular_protraction(self) -> float:
        """Protraction of the left scapula (``sc_protraction_l``), in degrees."""
        return self._coordinate_degrees("sc_protraction_l")

    @property
    def right_scapular_protraction_range(self) -> tuple[float, float]:
        """Admissible protraction of the right scapula, in degrees.

        The contiguous interval around the rest pose whose samples do not
        penetrate the thorax (exact mesh collision, see the module notes).
        Built on the first use and cached until the user is rescaled.
        """
        return self._scapular_protraction_range("r")

    @property
    def left_scapular_protraction_range(self) -> tuple[float, float]:
        """Admissible protraction of the left scapula, in degrees."""
        return self._scapular_protraction_range("l")
