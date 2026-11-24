"""Scapulothoracic contact of the shoulder girdle of :class:`User`.

The kinematic tree of the user has no scapulothoracic joint (see
:mod:`.shoulder`). The handle fit uses distances from three costal scapular
mesh vertices to the rib surface as soft objectives, with a separate mesh
intersection report. These unsigned distances do not guarantee contact or
non-penetration. This module also provides ellipsoidal residual functions for
other uses; neither method adds a kinematic constraint to the model.

The thorax is an axis-aligned ellipsoid fitted by the biomechanics specialist
to the ribs of the bundled model, expressed in the torso frame of the
*unscaled* model (x forward, y up, z to the right). Points are mapped back to
that frame with the scale factors of the torso body, so that the ellipsoidal
radius ``r`` (1 on the surface) does not depend on the user's size. Six
points of the scapula are used: the trigonum spinae (``TS``) and the angulus
inferior (``AI``) must touch the thorax, the acromion angle (``AA``), two
points of the medial border (``MB1``, ``MB2``) and the superior angle
(``SA``) must not penetrate it.

Notes
-----
All numbers are modelling values, not literature values: the ellipsoid and the
scapular points were measured on the mesh of the bundled model. The contact
radii of ``TS`` and ``AI`` are not constants: they are calibrated on each user
as the radii these points have when all girdle coordinates are zero (nominally
1.239 and 1.158, the values of the bundled model), so the contact residuals
vanish at the zero pose by construction. This is a hypothesis (the scapula
touches the thorax in the zero pose, with any soft tissue offset implicit in
the radii). The clearance radius 1.10 of the other points is a hypothesis of
the biomechanics specialist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import opensim
import vtk
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares
from vtk.util.numpy_support import vtk_to_numpy

from ...utils.collision import collision_detector, mesh_matrix, mesh_polydata
from ...utils.kinematics import KinematicChain
from ...utils.opensim import rotation_to_numpy, vec3_to_numpy
from .mapping import segment_scale_factors
from .shoulder import SHOULDER_GIRDLE_COORDINATES, SHOULDER_GIRDLE_REFERENCE_POSE

__all__ = []

# Torso frame of the unscaled model, right side, metres.
_ELLIPSOID_CENTER = np.array([-0.01548, 0.25861, 0.0])
# Semi-axes along x (antero-posterior), y (vertical) and z (lateral).
_ELLIPSOID_SEMI_AXES = np.array([0.07522, 0.15482, 0.13707])
# Scapular points, in this order, relative to the right glenohumeral centre
# (unscaled torso frame); the right glenohumeral centre is at (0.003155,
# 0.3715, 0.17). The first two are the contact points.
_POINT_NAMES = ("AI", "TS", "AA", "MB1", "MB2", "SA")
_POINT_OFFSETS = np.array(
    [
        [-0.0722, -0.1248, -0.0452],
        [-0.0864, -0.0225, -0.0850],
        [-0.0045, 0.0268, -0.0052],
        [-0.0896, -0.0530, -0.0786],
        [-0.0864, -0.0899, -0.0618],
        [-0.0520, 0.0149, -0.0804],
    ]
)
_CONTACT_COUNT = 2
# Points that must stay outside this ellipsoidal radius (hypothesis).
_CLEARANCE_RADIUS = 1.10
# Non-penetration residual: the points may sink into the thorax by this depth
# (soft tissue) at no cost, and beyond it the residual grows with a quadratic
# blend of this width, then linearly (both in metres of penetration).
_PENETRATION_ALLOWANCE = 0.4e-3
_PENETRATION_BLEND = 0.1e-3
_MIRROR = np.array([1.0, 1.0, -1.0])
_GIRDLE_NAMES = tuple(SHOULDER_GIRDLE_COORDINATES)
_MESH_CONTACT_TARGET_M = 0.005
_MESH_CONTACT_MAX_GAP_M = 0.0055
_SHOULDER_MAP_STEP_RAD = math.radians(5.0)
# Strong soft preference for collision-free map samples; exact post-fit
# collision checks remain the final guard.
_SHOULDER_COLLISION_PENALTY = 1000.0
# Rib-facing vertices on the inferior, superior and lateral costal surface of
# the bundled right scapula, in the unscaled mesh frame. The left mesh mirrors
# them. They are model landmarks, not measured anatomical contact locations.
_COSTAL_FIT_POINTS = np.array([
    [-0.04701870, 0.26429280, 0.12891562],
    [-0.04884731, 0.38570560, 0.08864388],
    [-0.02128667, 0.32236240, 0.12832849],
])


class _ScapulaMeshContact:
    """Measure proximity and intersections of the scapula and thorax meshes."""

    def __init__(self, user: Any, side: str) -> None:
        self.user = user
        self.reference = user._reference
        self.state = user.state
        frame = opensim.PhysicalOffsetFrame.safeDownCast(
            user.getBodySet().get(f"scapula_{side}").getComponent(
                f"scapula_{side}_mesh_frame"
            )
        )
        self.scapula = opensim.Mesh.safeDownCast(frame.get_attached_geometry(0))
        data = mesh_polydata(self.scapula, user.geometry_directories)
        vertices = vtk_to_numpy(data.GetPoints().GetData()).copy()
        self.vertices = np.column_stack((vertices, np.ones(len(vertices))))
        # Select the same three rib-facing landmarks independently of pose and
        # scale, using the original scapular mesh coordinates.
        landmarks = _COSTAL_FIT_POINTS * (np.ones(3) if side == "r" else _MIRROR)
        distances = np.linalg.norm(vertices[:, None, :] - landmarks[None, :, :], axis=2)
        indices = np.argmin(distances, axis=0)
        if len(set(indices)) != 3 or np.any(distances[indices, np.arange(3)] > 1e-3):
            raise ValueError(f"{side} scapula mesh does not contain the costal landmarks")
        self.costal_vertices = self.vertices[indices]
        self.ribs_locator = vtk.vtkStaticCellLocator()
        self.ribs = None
        self.detectors = []
        torso = user.getBodySet().get("torso")
        for index in range(torso.getPropertyByName("attached_geometry").size()):
            mesh = opensim.Mesh.safeDownCast(torso.get_attached_geometry(index))
            if mesh is None:
                continue
            name = Path(mesh.get_mesh_file()).name
            if name not in ("hat_ribs_torso.vtp", "hat_spine_torso.vtp"):
                continue
            thorax_data = mesh_polydata(mesh, user.geometry_directories)
            self.detectors.append(
                (
                    mesh,
                    collision_detector(data, thorax_data),
                    collision_detector(thorax_data, data),
                )
            )
            if name == "hat_ribs_torso.vtp":
                self.ribs = mesh
                scale = mesh.get_scale_factors()
                transform = vtk.vtkTransform()
                transform.Scale(scale.get(0), scale.get(1), scale.get(2))
                transformed = vtk.vtkTransformFilter()
                transformed.SetInputData(thorax_data)
                transformed.SetTransform(transform)
                transformed.Update()
                self.ribs_locator.SetDataSet(transformed.GetOutput())
                self.ribs_locator.BuildLocator()
        if self.ribs is None or len(self.detectors) != 2:
            raise ValueError("Scapula, rib and spine meshes are required")

    def _distances(self, vertices: np.ndarray) -> np.ndarray:
        matrix = mesh_matrix(self.scapula, self.user.state)
        transform = np.array(
            [[matrix.GetElement(row, column) for column in range(4)] for row in range(3)]
        )
        points_ground = vertices @ transform.T
        ribs_frame = self.ribs.getFrame().getTransformInGround(self.user.state)
        rotation = rotation_to_numpy(ribs_frame.R())
        position = vec3_to_numpy(ribs_frame.p())
        points = (points_ground - position) @ rotation
        distances = []
        for point in points:
            closest = [0.0, 0.0, 0.0]
            cell_id = vtk.reference(0)
            sub_id = vtk.reference(0)
            distance = vtk.reference(0.0)
            self.ribs_locator.FindClosestPoint(point, closest, cell_id, sub_id, distance)
            distances.append(math.sqrt(float(distance)))
        return np.asarray(distances)

    def distance(self) -> float:
        return float(np.min(self._distances(self.vertices)))

    def costal_distances(self) -> np.ndarray:
        return self._distances(self.costal_vertices)

    def intersects(self) -> bool:
        scapula_matrix = mesh_matrix(self.scapula, self.user.state)
        for mesh, forward, reverse in self.detectors:
            torso_matrix = mesh_matrix(mesh, self.user.state)
            for detector, first, second in (
                (forward, scapula_matrix, torso_matrix),
                (reverse, torso_matrix, scapula_matrix),
            ):
                detector.SetMatrix(0, first)
                detector.SetMatrix(1, second)
                detector.Update()
                if detector.GetNumberOfContacts():
                    return True
        return False


def _ellipsoid_radius(unscaled_point: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the ellipsoidal radius of a point and its gradient.

    Parameters
    ----------
    unscaled_point : numpy.ndarray
        Point of the right side, torso frame of the unscaled model, shape
        ``(3,)``.

    Returns
    -------
    radius : float
        ``||(u - c) / semi_axes||``; 1 on the surface, less than 1 inside.
    gradient : numpy.ndarray
        Analytic derivative of the radius with respect to the point,
        ``diag(1 / semi_axes**2) (u - c) / r``, shape ``(3,)``.
    """
    offset = unscaled_point - _ELLIPSOID_CENTER
    radius = float(np.linalg.norm(offset / _ELLIPSOID_SEMI_AXES))
    gradient = offset / _ELLIPSOID_SEMI_AXES**2 / max(radius, 1e-12)
    return radius, gradient


def _radii(points: np.ndarray, torso_scale: np.ndarray, side: str) -> np.ndarray:
    """Return the ellipsoidal radii of scapular points.

    Parameters
    ----------
    points : numpy.ndarray
        Positions in the torso frame of the scaled model, shape ``(6, 3)`` or
        ``(N, 6, 3)``, ordered as ``_POINT_NAMES``.
    torso_scale : numpy.ndarray
        Scale factors ``(x, y, z)`` of the torso body.
    side : {"r", "l"}
        Side of the scapula; the left one is mirrored onto the right one.

    Returns
    -------
    numpy.ndarray
        Radii, shape ``(6,)`` or ``(N, 6)``.
    """
    mirror = np.ones(3) if side == "r" else _MIRROR
    # Back to the unscaled torso frame, left side mirrored to the right.
    unscaled = points / torso_scale * mirror
    scaled = (unscaled - _ELLIPSOID_CENTER) / _ELLIPSOID_SEMI_AXES
    return np.linalg.norm(scaled, axis=-1)


def _residual_array(
    radii: np.ndarray, contact_radii: np.ndarray, length: float
) -> np.ndarray:
    """Return the residuals (m) of the six scapular points from their radii.

    Parameters
    ----------
    radii : numpy.ndarray
        Ellipsoidal radii ordered as ``_POINT_NAMES``, shape ``(6,)`` or
        ``(N, 6)``.
    contact_radii : numpy.ndarray
        Radii the two contact points (``AI``, ``TS``) must keep.
    length : float
        Mean scaled semi-axis, which converts radius into metres.

    Returns
    -------
    numpy.ndarray
        Residuals in metres, same shape as ``radii``.
    """
    residuals = np.empty(radii.shape)
    # Contact: equations, zero when the point is on its contact radius.
    residuals[..., :_CONTACT_COUNT] = (
        radii[..., :_CONTACT_COUNT] - contact_radii
    ) * length
    # Non-penetration: unilateral, exactly zero up to the allowed depth and
    # continuous with continuous derivative beyond it (Huber-like). With
    # u = penetration depth (m) - allowance: 0 for u <= 0, u^2 / (2 b) for
    # 0 < u < b and u - b / 2 for u >= b, where b is the blend width.
    depth = (_CLEARANCE_RADIUS - radii[..., _CONTACT_COUNT:]) * length
    u = np.maximum(depth - _PENETRATION_ALLOWANCE, 0.0)
    blend = _PENETRATION_BLEND
    residuals[..., _CONTACT_COUNT:] = np.where(
        u < blend, u * u / (2.0 * blend), u - 0.5 * blend
    )
    return residuals


@dataclass
class _Setup:
    """Quantities of one scapula that stay valid while the geometry does.

    The setup is rebuilt when the anthropometric reference or the state
    object of the user changes (any scaling or reinitialisation does both).
    """

    reference: Any
    state: Any
    chain: Any
    coordinate_order: list[int]
    station_array: np.ndarray
    torso_scale: np.ndarray
    length: float
    stations: list
    scapula: Any
    torso: Any
    contact_radii: np.ndarray
    contact_pose: dict[str, float] | None = None


@dataclass
class _ShoulderMap:
    """Interpolation of scapular coordinates for one scaled user and side."""

    reference: Any
    state: Any
    bounds: tuple[tuple[float, float], ...]
    interpolator: RegularGridInterpolator
    collision_risk: RegularGridInterpolator


class _ScapulothoracicMixin:
    """Scapulothoracic contact residuals, exposed on :class:`User`."""

    @property
    def scapula_contact_max_gap_m(self) -> float:
        """Return the maximum modeled scapula-to-rib gap, in metres."""
        return _MESH_CONTACT_MAX_GAP_M

    def _scapula_mesh_contact(self, side: str) -> _ScapulaMeshContact:
        if side not in ("r", "l"):
            raise ValueError("side must be 'r' or 'l'")
        cache = self.__dict__.setdefault("_scapula_mesh_contact_cache", {})
        contact = cache.get(side)
        if contact is None or contact.state is not self.state or contact.reference is not self._reference:
            contact = _ScapulaMeshContact(self, side)
            cache[side] = contact
        return contact

    def scapula_contact_residual(self, side: str) -> float:
        """Return the weighted scapular mesh-proximity residual.

        Parameters
        ----------
        side : {"r", "l"}
            Scapular side.

        Returns
        -------
        float
            Distance from the 5 mm target multiplied by 1000.
        """
        self.realizePosition(self.state)
        return 1000.0 * (self._scapula_mesh_contact(side).distance() - _MESH_CONTACT_TARGET_M)

    def scapula_costal_distances(self, side: str) -> np.ndarray:
        """Return distances from three costal scapular vertices to the ribs.

        Parameters
        ----------
        side : {"r", "l"}
            Scapular side.

        Returns
        -------
        numpy.ndarray
            Unsigned closest-surface distances in metres, ordered inferior,
            superior, lateral. They do not detect penetration or establish
            contact over the entire scapula.
        """
        self.realizePosition(self.state)
        return self._scapula_mesh_contact(side).costal_distances()

    def scapular_fit_residuals(
        self, side: str, protraction_weight: float, elevation_weight: float
    ) -> np.ndarray:
        """Return retraction, elevation and mesh-contact fitting residuals.

        Parameters
        ----------
        side : {"r", "l"}
            Scapular side.
        protraction_weight, elevation_weight : float
            Nonnegative weights for retraction and positive elevation.

        Returns
        -------
        numpy.ndarray
            Retraction, elevation, three equally weighted costal-point
            distances from the 5 mm target, and a soft collision-risk
            residual derived from the sampled shoulder map. The risk is an
            interpolation, not a collision test at the current pose.
        """
        if side not in ("r", "l"):
            raise ValueError("side must be 'r' or 'l'")
        if not all(math.isfinite(weight) and weight >= 0 for weight in (protraction_weight, elevation_weight)):
            raise ValueError("scapular weights must be finite and nonnegative")
        coordinates = self.getCoordinateSet()
        protraction = coordinates.get(f"sc_protraction_{side}")
        elevation = coordinates.get(f"sc_elevation_{side}")
        shoulder = (protraction.getValue(self.state), elevation.getValue(self.state))
        risk = float(self._shoulder_map(side).collision_risk(shoulder))
        return np.concatenate((
            np.array([
                math.sqrt(protraction_weight) * (shoulder[0] - protraction.getRangeMin()),
                math.sqrt(elevation_weight) * max(shoulder[1], 0.0),
            ]),
            1000.0 / math.sqrt(3.0) * (
                self.scapula_costal_distances(side) - _MESH_CONTACT_TARGET_M
            ),
            np.array([_SHOULDER_COLLISION_PENALTY * risk]),
        ))

    def _shoulder_map(self, side: str) -> _ShoulderMap:
        if side not in ("r", "l"):
            raise ValueError("side must be 'r' or 'l'")
        coordinates = [
            self.getCoordinateSet().get(f"{name}_{side}") for name in _GIRDLE_NAMES
        ]
        bounds = tuple(
            (coordinate.getRangeMin(), coordinate.getRangeMax())
            for coordinate in coordinates
        )
        cache = self.__dict__.setdefault("_shoulder_map_cache", {})
        built = cache.get(side)
        if (
            built is not None
            and built.reference is self._reference
            and built.state is self.state
            and built.bounds == bounds
        ):
            return built

        # Cover the full requested shoulder ranges for this scaled user.
        grid = [
            np.linspace(low, high, math.ceil((high - low) / _SHOULDER_MAP_STEP_RAD) + 1)
            for low, high in bounds[:2]
        ]
        lower = np.array([limit[0] for limit in bounds[2:]])
        upper = np.array([limit[1] for limit in bounds[2:]])
        values = np.empty((len(grid[0]), len(grid[1]), 3))
        crosses = np.empty(values.shape[:2], dtype=bool)
        contact = self._scapula_mesh_contact(side)
        saved = np.array([coordinate.getValue(self.state) for coordinate in coordinates])
        failed = 0

        def apply(shoulder: np.ndarray, scapula: np.ndarray) -> None:
            for coordinate, value in zip(coordinates, np.concatenate((shoulder, scapula))):
                coordinate.setValue(self.state, float(value), False)
            self.realizePosition(self.state)

        try:
            for i, protraction in enumerate(grid[0]):
                for j, elevation in enumerate(grid[1]):
                    shoulder = np.array([protraction, elevation])
                    previous = values[i, j - 1] if j else (
                        values[i - 1, j] if i else np.clip(np.zeros(3), lower, upper)
                    )

                    def residual(scapula: np.ndarray) -> np.ndarray:
                        apply(shoulder, scapula)
                        return np.concatenate((
                            1000.0 / math.sqrt(3.0) * (
                                contact.costal_distances() - _MESH_CONTACT_TARGET_M
                            ),
                            0.1 * scapula,
                        ))

                    # Check both the continued pose and a neutral start; prefer
                    # a non-intersecting candidate when one is found.
                    candidates = []
                    successful = False
                    neutral = np.clip(np.zeros(3), lower, upper)
                    starts = (
                        [previous] if np.array_equal(previous, neutral)
                        else [previous, neutral]
                    )
                    starts.append(np.clip(np.radians([10.0, 0.0, 15.0]), lower, upper))
                    for start in starts:
                        objective = residual(start)
                        candidates.append(
                            (contact.intersects(), float(objective @ objective), start.copy())
                        )
                        result = least_squares(
                            residual, start, bounds=(lower, upper), max_nfev=60,
                            ftol=1e-4, xtol=1e-4, gtol=1e-4,
                        )
                        successful = successful or result.success
                        objective = residual(result.x)
                        candidates.append(
                            (contact.intersects(), float(objective @ objective), result.x.copy())
                        )
                    if not successful:
                        failed += 1
                    clear = [candidate for candidate in candidates if not candidate[0]]
                    crossing = [candidate for candidate in candidates if candidate[0]]
                    if clear and crossing:
                        safe = min(clear, key=lambda item: item[1])
                        target = min(crossing, key=lambda item: item[1])
                        if target[1] < safe[1]:
                            # Search toward the closer solution without changing
                            # the requested shoulder angles.
                            low, high = 0.0, 1.0
                            for _ in range(12):
                                fraction = (low + high) / 2.0
                                candidate = safe[2] + fraction * (target[2] - safe[2])
                                objective = residual(candidate)
                                collides = contact.intersects()
                                if collides:
                                    high = fraction
                                else:
                                    low = fraction
                                    candidates.append(
                                        (False, float(objective @ objective), candidate.copy())
                                    )
                    chosen = min(candidates, key=lambda item: (item[0], item[1]))
                    crosses[i, j], values[i, j] = chosen[0], chosen[2]
        finally:
            for coordinate, value in zip(coordinates, saved):
                coordinate.setValue(self.state, float(value), False)
            self.realizePosition(self.state)

        if failed:
            warnings.warn(
                f"{side} shoulder map: optimization did not converge for {failed} "
                f"of {len(grid[0]) * len(grid[1])} samples; best candidates retained",
                RuntimeWarning,
                stacklevel=2,
            )
        if np.all(crosses):
            warnings.warn(
                f"{side} shoulder map: no collision-free sampled poses found",
                RuntimeWarning,
                stacklevel=2,
            )
            risk = np.ones(crosses.shape)
        else:
            spacing = tuple((axis[1] - axis[0]) / _SHOULDER_MAP_STEP_RAD for axis in grid)
            risk = distance_transform_edt(crosses, sampling=spacing)
        built = _ShoulderMap(
            reference=self._reference,
            state=self.state,
            bounds=bounds,
            interpolator=RegularGridInterpolator(tuple(grid), values),
            collision_risk=RegularGridInterpolator(tuple(grid), risk),
        )
        cache[side] = built
        return built

    def set_shoulder_girdle(
        self, side: str, protraction: float, elevation: float, *, check_contact: bool = True
    ) -> None:
        """Set shoulder angles and derive the three scapular coordinates.

        Parameters
        ----------
        side : {"r", "l"}
            Shoulder side.
        protraction, elevation : float
            Sternoclavicular angles in radians.
        check_contact : bool, default=True
            Warn about gaps or mesh intersections at the interpolated pose.

        Notes
        -----
        The cached map is fitted separately to each scaled user's mesh.
        Its 5 mm point targets, collision-guided candidate search and
        interpolation are approximations; none guarantees that the full
        scapular mesh remains outside the thorax.

        Raises
        ------
        ValueError
            If the side or either angle is invalid or outside its range.
        """
        if side not in ("r", "l"):
            raise ValueError("side must be 'r' or 'l'")
        angles = np.array([protraction, elevation], dtype=float)
        if not np.all(np.isfinite(angles)):
            raise ValueError("shoulder angles must be finite")
        coordinates = self.getCoordinateSet()
        for name, value in zip(_GIRDLE_NAMES[:2], angles):
            coordinate = coordinates.get(f"{name}_{side}")
            if not coordinate.getRangeMin() <= value <= coordinate.getRangeMax():
                raise ValueError("shoulder angle outside its range")
        scapula = self._shoulder_map(side).interpolator(tuple(angles))
        self.set_scapular_girdle_values(
            side, np.concatenate((angles, scapula)), check_contact=check_contact
        )

    def _set_shoulder_elevation(
        self,
        side: str,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
        *,
        check_contact: bool = True,
    ) -> None:
        if range_min is not None or range_max is not None:
            self._set_coordinate_degrees(
                f"sc_elevation_{side}", degrees, range_min, range_max
            )
        coordinates = self.getCoordinateSet()
        protraction = coordinates.get(f"sc_protraction_{side}").getValue(self.state)
        self.set_shoulder_girdle(
            side, protraction, math.radians(degrees), check_contact=check_contact
        )

    def set_left_shoulder_elevation(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
        *,
        check_contact: bool = True,
    ) -> None:
        """Set left shoulder-girdle elevation and adapt scapular orientation.

        Positive values elevate the shoulder; negative values depress it.
        This moves the shoulder girdle, not the humerus at the glenohumeral
        joint. The current protraction is preserved.

        Parameters
        ----------
        degrees : float
            Elevation in degrees, positive upward and negative downward.
        range_min, range_max : float or None, optional
            Coordinate bounds in degrees. Supplied bounds replace the
            corresponding current bounds; omitted bounds remain unchanged.
            When either bound is supplied, ``degrees`` must lie in the
            resulting range.
        check_contact : bool, default=True
            Report possible separation or intersection after setting the pose.

        Raises
        ------
        ValueError
            If the angle or requested coordinate range is invalid.
        """
        self._set_shoulder_elevation(
            "l", degrees, range_min, range_max, check_contact=check_contact
        )

    def set_right_shoulder_elevation(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
        *,
        check_contact: bool = True,
    ) -> None:
        """Set right shoulder-girdle elevation and adapt scapular orientation.

        Positive values elevate the shoulder; negative values depress it.
        This moves the shoulder girdle, not the humerus at the glenohumeral
        joint. The current protraction is preserved.

        Parameters
        ----------
        degrees : float
            Elevation in degrees, positive upward and negative downward.
        range_min, range_max : float or None, optional
            Coordinate bounds in degrees. Supplied bounds replace the
            corresponding current bounds; omitted bounds remain unchanged.
            When either bound is supplied, ``degrees`` must lie in the
            resulting range.
        check_contact : bool, default=True
            Report possible separation or intersection after setting the pose.

        Raises
        ------
        ValueError
            If the angle or requested coordinate range is invalid.
        """
        self._set_shoulder_elevation(
            "r", degrees, range_min, range_max, check_contact=check_contact
        )

    def _set_shoulder_protraction(
        self,
        side: str,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
        *,
        check_contact: bool = True,
    ) -> None:
        if range_min is not None or range_max is not None:
            self._set_coordinate_degrees(
                f"sc_protraction_{side}", degrees, range_min, range_max
            )
        coordinates = self.getCoordinateSet()
        elevation = coordinates.get(f"sc_elevation_{side}").getValue(self.state)
        self.set_shoulder_girdle(
            side, math.radians(degrees), elevation, check_contact=check_contact
        )

    def set_left_shoulder_protraction(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
        *,
        check_contact: bool = True,
    ) -> None:
        """Set left shoulder-girdle protraction and adapt scapular orientation.

        Positive values protract the shoulder forward; negative values
        retract it. This moves the shoulder girdle, not the humerus at the
        glenohumeral joint. The current elevation is preserved.

        Parameters
        ----------
        degrees : float
            Protraction in degrees, positive forward and negative backward.
        range_min, range_max : float or None, optional
            Coordinate bounds in degrees. Supplied bounds replace the
            corresponding current bounds; omitted bounds remain unchanged.
            When either bound is supplied, ``degrees`` must lie in the
            resulting range.
        check_contact : bool, default=True
            Report possible separation or intersection after setting the pose.

        Raises
        ------
        ValueError
            If the angle or requested coordinate range is invalid.
        """
        self._set_shoulder_protraction(
            "l", degrees, range_min, range_max, check_contact=check_contact
        )

    def set_right_shoulder_protraction(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
        *,
        check_contact: bool = True,
    ) -> None:
        """Set right shoulder-girdle protraction and adapt scapular orientation.

        Positive values protract the shoulder forward; negative values
        retract it. This moves the shoulder girdle, not the humerus at the
        glenohumeral joint. The current elevation is preserved.

        Parameters
        ----------
        degrees : float
            Protraction in degrees, positive forward and negative backward.
        range_min, range_max : float or None, optional
            Coordinate bounds in degrees. Supplied bounds replace the
            corresponding current bounds; omitted bounds remain unchanged.
            When either bound is supplied, ``degrees`` must lie in the
            resulting range.
        check_contact : bool, default=True
            Report possible separation or intersection after setting the pose.

        Raises
        ------
        ValueError
            If the angle or requested coordinate range is invalid.
        """
        self._set_shoulder_protraction(
            "r", degrees, range_min, range_max, check_contact=check_contact
        )

    def scapula_contact_report(self, side: str) -> dict[str, float | bool]:
        """Measure scapula-to-rib distance and scapula-thorax mesh crossing.

        Parameters
        ----------
        side : {"r", "l"}
            Scapular side.

        Returns
        -------
        dict
            ``gap_m`` is the minimum scapular-vertex-to-rib-surface distance;
            ``max_costal_gap_m`` is the largest of the three point distances;
            ``intersects`` reports triangle crossings with ribs or spine.
            These metrics do not establish anatomical contact over the entire
            scapula or detect all possible containment of open meshes.
        """
        self.realizePosition(self.state)
        contact = self._scapula_mesh_contact(side)
        gap = contact.distance()
        costal_gaps = contact.costal_distances()
        intersects = contact.intersects()
        if (
            intersects
            or gap > _MESH_CONTACT_MAX_GAP_M
            or np.any(np.abs(costal_gaps - _MESH_CONTACT_TARGET_M) > 0.0005)
        ):
            warnings.warn(
                f"{side} scapula: rib gap {gap * 1000:.2f} mm, "
                f"costal point gaps {np.round(costal_gaps * 1000, 2).tolist()} mm, "
                f"intersection with ribs or spine: {intersects}",
                RuntimeWarning,
                stacklevel=2,
            )
        return {
            "gap_m": gap,
            "max_costal_gap_m": float(np.max(costal_gaps)),
            "intersects": intersects,
        }

    def set_scapular_girdle_values(
        self, side: str, values: Sequence[float], *, check_contact: bool = True
    ) -> None:
        """Set the five scapular-girdle coordinates, in radians.

        Parameters
        ----------
        side : {"r", "l"}
            Scapular side.
        values : sequence of float
            Values in the order of ``SHOULDER_GIRDLE_COORDINATES``.
        check_contact : bool, default=True
            Report possible separation or intersection after setting the pose.
            Solvers may disable this for intermediate candidate poses.

        Raises
        ------
        ValueError
            If the side, values or coordinate ranges are invalid.
        """
        if side not in ("r", "l"):
            raise ValueError("side must be 'r' or 'l'")
        angles = np.asarray(values, dtype=float)
        if angles.shape != (len(_GIRDLE_NAMES),) or not np.all(np.isfinite(angles)):
            raise ValueError("five finite scapular coordinates are required")
        coordinates = [
            self.getCoordinateSet().get(f"{name}_{side}") for name in _GIRDLE_NAMES
        ]
        if any(
            not coordinate.getRangeMin() - 1e-10 <= value <= coordinate.getRangeMax() + 1e-10
            for coordinate, value in zip(coordinates, angles)
        ):
            raise ValueError("scapular coordinate outside its range")
        for coordinate, value in zip(coordinates, angles):
            coordinate.setValue(self.state, float(value), False)
        self.realizePosition(self.state)
        if check_contact:
            self.scapula_contact_report(side)

    def resolve_scapula_intersection(
        self, side: str, reference: Sequence[float]
    ) -> bool:
        """Move an intersecting scapula toward a non-intersecting reference.

        Parameters
        ----------
        side : {"r", "l"}
            Scapular side.
        reference : sequence of float
            Reference girdle coordinates in radians, in the order of
            ``SHOULDER_GIRDLE_COORDINATES``.

        Returns
        -------
        bool
            Whether the original pose was changed. If the reference also
            intersects, the original pose is restored and left for reporting.

        Notes
        -----
        Mesh crossing is tested at the resulting pose, not continuously over
        the interpolated motion. A non-intersecting pose is not necessarily
        within the modeled contact distance.
        """
        contact = self._scapula_mesh_contact(side)
        if not contact.intersects():
            self.scapula_contact_report(side)
            return False
        coordinates = [
            self.getCoordinateSet().get(f"{name}_{side}") for name in _GIRDLE_NAMES
        ]
        target = np.array([coordinate.getValue(self.state) for coordinate in coordinates])
        reference_values = np.asarray(reference, dtype=float)
        self.set_scapular_girdle_values(side, reference_values, check_contact=False)
        if contact.intersects():
            self.set_scapular_girdle_values(side, target, check_contact=False)
            self.scapula_contact_report(side)
            return False
        low, high = 0.0, 1.0
        for _ in range(20):
            fraction = (low + high) / 2.0
            self.set_scapular_girdle_values(
                side, reference_values + fraction * (target - reference_values),
                check_contact=False,
            )
            if contact.intersects():
                high = fraction
            else:
                low = fraction
        self.set_scapular_girdle_values(
            side, reference_values + low * (target - reference_values),
            check_contact=False,
        )
        self.scapula_contact_report(side)
        return True

    def _scapulothoracic_setup(self, side: str) -> _Setup:
        """Return the cached setup of one side, building it if outdated."""
        if side not in ("r", "l"):
            raise ValueError("side must be 'r' or 'l'")
        cache = self.__dict__.setdefault("_scapulothoracic_cache", {})
        setup = cache.get(side)
        if (
            setup is not None
            and setup.reference is self._reference
            and setup.state is self.state
        ):
            return setup
        factors = segment_scale_factors(self._reference, self._baseline)
        mirror = np.ones(3) if side == "r" else _MIRROR
        # Glenohumeral parent frame in the scapula, read after the scaling.
        parent_frame = self.opensim.PhysicalOffsetFrame.safeDownCast(
            self.getJointSet().get(f"acromial_{side}").getParentFrame()
        )
        translation = parent_frame.get_translation()
        glenohumeral = np.array([translation.get(i) for i in range(3)])
        # Points in the scapula frame, with the scale factors of the scapula.
        scapula_scale = np.asarray(factors["scapula"])
        points = scapula_scale * mirror * _POINT_OFFSETS + glenohumeral
        torso_scale = np.asarray(factors["torso"])
        # Closed-form chain torso -> scapula: the scapula pose in the torso
        # depends on the girdle coordinates only.
        chain = KinematicChain(self, "torso", f"scapula_{side}")
        order = [
            chain.coordinate_names.index(f"{name}_{side}") for name in _GIRDLE_NAMES
        ]
        setup = _Setup(
            reference=self._reference,
            state=self.state,
            chain=chain,
            coordinate_order=order,
            station_array=points,
            torso_scale=torso_scale,
            length=float(np.mean(_ELLIPSOID_SEMI_AXES * torso_scale)),
            stations=[self.opensim.Vec3(*point) for point in points],
            scapula=self.getBodySet().get(f"scapula_{side}"),
            torso=self.getBodySet().get("torso"),
            contact_radii=np.zeros(_CONTACT_COUNT),
        )
        # Calibrate the contact radii on the zero girdle pose of this user.
        zero = self._girdle_points(setup, np.zeros(len(_GIRDLE_NAMES)))
        setup.contact_radii = _radii(zero, torso_scale, side)[:_CONTACT_COUNT]
        cache[side] = setup
        return setup

    def _torso_points(self, setup: _Setup, state: Any) -> np.ndarray:
        """Return the scapular points in the torso frame for a given state."""
        self.realizePosition(state)
        locations = [
            setup.scapula.findStationLocationInAnotherFrame(state, station, setup.torso)
            for station in setup.stations
        ]
        return np.array([[location.get(i) for i in range(3)] for location in locations])

    def _girdle_points(self, setup: _Setup, girdle_degrees: np.ndarray) -> np.ndarray:
        """Return the scapular points in the torso frame, one set per row.

        Parameters
        ----------
        setup : _Setup
            Setup of the side.
        girdle_degrees : numpy.ndarray
            Girdle coordinates in degrees, shape ``(5,)`` or ``(N, 5)``.

        Returns
        -------
        numpy.ndarray
            Points, shape ``(6, 3)`` for one vector or ``(N, 6, 3)``.
        """
        values = np.atleast_2d(np.radians(girdle_degrees))
        # chain columns follow the joint order, which may differ from the
        # order of the girdle names
        chain_values = np.empty_like(values)
        chain_values[:, setup.coordinate_order] = values
        rotations, positions = setup.chain.poses(chain_values)
        # scapula pose in the torso -> points of the scapula in the torso
        points = (
            np.einsum("nij,kj->nki", rotations[:, -1], setup.station_array)
            + positions[:, -1][:, None, :]
        )
        return points[0] if np.ndim(girdle_degrees) == 1 else points

    def scapulothoracic_residual_vector(
        self, side: str, girdle_degrees: Sequence[float]
    ) -> np.ndarray:
        """Return the contact residuals for girdle coordinate values.

        Fast variant for solvers: the state of the user is not read or
        modified and the per-user quantities are cached until the user is
        rescaled.

        Parameters
        ----------
        side : {"r", "l"}
            Side of the scapula.
        girdle_degrees : sequence of float
            Values in degrees of ``sc_protraction``, ``sc_elevation``,
            ``ac_protraction``, ``ac_upward_rotation`` and
            ``ac_axial_rotation`` (the order of
            ``SHOULDER_GIRDLE_COORDINATES``).

        Returns
        -------
        numpy.ndarray
            Residuals in metres, shape ``(6,)``, ordered ``AI``, ``TS``,
            ``AA``, ``MB1``, ``MB2``, ``SA``. See
            :meth:`scapulothoracic_residuals`.

        Raises
        ------
        ValueError
            If ``side`` is invalid or the number of values is not five.
        """
        setup = self._scapulothoracic_setup(side)
        values = np.asarray(girdle_degrees, dtype=float)
        if values.shape != (len(_GIRDLE_NAMES),):
            raise ValueError(f"expected {len(_GIRDLE_NAMES)} girdle coordinate values")
        points = self._girdle_points(setup, values)
        radii = _radii(points, setup.torso_scale, side)
        return _residual_array(radii, setup.contact_radii, setup.length)

    def scapulothoracic_residual_array(
        self, side: str, girdle_degrees: np.ndarray
    ) -> np.ndarray:
        """Return the contact residuals for many girdle poses at once.

        Same quantities as :meth:`scapulothoracic_residual_vector`, evaluated
        in closed form for a batch (for instance a finite-difference
        Jacobian in one call).

        Parameters
        ----------
        side : {"r", "l"}
            Side of the scapula.
        girdle_degrees : numpy.ndarray
            Girdle coordinate values in degrees, shape ``(N, 5)``, in the
            order of ``SHOULDER_GIRDLE_COORDINATES``.

        Returns
        -------
        numpy.ndarray
            Residuals in metres, shape ``(N, 6)``, ordered ``AI``, ``TS``,
            ``AA``, ``MB1``, ``MB2``, ``SA``.

        Raises
        ------
        ValueError
            If ``side`` is invalid or ``girdle_degrees`` is not ``(N, 5)``.
        """
        setup = self._scapulothoracic_setup(side)
        values = np.asarray(girdle_degrees, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(_GIRDLE_NAMES):
            raise ValueError(f"expected shape (N, {len(_GIRDLE_NAMES)})")
        radii = _radii(self._girdle_points(setup, values), setup.torso_scale, side)
        return _residual_array(radii, setup.contact_radii, setup.length)

    def scapulothoracic_residuals(
        self, side: str, state: Any | None = None
    ) -> dict[str, float]:
        """Return the scapulothoracic contact residuals of one scapula.

        Parameters
        ----------
        side : {"r", "l"}
            Side of the scapula.
        state : opensim.State or None, optional
            State to evaluate, realized to Position by this method. Defaults
            to the current state of the user.

        Returns
        -------
        dict of str to float
            Residual in metres for ``TS`` and ``AI`` (contact: zero when the
            point is on its contact radius) and for ``AA``, ``MB1``, ``MB2``
            and ``SA`` (non-penetration: exactly zero while the point is
            outside the clearance radius or sunk into it by less than 0.4 mm,
            positive beyond: quadratic over the next 0.1 mm, then equal to
            the depth minus 0.45 mm). All residuals are continuous with
            continuous derivatives, so they can be stacked in a
            least-squares problem; the unilateral ones have no floor.

        Raises
        ------
        ValueError
            If ``side`` is not ``"r"`` or ``"l"``.

        Notes
        -----
        The thorax is an ellipsoid fixed in the torso frame; see the module
        notes for the origin and status of the numbers. The contact radii are
        calibrated on the zero girdle pose of each user, so the contact
        residuals are zero there by construction.
        """
        setup = self._scapulothoracic_setup(side)
        points = self._torso_points(setup, self.state if state is None else state)
        radii = _radii(points, setup.torso_scale, side)
        residuals = _residual_array(radii, setup.contact_radii, setup.length)
        return dict(zip(_POINT_NAMES, residuals.tolist()))

    def acromion_angle_location(
        self, side: str, state: Any | None = None
    ) -> tuple[float, float, float]:
        """Return the scapular acromion-angle landmark in Ground.

        Parameters
        ----------
        side : {"r", "l"}
            Side of the scapula.
        state : opensim.State or None, optional
            State at which to evaluate the landmark. Defaults to the current
            state of the user.

        Returns
        -------
        tuple of float
            ``(x, y, z)`` location in metres in the Ground frame.

        Notes
        -----
        This is the scapular ``AA`` landmark, distinct from the
        glenohumeral joint centre represented by the ``acromial`` joint.

        Raises
        ------
        ValueError
            If ``side`` is not ``"r"`` or ``"l"``.
        """
        if side not in ("r", "l"):
            raise ValueError("side must be 'r' or 'l'")
        current_state = self.state if state is None else state
        self.realizePosition(current_state)
        setup = self._scapulothoracic_setup(side)
        station = setup.stations[_POINT_NAMES.index("AA")]
        location = setup.scapula.findStationLocationInGround(current_state, station)
        return tuple(float(value) for value in vec3_to_numpy(location))

    def scapulothoracic_residuals_for(
        self, side: str, coordinates: Mapping[str, float]
    ) -> dict[str, float]:
        """Return the contact residuals for given shoulder girdle coordinates.

        Parameters
        ----------
        side : {"r", "l"}
            Side of the scapula.
        coordinates : mapping of str to float
            Girdle coordinate values in degrees, by name without the side
            suffix (for example ``"sc_elevation"``). Coordinates not listed
            keep their value in the current state.

        Returns
        -------
        dict of str to float
            See :meth:`scapulothoracic_residuals`. The state of the user is
            not modified.

        Raises
        ------
        ValueError
            If ``side`` is invalid or a name is not a girdle coordinate.
        """
        unknown = set(coordinates) - set(_GIRDLE_NAMES)
        if unknown:
            raise ValueError(f"not shoulder girdle coordinates: {sorted(unknown)}")
        values = [
            coordinates.get(name, self._coordinate_degrees(f"{name}_{side}"))
            for name in _GIRDLE_NAMES
        ]
        residuals = self.scapulothoracic_residual_vector(side, values)
        return dict(zip(_POINT_NAMES, residuals.tolist()))

    def shoulder_girdle_reference_pose(self, contact: bool = True) -> dict[str, float]:
        """Return the girdle pose used to regularise the shoulder.

        Parameters
        ----------
        contact : bool, optional
            If True (default), return the pose closest to the reference pose
            (in degrees, within the coordinate ranges) at which the
            scapulothoracic contact residuals of this user vanish. If False,
            return the raw reference pose.

        Returns
        -------
        dict of str to float
            Coordinate names without side suffix mapped to degrees. The raw
            reference pose is a hypothesis of the biomechanics specialist
            (not measured data), lies within the coordinate ranges and does
            not satisfy the contact. The contact-consistent pose is computed
            by least squares (contact residuals weighted almost rigidly) and
            is the same for both sides because the contact is mirrored.
        """
        if not contact:
            return dict(SHOULDER_GIRDLE_REFERENCE_POSE)
        setup = self._scapulothoracic_setup("r")
        if setup.contact_pose is None:
            reference = np.array(
                [SHOULDER_GIRDLE_REFERENCE_POSE[name] for name in _GIRDLE_NAMES]
            )
            lower, upper = np.array(
                [SHOULDER_GIRDLE_COORDINATES[name] for name in _GIRDLE_NAMES]
            ).T

            def objective(values: np.ndarray) -> np.ndarray:
                # Distance to the reference (degrees) plus the residuals in
                # metres with a weight that makes them practically rigid.
                residuals = self.scapulothoracic_residual_vector("r", values)
                return np.concatenate([values - reference, 1e6 * residuals])

            solution = least_squares(objective, reference, bounds=(lower, upper))
            setup.contact_pose = dict(zip(_GIRDLE_NAMES, solution.x.tolist()))
        return dict(setup.contact_pose)

    def set_girdle_pose(self, contact: bool = True) -> None:
        """Set both shoulder girdles to the user's reference pose.

        Parameters
        ----------
        contact : bool, default=True
            Use the contact-consistent reference pose when true; otherwise
            apply the raw reference pose.
        """
        pose = self.shoulder_girdle_reference_pose(contact=contact)
        values = np.deg2rad([pose[name] for name in _GIRDLE_NAMES])
        for side in ("r", "l"):
            self.set_scapular_girdle_values(
                side, values, check_contact=False
            )
