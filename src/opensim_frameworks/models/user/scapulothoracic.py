"""Scapulothoracic contact of the shoulder girdle of :class:`User`.

The kinematic tree of the user has no scapulothoracic joint (see
:mod:`.shoulder`): the scapula is only prevented from leaving the thorax by
conditions on a few of its points. This module provides those conditions as
smooth residual functions suitable for least-squares solvers (the unilateral
ones are exactly zero while the condition holds); it neither adds
constraints to the model nor modifies its state.

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
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from ...utils.kinematics import KinematicChain
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


class _ScapulothoracicMixin:
    """Scapulothoracic contact residuals, exposed on :class:`User`."""

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
