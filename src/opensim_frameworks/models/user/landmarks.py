"""Aggregated joint-centre and marker landmarks, in the ground frame.

complements :mod:`.joint_centers_generated` (per-joint named properties such
as ``pelvis``/``left_hip``, each a thin wrapper around :meth:`joint_center`):
this module defines ``joint_center`` itself, the aggregated
``joint_centers`` dict, marker-based landmarks (eye/palm/foot markers,
individual heel/toe/metatarsal properties), the whole-body centre of mass
and its ground projection, and :meth:`User.set_position`.
"""

from __future__ import annotations

import math
import itertools

import numpy as np
import opensim as osim
from scipy.optimize import least_squares

from ...utils.geometry import (
    get_mesh_vertices_in_ground,
    rotation_to_numpy,
    vec3_to_numpy,
)

from .joint_center_table import JOINT_CENTER_NAMES as _JOINT_CENTER_NAMES

__all__ = []

_PELVIS_TRANSLATION_COORDINATES = ("pelvis_tx", "pelvis_ty", "pelvis_tz")

# Friendly name -> OpenSim marker name. The foot has no 1st-metatarsal
# marker in this model (only the 5th): no landmark is exposed for it rather
# than guessing one.
_FOOT_MARKER_NAMES = {
    "left_heel": "LCAL",
    "right_heel": "RCAL",
    "left_toe": "LTOE",
    "right_toe": "RTOE",
    "left_mt5": "LMT5",
    "right_mt5": "RMT5",
}


class _LandmarkMixin:
    """Aggregated joint-centre and marker landmark properties."""

    def joint_center(self, joint_name: str) -> tuple[float, float, float]:
        """Return a named joint's centre in Ground coordinates, in metres.

        Parameters
        ----------
        joint_name : str
            Name of the OpenSim joint whose child-frame position is returned.

        Returns
        -------
        tuple[float, float, float]
            The joint centre as an ``(x, y, z)`` position in the ground frame.
        """
        self.realizePosition(self.state)
        frame = self.getJointSet().get(joint_name).getChildFrame()
        position = frame.getPositionInGround(self.state)
        return (position.get(0), position.get(1), position.get(2))

    def _joint_center(self, joint_name: str) -> tuple[float, float, float]:
        """Backward-compatible alias for :meth:`joint_center`."""
        return self.joint_center(joint_name)

    @property
    def joint_centers(self) -> dict[str, tuple[float, float, float]]:
        """Return every named joint centre at once, in the ground frame.

        Equivalent to calling every individual joint-centre property below
        (``pelvis``, ``left_hip``, ``right_knee``, ...) and collecting the
        results into one dictionary, keyed by the same friendly names.
        Reflects the model's current posture: each position is computed via
        ``getPositionInGround`` with an automatic ``realizePosition``, so it
        is correct immediately after a posture setter, with no need to call
        :meth:`update_state` first.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Mapping from friendly joint name (``"pelvis"``, ``"left_hip"``,
            ``"right_ankle"``, ...) to that joint's centre as an ``(x, y,
            z)`` position in the ground frame, in metres.
        """
        return {
            name: self.joint_center(joint)
            for name, joint in _JOINT_CENTER_NAMES.items()
        }

    def _marker_location(self, marker_name: str) -> tuple[float, float, float]:
        self.realizePosition(self.state)
        position = self.getMarkerSet().get(marker_name).getLocationInGround(self.state)
        return (position.get(0), position.get(1), position.get(2))

    @property
    def foot_markers(self) -> dict[str, tuple[float, float, float]]:
        """Return every foot landmark marker at once, in the ground frame.

        Equivalent to calling each of ``left_heel``/``right_heel``,
        ``left_toe``/``right_toe``, and ``left_mt5``/``right_mt5`` and
        collecting the results into one dictionary. Reflects the model's
        current posture (``realizePosition`` is called automatically; no
        need to call :meth:`update_state` first). There is no 1st-metatarsal
        marker in this model (only the 5th, see :attr:`left_mt5`), so none
        is exposed here either.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Mapping from friendly marker name (``"left_heel"``,
            ``"right_toe"``, ``"left_mt5"``, ...) to that marker's location
            as an ``(x, y, z)`` position in the ground frame, in metres.
        """
        return {
            name: self._marker_location(marker)
            for name, marker in _FOOT_MARKER_NAMES.items()
        }

    @property
    def eye_markers(self) -> dict[str, tuple[float, float, float]]:
        """Return the right and left orbital marker positions.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Mapping from ``"right"`` and ``"left"`` to the corresponding eye
            marker's ``(x, y, z)`` position in metres in the ground frame.
        """
        return {
            side: self._marker_location(marker)
            for side, marker in (("right", "REYE"), ("left", "LEYE"))
        }

    @property
    def palm_markers(self) -> dict[str, tuple[float, float, float]]:
        """Return the articulated right and left palm-centre marker positions.

        Each marker is attached to its corresponding hand body at the centre
        of the resolved palm, halfway from the wrist along the hand's local
        negative y axis. The hand bodies are connected to the radii by
        universal wrist joints with flexion/extension and radial/ulnar
        deviation coordinates. Marker positions therefore follow the current
        wrist posture and are expressed in the ground frame.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Mapping from ``"right"`` and ``"left"`` to the corresponding
            palm-centre ``(x, y, z)`` positions in metres in the ground frame.
        """
        return {
            side: self._marker_location(marker)
            for side, marker in (("right", "RPALM"), ("left", "LPALM"))
        }

    @property
    def left_heel(self) -> tuple[float, float, float]:
        """Return the left heel marker location, in the ground frame.

        Reads the OpenSim marker ``LCAL``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("LCAL")

    @property
    def right_heel(self) -> tuple[float, float, float]:
        """Return the right heel marker location, in the ground frame.

        Reads the OpenSim marker ``RCAL``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("RCAL")

    @property
    def left_toe(self) -> tuple[float, float, float]:
        """Return the left toe marker location, in the ground frame.

        Reads the OpenSim marker ``LTOE``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("LTOE")

    @property
    def right_toe(self) -> tuple[float, float, float]:
        """Return the right toe marker location, in the ground frame.

        Reads the OpenSim marker ``RTOE``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("RTOE")

    @property
    def left_mt5(self) -> tuple[float, float, float]:
        """Return the left 5th-metatarsal marker location, in the ground frame.

        Reads the OpenSim marker ``LMT5``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first). There is no 1st-metatarsal marker in
        this model, so this is the only metatarsal landmark available.

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("LMT5")

    @property
    def right_mt5(self) -> tuple[float, float, float]:
        """Return the right 5th-metatarsal marker location, in the ground frame.

        Reads the OpenSim marker ``RMT5``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first). There is no 1st-metatarsal marker in
        this model, so this is the only metatarsal landmark available.

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("RMT5")

    @property
    def com(self) -> tuple[float, float, float]:
        """Return the whole-body centre of mass, in the ground frame.

        Computed natively by OpenSim (``Model.calcMassCenterPosition``),
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need to call :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        self.realizePosition(self.state)
        position = self.calcMassCenterPosition(self.state)
        return (position.get(0), position.get(1), position.get(2))

    @property
    def cop(self) -> tuple[float, float, float]:
        """Return the ground projection of the centre of mass.

        This is a purely kinematic projection of :attr:`com` straight down
        onto the ground plane (``y = 0``): the ``x``/``z`` coordinates are
        copied unchanged from :attr:`com` and ``y`` is forced to ``0.0``.
        Despite the name, this is **not** a dynamically computed centre of
        pressure derived from contact/ground-reaction forces.

        Returns
        -------
        tuple[float, float, float]
            ``(x, 0.0, z)`` position in metres, in the ground frame, where
            ``x``/``z`` match :attr:`com`.
        """
        x, _, z = self.com
        return (x, 0.0, z)

    def set_position(
        self, reference: tuple[float, float, float], x: float, y: float, z: float
    ) -> None:
        """Translate the whole model so that ``reference`` ends up at ``(x, y, z)``.

        ``reference`` is any ground-frame point belonging to this model --
        e.g. :attr:`com`, :attr:`cop`, a :attr:`joint_centers` entry, or one
        of the named joint-centre properties (``left_ankle``, ``pelvis``,
        ...). The whole model is translated rigidly through its root
        (``pelvis_tx``/``pelvis_ty``/``pelvis_tz``): relative posture and
        joint angles are unaffected. Calls :meth:`update_state` internally,
        so derived quantities are immediately up to date.

        Parameters
        ----------
        reference : tuple[float, float, float]
            Current ground-frame position, in metres, of the point to move
            (read it right before calling, since it depends on the current
            posture).
        x, y, z : float
            Target ground-frame coordinates for that point, in metres.

        Raises
        ------
        ValueError
            If any coordinate is not finite.
        """
        if not all(math.isfinite(value) for value in (x, y, z, *reference)):
            raise ValueError("position must be finite")
        targets = (x, y, z)
        for name, target, current in zip(
            _PELVIS_TRANSLATION_COORDINATES, targets, reference
        ):
            coordinate = self.getCoordinateSet().get(name)
            coordinate.setValue(
                self.state,
                coordinate.getValue(self.state) + (target - current),
                False,
            )
        self.update_state()

    def feet_reference(self) -> dict[str, dict[str, float]]:
        """Capture each foot's initial ground height, pitch, and roll.

        Returns
        -------
        dict
            Per-side ground height and calcaneus orientation references.
        """
        self.realizePosition(self.state)
        directories = list(self.geometry_directories)
        reference = {}
        for side in ("r", "l"):
            rotation = rotation_to_numpy(
                self.getBodySet()
                .get(f"calcn_{side}")
                .getTransformInGround(self.state)
                .R()
            )
            vertices = np.vstack(
                [
                    get_mesh_vertices_in_ground(
                        self,
                        self.state,
                        f"{kind}_{side}",
                        directories,
                    )
                    for kind in ("calcn", "toes")
                ]
            )
            reference[side] = {
                "pitch": float(rotation[1, 0]),
                "roll": float(rotation[1, 2]),
                "height": float(vertices[:, 1].min()),
            }
        return reference

    def ground_feet(
        self,
        reference: dict[str, dict[str, float]],
        tolerance: float = 1e-6,
    ) -> dict[str, dict[str, float]]:
        """Restore both feet to their captured ground height and orientation.

        Parameters
        ----------
        reference : dict
            Per-foot height, pitch, and roll values from :meth:`feet_reference`.
        tolerance : float, default=1e-6
            Maximum acceptable residual for height, pitch, and roll.

        Returns
        -------
        dict
            Final leg coordinates and foot height and orientation errors.

        Raises
        ------
        RuntimeError
            If either foot cannot be placed within the specified tolerance.
        """
        coordinates = self.getCoordinateSet()
        directories = list(self.geometry_directories)
        report = {}
        for side in ("r", "l"):
            names = (
                f"hip_flexion_{side}",
                f"knee_angle_{side}",
                f"hip_rotation_{side}",
            )
            targets = [coordinates.get(name) for name in names]
            lower = np.array([coordinate.getRangeMin() for coordinate in targets])
            upper = np.array([coordinate.getRangeMax() for coordinate in targets])

            self.realizePosition(self.state)
            local = {}
            for kind in ("calcn", "toes"):
                name = f"{kind}_{side}"
                body = self.getBodySet().get(name)
                transform = body.getTransformInGround(self.state)
                rotation = rotation_to_numpy(transform.R())
                position = vec3_to_numpy(transform.p())
                ground = get_mesh_vertices_in_ground(
                    self, self.state, name, directories
                )
                local[name] = (body, (ground - position) @ rotation)

            def pose(x: np.ndarray) -> tuple[float, float, float]:
                for coordinate, value in zip(targets, x, strict=True):
                    coordinate.setValue(self.state, float(value), False)
                self.realizePosition(self.state)
                lowest = np.inf
                pitch = roll = 0.0
                for name, (body, vertices) in local.items():
                    transform = body.getTransformInGround(self.state)
                    rotation = rotation_to_numpy(transform.R())
                    position = vec3_to_numpy(transform.p())
                    lowest = min(
                        lowest,
                        float((vertices @ rotation.T + position)[:, 1].min()),
                    )
                    if name.startswith("calcn"):
                        pitch = float(rotation[1, 0])
                        roll = float(rotation[1, 2])
                return lowest, pitch, roll

            def residuals(x: np.ndarray) -> np.ndarray:
                lowest, pitch, roll = pose(x)
                return np.array(
                    [
                        lowest - reference[side]["height"],
                        pitch - reference[side]["pitch"],
                        roll - reference[side]["roll"],
                    ]
                )

            best = None
            for hip, knee, rotation in itertools.product(
                (0.3, 0.8, 1.3),
                (0.3, 0.8, 1.3),
                (-0.4, 0.0, 0.4),
            ):
                result = least_squares(
                    residuals,
                    np.clip([hip, knee, rotation], lower, upper),
                    bounds=(lower, upper),
                    xtol=1e-12,
                    ftol=1e-12,
                    gtol=1e-12,
                )
                if best is None or result.cost < best.cost:
                    best = result
            if best is None or np.abs(best.fun).max() > tolerance:
                residual = None if best is None else best.fun
                raise RuntimeError(
                    f"Could not place the {side} foot on the ground: "
                    f"residuals {residual}"
                )
            lowest, pitch, roll = pose(best.x)
            calcn = self.getBodySet().get(f"calcn_{side}")
            sole_up = rotation_to_numpy(
                calcn.getTransformInGround(self.state).R()
            )[1, 1]
            report[side] = {
                "hip_flexion_deg": math.degrees(best.x[0]),
                "knee_angle_deg": math.degrees(best.x[1]),
                "hip_rotation_deg": math.degrees(best.x[2]),
                "height_error_m": lowest - reference[side]["height"],
                "pitch_error": pitch - reference[side]["pitch"],
                "roll_error": roll - reference[side]["roll"],
                "sole_tilt_deg": math.degrees(
                    math.acos(min(1.0, float(sole_up)))
                ),
            }
        self._synchronize_coupled_coordinates()
        return report

    def _synchronize_coupled_coordinates(self) -> None:
        coordinates = self.getCoordinateSet()
        constraints = self.getConstraintSet()
        for index in range(constraints.getSize()):
            coupler = osim.CoordinateCouplerConstraint.safeDownCast(
                constraints.get(index)
            )
            if coupler is None:
                continue
            names = coupler.getIndependentCoordinateNames()
            values = osim.Vector(names.getSize(), 0.0)
            for index in range(names.getSize()):
                values.set(
                    index,
                    coordinates.get(names.get(index)).getValue(self.state),
                )
            coordinates.get(coupler.getDependentCoordinateName()).setValue(
                self.state, coupler.getFunction().calcValue(values), False
            )
        self.realizePosition(self.state)
