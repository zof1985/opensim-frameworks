"""Position a user on the chest-press equipment."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import opensim as osim
from scipy.optimize import least_squares

from opensim_frameworks.utils.geometry import (
    body_mass_center_array,
    body_transform_array,
    get_mesh_vertices_in_ground,
    numpy_to_vec3,
    rotation_to_numpy,
    translate_model,
    vec3_to_numpy,
)
from opensim_frameworks import SHOULDER_GIRDLE_COORDINATES

__all__ = ["position_user", "set_girdle_pose"]

_PELVIS_AXES = {
    "pelvis_tx": (1.0, 0.0, 0.0),
    "pelvis_ty": (0.0, 1.0, 0.0),
    "pelvis_tz": (0.0, 0.0, 1.0),
}
_TRUNK_COORDINATES = (
    "pelvis_tilt",
    "lumbar_extension",
    "lumbar_bending",
    "lumbar_rotation",
)


def position_user(
    user,
    equipment,
    backrest_angle_deg: float,
) -> dict[str, Any]:
    """Position the user against the backrest, seat, and floor.

    The user and equipment must already belong to the same initialized
    ``opensim_frameworks.Framework``. The models are modified in place.

    The setup applies the initial shoulder-girdle pose, aligns the trunk and
    shoulder midpoint to the backrest, seats the pelvis, abducts the hips by
    30 degrees, restores the feet to their initial ground contact, and
    internally rotates the shoulders and pronates the forearms by 90 degrees.

    Parameters
    ----------
    user : opensim_frameworks.User
        User to position.
    equipment : opensim_frameworks.Equipment
        Equipment providing the backrest, seat, and ground references.
    backrest_angle_deg : float
        Backrest inclination relative to the Ground x axis, in degrees.

    Returns
    -------
    dict
        Positioning report with ``back``, ``seat``, ``hips``, and ``feet``
        entries.
    """
    if not hasattr(user, 'getCoordinateSet'):
        raise TypeError("user must be an opensim_frameworks.User")
    if not hasattr(equipment, 'getMarkerSet'):
        raise TypeError("equipment must be an opensim_frameworks.Equipment")
    if not math.isfinite(backrest_angle_deg) or not 0.0 < backrest_angle_deg < 180.0:
        raise ValueError("backrest_angle_deg must be finite and between 0 and 180")

    set_girdle_pose(user)
    coordinates = user.getCoordinateSet()

    def lock_trunk(locked: bool) -> None:
        """Set the lock state for all trunk coordinates.

        Parameters
        ----------
        locked : bool
            Whether the trunk coordinates should be locked.
        """
        for name in _TRUNK_COORDINATES:
            coordinate = coordinates.get(name)
            coordinate.set_locked(locked)
            coordinate.setLocked(user.state, locked)

    lock_trunk(False)
    feet_reference = get_feet_reference(user)

    back_report = align_user_to_backrest(user, equipment, backrest_angle_deg)
    lock_trunk(True)
    back_report["locked_coordinates"] = _TRUNK_COORDINATES
    user.set_right_shoulder_rotation(90.0)
    user.set_left_shoulder_rotation(90.0)
    user.set_right_forearm_pronation(90.0)
    user.set_left_forearm_pronation(90.0)
    
    # Unlock finger coordinates for fitting
    finger_prefixes = ("index_", "middle_", "ring_", "little_", "thumb_")
    for i in range(coordinates.getSize()):
        coord = coordinates.get(i)
        name = coord.getName()
        if any(name.startswith(prefix) for prefix in finger_prefixes):
            coord.set_locked(False)
            coord.setLocked(user.state, False)
    
    user.realizePosition(user.state)
    return {
        "back": back_report,
        "seat": seat_to_pelvis(user, equipment, backrest_angle_deg),
        "hips": abduct_hips(user),
        "feet": ground_feet(user, feet_reference),
    }


def set_girdle_pose(
    user,
) -> None:
    """Apply the user's shoulder-girdle reference pose to both sides.

    Parameters
    ----------
    user : opensim_frameworks.User
        User whose reference pose is applied.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If a reference angle is non-finite or outside its coordinate range.
    """
    pose = user.shoulder_girdle_reference_pose()
    coordinates = user.getCoordinateSet()
    for side in ("r", "l"):
        for name in SHOULDER_GIRDLE_COORDINATES:
            degrees = pose[name]
            value = math.radians(degrees)
            coordinate = coordinates.get(f"{name}_{side}")
            if not math.isfinite(value) or not (
                coordinate.getRangeMin() <= value <= coordinate.getRangeMax()
            ):
                raise ValueError(
                    f"{name}_{side} reference pose is outside its coordinate range"
                )
            coordinate.setValue(user.state, value, False)
    user.realizePosition(user.state)


def align_user_to_backrest(
    user,
    equipment,
    backrest_angle_deg: float,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Align the user's trunk and shoulders with the backrest.

    Parameters
    ----------
    user : opensim_frameworks.User
        User to align.
    equipment : opensim_frameworks.Equipment
        Equipment providing the backrest marker and mesh.
    backrest_angle_deg : float
        Backrest inclination relative to the Ground x axis, in degrees.
    tolerance : float, default=1e-6
        Maximum acceptable positional or angular residual.

    Returns
    -------
    dict
        Trunk angle, applied rotation and translation, and contact gaps.

    Raises
    ------
    ValueError
        If no torso or pelvis mesh vertex faces the back pad.
    RuntimeError
        If the trunk, shoulders, or back contact cannot be aligned.
    """
    angle = math.radians(backrest_angle_deg)
    normal = np.array([math.sin(angle), -math.cos(angle), 0.0])
    up = np.array([math.cos(angle), math.sin(angle), 0.0])
    lateral = np.array([0.0, 0.0, 1.0])

    user.realizePosition(user.state)
    marker = vec3_to_numpy(
        equipment.getMarkerSet().get("back_pad").getLocationInGround(equipment.state)
    )
    pad = get_mesh_vertices_in_ground(
        equipment,
        equipment.state,
        "back_pad",
        [Path(equipment.mesh_path)],
    )
    pad_relative = pad - marker
    pad_surface = float((pad_relative @ normal).max())
    pad_up = pad_relative @ up
    pad_lat = pad_relative @ lateral
    coordinates = user.getCoordinateSet()
    directories = list(user.geometry_directories)

    def trunk_angle() -> float:
        """Return the hip-to-shoulder angle in Ground.

        Returns
        -------
        float
            Trunk angle in radians.
        """
        user.realizePosition(user.state)
        centers = user.joint_centers
        hips = (np.asarray(centers["left_hip"]) + np.asarray(centers["right_hip"])) / 2
        shoulders = (
            np.asarray(centers["left_shoulder"]) + np.asarray(centers["right_shoulder"])
        ) / 2
        line = shoulders - hips
        return math.atan2(line[1], line[0])

    def shoulder_midpoint() -> np.ndarray:
        """Return the midpoint between shoulder joints in Ground.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Shoulder midpoint in metres.
        """
        user.realizePosition(user.state)
        centers = user.joint_centers
        return (
            np.asarray(centers["left_shoulder"]) + np.asarray(centers["right_shoulder"])
        ) / 2

    def translate(vector: np.ndarray) -> None:
        """Translate the user along the pelvis translation coordinates.

        Parameters
        ----------
        vector : numpy.ndarray, shape (3,)
            Translation vector in Ground, in metres.
        """
        translate_model(user, user.state, _PELVIS_AXES, vector)

    def contact_gaps() -> tuple[float, float]:
        """Return the torso and pelvis gaps from the back pad.

        Returns
        -------
        tuple of float
            Torso and pelvis gaps in metres.
        """
        gaps = []
        for body_name in ("torso", "pelvis"):
            vertices = (
                get_mesh_vertices_in_ground(
                    user,
                    user.state,
                    body_name,
                    directories,
                )
                - marker
            )
            facing = (
                (vertices @ up >= pad_up.min())
                & (vertices @ up <= pad_up.max())
                & (vertices @ lateral >= pad_lat.min())
                & (vertices @ lateral <= pad_lat.max())
            )
            if not np.any(facing):
                raise ValueError(f"No {body_name} vertex faces the back pad")
            gaps.append(float((vertices[facing] @ normal).min() - pad_surface))
        return gaps[0], gaps[1]

    start = trunk_angle()
    rotation = angle - start
    tilt = coordinates.get("pelvis_tilt")
    tilt_start = tilt.getValue(user.state)
    tilt.setValue(user.state, tilt_start + rotation, False)
    if abs(trunk_angle() - angle) > tolerance:
        tilt.setValue(user.state, tilt_start - rotation, False)
        if abs(trunk_angle() - angle) > tolerance:
            raise RuntimeError("Could not align the trunk to the backrest angle")

    offset = shoulder_midpoint() - marker
    in_plane = offset - (offset @ normal) * normal
    translation = -in_plane
    translate(translation)

    torso_gap, pelvis_gap = contact_gaps()
    shift = -min(torso_gap, pelvis_gap) * normal
    translate(shift)
    translation = translation + shift

    offset = shoulder_midpoint() - marker
    in_plane = offset - (offset @ normal) * normal
    torso_gap, pelvis_gap = contact_gaps()
    if abs(trunk_angle() - angle) > tolerance:
        raise RuntimeError("Trunk angle changed while translating the user")
    if np.linalg.norm(in_plane) > tolerance:
        raise RuntimeError("Shoulder midpoint is not normal to the marker")
    if (
        min(torso_gap, pelvis_gap) < -tolerance
        or min(torso_gap, pelvis_gap) > tolerance
    ):
        raise RuntimeError("Back contact with the pad was not achieved")

    return {
        "trunk_angle_deg": math.degrees(trunk_angle()),
        "trunk_rotation_deg": math.degrees(rotation),
        "translation_m": translation,
        "shoulder_marker_offset_m": in_plane,
        "shoulder_normal_distance_m": float(offset @ normal),
        "torso_gap_m": torso_gap,
        "pelvis_gap_m": pelvis_gap,
    }


def get_feet_reference(user) -> dict[str, dict[str, float]]:
    """Capture each foot's initial height, pitch, and roll.

    Parameters
    ----------
    user : opensim_frameworks.User
        User whose initial foot pose is recorded.

    Returns
    -------
    dict
        Per-side foot height, pitch, and roll references.
    """
    user.realizePosition(user.state)
    directories = list(user.geometry_directories)
    reference = {}
    for side in ("r", "l"):
        rotation = rotation_to_numpy(
            user.getBodySet().get(f"calcn_{side}").getTransformInGround(user.state).R()
        )
        vertices = np.vstack(
            [
                get_mesh_vertices_in_ground(
                    user,
                    user.state,
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


def seat_to_pelvis(
    user,
    equipment,
    backrest_angle_deg: float,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Move the seat and pelvis until the pelvis contacts the seat pad.

    Parameters
    ----------
    user : opensim_frameworks.User
        User whose pelvis is positioned.
    equipment : opensim_frameworks.Equipment
        Equipment providing the seat pad and seat-position coordinate.
    backrest_angle_deg : float
        Backrest inclination relative to the Ground x axis, in degrees.
    tolerance : float, default=1e-6
        Maximum acceptable pelvis-to-seat gap, in metres.

    Returns
    -------
    dict
        Seat coordinate, user translation, and final contact gap.

    Raises
    ------
    ValueError
        If no pelvis mesh vertex faces the seat pad.
    RuntimeError
        If contact with the seat pad cannot be achieved.
    """
    angle = math.radians(backrest_angle_deg)
    slide = np.array([math.cos(angle), math.sin(angle), 0.0])
    directories = list(user.geometry_directories)
    equipment.realizePosition(equipment.state)
    pad = get_mesh_vertices_in_ground(
        equipment,
        equipment.state,
        "seat_pad",
        [Path(equipment.mesh_path)],
    )
    _, _, axes = np.linalg.svd(pad - pad.mean(axis=0), full_matrices=False)
    normal = axes[2] if axes[2][1] > 0 else -axes[2]
    lateral = np.array([0.0, 0.0, 1.0])
    along = np.cross(lateral, normal)

    def gap() -> float:
        """Return the minimum pelvis-to-seat surface gap.

        Returns
        -------
        float
            Minimum pelvis-to-seat gap in metres.
        """
        pad_vertices = get_mesh_vertices_in_ground(
            equipment,
            equipment.state,
            "seat_pad",
            [Path(equipment.mesh_path)],
        )
        pelvis = get_mesh_vertices_in_ground(
            user,
            user.state,
            "pelvis",
            directories,
        )
        facing = (
            (pelvis @ along >= (pad_vertices @ along).min())
            & (pelvis @ along <= (pad_vertices @ along).max())
            & (pelvis @ lateral >= (pad_vertices @ lateral).min())
            & (pelvis @ lateral <= (pad_vertices @ lateral).max())
        )
        if not np.any(facing):
            raise ValueError("No pelvis vertex faces the seat pad")
        return float((pelvis[facing] @ normal).min() - (pad_vertices @ normal).max())

    seat = equipment.getCoordinateSet().get("seat_position")
    slope = float(slide @ normal)
    user_translation = np.zeros(3)
    for _ in range(30):
        current = gap()
        if abs(current) <= tolerance:
            break
        position = seat.getValue(equipment.state)
        target = min(
            max(position + current * slope, seat.getRangeMin()),
            seat.getRangeMax(),
        )
        if abs(target - position) > 1e-12:
            translate_model(
                equipment,
                equipment.state,
                {"seat_position": tuple(slide)},
                (target - position) * slide,
            )
        else:
            step = -current / slope * slide
            translate_model(user, user.state, _PELVIS_AXES, step)
            user_translation += step
    else:
        raise RuntimeError("Seat pad contact with the pelvis was not achieved")

    return {
        "seat_position_m": seat.getValue(equipment.state),
        "user_translation_m": user_translation,
        "pelvis_gap_m": gap(),
    }


def abduct_hips(user, degrees: float = 30.0) -> dict[str, Any]:
    """Abduct both hips and verify that the knees move farther apart.

    Parameters
    ----------
    user : opensim_frameworks.User
        User whose hip coordinates are changed.
    degrees : float, default=30.0
        Hip-abduction magnitude in degrees.

    Returns
    -------
    dict
        Applied hip-abduction angles and final knee separation.

    Raises
    ------
    RuntimeError
        If a coordinate is out of range or the knees do not separate.
    """
    def knee_separation() -> float:
        """Return the lateral distance between the knees.

        Returns
        -------
        float
            Knee separation in metres.
        """
        user.realizePosition(user.state)
        centers = user.joint_centers
        return abs(centers["left_knee"][2] - centers["right_knee"][2])

    before = knee_separation()
    coordinates = user.getCoordinateSet()
    for side in ("r", "l"):
        coordinate = coordinates.get(f"hip_adduction_{side}")
        value = -math.radians(degrees)
        if not coordinate.getRangeMin() <= value <= coordinate.getRangeMax():
            raise RuntimeError(f"hip_adduction_{side}={value:.6g} outside its range")
        coordinate.setValue(user.state, value, False)
    after = knee_separation()
    if after <= before:
        raise RuntimeError("Hip abduction did not move the knees apart")
    return {
        "hip_abduction_deg": {
            side: -math.degrees(
                coordinates.get(f"hip_adduction_{side}").getValue(user.state)
            )
            for side in ("r", "l")
        },
        "knee_separation_m": float(after),
    }


def synchronize_coupled_coordinates(user) -> None:
    """Recompute dependent coordinates for all coordinate couplers.

    Parameters
    ----------
    user : opensim_frameworks.User
        User model whose dependent coordinates are synchronized.
    """
    coordinates = user.getCoordinateSet()
    constraints = user.getConstraintSet()
    for index in range(constraints.getSize()):
        coupler = osim.CoordinateCouplerConstraint.safeDownCast(constraints.get(index))
        if coupler is None:
            continue
        names = coupler.getIndependentCoordinateNames()
        values = osim.Vector(names.getSize(), 0.0)
        for index in range(names.getSize()):
            values.set(index, coordinates.get(names.get(index)).getValue(user.state))
        coordinates.get(coupler.getDependentCoordinateName()).setValue(
            user.state, coupler.getFunction().calcValue(values), False
        )
    user.realizePosition(user.state)


def ground_feet(
    user,
    reference: dict[str, dict[str, float]],
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Restore both feet to their initial ground height and orientation.

    Parameters
    ----------
    user : opensim_frameworks.User
        User whose feet are positioned.
    reference : dict
        Per-foot height, pitch, and roll values returned by
        :func:`get_feet_reference`.
    tolerance : float, default=1e-6
        Maximum acceptable height, pitch, and roll residual.

    Returns
    -------
    dict
        Final leg coordinates and foot height and orientation errors.

    Raises
    ------
    RuntimeError
        If either foot cannot be placed within the specified tolerance.
    """
    coordinates = user.getCoordinateSet()
    directories = list(user.geometry_directories)
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

        user.realizePosition(user.state)
        local = {}
        for kind in ("calcn", "toes"):
            name = f"{kind}_{side}"
            body = user.getBodySet().get(name)
            transform = body.getTransformInGround(user.state)
            rotation = rotation_to_numpy(transform.R())
            position = vec3_to_numpy(transform.p())
            ground = get_mesh_vertices_in_ground(
                user,
                user.state,
                name,
                directories,
            )
            local[name] = (body, (ground - position) @ rotation)

        def pose(x: np.ndarray) -> tuple[float, float, float]:
            """Set candidate leg angles and measure foot height and tilt.

            Parameters
            ----------
            x : numpy.ndarray, shape (3,)
                Hip flexion, knee flexion, and hip rotation in radians.

            Returns
            -------
            tuple of float
                Lowest foot height, pitch component, and roll component.
            """
            for coordinate, value in zip(targets, x):
                coordinate.setValue(user.state, float(value), False)
            user.realizePosition(user.state)
            lowest = np.inf
            pitch = roll = 0.0
            for name, (body, vertices) in local.items():
                transform = body.getTransformInGround(user.state)
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
            """Return foot-height, pitch, and roll errors.

            Parameters
            ----------
            x : numpy.ndarray, shape (3,)
                Candidate leg coordinates in radians.

            Returns
            -------
            numpy.ndarray, shape (3,)
                Residuals relative to the captured foot reference.
            """
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
        calcn = user.getBodySet().get(f"calcn_{side}")
        sole_up = rotation_to_numpy(calcn.getTransformInGround(user.state).R())[
            1, 1
        ]
        report[side] = {
            "hip_flexion_deg": math.degrees(best.x[0]),
            "knee_angle_deg": math.degrees(best.x[1]),
            "hip_rotation_deg": math.degrees(best.x[2]),
            "height_error_m": lowest - reference[side]["height"],
            "pitch_error": pitch - reference[side]["pitch"],
            "roll_error": roll - reference[side]["roll"],
            "sole_tilt_deg": math.degrees(math.acos(min(1.0, float(sole_up)))),
        }
    synchronize_coupled_coordinates(user)
    return report
