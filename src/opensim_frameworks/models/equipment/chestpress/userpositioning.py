"""Position a user on the chest-press equipment."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from opensim_frameworks.utils.geometry import (
    get_mesh_vertices_in_ground,
    translate_model,
    vec3_to_numpy,
)

__all__ = ["position_user"]

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
    scapular acromion-angle midpoint to the backrest flags, seats the pelvis,
    abducts the hips by 30 degrees, restores the feet to their initial ground
    contact, and internally rotates the shoulders and pronates the forearms
    by 90 degrees.

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
    if not hasattr(user, "getCoordinateSet"):
        raise TypeError("user must be an opensim_frameworks.User")
    if not hasattr(equipment, "getMarkerSet"):
        raise TypeError("equipment must be an opensim_frameworks.Equipment")
    if not math.isfinite(backrest_angle_deg) or not 0.0 < backrest_angle_deg < 180.0:
        raise ValueError("backrest_angle_deg must be finite and between 0 and 180")

    user.set_girdle_pose()
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
    feet_reference = user.feet_reference()

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
    seat_report = seat_to_pelvis(user, equipment, backrest_angle_deg)
    return {
        "back": back_report,
        "seat": seat_report,
        "hips": abduct_hips(user),
        "feet": user.ground_feet(feet_reference),
    }


def align_user_to_backrest(
    user,
    equipment,
    backrest_angle_deg: float,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Align the user's trunk and acromion-angle landmarks with the backrest.

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
        If the trunk, acromion-angle landmarks, or back contact cannot be aligned.
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

    def acromion_midpoint() -> np.ndarray:
        """Return the midpoint between the scapular AA landmarks in Ground.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Midpoint of the left and right acromion-angle landmarks in metres.
        """
        return np.mean(
            [
                np.asarray(user.acromion_angle_location(side))
                for side in ("l", "r")
            ],
            axis=0,
        )

    def translate(vector: np.ndarray) -> None:
        """Translate the user in Ground.

        Parameters
        ----------
        vector : numpy.ndarray, shape (3,)
            Translation vector in Ground, in metres.
        """
        translate_model(user, user.state, vector, inplace=True)

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

    offset = acromion_midpoint() - marker
    in_plane = offset - (offset @ normal) * normal
    translation = -in_plane
    translate(translation)

    torso_gap, pelvis_gap = contact_gaps()
    shift = -min(torso_gap, pelvis_gap) * normal
    translate(shift)
    translation = translation + shift

    offset = acromion_midpoint() - marker
    in_plane = offset - (offset @ normal) * normal
    torso_gap, pelvis_gap = contact_gaps()
    if abs(trunk_angle() - angle) > tolerance:
        raise RuntimeError("Trunk angle changed while translating the user")
    if np.linalg.norm(in_plane) > tolerance:
        raise RuntimeError("Acromion midpoint is not normal to the backrest marker")
    if (
        min(torso_gap, pelvis_gap) < -tolerance
        or min(torso_gap, pelvis_gap) > tolerance
    ):
        raise RuntimeError("Back contact with the pad was not achieved")

    return {
        "trunk_angle_deg": math.degrees(trunk_angle()),
        "trunk_rotation_deg": math.degrees(rotation),
        "translation_m": translation,
        "acromion_marker_offset_m": in_plane,
        "acromion_normal_distance_m": float(offset @ normal),
        "torso_gap_m": torso_gap,
        "pelvis_gap_m": pelvis_gap,
    }


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
            seat.setValue(equipment.state, target, False)
            equipment.realizePosition(equipment.state)
        else:
            step = -current / slope * slide
            translate_model(user, user.state, step, inplace=True)
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
