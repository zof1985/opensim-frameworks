"""Shoulder girdle (clavicle and scapula) added to the bundled user model.

The bundled model joins the humerus directly to the torso with the three
rotations of the glenohumeral joint (``acromial_*``), so the shoulder centre
cannot move on the thorax. :func:`add_shoulder_girdle` inserts, on each side,
a clavicle and a scapula between the torso and the humerus::

    torso -> sternoclavicular -> clavicle -> acromioclavicular -> scapula
          -> acromial (glenohumeral, unchanged) -> humerus

All the new coordinates are zero in the pose of the bundled model, so a user
whose girdle coordinates are zero is posed exactly as before (the glenohumeral
centre does not move). The coordinates are *relative to that rest pose*, not
absolute anatomical angles.

The kinematic tree has no loop: the contact of the scapula with the thorax is
not a joint of the model (see :func:`scapula_thorax_points`).

Notes
-----
The geometry, ranges and masses below are modelling estimates, not measured
data: the positions are derived from the markers and joint frames of the
bundled model; the ranges are order-of-magnitude values for pressing tasks
(sternoclavicular elevation and protraction roughly after Kapandji and
Ludewig et al. 2009); the masses are 0.9 % (scapula) and 0.2 % (clavicle) of
the body mass of the unscaled model. They must be validated before being used
as anatomical results.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = [
    "CLAVICLE_TO_BIACROMIAL_RATIO",
    "SHOULDER_GIRDLE_COORDINATES",
    "SHOULDER_GIRDLE_REFERENCE_POSE",
    "add_shoulder_girdle",
]

# Straight sternoclavicular-acromioclavicular distance over the biacromial
# breadth. Estimate from the literature recalled by the biomechanics
# specialist, not verified against a primary source.
CLAVICLE_TO_BIACROMIAL_RATIO = 0.37

# name -> (minimum, maximum), degrees, relative to the rest pose
SHOULDER_GIRDLE_COORDINATES: dict[str, tuple[float, float]] = {
    "sc_protraction": (-30.0, 25.0),
    "sc_elevation": (-10.0, 35.0),
    "ac_protraction": (-25.0, 25.0),
    "ac_upward_rotation": (-15.0, 40.0),
    "ac_axial_rotation": (-35.0, 30.0),
}

# Pose towards which a solver may weakly regularise the girdle, degrees.
# Hypothesis of the biomechanics specialist (not measured data); all values
# lie within SHOULDER_GIRDLE_COORDINATES.
SHOULDER_GIRDLE_REFERENCE_POSE: dict[str, float] = {
    "sc_protraction": 0.0,
    "sc_elevation": 8.0,
    "ac_protraction": 0.0,
    "ac_upward_rotation": 20.0,
    "ac_axial_rotation": 10.0,
}

# Right side, torso frame (x forward, y up, z to the right), metres.
# Sternoclavicular centre: from the sternal marker, towards the right.
_STERNOCLAVICULAR_FROM_MARKER = np.array([-0.005, -0.008, 0.003])  # added to CLAV
# Acromioclavicular centre relative to the glenohumeral centre along x
# (behind it) and y (above it); the lateral (z) offset is not fixed: it is
# solved so that the clavicle has the requested length.
_ACROMIOCLAVICULAR_FROM_GLENOHUMERAL_XY = np.array([-0.012, 0.030])

# Segment masses of the unscaled model (kg) and rough dimensions (m).
_CLAVICLE_MASS = 0.15
_SCAPULA_MASS = 0.68
_SCAPULA_LENGTH = 0.15
_SCAPULA_WIDTH = 0.10
_SCAPULA_CENTER_OFFSET = np.array([-0.03, -0.04, -0.03])

_MIRROR = np.array([1.0, 1.0, -1.0])


def _mirrored_axis(axis: np.ndarray) -> np.ndarray:
    """Rotation axis of the left side that mirrors a right-side axis.

    Mirroring across the sagittal plane flips the sign of a rotation about
    an axis (a pseudovector), so the left axis is ``-M @ axis``.
    """
    return -_MIRROR * axis


def _rotation_joint(
    opensim: Any,
    name: str,
    parent: Any,
    parent_location: np.ndarray,
    child: Any,
    axes: list[tuple[str, np.ndarray]],
    ranges_deg: dict[str, tuple[float, float]],
) -> Any:
    """Build a joint with the rotations ``axes`` (body-fixed, in order)."""
    transform = opensim.SpatialTransform()
    directions = [np.asarray(direction, dtype=float) for _, direction in axes]
    if len(directions) == 2:
        # the unused third rotation axis must not be collinear with the others
        third = np.cross(*directions)
        directions.append(third / np.linalg.norm(third))
    for index in range(6):
        axis = transform.updTransformAxis(index)
        # property setters copy the function (setFunction would adopt it)
        axis.set_function(
            opensim.LinearFunction() if index < len(axes) else opensim.Constant(0.0)
        )
        if index < len(axes):
            axis.setCoordinateNames(opensim.ArrayStr(axes[index][0], 1))
        if index < len(directions):
            axis.setAxis(opensim.Vec3(*directions[index]))
    joint = opensim.CustomJoint(
        name,
        parent,
        opensim.Vec3(*parent_location),
        opensim.Vec3(0),
        child,
        opensim.Vec3(0),
        opensim.Vec3(0),
        transform,
    )
    return joint


def add_shoulder_girdle(
    model: Any,
    opensim: Any,
    split_mass: Callable[[Any, float, np.ndarray], None],
    clavicle_length: float,
) -> None:
    """Insert a clavicle and a scapula between torso and humerus, both sides.

    Parameters
    ----------
    model : opensim.Model
        Model with the ``torso`` body, the ``acromial_r``/``acromial_l``
        joints and the ``CLAV`` marker, before initialisation of the system.
    opensim : module
        The ``opensim`` module.
    split_mass : callable
        ``split_mass(body, mass, center)`` removes ``mass`` (kg) located at
        ``center`` (body frame) from ``body``, preserving its first moment.
    clavicle_length : float
        Straight distance (m) between the sternoclavicular and the
        acromioclavicular centres of the unscaled model. The acromioclavicular
        centre is placed laterally so that this length is met while the
        glenohumeral centre stays where the bundled model has it.

    Raises
    ------
    RuntimeError
        If the model does not have the expected bodies, joints or marker.
    ValueError
        If ``clavicle_length`` is not positive or is too short to reach
        the acromioclavicular centre height and depth.

    Notes
    -----
    Adds, for ``side`` in ``r``/``l``: bodies ``clavicle_<side>`` and
    ``scapula_<side>`` (no geometry); joints ``sternoclavicular_<side>`` and
    ``acromioclavicular_<side>``; coordinates ``sc_protraction``,
    ``sc_elevation``, ``ac_protraction``, ``ac_upward_rotation`` and
    ``ac_axial_rotation`` (suffix ``_<side>``), in radians in the model.
    Positive is protraction (lateral end forward), elevation (lateral end up),
    upward rotation of the scapula and posterior rotation about the
    clavicle's long axis. The left axes mirror the right ones. The
    glenohumeral joint ``acromial_<side>`` keeps its name, coordinates and
    pose; it is re-attached to the scapula at the same point of space.
    """
    bodies = model.getBodySet()
    if not bodies.contains("torso") or not model.getMarkerSet().contains("CLAV"):
        raise RuntimeError("The model has no torso body or CLAV marker")
    torso = bodies.get("torso")
    marker = model.getMarkerSet().get("CLAV").get_location()
    notch = np.array([marker.get(i) for i in range(3)])

    for side, letter in (("right", "r"), ("left", "l")):
        mirror = np.ones(3) if side == "right" else _MIRROR
        joint_name = f"acromial_{letter}"
        if not model.getJointSet().contains(joint_name):
            raise RuntimeError(f"The model has no joint {joint_name!r}")
        glenohumeral = model.getJointSet().get(joint_name)
        parent_frame = opensim.PhysicalOffsetFrame.safeDownCast(
            glenohumeral.getParentFrame()
        )
        if parent_frame is None or parent_frame.getParentFrame().getName() != "torso":
            raise RuntimeError(f"{joint_name!r} is not attached to the torso")
        center = np.array([parent_frame.get_translation().get(i) for i in range(3)])

        sc = mirror * (notch + _STERNOCLAVICULAR_FROM_MARKER)
        # Acromioclavicular centre: x and y from the glenohumeral centre (the
        # mirror only flips z), lateral coordinate from the clavicle length.
        ac = center.copy()
        ac[:2] += _ACROMIOCLAVICULAR_FROM_GLENOHUMERAL_XY
        in_plane = ac[:2] - sc[:2]
        lateral_squared = clavicle_length**2 - float(in_plane @ in_plane)
        if clavicle_length <= 0 or lateral_squared <= 0:
            raise ValueError(
                f"clavicle_length={clavicle_length} m is too short for the "
                "acromioclavicular centre position"
            )
        ac[2] = sc[2] + mirror[2] * math.sqrt(lateral_squared)
        clavicle_vector = ac - sc

        # axes defined for the right side; the left side mirrors them
        right_axes = {
            "protraction": np.array([0.0, 1.0, 0.0]),
            "elevation": np.array([-1.0, 0.0, 0.0]),
            "upward": np.array([-1.0, 0.0, 0.0]),
            "axial": mirror * clavicle_vector / clavicle_length,
        }
        axes = (
            right_axes
            if side == "right"
            else {name: _mirrored_axis(axis) for name, axis in right_axes.items()}
        )

        # clavicle
        clavicle_center = clavicle_vector / 2
        perpendicular = _CLAVICLE_MASS * clavicle_length**2 / 12
        clavicle = opensim.Body(
            f"clavicle_{letter}",
            _CLAVICLE_MASS,
            opensim.Vec3(*clavicle_center),
            opensim.Inertia(perpendicular, perpendicular, 1e-5),
        )
        split_mass(torso, _CLAVICLE_MASS, sc + clavicle_center)
        model.addBody(clavicle)

        # scapula, with its frame at the acromioclavicular centre
        scapula_center = (center - ac) / 2 + mirror * _SCAPULA_CENTER_OFFSET
        a, b = _SCAPULA_LENGTH, _SCAPULA_WIDTH
        scapula = opensim.Body(
            f"scapula_{letter}",
            _SCAPULA_MASS,
            opensim.Vec3(*scapula_center),
            opensim.Inertia(
                _SCAPULA_MASS * (a**2 + b**2) / 12,
                _SCAPULA_MASS * b**2 / 12,
                _SCAPULA_MASS * a**2 / 12,
            ),
        )
        split_mass(torso, _SCAPULA_MASS, ac + scapula_center)
        model.addBody(scapula)

        ranges = {
            f"{name}_{letter}": limits
            for name, limits in SHOULDER_GIRDLE_COORDINATES.items()
        }
        sternoclavicular = _rotation_joint(
            opensim,
            f"sternoclavicular_{letter}",
            torso,
            sc,
            clavicle,
            [
                (f"sc_protraction_{letter}", axes["protraction"]),
                (f"sc_elevation_{letter}", axes["elevation"]),
            ],
            ranges,
        )
        acromioclavicular = _rotation_joint(
            opensim,
            f"acromioclavicular_{letter}",
            clavicle,
            clavicle_vector,
            scapula,
            [
                (f"ac_protraction_{letter}", axes["protraction"]),
                (f"ac_upward_rotation_{letter}", axes["upward"]),
                (f"ac_axial_rotation_{letter}", axes["axial"]),
            ],
            ranges,
        )
        for joint in (sternoclavicular, acromioclavicular):
            model.addJoint(joint)
            for index in range(joint.numCoordinates()):
                coordinate = joint.upd_coordinates(index)
                low, high = ranges[coordinate.getName()]
                coordinate.setRangeMin(math.radians(low))
                coordinate.setRangeMax(math.radians(high))
                coordinate.set_clamped(True)
                coordinate.set_locked(False)

        # glenohumeral joint: same point of space, now carried by the scapula
        parent_frame.connectSocket_parent(scapula)
        parent_frame.set_translation(opensim.Vec3(*(center - ac)))
