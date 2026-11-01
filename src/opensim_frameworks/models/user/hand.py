"""Approximate finger joints for the bundled bone geometries.

The joint centres, axes, rest angles and inertial properties in this module
are geometric approximations, not a subject-specific anatomical calibration.
The corrected reference pose is the zero of each coordinate, not an anatomical
zero.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ...utils.opensim import read_vtp_vertices

__all__ = [
    "FINGERS",
    "FINGER_SEGMENTS",
    "MOVING_HAND_SEGMENTS",
    "JOINT_SEPARATION_MM",
    "add_articulated_hands",
]

FINGERS = ("index", "middle", "ring", "little")
FINGER_SEGMENTS = ("proximal", "medial", "distal")
MOVING_HAND_SEGMENTS = (
    "metacarpal1",
    "thumb_proximal",
    "thumb_distal",
    *(f"{finger}_{segment}" for finger in FINGERS for segment in FINGER_SEGMENTS),
)

# Distal translations in millimetres, relative to each parent joint. These
# were selected against the bundled meshes by sampling the coordinate ranges
# and checking triangle intersections; they are not anatomical measurements.
JOINT_SEPARATION_MM = {
    "index": (1.25, 0.0, 0.0),
    "middle": (2.75, 0.0, 1.0),
    "ring": (0.0, 0.75, 0.75),
    "little": (2.5, 0.0, 0.0),
    "thumb": (0.0, 1.25, 2.5),
}


def _vertices(mesh: Any, directory: Path) -> np.ndarray:
    """Read mesh vertices and apply their OpenSim scale factors.

    Parameters
    ----------
    mesh : opensim.Mesh
        Attached bone mesh.
    directory : pathlib.Path
        Directory containing the mesh file.

    Returns
    -------
    numpy.ndarray
        Scaled vertices in the mesh frame, in metres.
    """
    scale = np.array([mesh.get_scale_factors().get(i) for i in range(3)])
    return read_vtp_vertices(directory / mesh.get_mesh_file()) * scale


def _long_axis(vertices: np.ndarray) -> np.ndarray:
    """Estimate a bone's principal longitudinal axis.

    Parameters
    ----------
    vertices : numpy.ndarray
        Bone vertices in the hand frame.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Unit principal axis oriented towards negative hand-frame y.
    """
    _, _, axes = np.linalg.svd(vertices - vertices.mean(axis=0), full_matrices=False)
    axis = axes[0]
    return -axis if axis[1] > 0 else axis


def _head_center(vertices: np.ndarray) -> np.ndarray:
    """Fit a sphere to the distal 20% of a long bone's vertices.

    Parameters
    ----------
    vertices : numpy.ndarray
        Bone vertices in the hand frame, in metres.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Estimated bone-head centre in the hand frame, in metres.

    Raises
    ------
    ValueError
        If a finite sphere with positive radius cannot be estimated.
    """
    # Select distal vertices along the estimated longitudinal axis.
    projection = vertices @ _long_axis(vertices)
    selected = vertices[projection >= np.quantile(projection, 0.8)]
    # Solve the linearized sphere fit and reject degenerate estimates.
    matrix = np.column_stack((2 * selected, np.ones(len(selected))))
    solution, _, rank, _ = np.linalg.lstsq(matrix, np.sum(selected**2, axis=1), rcond=None)
    center = solution[:3]
    radius_squared = float(solution[3] + center @ center)
    if rank != 4 or not np.all(np.isfinite(center)) or radius_squared <= 0:
        raise ValueError("Cannot estimate the joint centre from the bone mesh")
    return center


def _initial_flexion(parent: np.ndarray, child: np.ndarray) -> float:
    """Approximate the sagittal angle of the two bone principal axes.

    Parameters
    ----------
    parent, child : numpy.ndarray
        Parent and child bone vertices in the hand frame.

    Returns
    -------
    float
        Difference between the projected principal-axis angles, in radians.
    """
    a, b = _long_axis(parent), _long_axis(child)
    return math.atan2(b[0], -b[1]) - math.atan2(a[0], -a[1])


def _orientation(first: np.ndarray, second: np.ndarray) -> tuple[float, float, float]:
    """Convert two perpendicular joint axes to body-fixed XYZ Euler angles.

    Parameters
    ----------
    first, second : numpy.ndarray, shape (3,)
        Orthogonal unit axes expressed in the hand frame.

    Returns
    -------
    tuple of float
        Joint-frame orientation in radians.
    """
    rotation = np.column_stack((first, second, np.cross(first, second)))
    return tuple(float(value) for value in Rotation.from_matrix(rotation).as_euler("XYZ"))


def _separation(center: np.ndarray, vertices: np.ndarray, distance_mm: float) -> np.ndarray:
    """Calculate a relative bone translation away from its joint centre.

    Parameters
    ----------
    center : numpy.ndarray, shape (3,)
        Estimated joint centre in the hand frame, in metres.
    vertices : numpy.ndarray
        Child bone vertices in the hand frame, in metres.
    distance_mm : float
        Translation magnitude in millimetres.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Translation in the hand frame, in metres.

    Raises
    ------
    ValueError
        If a nonzero translation direction cannot be determined.
    """
    if distance_mm == 0:
        return np.zeros(3)
    direction = vertices.mean(axis=0) - center
    length = float(np.linalg.norm(direction))
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError("Cannot determine the direction of a finger-bone correction")
    return direction * (distance_mm / (1000 * length))


def _vec3(opensim: Any, values: np.ndarray | tuple[float, ...]) -> Any:
    """Convert three numeric values to an OpenSim vector.

    Parameters
    ----------
    opensim : module
        OpenSim Python module.
    values : numpy.ndarray or tuple of float
        Three vector components.

    Returns
    -------
    opensim.Vec3
        OpenSim vector containing the supplied components.
    """
    return opensim.Vec3(*(float(value) for value in values))


def add_articulated_hands(model: Any, opensim: Any, mesh_directory: Path) -> None:
    """Split the bundled rigid hands into palms and articulated bone segments.

    Parameters
    ----------
    model : opensim.Model
        Model whose original hand bodies are modified in place.
    opensim : module
        OpenSim Python module.
    mesh_directory : pathlib.Path
        Directory containing the original bone meshes.

    Raises
    ------
    ValueError
        If a required mesh is missing, degenerate or cannot provide a joint
        centre or separation direction.

    Notes
    -----
    The distal joint centres are sphere fits to the parent bone head. The
    thumb carpometacarpal centre is the midpoint of the closest mesh vertices.
    Flexion axes and thumb opposition axes are fixed approximations in the
    original hand frame. Ranges are adapted from the public AAH reference
    model, with the nominal flexion of the four fingers approximated from
    bone principal axes. Neither the ranges nor the axes are clinically
    validated for these meshes.

    Some bones are translated along their estimated longitudinal directions
    to reduce intersections in the bundled reference pose and sampled joint
    movements. These translations are not anatomical measurements and
    sampled tests do not guarantee collision-free motion.

    Finger masses total 25% of the original hand mass and are distributed
    using bone bounding-box volumes; cuboid inertias are approximations.
    The original hand's mass and first moment are preserved at the reference
    pose. Its original inertia tensor is not preserved exactly.
    """
    for side in ("r", "l"):
        # Record the original hand meshes and appearance for later reassignment.
        hand = model.updBodySet().get(f"hand_{side}")
        source = {}
        specifications = []
        for index in range(hand.getPropertyByName("attached_geometry").size()):
            mesh = opensim.Mesh.safeDownCast(hand.get_attached_geometry(index))
            if mesh is None:
                raise ValueError(f"Unexpected non-mesh geometry on hand_{side}")
            name = mesh.get_mesh_file()
            stem = name.removesuffix(f"_{side}vs.vtp")
            appearance = mesh.get_Appearance()
            specifications.append(
                (
                    stem,
                    name,
                    mesh.getName(),
                    tuple(mesh.get_scale_factors().get(i) for i in range(3)),
                    tuple(mesh.getColor().get(i) for i in range(3)),
                    mesh.getOpacity(),
                    appearance.get_visible(),
                    appearance.get_representation(),
                )
            )
            source[stem] = _vertices(mesh, mesh_directory)

        # Check that all bones needed to estimate the finger joints are present.
        required = set(MOVING_HAND_SEGMENTS) | {
            "trapezium",
            "metacarpal2",
            "metacarpal3",
            "metacarpal4",
            "metacarpal5",
        }
        missing = required - source.keys()
        if missing:
            raise ValueError(f"Missing hand_{side} bone meshes: {sorted(missing)}")

        # Estimate finger joint centres and accumulate the geometric corrections.
        joint_centers = {}
        relative_offsets = {}
        cumulative_offsets = {}
        for digit, number in zip(FINGERS, (2, 3, 4, 5)):
            parent = f"metacarpal{number}"
            parent_offset = np.zeros(3)
            for segment, separation in zip(FINGER_SEGMENTS, JOINT_SEPARATION_MM[digit]):
                child = f"{digit}_{segment}"
                center = _head_center(source[parent])
                relative_offset = _separation(center, source[child], separation)
                joint_centers[child] = center
                relative_offsets[child] = relative_offset
                cumulative_offsets[child] = parent_offset + relative_offset
                parent, parent_offset = child, cumulative_offsets[child]

        # Estimate the thumb CMC centre and corrections along the thumb chain.
        trapezium = source["trapezium"]
        metacarpal = source["metacarpal1"]
        distances, indices = cKDTree(trapezium).query(metacarpal)
        index = int(np.argmin(distances))
        thumb_center = (metacarpal[index] + trapezium[indices[index]]) / 2
        parent_offset = np.zeros(3)
        for child, center, separation in (
            ("metacarpal1", thumb_center, JOINT_SEPARATION_MM["thumb"][0]),
            (
                "thumb_proximal",
                _head_center(source["metacarpal1"]),
                JOINT_SEPARATION_MM["thumb"][1],
            ),
            (
                "thumb_distal",
                _head_center(source["thumb_proximal"]),
                JOINT_SEPARATION_MM["thumb"][2],
            ),
        ):
            relative_offset = _separation(center, source[child], separation)
            joint_centers[child] = center
            relative_offsets[child] = relative_offset
            cumulative_offsets[child] = parent_offset + relative_offset
            parent_offset = cumulative_offsets[child]

        # Distribute part of the hand mass while preserving its first moment.
        volumes = {
            name: float(np.prod(np.ptp(source[name], axis=0)))
            for name in MOVING_HAND_SEGMENTS
        }
        if any(volume <= 0 or not math.isfinite(volume) for volume in volumes.values()):
            raise ValueError(f"Degenerate finger mesh on hand_{side}")
        finger_mass = 0.25 * hand.getMass()
        masses = {
            name: finger_mass * volume / sum(volumes.values())
            for name, volume in volumes.items()
        }
        centers = {name: source[name].mean(axis=0) for name in MOVING_HAND_SEGMENTS}
        original_mass = hand.getMass()
        original_center = np.array([hand.getMassCenter().get(i) for i in range(3)])
        removed_moment = sum(
            masses[name] * (centers[name] + cumulative_offsets[name])
            for name in MOVING_HAND_SEGMENTS
        )
        remaining_center = (original_mass * original_center - removed_moment) / (
            original_mass - finger_mass
        )
        hand.scaleMass((original_mass - finger_mass) / original_mass)
        hand.setMassCenter(_vec3(opensim, remaining_center))

        # Create separate finger bodies with approximate cuboid inertias.
        hand.updPropertyByName("attached_geometry").clear()
        bodies = {}
        for name in MOVING_HAND_SEGMENTS:
            lengths = np.ptp(source[name], axis=0)
            mass = masses[name]
            moments = mass * np.array(
                [
                    lengths[1] ** 2 + lengths[2] ** 2,
                    lengths[0] ** 2 + lengths[2] ** 2,
                    lengths[0] ** 2 + lengths[1] ** 2,
                ]
            ) / 12
            body = opensim.Body(
                f"{name}_{side}",
                mass,
                _vec3(opensim, centers[name]),
                opensim.Inertia(*(float(value) for value in moments)),
            )
            model.addBody(body)
            bodies[name] = body

        # Attach each saved mesh to its corresponding finger or palm body.
        for stem, filename, name, scale, color, opacity, visible, representation in specifications:
            mesh = opensim.Mesh(filename)
            mesh.setName(name)
            mesh.set_scale_factors(_vec3(opensim, scale))
            mesh.setColor(_vec3(opensim, color))
            mesh.setOpacity(opacity)
            mesh.upd_Appearance().set_visible(visible)
            mesh.upd_Appearance().set_representation(representation)
            (bodies[stem] if stem in bodies else hand).attachGeometry(mesh)

        # Connect the four fingers with MCP, PIP and DIP joints.
        z_axis = np.array([0.0, 0.0, 1.0])
        for digit, number in zip(FINGERS, (2, 3, 4, 5)):
            parent_body = hand
            parent_mesh = f"metacarpal{number}"
            for segment, joint_name in zip(FINGER_SEGMENTS, ("mcp", "pip", "dip")):
                child_name = f"{digit}_{segment}"
                center = joint_centers[child_name]
                flexion = _initial_flexion(source[parent_mesh], source[child_name])
                location = _vec3(opensim, center)
                child_location = _vec3(opensim, center - relative_offsets[child_name])
                if joint_name == "mcp":
                    lateral_sign = (1 if digit in ("ring", "little") else -1) * (
                        1 if side == "r" else -1
                    )
                    orientation = _vec3(
                        opensim, _orientation(z_axis, np.array([lateral_sign, 0.0, 0.0]))
                    )
                    joint = opensim.UniversalJoint(
                        f"{digit}_{joint_name}_{side}",
                        parent_body, location, orientation,
                        bodies[child_name], child_location, orientation,
                    )
                    abduction = joint.updCoordinate(1)
                    abduction.setName(f"{digit}_{joint_name}_abduction_{side}")
                    abduction.setRangeMin(math.radians(-15))
                    abduction.setRangeMax(math.radians(30 if digit in ("index", "little") else 15))
                    abduction.set_locked(True)
                    limits = (-50, 90)
                else:
                    joint = opensim.PinJoint(
                        f"{digit}_{joint_name}_{side}",
                        parent_body, location, opensim.Vec3(0),
                        bodies[child_name], child_location, opensim.Vec3(0),
                    )
                    limits = (0, 120) if joint_name == "pip" else (-30, 90)
                coordinate = joint.updCoordinate(0)
                coordinate.setName(f"{digit}_{joint_name}_flexion_{side}")
                coordinate.setRangeMin(math.radians(limits[0]) - flexion)
                coordinate.setRangeMax(math.radians(limits[1]) - flexion)
                coordinate.set_locked(True)
                model.addJoint(joint)
                parent_body = bodies[child_name]
                parent_mesh = child_name

        # Connect the thumb metacarpal with opposition and flexion coordinates.
        location = _vec3(opensim, thumb_center)
        child_location = _vec3(opensim, thumb_center - relative_offsets["metacarpal1"])
        opposition_sign = 1 if side == "r" else -1
        orientation = _vec3(
            opensim,
            _orientation(np.array([opposition_sign, 0.0, 0.0]), z_axis),
        )
        joint = opensim.UniversalJoint(
            f"thumb_cmc_{side}", hand, location, orientation,
            bodies["metacarpal1"], child_location, orientation,
        )
        for index, motion, limits in (
            (0, "opposition", (-45, 45)),
            (1, "flexion", (-15, 45)),
        ):
            coordinate = joint.updCoordinate(index)
            coordinate.setName(f"thumb_cmc_{motion}_{side}")
            coordinate.setRangeMin(math.radians(limits[0]))
            coordinate.setRangeMax(math.radians(limits[1]))
            coordinate.set_locked(True)
        model.addJoint(joint)

        # Connect the thumb phalanges through its MCP and IP joints.
        for parent, child, joint_name, limits in (
            ("metacarpal1", "thumb_proximal", "mcp", (-30, 40)),
            ("thumb_proximal", "thumb_distal", "ip", (-75, 75)),
        ):
            center = joint_centers[child]
            location = _vec3(opensim, center)
            child_location = _vec3(opensim, center - relative_offsets[child])
            joint = opensim.PinJoint(
                f"thumb_{joint_name}_{side}",
                bodies[parent], location, opensim.Vec3(0),
                bodies[child], child_location, opensim.Vec3(0),
            )
            coordinate = joint.updCoordinate()
            coordinate.setName(f"thumb_{joint_name}_flexion_{side}")
            coordinate.setRangeMin(math.radians(limits[0]))
            coordinate.setRangeMax(math.radians(limits[1]))
            coordinate.set_locked(True)
            model.addJoint(joint)
