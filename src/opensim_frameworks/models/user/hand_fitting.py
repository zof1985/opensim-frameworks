"""Hand-to-grip fitting: mesh/capsule proxies and the fit_handle solver.

Everything here supports three public operations exposed on :class:`User`
through :class:`_HandFittingMixin`: reading/writing individual finger
coordinates, exact triangle-surface intersection checks
(``hand_surface_intersections``/``hand_handle_intersections``), and
``fit_handle``, a staged solver that fits shoulder, elbow, forearm, wrist and
every finger coordinate to a grip body. ``fit_handle`` is built on an inner
engine, :meth:`_HandFittingMixin._fit_handle_grip` (the original, narrower
``fit_handle`` validated first, before shoulder/elbow were in scope): a
cheap-PCA-capsule-proxy SLSQP solver that wraps a hand around a grip body
assuming the arm is already coarsely positioned nearby. ``fit_handle`` itself
adds three stages ahead of that engine (cylinder fit to the grip mesh, MCP-
axis alignment, palm-cylinder tangency) so no prior arm positioning is
required; see its own docstring. This is fitting-time logic only: the
anatomical construction of the articulated hand bodies themselves (joints,
axes, rest pose) lives in :mod:`.hand`.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.optimize import least_squares, minimize

from ...utils.geometry import enclosing_capsule, segment_pair_distance
from ...utils.opensim import (
    import_opensim,
    read_vtp_vertices,
    rotation_to_numpy,
    vec3_to_numpy,
)
from .hand import FINGERS, MOVING_HAND_SEGMENTS

__all__ = []


# -- Shared mesh-surface loading, used by hand_surface_intersections,
# -- hand_handle_intersections and the capsule proxies built for fit_handle.


def _resolve_body_mesh_path(
    filename: Path, directories: Iterable[Path], label: str
) -> Path:
    """Resolve a mesh filename as an absolute path or within given directories.

    Parameters
    ----------
    filename : pathlib.Path
        Mesh path as stored on an OpenSim geometry component.
    directories : iterable of pathlib.Path
        Directories searched, in order, when `filename` is relative. An empty
        iterable means only a literal absolute path (or one resolvable
        relative to the current working directory) is accepted.
    label : str
        Capitalized noun describing the mesh owner, used in error messages
        (for example ``"Hand"``).

    Returns
    -------
    pathlib.Path
        Existing mesh path.

    Raises
    ------
    FileNotFoundError
        If `filename` cannot be resolved to an existing file. This never
        falls back to OpenSim's native geometry search path, which silently
        fails to resolve some valid absolute paths under the OpenSim 4.6
        Python bindings.
    """
    if not filename.is_absolute():
        source = next(
            (
                directory / filename
                for directory in directories
                if (directory / filename).is_file()
            ),
            None,
        )
        if source is None:
            raise FileNotFoundError(f"{label} mesh not found: {filename}")
        filename = source
    if not filename.is_file():
        raise FileNotFoundError(f"{label} mesh not found: {filename}")
    return filename


def _load_mesh_surface(
    mesh: Any, state: Any, directories: Iterable[Path], label: str
) -> tuple[Any, Any]:
    """Load one attached OpenSim mesh as VTK polydata with its ground transform.

    Parameters
    ----------
    mesh : opensim.Mesh
        Attached mesh geometry.
    state : opensim.State
        State realized through Position, used for the mesh frame transform.
    directories : iterable of pathlib.Path
        Directories searched when the mesh file path is relative.
    label : str
        Capitalized noun describing the mesh owner, used in error messages.

    Returns
    -------
    tuple of (vtk.vtkPolyData, vtk.vtkMatrix4x4)
        Loaded triangular surface and its scaled mesh-to-ground transform.

    Raises
    ------
    FileNotFoundError
        If the mesh file cannot be resolved.
    ValueError
        If the mesh is not a triangulated VTP surface with finite geometry
        and a valid scale.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    filename = _resolve_body_mesh_path(Path(mesh.get_mesh_file()), directories, label)
    if filename.suffix.lower() != ".vtp":
        raise ValueError(f"{label} mesh must be a VTP file: {filename}")
    scale = np.array([mesh.get_scale_factors().get(i) for i in range(3)])
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError(f"Invalid scale factors for {label.lower()} mesh: {filename}")
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(filename))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata.GetNumberOfPoints() == 0 or polydata.GetNumberOfPolys() == 0:
        raise ValueError(f"{label} mesh has no polygonal surface: {filename}")
    if not np.all(np.isfinite(vtk_to_numpy(polydata.GetPoints().GetData()))):
        raise ValueError(f"{label} mesh contains nonfinite vertices: {filename}")
    offsets = vtk_to_numpy(polydata.GetPolys().GetOffsetsArray())
    if np.any(np.diff(offsets) != 3):
        raise ValueError(f"{label} mesh contains non-triangular polygons: {filename}")
    transform = mesh.getFrame().getTransformInGround(state)
    matrix = vtk.vtkMatrix4x4()
    for row in range(3):
        for column in range(3):
            matrix.SetElement(
                row, column, float(transform.R().get(row, column) * scale[column])
            )
        matrix.SetElement(row, 3, float(transform.p().get(row)))
    return polydata, matrix


def _surfaces_intersecting_pairs(
    first_surfaces: list[tuple[str, str, Any, Any]],
    second_surfaces: list[tuple[str, str, Any, Any]] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return mesh filename pairs whose VTK surfaces touch or cross.

    Parameters
    ----------
    first_surfaces : list of (body name, mesh filename, vtkPolyData, vtkMatrix4x4)
        Surfaces to compare.
    second_surfaces : list of the same shape, or None, default=None
        When given, every surface in `first_surfaces` is compared against
        every surface in `second_surfaces` (a cross-model check). When
        omitted, every pair within `first_surfaces` belonging to different
        bodies is compared instead (a self-check).

    Returns
    -------
    tuple of tuple of str
        Pairs of mesh filenames whose surfaces touch or cross, in the order
        checked.
    """
    import vtk

    if second_surfaces is None:
        pairs = (
            (first, second)
            for index, first in enumerate(first_surfaces)
            for second in first_surfaces[index + 1 :]
            if first[0] != second[0]
        )
    else:
        pairs = ((first, second) for first in first_surfaces for second in second_surfaces)

    intersections = []
    for first, second in pairs:
        collision = vtk.vtkCollisionDetectionFilter()
        collision.SetInputData(0, first[2])
        collision.SetInputData(1, second[2])
        collision.SetMatrix(0, first[3])
        collision.SetMatrix(1, second[3])
        collision.SetCollisionModeToFirstContact()
        collision.SetBoxTolerance(0)
        collision.SetCellTolerance(0)
        collision.Update()
        if collision.GetNumberOfContacts():
            intersections.append((first[1], second[1]))
    return tuple(intersections)


# -- Cheap capsule proxies used inside the fit_handle solver loop. Exact
# -- triangle-surface checks (above) are reserved for final accept/reject.

#: Palm capsules nearest the grip, used as the operational definition of
#: "palm contact" (the metacarpal heads, not the wrist-side carpals). This is
#: inherited from the exploratory grip_capsule_study prototype, not an
#: anatomically validated contact region.
_PALM_CONTACT_CAPSULES = ("metacarpal2", "metacarpal3", "metacarpal4", "metacarpal5")

#: The 15 phalanx-like capsules (4 fingers x 3 segments, plus the 3-link
#: thumb chain) whose distance to the grip drives the wrap objective.
_WRAP_CAPSULES = tuple(MOVING_HAND_SEGMENTS)

#: Wrist and forearm coordinates fit_handle may adjust, besides the finger
#: coordinates; arm_flex/elbow_flex/etc. (shoulder and elbow) are excluded.
_WRIST_COORDINATE_SUFFIXES = ("wrist_flex", "wrist_dev", "pro_sup")

#: (finger, joint, motion) triples enumerating every finger coordinate
#: unlocked by fit_handle, in a fixed, deterministic order.
_FINGER_COORDINATE_KEYS = tuple(
    (finger, joint, "flexion")
    for finger in FINGERS
    for joint in ("mcp", "pip", "dip")
) + tuple(
    (finger, "mcp", "abduction") for finger in FINGERS
) + (
    ("thumb", "cmc", "opposition"),
    ("thumb", "cmc", "flexion"),
    ("thumb", "mcp", "flexion"),
    ("thumb", "ip", "flexion"),
)

#: fit_handle tuning weights and tolerances. These are engineering choices
#: that balance the wrap objective against secondary preferences; they are
#: not derived from a validated biomechanical or comfort source.
_WRAP_REGULARIZATION_WEIGHT = 1e-6
_PUSH_ADDUCTION_WEIGHT = 0.05
_PULL_PALM_APPROACH_WEIGHT = 0.1
_PULL_THUMB_WRAP_WEIGHT = 0.2
_FIT_HANDLE_CONSTRAINT_TOLERANCE_M = 1e-6

#: Points per axis for the pre-solve wrist/forearm grid search (see
#: fit_handle). Pure forward kinematics over 3 coordinates with no mesh I/O,
#: so a finer grid is cheap; a 6-point grid was confirmed too coarse to find
#: a wrist pose with adequate range-of-motion headroom on a real handle.
_WRIST_GRID_POINTS_PER_AXIS = 16

#: Absolute tolerance, in metres, used to treat two wrist-grid points as
#: equally (near-)feasible when choosing between them by range-of-motion
#: margin instead of by violation alone.
_WRIST_GRID_VIOLATION_ATOL = 1e-6

#: Side-inference sanity bounds: the nearer palm marker must be closer than
#: this to the handle's capsule centre, and clearly closer than the other
#: palm, otherwise fit_handle refuses to guess which hand to move.
_SIDE_INFERENCE_MAX_DISTANCE_M = 0.3
_SIDE_INFERENCE_MARGIN_M = 0.02


# -- fit_handle tuning (the staged cylinder-fit + two-stage IK ahead of the
# -- _fit_handle_grip engine below, that also moves the shoulder and elbow,
# -- instead of assuming the arm is already coarsely positioned).

#: Fingers whose MCP joint centres define the least-squares line used in
#: fit_handle's step 1. Each MCP joint's child-frame location is the
#: joint centre itself, which does not move under that joint's own
#: coordinates (a revolute/universal joint's coordinates rotate the child
#: body about its own fixed attachment point), so this line moves rigidly
#: with the hand body under any shoulder/elbow/forearm/wrist coordinate
#: change, independent of the current finger pose.
_MCP_LINE_FINGERS = ("middle", "ring", "little")

#: Shoulder, elbow, forearm and wrist coordinates fit_handle moves in its
#: steps 2-3, on top of every finger coordinate _fit_handle_grip itself
#: adjusts in step 4. Distinct from _WRIST_COORDINATE_SUFFIXES (forearm and
#: wrist only, no shoulder/elbow), which _fit_handle_grip keeps using
#: unchanged when called directly (not through fit_handle).
_FULL_ARM_COORDINATE_SUFFIXES = (
    "arm_flex",
    "arm_add",
    "arm_rot",
    "elbow_flex",
    "pro_sup",
    "wrist_flex",
    "wrist_dev",
)

#: fit_handle step 2 (MCP-axis alignment) objective weights: the
#: point-to-segment distance term dominates over the angle-misalignment
#: term, which in turn dominates over the regularization towards the pose
#: fit_handle was called with. Tuned on one real case (ANSUR M-150,
#: High Row equipment, neutral grip); not derived from a validated source,
#: like the comparable _fit_handle_grip weights above.
_AXIS_DISTANCE_WEIGHT = 50.0
_AXIS_ANGLE_WEIGHT = 5.0
_AXIS_ALIGNMENT_REGULARIZATION_WEIGHT = 1e-3

#: fit_handle step 3 (palm-to-cylinder tangency) regularization
#: weight, towards the step 2 solution. Same tuning caveat as above.
_PALM_TANGENCY_REGULARIZATION_WEIGHT = 1e-2


@dataclass(frozen=True)
class _HandCapsule:
    """A bone-enclosing PCA capsule for one attached hand mesh, in its own frame."""

    name: str
    body: str
    frame: Any
    start: np.ndarray
    end: np.ndarray
    radius: float


def _collect_hand_capsules(user: Any, suffix: str, label: str) -> list[_HandCapsule]:
    """Build one PCA capsule per attached palm or finger mesh.

    Parameters
    ----------
    user : User
        Initialized user whose bone meshes are measured.
    suffix : {'r', 'l'}
        Hand side.
    label : str
        Capitalized noun used in error messages (for example ``"Hand"``).

    Returns
    -------
    list of _HandCapsule
        One capsule per attached palm or finger mesh, in a deterministic
        body and mesh order (``"hand"`` first, then each of
        :data:`MOVING_HAND_SEGMENTS`).

    Raises
    ------
    ValueError
        If a body has no attached mesh, a non-mesh geometry is attached, or
        a mesh scale is invalid.
    FileNotFoundError
        If a referenced mesh cannot be resolved.
    """
    capsules = []
    for stem in ("hand", *MOVING_HAND_SEGMENTS):
        body = user.getBodySet().get(f"{stem}_{suffix}")
        count = body.getPropertyByName("attached_geometry").size()
        if count == 0:
            raise ValueError(f"Body {body.getName()!r} has no attached mesh")
        for index in range(count):
            mesh = user.opensim.Mesh.safeDownCast(body.get_attached_geometry(index))
            if mesh is None:
                raise ValueError(f"Non-mesh geometry on body {body.getName()!r}")
            path = _resolve_body_mesh_path(
                Path(mesh.get_mesh_file()), user.geometry_directories, label
            )
            scale = np.array([mesh.get_scale_factors().get(i) for i in range(3)])
            if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
                raise ValueError(f"Invalid scale factors for {label.lower()} mesh: {path}")
            vertices = read_vtp_vertices(path) * scale
            start, end, radius = enclosing_capsule(vertices)
            name = Path(mesh.get_mesh_file()).name.removesuffix(f"_{suffix}vs.vtp")
            capsules.append(_HandCapsule(name, stem, mesh.getFrame(), start, end, radius))
    return capsules


def _capsule_segments_in_ground(
    capsules: list[_HandCapsule], state: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Transform capsule centre segments from their mesh frame into Ground.

    Parameters
    ----------
    capsules : list of _HandCapsule
        Capsules to transform.
    state : opensim.State
        State realized through Position.

    Returns
    -------
    starts, ends : numpy.ndarray, shape (n, 3)
        Capsule segment endpoints in Ground, in metres.
    """
    starts, ends = [], []
    for capsule in capsules:
        transform = capsule.frame.getTransformInGround(state)
        rotation = rotation_to_numpy(transform.R())
        translation = vec3_to_numpy(transform.p())
        starts.append(rotation @ capsule.start + translation)
        ends.append(rotation @ capsule.end + translation)
    return np.asarray(starts), np.asarray(ends)


def _self_collision_pairs(
    capsules: list[_HandCapsule], state: Any
) -> tuple[np.ndarray, np.ndarray]:
    """List checked self-collision index pairs and their reference-pose floor.

    Every pair whose capsules belong to different rigid bodies is checked;
    same-body pairs are excluded because their relative transform cannot
    change under finger motion. A single rigid capsule cannot represent a
    joint's articulation surface, so capsules on either side of a finger or
    thumb joint already overlap at the locked reference pose even though the
    actual bone meshes do not (confirmed independently by
    :meth:`User.hand_surface_intersections`); the same proxy overestimation
    also affects a handful of nearby-but-unconnected bones close to the thumb
    base (for example ``trapezoid``-``metacarpal1``), which a purely
    joint-based exclusion list would miss. Rather than hardcoding which pairs
    are anatomically adjacent, each pair's own reference-pose gap is used as
    its floor whenever that gap is already at or below zero: the hard
    constraint then only forbids a pair from overlapping *more* than it
    already does at the verified collision-free reference pose, instead of
    forbidding the proxy overlap outright. Pairs with a healthy positive
    reference gap keep the ordinary zero floor.

    Parameters
    ----------
    capsules : list of _HandCapsule
        Complete hand capsule set for one side.
    state : opensim.State
        Reference-pose state (finger coordinates at their locked default)
        used to compute each pair's floor once.

    Returns
    -------
    pairs : numpy.ndarray, shape (m, 2)
        Checked capsule index pairs.
    floor : numpy.ndarray, shape (m,)
        Per-pair gap floor in metres: ``min(reference_gap, 0.0)``.
    """
    pairs = np.asarray(
        [
            (i, j)
            for i, j in itertools.combinations(range(len(capsules)), 2)
            if capsules[i].body != capsules[j].body
        ],
        dtype=int,
    ).reshape(-1, 2)
    starts, ends = _capsule_segments_in_ground(capsules, state)
    radii = np.array([capsule.radius for capsule in capsules])
    reference_gaps = (
        segment_pair_distance(starts[pairs[:, 0]], ends[pairs[:, 0]], starts[pairs[:, 1]], ends[pairs[:, 1]])
        - radii[pairs[:, 0]]
        - radii[pairs[:, 1]]
    )
    return pairs, np.minimum(reference_gaps, 0.0)


def _handle_mesh_vertices(handle: Any, label: str) -> np.ndarray:
    """Return a grip body's attached mesh vertices, in Ground.

    Shared by :func:`_handle_capsule` (PCA axis proxy) and
    :func:`_handle_cylinder` (least-squares cylinder fit): both need the same
    scaled, ground-frame vertices, just fitted with a different model.

    Parameters
    ----------
    handle : opensim.Body
        Grip body belonging to a separately initialized OpenSim model. Its
        pose is read from that model's own working state, not from any
        caller's Python-level ``.state`` attribute, so this works even when
        `handle` was retrieved through a different Python wrapper than the
        one that last moved its owning model.
    label : str
        Capitalized noun used in error messages (for example ``"Handle"``).

    Returns
    -------
    numpy.ndarray, shape (n, 3)
        Scaled vertices of every mesh attached to `handle`, in Ground, in
        metres.

    Raises
    ------
    TypeError
        If `handle` is not an ``opensim.Body``.
    ValueError
        If `handle` is not connected to an initialized model, has no
        attached mesh, or its geometry is invalid.
    FileNotFoundError
        If a referenced mesh file is not an absolute path (or one resolvable
        relative to the current working directory): this does not search
        any geometry directory registered for the handle's own model.
    """
    opensim_module = import_opensim()
    if not isinstance(handle, opensim_module.Body):
        raise TypeError("handle must be an opensim.Body")
    if not handle.hasModel():
        raise ValueError("handle is not connected to an initialized OpenSim model")
    handle_model = handle.getModel()
    handle_state = handle_model.getWorkingState()
    handle_model.realizePosition(handle_state)
    count = handle.getPropertyByName("attached_geometry").size()
    if count == 0:
        raise ValueError(f"Body {handle.getName()!r} has no attached mesh")
    vertex_arrays = []
    for index in range(count):
        mesh = opensim_module.Mesh.safeDownCast(handle.get_attached_geometry(index))
        if mesh is None:
            raise ValueError(f"Non-mesh geometry on body {handle.getName()!r}")
        path = _resolve_body_mesh_path(Path(mesh.get_mesh_file()), (), label)
        scale = np.array([mesh.get_scale_factors().get(i) for i in range(3)])
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError(f"Invalid scale factors for {label.lower()} mesh: {path}")
        vertices = read_vtp_vertices(path) * scale
        transform = mesh.getFrame().getTransformInGround(handle_state)
        rotation = rotation_to_numpy(transform.R())
        translation = vec3_to_numpy(transform.p())
        vertex_arrays.append(vertices @ rotation.T + translation)
    return np.vstack(vertex_arrays)


def _handle_capsule(handle: Any, label: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a single PCA capsule enclosing a grip body's attached mesh, in Ground.

    Parameters
    ----------
    handle : opensim.Body
        Grip body; see :func:`_handle_mesh_vertices`.
    label : str
        Capitalized noun used in error messages (for example ``"Handle"``).

    Returns
    -------
    start, end : numpy.ndarray, shape (3,)
        Capsule centre-segment endpoints in Ground, in metres.
    radius : float
        Capsule radius in metres.

    Raises
    ------
    TypeError, ValueError, FileNotFoundError
        See :func:`_handle_mesh_vertices`.
    """
    return enclosing_capsule(_handle_mesh_vertices(handle, label))


def _handle_cylinder(
    handle: Any, label: str
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit a least-squares cylinder to a grip body's attached mesh, in Ground.

    Unlike :func:`_handle_capsule`'s PCA axis, which only orients a capsule
    without estimating a surface-fitting radius, this minimizes every
    vertex's perpendicular distance to the fitted cylindrical surface over
    its axis direction, a point on the axis, and the radius jointly
    (:func:`scipy.optimize.least_squares`). Initialized from the same PCA
    axis and capsule radius used elsewhere, so a degenerate or near-planar
    point cloud is rejected the same way before any nonlinear solve is
    attempted.

    Parameters
    ----------
    handle : opensim.Body
        Grip body; see :func:`_handle_mesh_vertices`.
    label : str
        Capitalized noun used in error messages.

    Returns
    -------
    axis_point, axis : numpy.ndarray, shape (3,)
        A point on the fitted axis, close to the mesh centroid, and its unit
        direction (arbitrary sign).
    radius : float
        Fitted cylinder radius, in metres.
    segment_start, segment_end : numpy.ndarray, shape (3,)
        Extremes of the vertex projections onto the fitted axis: the
        physical extent actually covered by the mesh along that axis, as
        opposed to the mathematically infinite line the axis alone implies.
        Used by :meth:`_HandFittingMixin.fit_handle` to bound its
        alignment target -- confirmed necessary on a real case, see that
        method's Notes.

    Raises
    ------
    TypeError, ValueError, FileNotFoundError
        See :func:`_handle_mesh_vertices`.
    RuntimeError
        If the nonlinear least-squares cylinder fit does not converge.
    ValueError
        If the fitted radius is not finite and positive.
    """
    vertices = _handle_mesh_vertices(handle, label)
    pca_start, pca_end, pca_radius = enclosing_capsule(vertices)
    axis0 = pca_end - pca_start
    axis0 = axis0 / np.linalg.norm(axis0)
    point0 = (pca_start + pca_end) / 2

    def residuals(params: np.ndarray) -> np.ndarray:
        point, theta, phi, radius = params[:3], params[3], params[4], params[5]
        direction = np.array(
            [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]
        )
        offset = vertices - point
        perpendicular = offset - np.outer(offset @ direction, direction)
        return np.linalg.norm(perpendicular, axis=1) - radius

    theta0 = math.acos(float(np.clip(axis0[2], -1.0, 1.0)))
    phi0 = math.atan2(float(axis0[1]), float(axis0[0]))
    result = least_squares(
        residuals, np.array([*point0, theta0, phi0, pca_radius]), method="lm"
    )
    if not result.success:
        raise RuntimeError(
            f"Cylinder least-squares fit did not converge for {label.lower()} "
            f"{handle.getName()!r}: {result.message}"
        )
    point, theta, phi, radius = result.x[:3], result.x[3], result.x[4], result.x[5]
    axis = np.array(
        [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]
    )
    if radius <= 0 or not math.isfinite(radius):
        raise ValueError(
            f"Degenerate cylinder radius fitted for {label.lower()} {handle.getName()!r}"
        )
    # Re-anchor the axis point near the mesh centroid (any point on the line
    # is equivalent for a direction/line representation; this choice keeps
    # it numerically close to the data instead of wherever the unconstrained
    # least-squares solve happened to place it along the infinite line).
    centroid = vertices.mean(axis=0)
    axis_point = point + ((centroid - point) @ axis) * axis
    projections = (vertices - axis_point) @ axis
    segment_start = axis_point + float(projections.min()) * axis
    segment_end = axis_point + float(projections.max()) * axis
    return axis_point, axis, float(radius), segment_start, segment_end


def _mcp_axis_in_ground(user: Any, suffix: str) -> tuple[np.ndarray, np.ndarray]:
    """Fit a least-squares line to the middle/ring/little MCP joint centres.

    Parameters
    ----------
    user : User
        User whose state is realized through Position.
    suffix : {'r', 'l'}
        Hand side.

    Returns
    -------
    point, direction : numpy.ndarray, shape (3,)
        The centroid of the three MCP joint centres and the line's unit
        direction (arbitrary sign), in Ground, in metres.

    Raises
    ------
    ValueError
        If the three joint centres do not define a unique direction (for
        example if they are nearly coincident).

    Notes
    -----
    Each MCP joint's child-frame location is the joint centre itself, which
    does not move under that joint's own coordinates: a revolute or
    universal joint's coordinates rotate the child body about its own fixed
    attachment point, which stays fixed in Ground regardless of the
    rotation. So this line moves rigidly with the hand body under any
    shoulder/elbow/forearm/wrist coordinate change, independent of the
    current finger pose -- no reference finger pose needs to be fixed to
    compute it, confirmed against this model's actual joint definitions
    (:func:`opensim_frameworks.models.user.hand.add_articulated_hands`).
    """
    user.realizePosition(user.state)
    points = np.array(
        [user._joint_center(f"{finger}_mcp_{suffix}") for finger in _MCP_LINE_FINGERS]
    )
    center = points.mean(axis=0)
    _, singular, axes = np.linalg.svd(points - center, full_matrices=False)
    if singular[0] <= 0:
        raise ValueError("MCP joint centres do not define a unique direction")
    return center, axes[0]


def _validate_soft_tissue_distance(value: Any) -> float:
    """Validate ``soft_tissues_approximation_distance`` and return it as a float.

    Parameters
    ----------
    value : Any
        Candidate value, expected to be a positive finite number in metres.

    Returns
    -------
    float
        The validated value.

    Raises
    ------
    ValueError
        If `value` is not a positive finite number (bools are rejected even
        though ``bool`` is a subclass of ``int``).
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            "soft_tissues_approximation_distance must be a positive "
            "finite number in metres"
        )
    return float(value)


class _HandFittingMixin:
    """Finger coordinates, surface-intersection checks, and fit_handle."""

    def finger_coordinate(
        self, side: str, finger: str, joint: str, motion: str
    ) -> Any:
        """Return a finger coordinate for explicit state and lock control.

        Parameters
        ----------
        side : {'left', 'right'}
            Anatomical side.
        finger : {'thumb', 'index', 'middle', 'ring', 'little'}
            Digit containing the joint.
        joint : {'cmc', 'mcp', 'pip', 'dip', 'ip'}
            Joint name. Available joints depend on the digit.
        motion : {'flexion', 'abduction', 'opposition'}
            Coordinate name. Available motions depend on the joint.

        Returns
        -------
        opensim.Coordinate
            The coordinate belonging to this user.

        Raises
        ------
        ValueError
            If the side or the requested coordinate is not available.
        """
        # Resolve the requested anatomical side and validate the coordinate name.
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        name = f"{finger}_{joint}_{motion}_{'l' if side == 'left' else 'r'}"
        coordinates = self.getCoordinateSet()
        if not coordinates.contains(name):
            raise ValueError(f"Unknown finger coordinate {name!r}")
        return coordinates.get(name)

    def set_finger_coordinate_locked(
        self, side: str, finger: str, joint: str, motion: str, locked: bool
    ) -> None:
        """Set a finger coordinate's current and default lock state.

        Parameters
        ----------
        side, finger, joint, motion : str
            Identify a coordinate as described by :meth:`finger_coordinate`.
        locked : bool
            Whether the coordinate should be locked.

        Raises
        ------
        ValueError
            If the coordinate is unavailable or ``locked`` is not a boolean.
        """
        # Validate the lock request and apply it to both the model and state.
        if not isinstance(locked, bool):
            raise ValueError("locked must be a bool")
        coordinate = self.finger_coordinate(side, finger, joint, motion)
        coordinate.set_locked(locked)
        coordinate.setLocked(self.state, locked)

    def set_finger_angle(
        self, side: str, finger: str, joint: str, motion: str, degrees: float
    ) -> None:
        """Set an unlocked finger coordinate relative to the corrected reference pose.

        Parameters
        ----------
        side, finger, joint, motion : str
            Identify a coordinate as described by :meth:`finger_coordinate`.
        degrees : float
            Coordinate value in degrees relative to the corrected reference pose.

        Raises
        ------
        ValueError
            If the coordinate is locked or the value is nonfinite or outside
            the approximate coordinate range.
        """
        # Resolve the coordinate and validate its lock state and requested angle.
        coordinate = self.finger_coordinate(side, finger, joint, motion)
        if not math.isfinite(degrees):
            raise ValueError("finger angle must be finite")
        if coordinate.getLocked(self.state):
            raise ValueError(f"Finger coordinate {coordinate.getName()!r} is locked")
        lower = coordinate.getRangeMin()
        upper = coordinate.getRangeMax()
        lower_degrees = math.degrees(lower)
        upper_degrees = math.degrees(upper)
        if not lower_degrees <= degrees <= upper_degrees:
            raise ValueError(f"Finger angle is outside the range of {coordinate.getName()!r}")
        # Preserve exact bounds when converting endpoint values back to radians.
        if degrees == lower_degrees:
            angle = lower
        elif degrees == upper_degrees:
            angle = upper
        else:
            angle = math.radians(degrees)
        coordinate.setValue(self.state, angle, False)

    def finger_angle(self, side: str, finger: str, joint: str, motion: str) -> float:
        """Return a finger coordinate in degrees relative to the reference pose.

        Parameters
        ----------
        side, finger, joint, motion : str
            Identify a coordinate as described by :meth:`finger_coordinate`.

        Returns
        -------
        float
            Coordinate value in degrees relative to the corrected reference pose.

        Raises
        ------
        ValueError
            If the requested coordinate is unavailable.
        """
        coordinate = self.finger_coordinate(side, finger, joint, motion)
        return math.degrees(coordinate.getValue(self.state))

    def hand_surface_intersections(
        self, side: str
    ) -> tuple[tuple[str, str], ...]:
        """Find intersections between bone surfaces on different hand bodies.

        Parameters
        ----------
        side : {'left', 'right'}
            Hand to inspect in the current user state.

        Returns
        -------
        tuple of tuple of str
            Pairs of intersecting mesh filenames. A pair is also reported
            when its surfaces touch.

        Raises
        ------
        ValueError
            If the side, mesh, or mesh scale is invalid.
        FileNotFoundError
            If a referenced bone mesh cannot be found in the registered
            geometry directories.

        Notes
        -----
        This checks triangle-surface intersections only. It does not detect
        complete containment without surface intersection, distinguish
        tangency from penetration, or model skin and soft tissue. The meshes
        belonging to the same rigid body are not compared. It neither changes
        the pose nor enforces a collision-free trajectory.
        """
        # Resolve the selected side and realize the current body transforms.
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        suffix = "l" if side == "left" else "r"
        self.realizePosition(self.state)
        surfaces = []

        # Load the palm and finger meshes and build their transforms to ground.
        for name in ("hand", *MOVING_HAND_SEGMENTS):
            body = self.getBodySet().get(f"{name}_{suffix}")
            count = body.getPropertyByName("attached_geometry").size()
            if count == 0:
                raise ValueError(f"Body {body.getName()!r} has no attached mesh")
            for index in range(count):
                mesh = self.opensim.Mesh.safeDownCast(body.get_attached_geometry(index))
                if mesh is None:
                    raise ValueError(f"Non-mesh geometry on body {body.getName()!r}")
                polydata, matrix = _load_mesh_surface(
                    mesh, self.state, self.geometry_directories, "Hand"
                )
                surfaces.append((body.getName(), mesh.get_mesh_file(), polydata, matrix))

        # Compare different bodies and report the mesh pairs that touch or cross.
        return _surfaces_intersecting_pairs(surfaces)

    def hand_handle_intersections(
        self, side: str, handle: Any
    ) -> tuple[tuple[str, str], ...]:
        """Find exact triangle-surface intersections between a hand and a grip body.

        Parameters
        ----------
        side : {'left', 'right'}
            Hand to inspect in the current user state.
        handle : opensim.Body
            Grip body belonging to a separately initialized OpenSim model
            (for example a piece of equipment). Its pose is read from that
            model's own working state (``handle.getModel().getWorkingState()``),
            not from this user's state, so `handle` may come from a Python
            wrapper distinct from whichever object last moved its model.

        Returns
        -------
        tuple of tuple of str
            Pairs of ``(hand mesh filename, handle mesh filename)`` whose
            surfaces touch or cross.

        Raises
        ------
        ValueError
            If `side` is invalid, `handle` is not connected to an
            initialized model, or a mesh/scale is invalid.
        TypeError
            If `handle` is not an ``opensim.Body``.
        FileNotFoundError
            If a referenced mesh cannot be resolved. Hand meshes are searched
            in :attr:`geometry_directories`; the handle's mesh file must be
            an absolute path or resolvable relative to the current working
            directory -- this does not search directories registered for the
            handle's own model.

        Notes
        -----
        Same limitations as :meth:`hand_surface_intersections`: triangle
        surface intersections only, no skin/soft-tissue model, no check of
        intermediate poses along a trajectory.
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        suffix = "l" if side == "left" else "r"
        if not isinstance(handle, self.opensim.Body):
            raise TypeError("handle must be an opensim.Body")
        if not handle.hasModel():
            raise ValueError("handle is not connected to an initialized OpenSim model")
        handle_model = handle.getModel()
        handle_state = handle_model.getWorkingState()
        handle_model.realizePosition(handle_state)

        self.realizePosition(self.state)
        hand_surfaces = []
        for name in ("hand", *MOVING_HAND_SEGMENTS):
            body = self.getBodySet().get(f"{name}_{suffix}")
            count = body.getPropertyByName("attached_geometry").size()
            if count == 0:
                raise ValueError(f"Body {body.getName()!r} has no attached mesh")
            for index in range(count):
                mesh = self.opensim.Mesh.safeDownCast(body.get_attached_geometry(index))
                if mesh is None:
                    raise ValueError(f"Non-mesh geometry on body {body.getName()!r}")
                polydata, matrix = _load_mesh_surface(
                    mesh, self.state, self.geometry_directories, "Hand"
                )
                hand_surfaces.append((body.getName(), mesh.get_mesh_file(), polydata, matrix))

        handle_count = handle.getPropertyByName("attached_geometry").size()
        if handle_count == 0:
            raise ValueError(f"Body {handle.getName()!r} has no attached mesh")
        handle_surfaces = []
        for index in range(handle_count):
            mesh = self.opensim.Mesh.safeDownCast(handle.get_attached_geometry(index))
            if mesh is None:
                raise ValueError(f"Non-mesh geometry on body {handle.getName()!r}")
            polydata, matrix = _load_mesh_surface(mesh, handle_state, (), "Handle")
            handle_surfaces.append((handle.getName(), mesh.get_mesh_file(), polydata, matrix))

        return _surfaces_intersecting_pairs(hand_surfaces, handle_surfaces)

    def _fit_handle_grip(
        self,
        handle: Any,
        function: Literal["push", "pull"] = "push",
        soft_tissues_approximation_distance: float = 0.01,
    ) -> dict[str, Any]:
        """Wrap one hand around a grip body without shoulder/elbow motion.

        This is the original ``fit_handle`` solver, validated first and kept
        unchanged: a cheap-PCA-capsule-proxy SLSQP engine that only unlocks
        the wrist, forearm and fingers, assuming the arm is already coarsely
        positioned near `handle`. It is now a private engine reused by the
        public :meth:`fit_handle` as that method's final stage (after
        shoulder/elbow/wrist have already been moved close to `handle`), but
        every detail below -- its push/pull strategies, its side-inference
        rule and its capsule-proxy/exact-mesh guarantees -- is unchanged and
        still applies exactly as described, whether called directly (if an
        arm is already known to be close to `handle`) or through
        :meth:`fit_handle`.

        Only the wrist (flexion and deviation), forearm pronation/supination
        and every finger coordinate of the inferred hand are unlocked and
        optimized; shoulder and elbow coordinates are never touched. The side
        is not a parameter: it is inferred as whichever palm marker
        (:attr:`palm_markers`) is nearest `handle`'s capsule centre. Call this
        after positioning the arm near `handle` (for example with an
        equipment-specific ``fit_user``, or with :meth:`fit_handle` itself);
        it does not search for the handle.

        For ``function="push"``, the palm (its four metacarpal-head capsules)
        is held in hard contact with `handle` within
        `soft_tissues_approximation_distance`, and finger wrap is a soft
        objective (not a hard "all phalanges touch" requirement, which an
        earlier exploratory study found infeasible in every tested case).
        The thumb participates in the same wrap objective as the other
        fingers (a stabilizing preference, not a forced target), and MCP
        abduction is softly pulled towards neutral/adduction.

        For ``function="pull"``, finger wrap is the primary soft objective
        (hook grip), the thumb's contribution to it is reduced (secondary,
        neutral role), and bringing the palm closer to `handle` is only a
        secondary soft objective, never a hard contact requirement.

        In both modes, no finger-finger, finger-palm or finger/palm-`handle`
        penetration is allowed (hard constraints); the exact trajectory
        between poses is not checked.

        Parameters
        ----------
        handle : opensim.Body
            Grip body belonging to a separately initialized OpenSim model
            (for example a piece of equipment). Its pose is read from that
            model's own working state, not from this user's state.
        function : {'push', 'pull'}, default='push'
            Grip strategy; see above for the difference.
        soft_tissues_approximation_distance : float, default=0.01
            Allowed gap, in metres, between a bone-capsule proxy and
            `handle`'s capsule, approximating the skin thickness that lets
            bone stop short of the rigid surface while still being "in
            contact" through it. This is an unvalidated placeholder (no
            source was found to justify a specific skin thickness for this
            use): it is applied only to the palmar/volar bone-to-`handle`
            distance (the push contact constraint and the wrap objective's
            target gap), isotropically around each capsule rather than only
            along the true bone-to-skin outward normal, and never to the
            finger-finger/finger-palm self-contact checks, which keep zero
            clearance.

        Returns
        -------
        dict
            Report with keys ``side`` (the inferred ``'left'``/``'right'``),
            ``function``, ``soft_tissues_approximation_distance_m``,
            ``solver_success``, ``wrap_objective`` (final soft-objective
            value), ``finger_handle_gaps_m`` and ``palm_handle_gaps_m``
            (per-capsule bone-to-`handle` capsule gaps in metres, before
            subtracting `soft_tissues_approximation_distance`), and
            ``minimum_self_gap_m``/``minimum_handle_gap_m`` (worst-case
            capsule-proxy clearances actually achieved).

        Raises
        ------
        TypeError
            If `handle` is not an ``opensim.Body``.
        ValueError
            If `function` or `soft_tissues_approximation_distance` is
            invalid, `handle` is not connected to an initialized model, or
            neither palm is unambiguously close to `handle`.
        FileNotFoundError
            If a referenced mesh cannot be resolved (see
            :meth:`hand_handle_intersections`).
        RuntimeError
            If no solver start satisfies the capsule-proxy hard constraints
            within tolerance, or if the final exact triangle-surface check
            (:meth:`hand_surface_intersections`, :meth:`hand_handle_intersections`)
            still reports an intersection; the state is left at the best
            (infeasible) solution found, in place, rather than restored.

        Notes
        -----
        Self-collision and finger/palm-to-`handle` non-penetration are
        enforced in the solver loop with a cheap PCA-capsule proxy (one
        capsule per attached bone mesh and one for the whole of `handle`),
        not the exact mesh; capsules can overestimate a non-convex or
        strongly tapered bone, so a capsule pair across (or near) an
        anatomical joint typically already overlaps at the locked reference
        pose even though the bone meshes do not. Rather than excluding such
        pairs outright, each pair's own reference-pose gap is used as its
        floor whenever that gap is already at or below zero (see
        ``_self_collision_pairs``), so the solver may not make the overlap
        worse but is not blocked by overlap that was already present and
        independently verified harmless at the start.

        An upstream arm/wrist fit may only guarantee a marker-level distance
        to `handle`, not that the palm and finger bones actually clear it
        (confirmed on a real equipment model, whose own contact marker sits
        roughly 15 mm inside the handle mesh): the hand-to-`handle` capsule
        gaps, unlike self-collision, keep the ordinary zero floor (there is
        no independent evidence the starting overlap, if any, is harmless),
        so before running the solver a coarse grid search over only the
        wrist/forearm coordinates (the only ones that move every hand
        capsule, including the palm) looks for a configuration that already
        clears `handle`, and the full SLSQP search for wrist and fingers
        together is started from there in addition to the original pose.
        This still cannot resolve an overlap that only shoulder/elbow motion
        (outside this method's scope, but within :meth:`fit_handle`'s) could
        clear; such cases surface as the ``RuntimeError`` below.

        The exact triangle-surface check only runs once, after the solver
        converges, as an accept/reject gate; it is not re-applied along a
        motion.
        """
        if function not in ("push", "pull"):
            raise ValueError("function must be 'push' or 'pull'")
        soft_tissue = _validate_soft_tissue_distance(soft_tissues_approximation_distance)

        # Infer which hand to fit from palm-marker proximity to the handle.
        handle_start, handle_end, handle_radius = _handle_capsule(handle, "Handle")
        handle_center = (handle_start + handle_end) / 2
        palm_markers = self.palm_markers
        distances = {
            name: float(np.linalg.norm(np.array(position) - handle_center))
            for name, position in palm_markers.items()
        }
        nearest_side = min(distances, key=distances.get)
        other_side = "left" if nearest_side == "right" else "right"
        if (
            distances[nearest_side] > _SIDE_INFERENCE_MAX_DISTANCE_M
            or distances[other_side] - distances[nearest_side] < _SIDE_INFERENCE_MARGIN_M
        ):
            raise ValueError(
                f"Cannot determine which hand to fit to handle {handle.getName()!r}: "
                f"right palm is {distances['right']:.4f} m and left palm is "
                f"{distances['left']:.4f} m from its capsule centre. Position the arm "
                "near the handle (for example with an equipment fit_user, or with "
                "fit_handle itself, which can reposition the arm from further away) "
                "before calling _fit_handle_grip."
            )
        side = nearest_side
        suffix = "l" if side == "left" else "r"

        # Unlock the wrist/forearm and every finger coordinate of this hand.
        wrist_coordinates = []
        for name in _WRIST_COORDINATE_SUFFIXES:
            coordinate = self.getCoordinateSet().get(f"{name}_{suffix}")
            coordinate.set_locked(False)
            coordinate.setLocked(self.state, False)
            wrist_coordinates.append(coordinate)
        finger_coordinates = []
        for finger, joint, motion in _FINGER_COORDINATE_KEYS:
            self.set_finger_coordinate_locked(side, finger, joint, motion, False)
            finger_coordinates.append(self.finger_coordinate(side, finger, joint, motion))
        coordinates = wrist_coordinates + finger_coordinates
        # Unlocking changes the system's Model stage; re-realize before reading
        # any transform or coordinate value below.
        self.realizePosition(self.state)
        bounds = [(c.getRangeMin(), c.getRangeMax()) for c in coordinates]
        bound_array = np.array(bounds)
        x0 = np.array([c.getValue(self.state) for c in coordinates])

        # Build the capsule proxies: hand bones at the reference pose for the
        # self-collision adjacency, and the handle once (it does not move).
        hand_capsules = _collect_hand_capsules(self, suffix, "Hand")
        pairs, self_gap_floor = _self_collision_pairs(hand_capsules, self.state)
        hand_radii = np.array([capsule.radius for capsule in hand_capsules])
        capsule_names = [capsule.name for capsule in hand_capsules]
        wrap_indices = np.array([capsule_names.index(name) for name in _WRAP_CAPSULES])
        palm_contact_indices = np.array(
            [capsule_names.index(name) for name in _PALM_CONTACT_CAPSULES]
        )
        abduction_indices = np.array(
            [
                len(wrist_coordinates) + _FINGER_COORDINATE_KEYS.index((finger, "mcp", "abduction"))
                for finger in FINGERS
            ]
        )
        thumb_wrap_positions = np.array(
            [
                list(_WRAP_CAPSULES).index(name)
                for name in ("metacarpal1", "thumb_proximal", "thumb_distal")
            ]
        )
        wrap_weights = np.ones(len(_WRAP_CAPSULES))
        if function == "pull":
            wrap_weights[thumb_wrap_positions] = _PULL_THUMB_WRAP_WEIGHT

        cache: dict[str, Any] = {}

        def apply(x: np.ndarray) -> None:
            for coordinate, value in zip(coordinates, x):
                coordinate.setValue(self.state, float(value), False)
            self.realizePosition(self.state)

        def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if "x" in cache and np.array_equal(x, cache["x"]):
                return cache["value"]
            apply(x)
            starts, ends = _capsule_segments_in_ground(hand_capsules, self.state)
            self_gaps = (
                segment_pair_distance(
                    starts[pairs[:, 0]], ends[pairs[:, 0]], starts[pairs[:, 1]], ends[pairs[:, 1]]
                )
                - hand_radii[pairs[:, 0]]
                - hand_radii[pairs[:, 1]]
            )
            handle_gaps = (
                segment_pair_distance(
                    starts,
                    ends,
                    np.broadcast_to(handle_start, starts.shape),
                    np.broadcast_to(handle_end, ends.shape),
                )
                - hand_radii
                - handle_radius
            )
            cache["x"], cache["value"] = x.copy(), (self_gaps, handle_gaps)
            return cache["value"]

        def constraints_fun(x: np.ndarray) -> np.ndarray:
            self_gaps, handle_gaps = evaluate(x)
            parts = [self_gaps - self_gap_floor, handle_gaps]
            if function == "push":
                parts.append(soft_tissue - handle_gaps[palm_contact_indices])
            return np.concatenate(parts)

        def objective_fun(x: np.ndarray) -> float:
            _, handle_gaps = evaluate(x)
            wrap_gaps = handle_gaps[wrap_indices] - soft_tissue
            wrap_term = float(np.sum(wrap_weights * wrap_gaps**2))
            regularization = _WRAP_REGULARIZATION_WEIGHT * float(np.dot(x - x0, x - x0))
            if function == "push":
                extra = _PUSH_ADDUCTION_WEIGHT * float(np.sum(x[abduction_indices] ** 2))
            else:
                extra = _PULL_PALM_APPROACH_WEIGHT * float(
                    np.mean(handle_gaps[palm_contact_indices])
                )
            return wrap_term + regularization + extra

        # An upstream arm/wrist fit may only guarantee a marker-level distance,
        # not that the palm and finger bones actually clear `handle` (confirmed
        # on a real equipment model, whose own contact marker sits roughly
        # 15 mm inside the handle mesh): the wrist-only starting pose handed to
        # this method can already overlap `handle`. Since the wrist/forearm
        # coordinates alone move every hand capsule (finger coordinates move
        # only the fingers, never the palm), a grid search over just those 3
        # coordinates cheaply finds a wrist pose with the least combined
        # constraint violation, if a fully clearing one exists, before the
        # full SLSQP search refines wrist and fingers together from there.
        # For push, simply maximizing clearance can overshoot the palm-contact
        # upper bound by tens of millimetres (observed on a real handle) and
        # leave SLSQP no better off than the original pose; the score below
        # instead balances non-penetration against that upper bound so the
        # starting point already lies inside, or very close to, the full
        # feasible region.
        #
        # The 3-coordinate constraint is typically under-determined, so many
        # wrist poses can be equally (near-)feasible for the palm alone: on a
        # real handle, picking arbitrarily among them by violation alone once
        # left the solver pinned within 1.1% of the wrist deviation limit,
        # while a wrist pose with much more range-of-motion headroom was
        # equally feasible and led SLSQP to a markedly better final wrap
        # quality besides. Among poses within `_WRIST_GRID_VIOLATION_ATOL` of
        # the minimum violation found, the one with the largest minimum
        # fractional margin to its own coordinate limits is kept; the
        # violation-only minimizer is also kept as a second, independent
        # start, since the two can differ and either may refine better.
        wrist_count = len(wrist_coordinates)
        wrist_bounds = bound_array[:wrist_count]
        wrist_spans = wrist_bounds[:, 1] - wrist_bounds[:, 0]
        wrist_grid_fractions = np.linspace(0.0, 1.0, _WRIST_GRID_POINTS_PER_AXIS)
        grid_candidates, grid_violations, grid_margins = [], [], []
        for alpha in wrist_grid_fractions:
            for beta in wrist_grid_fractions:
                for gamma in wrist_grid_fractions:
                    candidate = wrist_bounds[:, 0] + np.array((alpha, beta, gamma)) * wrist_spans
                    for coordinate, value in zip(wrist_coordinates, candidate):
                        coordinate.setValue(self.state, float(value), False)
                    self.realizePosition(self.state)
                    grid_starts, grid_ends = _capsule_segments_in_ground(hand_capsules, self.state)
                    grid_gaps = (
                        segment_pair_distance(
                            grid_starts,
                            grid_ends,
                            np.broadcast_to(handle_start, grid_starts.shape),
                            np.broadcast_to(handle_end, grid_ends.shape),
                        )
                        - hand_radii
                        - handle_radius
                    )
                    violation = max(0.0, -float(np.min(grid_gaps)))
                    if function == "push":
                        violation += max(
                            0.0, float(np.max(grid_gaps[palm_contact_indices])) - soft_tissue
                        )
                    # Fractional margin to the nearer bound, computed per
                    # coordinate (each has its own span) before taking the
                    # worst case across the three -- an absolute distance is
                    # not comparable between coordinates with different spans.
                    fractional_margins = (
                        np.minimum(candidate - wrist_bounds[:, 0], wrist_bounds[:, 1] - candidate)
                        / wrist_spans
                    )
                    margin = float(np.min(fractional_margins))
                    grid_candidates.append(candidate)
                    grid_violations.append(violation)
                    grid_margins.append(margin)
        cache.clear()

        grid_violations = np.array(grid_violations)
        grid_margins = np.array(grid_margins)
        least_violation_index = int(np.argmin(grid_violations))
        near_minimal = grid_violations <= grid_violations[least_violation_index] + _WRIST_GRID_VIOLATION_ATOL
        best_margin_index = int(np.flatnonzero(near_minimal)[np.argmax(grid_margins[near_minimal])])
        wrist_candidates = [grid_candidates[best_margin_index]]
        if best_margin_index != least_violation_index:
            wrist_candidates.append(grid_candidates[least_violation_index])

        # Try the reference pose, each clearing wrist candidate on its own,
        # and each clearing wrist candidate combined with a half-flexed
        # finger pose.
        starts = [x0]
        for wrist_values in wrist_candidates:
            cleared = x0.copy()
            cleared[:wrist_count] = wrist_values
            cleared = np.clip(cleared, bound_array[:, 0], bound_array[:, 1])
            starts.append(cleared)
            flexed = cleared.copy()
            for offset, (_, _, motion) in enumerate(_FINGER_COORDINATE_KEYS):
                if motion == "flexion":
                    index = wrist_count + offset
                    lower, upper = bounds[index]
                    flexed[index] = lower + 0.5 * (upper - lower)
            starts.append(np.clip(flexed, bound_array[:, 0], bound_array[:, 1]))

        solutions = []
        for start in starts:
            result = minimize(
                objective_fun,
                start,
                method="SLSQP",
                bounds=bounds,
                constraints=[{"type": "ineq", "fun": constraints_fun}],
                options={"maxiter": 200, "ftol": 1e-9},
            )
            residual = constraints_fun(result.x)
            if result.success and np.min(residual) >= -_FIT_HANDLE_CONSTRAINT_TOLERANCE_M:
                solutions.append(result)
        if not solutions:
            apply(starts[0])
            raise RuntimeError(
                f"Unable to fit a {function} pose for handle {handle.getName()!r} "
                "within the capsule-proxy constraints"
            )
        best = min(solutions, key=lambda item: item.fun)
        apply(best.x)

        # Final accept/reject gate: exact triangle-surface intersection checks.
        self_intersections = self.hand_surface_intersections(side)
        handle_intersections = self.hand_handle_intersections(side, handle)
        if self_intersections or handle_intersections:
            raise RuntimeError(
                "Grip solution failed the exact surface-intersection check: "
                f"{len(self_intersections)} self intersection(s), "
                f"{len(handle_intersections)} hand-handle intersection(s)"
            )

        self_gaps, handle_gaps = evaluate(best.x)
        return {
            "side": side,
            "function": function,
            "soft_tissues_approximation_distance_m": soft_tissue,
            "solver_success": True,
            "wrap_objective": float(best.fun),
            "finger_handle_gaps_m": dict(
                zip(_WRAP_CAPSULES, handle_gaps[wrap_indices].tolist())
            ),
            "palm_handle_gaps_m": dict(
                zip(_PALM_CONTACT_CAPSULES, handle_gaps[palm_contact_indices].tolist())
            ),
            "minimum_self_gap_m": float(np.min(self_gaps)),
            "minimum_handle_gap_m": float(np.min(handle_gaps)),
        }

    def fit_handle(
        self,
        handle: Any,
        side: Literal["left", "right"],
        function: Literal["push", "pull"] = "push",
        soft_tissues_approximation_distance: float = 0.01,
    ) -> dict[str, Any]:
        """Fit shoulder, elbow, forearm, wrist and finger coordinates to a grip body.

        This is the framework's primary public grip-fitting entry point. It
        extends :meth:`_fit_handle_grip` -- the original ``fit_handle``,
        validated first and kept unchanged as the internal engine for its
        own stage below -- with three stages ahead of it that also move the
        shoulder (``arm_flex``, ``arm_add``, ``arm_rot``) and elbow
        (``elbow_flex``), so no prior arm positioning is required: only
        `handle` itself must already be placed in its own model's working
        state (for example by an equipment-specific lever mechanism). `side`
        is an explicit parameter here, not inferred by palm-marker proximity
        like :meth:`_fit_handle_grip`: before this method runs, the arm can
        be arbitrarily far from `handle`, which is exactly the regime where
        that proximity-based inference (kept unchanged in
        :meth:`_fit_handle_grip` itself, still appropriate once the arm is
        already coarsely positioned) is not reliable.

        The fit proceeds in four stages:

        1. A least-squares cylinder (axis direction, a point on the axis,
           and radius; :func:`_handle_cylinder`) is fitted to `handle`'s
           attached mesh vertices, together with the real extent of those
           vertices along the fitted axis (a finite segment, not the
           mathematically infinite line the axis direction alone implies).
        2. A least-squares line is fitted to the middle/ring/little MCP
           joint centres (:func:`_mcp_axis_in_ground`); it moves rigidly
           with the hand body under any shoulder/elbow/forearm/wrist
           coordinate change, independent of the current finger pose. One
           SLSQP run, over the seven continuous shoulder/elbow/forearm/wrist
           coordinates, minimizes this line's distance to the cylinder's
           finite axis *segment* plus its angular misalignment to the
           (sign-free) axis direction, regularized towards the pose
           `fit_handle` was called with.
        3. A second SLSQP run, from stage 2's solution, brings the palm
           marker (``{prefix}PALM``) to tangency with the cylinder (its
           distance to the axis equal to the fitted radius), regularized
           towards stage 2's pose, over the same seven coordinates.
        4. :meth:`_fit_handle_grip` refines the wrist, forearm and finger
           coordinates against the real palm, finger and handle meshes: it
           already enforces a palmar contact target (gap at most
           `soft_tissues_approximation_distance`, for ``function="push"``)
           as a hard capsule-proxy constraint, finger-wrap closure as a soft
           objective, and the exact mesh non-interpenetration gate -- all
           reused unchanged here, not reimplemented.

        Parameters
        ----------
        handle : opensim.Body
            Grip body belonging to a separately initialized OpenSim model,
            with its pose already set in that model's own working state.
        side : {'left', 'right'}
            Hand to fit; not inferred (see above).
        function : {'push', 'pull'}, default='push'
            Forwarded to the stage-4 :meth:`_fit_handle_grip` call; see its
            docstring for the push/pull difference. Only that final stage
            uses it (stages 1-3 only align the arm to the handle geometry,
            independent of grip strategy).
        soft_tissues_approximation_distance : float, default=0.01
            Forwarded to the stage-4 :meth:`_fit_handle_grip` call; see its
            docstring.

        Returns
        -------
        dict
            Report with keys ``side``, ``cylinder_axis_point_m``,
            ``cylinder_axis_direction``, ``cylinder_radius_m`` (stage 1),
            ``axis_alignment_distance_m`` and
            ``axis_alignment_angle_error_deg`` (stage 2 residuals),
            ``palm_axis_gap_m`` (stage 3 residual: the palm's perpendicular
            distance to the cylinder axis minus its radius, near zero at
            tangency), ``full_arm_coordinates_rad`` (stage 3 solution for
            the seven shoulder/elbow/forearm/wrist coordinates, keyed by
            coordinate name), ``grip`` (the full dict returned by the
            stage-4 :meth:`_fit_handle_grip` call) and
            ``palm_axis_gap_after_wrap_m`` (the same signed gap recomputed
            after stage 4; diagnostic only, see Notes).

        Raises
        ------
        TypeError
            If `handle` is not an ``opensim.Body``.
        ValueError
            If `side`, `function` or `soft_tissues_approximation_distance`
            is invalid, or if `handle`'s mesh or the MCP joint centres are
            degenerate.
        RuntimeError
            If the cylinder fit does not converge, if either SLSQP stage
            fails to converge, if the stage-4 :meth:`_fit_handle_grip` call
            infers the wrong side (the stage-3 pose placed the opposite palm
            closer to `handle`), or if :meth:`_fit_handle_grip` itself
            raises (no feasible capsule-proxy wrap, or a failed exact-mesh
            gate).

        Notes
        -----
        Validated so far on the complete High Row reference dataset (ANSUR
        M/F x 150/175/205 cm, both sides, neutral grip, ``function="push"``):
        all 12 neutral-grip fits converged with zero mesh intersection and
        sub-millimetre residuals. The supine grip on that same equipment
        failed in every one of the 6 tested cases with ``function="push"``
        (stage 4 could not find a feasible hard-palm-contact pose); this is
        consistent with a previously observed push/pull asymmetry on
        supinated handles (see the project's grip-ergonomics notes) and is
        expected to require ``function="pull"`` for that handle -- not yet
        re-verified end to end at the time of this note. See this module's
        test coverage before relying on this method for other anthropometry
        or equipment. Stages 2 and 3 each run a single SLSQP from the pose
        `fit_handle` was called with (no multistart), unlike
        :meth:`_fit_handle_grip`'s own wrist/finger search; a poor starting
        pose could converge to a worse local optimum or fail outright.
        Coordinate ranges are read as currently set on the model and never
        widened (as some equipment-specific coarse arm fits do for
        ``arm_flex``); a `handle` unreachable within the current range fails
        stage 2 or 3 rather than silently ignoring the limit. Like
        :meth:`_fit_handle_grip`, this method does not lock any coordinate
        back after moving it -- a caller's own pipeline is responsible for
        that. It does not check for self-collision between the
        forearm/upper arm and the rest of the body or the equipment frame,
        the same unaddressed scope as the existing coarse arm fits in this
        project. `palm_axis_gap_after_wrap_m` is purely diagnostic:
        :meth:`_fit_handle_grip`'s own hard capsule-proxy constraint and the
        exact mesh gate are the actual acceptance criteria for stage 4, so a
        small deviation here is expected and is not itself cause for
        failure.
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if function not in ("push", "pull"):
            raise ValueError("function must be 'push' or 'pull'")
        suffix = "l" if side == "left" else "r"
        prefix = "L" if side == "left" else "R"
        soft_tissue = _validate_soft_tissue_distance(soft_tissues_approximation_distance)

        # Stage 0: least-squares cylinder fit to the handle mesh, with the
        # real vertex extent along the fitted axis (see _handle_cylinder).
        axis_point, axis, radius, segment_start, segment_end = _handle_cylinder(
            handle, "Handle"
        )

        # Unlock the seven shoulder/elbow/forearm/wrist coordinates moved by
        # stages 2-3 (finger coordinates are unlocked later, by stage 4's
        # own _fit_handle_grip call).
        coordinates = [
            self.getCoordinateSet().get(f"{name}_{suffix}")
            for name in _FULL_ARM_COORDINATE_SUFFIXES
        ]
        for coordinate in coordinates:
            coordinate.set_locked(False)
            coordinate.setLocked(self.state, False)
        self.realizePosition(self.state)
        bounds = [(c.getRangeMin(), c.getRangeMax()) for c in coordinates]
        x0 = np.array([c.getValue(self.state) for c in coordinates])

        def apply(x: np.ndarray) -> None:
            for coordinate, value in zip(coordinates, x):
                coordinate.setValue(self.state, float(value), False)
            self.realizePosition(self.state)

        def axis_segment_distance(point: np.ndarray) -> float:
            return float(
                segment_pair_distance(
                    point[None, :], point[None, :], segment_start[None, :], segment_end[None, :]
                )[0]
            )

        # Stage 2: align the MCP line to the cylinder's finite axis segment.
        def step2_objective(x: np.ndarray) -> float:
            apply(x)
            point, direction = _mcp_axis_in_ground(self, suffix)
            distance = axis_segment_distance(point)
            angle_term = 1.0 - (direction @ axis) ** 2
            regularization = float(np.dot(x - x0, x - x0))
            return (
                _AXIS_DISTANCE_WEIGHT * distance**2
                + _AXIS_ANGLE_WEIGHT * angle_term
                + _AXIS_ALIGNMENT_REGULARIZATION_WEIGHT * regularization
            )

        stage2 = minimize(
            step2_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-10},
        )
        if not stage2.success:
            raise RuntimeError(
                "fit_handle stage 2 (MCP-axis alignment) failed for "
                f"handle {handle.getName()!r}: {stage2.message}"
            )
        apply(stage2.x)
        point, direction = _mcp_axis_in_ground(self, suffix)
        axis_alignment_distance = axis_segment_distance(point)
        axis_alignment_angle_error = math.degrees(
            math.acos(min(1.0, abs(float(direction @ axis))))
        )
        x1 = stage2.x

        # Stage 3: bring the palm marker to tangency with the cylinder.
        def palm_axis_gap() -> float:
            palm = np.array(self._marker_location(f"{prefix}PALM"))
            offset = palm - axis_point
            perpendicular = offset - (offset @ axis) * axis
            return float(np.linalg.norm(perpendicular)) - radius

        def step3_objective(x: np.ndarray) -> float:
            apply(x)
            gap = palm_axis_gap()
            regularization = float(np.dot(x - x1, x - x1))
            return gap**2 + _PALM_TANGENCY_REGULARIZATION_WEIGHT * regularization

        stage3 = minimize(
            step3_objective,
            x1,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-12},
        )
        if not stage3.success:
            raise RuntimeError(
                "fit_handle stage 3 (palm-cylinder tangency) failed "
                f"for handle {handle.getName()!r}: {stage3.message}"
            )
        apply(stage3.x)
        stage3_palm_axis_gap = palm_axis_gap()

        # Stage 4: delegate the wrist/finger wrap entirely to the existing,
        # already-validated _fit_handle_grip engine (already enforces the
        # palmar contact target for push, finger wrap as a soft objective,
        # and the exact mesh gate, for whichever `function` was requested).
        grip_report = self._fit_handle_grip(
            handle, function=function, soft_tissues_approximation_distance=soft_tissue
        )
        if grip_report["side"] != side:
            raise RuntimeError(
                f"_fit_handle_grip inferred side {grip_report['side']!r} while "
                f"fitting the {side!r} arm with fit_handle: the stage-3 pose "
                "placed the opposite palm closer to handle than the intended one"
            )

        return {
            "side": side,
            "cylinder_axis_point_m": tuple(axis_point.tolist()),
            "cylinder_axis_direction": tuple(axis.tolist()),
            "cylinder_radius_m": radius,
            "axis_alignment_distance_m": axis_alignment_distance,
            "axis_alignment_angle_error_deg": axis_alignment_angle_error,
            "palm_axis_gap_m": float(stage3_palm_axis_gap),
            "full_arm_coordinates_rad": dict(
                zip(
                    (f"{name}_{suffix}" for name in _FULL_ARM_COORDINATE_SUFFIXES),
                    stage3.x.tolist(),
                )
            ),
            "grip": grip_report,
            "palm_axis_gap_after_wrap_m": float(palm_axis_gap()),
        }
