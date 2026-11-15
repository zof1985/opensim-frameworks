"""Geometry helpers for point clouds, mesh transforms, lines, and surfaces."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .opensim import (
    import_opensim,
    read_vtp_vertices,
    rotation_to_numpy,
    vec3_to_numpy,
    body_mass_center_array,
    body_transform_array,
    numpy_to_vec3,
    translate_model,
)

__all__ = [
    "enclosing_capsule",
    "segment_pair_distance",
    "get_mesh_vertices_in_ground",
    "enclosing_parallelepiped",
    "fit_parallelepiped",
    "enclosing_cylinder",
    "fit_cylinder",
    "get_plane_from_axis_and_point",
    "get_projection_point_on_plane",
    "get_projection_point_on_axis",
    "get_distance_to_plane",
    "get_distance_to_axis",
    "get_distance_to_mesh",
    "get_ls_line",
    "get_angles_between_lines",
    "body_mass_center_array",
    "body_transform_array",
    "numpy_to_vec3",
    "translate_model",
]


def get_mesh_vertices_in_ground(
    model: Any,
    state: Any | None = None,
    body_name: str | None = None,
    directories: Sequence[str | Path] = (),
    *,
    owner_label: str = "Mesh",
) -> np.ndarray:
    """Return scaled mesh vertices attached to a body, expressed in Ground.

    ``model`` may be an initialized model with ``state`` and ``body_name``, or
    an ``opensim.Body`` directly. For a body, its owning model's working state
    is used when available; a detached body's local frame is treated as Ground.
    Relative mesh filenames are searched in ``directories`` in order.

    Parameters
    ----------
    model : opensim.Model or opensim.Body
        Initialized model or body containing the attached geometry.
    state : opensim.State or None, optional
        State at which model and mesh-frame transforms are evaluated. Required
        with a model; optional with a body.
    body_name : str or None, optional
        Name of the body to read when ``model`` is a model. Omit when a body is
        supplied directly.
    directories : sequence of str or pathlib.Path
        Directories searched for relative mesh filenames.
    owner_label : str, default="Mesh"
        Label used in missing-file errors.

    Returns
    -------
    numpy.ndarray, shape (n, 3)
        Scaled mesh vertices expressed in Ground, in metres.

    Raises
    ------
    ValueError
        If a mesh filename is empty, geometry is invalid, or the body has no
        attached mesh.
    FileNotFoundError
        If a mesh file cannot be found.
    """
    opensim = import_opensim()
    if isinstance(model, opensim.Body):
        body = model
        if body_name is not None:
            raise TypeError("body_name must be omitted when passing a body")
        owner_model = body.getModel() if body.hasModel() else None
        if owner_model is not None:
            if state is None:
                state = (
                    owner_model.getWorkingState()
                    if owner_model.hasSystem()
                    else owner_model.initSystem()
                )
    else:
        if state is None or body_name is None:
            raise TypeError("model calls require both state and body_name")
        model.realizePosition(state)
        body = model.getBodySet().get(body_name)

    if not body.hasModel():
        body.finalizeFromProperties()
    frames = [body]
    frames.extend(
        frame
        for component in body.getComponentsList()
        if (frame := opensim.PhysicalFrame.safeDownCast(component)) is not None
    )
    vertex_arrays = []
    label = body.getName() if body_name is None else body_name
    for frame in frames:
        geometry_property = frame.getPropertyByName("attached_geometry")
        for index in range(geometry_property.size()):
            mesh = opensim.Mesh.safeDownCast(frame.get_attached_geometry(index))
            if mesh is None:
                continue
            filename = Path(mesh.get_mesh_file()).expanduser()
            if not filename.name:
                raise ValueError(f"Empty mesh filename on body {label}")
            if not filename.is_absolute():
                resolved = next(
                    (
                        Path(directory) / filename
                        for directory in directories
                        if (Path(directory) / filename).is_file()
                    ),
                    None,
                )
                if resolved is None and not filename.is_file():
                    raise FileNotFoundError(
                        f"{owner_label} mesh not found: {filename}"
                    )
                filename = filename if resolved is None else resolved
            elif not filename.is_file():
                raise FileNotFoundError(f"{owner_label} mesh not found: {filename}")

            vertices = read_vtp_vertices(filename)
            factors = mesh.get_scale_factors()
            scale = np.array([factors.get(index) for index in range(3)])
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or not np.all(np.isfinite(vertices))
                or not np.all(np.isfinite(scale))
                or np.any(scale <= 0)
            ):
                raise ValueError(f"Invalid mesh geometry for body {label}")
            frame_transform = frame.findTransformInBaseFrame()
            frame_rotation = rotation_to_numpy(frame_transform.R())
            frame_translation = vec3_to_numpy(frame_transform.p())
            if body.hasModel() and state is not None:
                body_transform = body.getTransformInGround(state)
                body_rotation = rotation_to_numpy(body_transform.R())
                body_translation = vec3_to_numpy(body_transform.p())
                rotation = body_rotation @ frame_rotation
                translation = body_rotation @ frame_translation + body_translation
            else:
                rotation = frame_rotation
                translation = frame_translation
            vertex_arrays.append(
                (vertices * scale) @ rotation.T + translation
            )
    if not vertex_arrays:
        raise ValueError(f"No mesh is attached to body {label}")
    return np.vstack(vertex_arrays)


def enclosing_parallelepiped(
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a circumscribing oriented parallelepiped to 3D vertices.

    Principal component axes define a right-handed frame; the axis most aligned
    with ``+y`` is used as height, the smallest extent as depth, and the
    remaining axis as width.

    Parameters
    ----------
    vertices : numpy.ndarray, shape (n, 3)
        Non-empty vertices expressed in a common frame.

    Returns
    -------
    dimensions : numpy.ndarray, shape (3,)
        ``(depth, height, width)`` extents in metres.
    orientation : numpy.ndarray, shape (3, 3)
        Right-handed orientation matrix whose columns are the depth, height,
        and width axes in the input frame.
    center : numpy.ndarray, shape (3,)
        Centre of the fitted parallelepiped in the input frame.
    corners : numpy.ndarray, shape (8, 3)
        Its eight corners in the input frame.

    Raises
    ------
    ValueError
        If the vertices cannot define a three-dimensional parallelepiped.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise ValueError("mesh vertices must be an array with shape (n, 3)")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("mesh vertices must be finite")

    centered = vertices - vertices.mean(axis=0)
    covariance = centered.T @ centered / len(vertices)
    _, eigenvectors = np.linalg.eigh(covariance)
    axes = eigenvectors[:, ::-1]
    principal_extents = np.ptp(centered @ axes, axis=0)

    depth_index = int(np.argmin(principal_extents))
    panel_indices = [index for index in range(3) if index != depth_index]
    height_index = max(
        panel_indices,
        key=lambda index: (
            abs(float(axes[1, index])),
            float(principal_extents[index]),
        ),
    )
    width_index = next(index for index in panel_indices if index != height_index)

    height_axis = axes[:, height_index].copy()
    if height_axis[1] < 0.0 or (
        abs(height_axis[1]) <= 1e-12 and height_axis[0] < 0.0
    ):
        height_axis *= -1.0
    width_axis = axes[:, width_index].copy()
    if width_axis[2] < 0.0 or (
        abs(width_axis[2]) <= 1e-12 and width_axis[0] < 0.0
    ):
        width_axis *= -1.0
    depth_axis = np.cross(height_axis, width_axis)
    depth_axis /= np.linalg.norm(depth_axis)
    orientation = np.column_stack((depth_axis, height_axis, width_axis))

    projected = vertices @ orientation
    lower = projected.min(axis=0)
    upper = projected.max(axis=0)
    dimensions = upper - lower
    if np.any(dimensions <= 1e-9):
        raise ValueError("mesh must have non-zero width, height, and depth")
    center = orientation @ ((lower + upper) / 2.0)
    signs = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    corners = center + (signs * dimensions / 2.0) @ orientation.T
    return dimensions, orientation, center, corners


fit_parallelepiped = enclosing_parallelepiped


def _as_finite_vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    """Validate and convert a finite vector of a specified length."""
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return vector


def _line_parts(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate line coefficients and return its point and unit direction."""
    coefficients = _as_finite_vector(axis, 6, "axis")
    point = coefficients[:3]
    direction = coefficients[3:]
    scale = float(np.max(np.abs(direction)))
    if scale == 0.0:
        raise ValueError("axis direction must be non-zero")
    direction = direction / scale
    return point, direction / np.linalg.norm(direction)


def enclosing_cylinder(
    points: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Return a circular cylinder enclosing a point cloud.

    The cylinder axis follows the point cloud's principal axis with the largest
    variance. Its radius is the maximum radial distance from that axis, and
    its end disks are placed at the minimum and maximum axial projections.

    Parameters
    ----------
    points : numpy.ndarray, shape (n, 3)
        At least two finite vertices in a common coordinate frame.

    Returns
    -------
    tuple
        ``(axis, radius, endpoint_min, endpoint_max)``. ``axis`` contains the
        line coefficients ``[x0, y0, z0, dx, dy, dz]`` with unit direction.
        Each endpoint is the centre of an end disk through the vertices with
        the minimum or maximum projection onto the cylinder axis.

    Raises
    ------
    ValueError
        If the point cloud is invalid or does not define a cylinder with
        positive radius and length.
    """
    vertices = np.asarray(points, dtype=float)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or vertices.shape[0] < 2
        or not np.all(np.isfinite(vertices))
    ):
        raise ValueError("points must contain at least two finite 3D vertices")

    center = vertices.mean(axis=0)
    centered = vertices - center
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError("point cloud must span a nondegenerate cylinder")
    _, eigenvectors = np.linalg.eigh(centered.T @ centered / len(vertices))
    axis = eigenvectors[:, -1]
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0.0:
        axis = -axis
    axial_coordinates = centered @ axis
    radial_offsets = centered - np.outer(axial_coordinates, axis)
    radius = float(np.max(np.linalg.norm(radial_offsets, axis=1)))
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("point cloud does not define a positive-radius cylinder")
    axial_min = float(axial_coordinates.min())
    axial_max = float(axial_coordinates.max())
    if axial_max - axial_min <= 1e-12:
        raise ValueError("point cloud does not define a positive-length cylinder")
    endpoints = center + np.array([axial_min, axial_max])[:, np.newaxis] * axis
    axis_point = center + (axial_min + axial_max) / 2.0 * axis
    return (
        np.concatenate((axis_point, axis)).astype(float).flatten(),
        radius,
        endpoints[0],
        endpoints[1],
    )


fit_cylinder = enclosing_cylinder


def get_plane_from_axis_and_point(
    axis: np.ndarray,
    point: np.ndarray,
) -> np.ndarray:
    """Return normalized plane coefficients through a line and a point.

    The line uses ``[x0, y0, z0, dx, dy, dz]`` coefficients. The supplied
    point must not lie on the line, since that would leave the plane
    undetermined.

    Parameters
    ----------
    axis : array-like, shape (6,)
        Point and direction coefficients of the line.
    point : array-like, shape (3,)
        Point on the plane, not on the line.

    Returns
    -------
    numpy.ndarray, shape (4,)
        Normalized plane coefficients ``[a, b, c, d]``.

    Raises
    ------
    ValueError
        If the line is invalid or ``point`` lies on the line.
    """
    axis_point, direction = _line_parts(axis)
    plane_point = _as_finite_vector(point, 3, "point")
    normal = np.cross(direction, plane_point - axis_point)
    scale = float(np.max(np.abs(normal)))
    if scale == 0.0:
        raise ValueError("point must not lie on the axis")
    normal /= scale
    normal /= np.linalg.norm(normal)
    return np.append(normal, -float(normal @ axis_point))


def get_projection_point_on_plane(point: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Project a point perpendicularly onto a plane.

    Parameters
    ----------
    point : array-like, shape (3,)
        Point to project.
    plane : array-like, shape (4,)
        Plane coefficients ``[a, b, c, d]``.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Perpendicular projection of ``point`` onto ``plane``.

    Raises
    ------
    ValueError
        If the point or plane coefficients are invalid.
    """
    position = _as_finite_vector(point, 3, "point")
    coefficients = _as_finite_vector(plane, 4, "plane")
    scale = float(np.max(np.abs(coefficients[:3])))
    if scale == 0.0:
        raise ValueError("plane normal coefficients must be non-zero")
    normal = coefficients[:3] / scale
    offset = coefficients[3] / scale
    return position - normal * (
        float(normal @ position + offset) / float(normal @ normal)
    )


def get_projection_point_on_axis(point: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Project a point perpendicularly onto a line.

    Parameters
    ----------
    point : array-like, shape (3,)
        Point to project.
    axis : array-like, shape (6,)
        Line coefficients ``[x0, y0, z0, dx, dy, dz]``.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Perpendicular projection of ``point`` onto ``axis``.

    Raises
    ------
    ValueError
        If the point or line coefficients are invalid.
    """
    position = _as_finite_vector(point, 3, "point")
    axis_point, direction = _line_parts(axis)
    return axis_point + direction * float((position - axis_point) @ direction)


def get_distance_to_plane(plane: np.ndarray, point: np.ndarray) -> float:
    """Return the perpendicular distance from a point to a plane.

    Parameters
    ----------
    plane : array-like, shape (4,)
        Plane coefficients ``[a, b, c, d]``.
    point : array-like, shape (3,)
        Point whose distance is measured.

    Returns
    -------
    float
        Non-negative perpendicular distance in the input coordinate units.

    Raises
    ------
    ValueError
        If the point or plane coefficients are invalid.
    """
    position = _as_finite_vector(point, 3, "point")
    projection = get_projection_point_on_plane(position, plane)
    return float(np.linalg.norm(position - projection))


def get_distance_to_axis(axis: np.ndarray, point: np.ndarray) -> float:
    """Return the perpendicular distance from a point to a line.

    Parameters
    ----------
    axis : array-like, shape (6,)
        Line coefficients ``[x0, y0, z0, dx, dy, dz]``.
    point : array-like, shape (3,)
        Point whose distance is measured.

    Returns
    -------
    float
        Non-negative perpendicular distance in the input coordinate units.

    Raises
    ------
    ValueError
        If the point or line coefficients are invalid.
    """
    position = _as_finite_vector(point, 3, "point")
    projection = get_projection_point_on_axis(position, axis)
    return float(np.linalg.norm(position - projection))


def get_distance_to_mesh(
    point: np.ndarray,
    mesh: np.ndarray,
    *,
    cylinder_fit: tuple[np.ndarray, float, np.ndarray, np.ndarray] | None = None,
) -> float:
    """Return distance to the lateral surface of a mesh's enclosing cylinder.

    ``mesh`` is an ``(n, 3)`` array of vertices without face connectivity, so
    the surface is approximated by a cylinder enclosing those vertices. End
    caps are not modeled. A precomputed ``cylinder_fit`` can be supplied when
    evaluating many points against the same mesh.

    Parameters
    ----------
    point : array-like, shape (3,)
        Point whose distance from the enclosing cylindrical surface is measured.
    mesh : array-like, shape (n, 3)
        Mesh vertices in a common coordinate frame.
    cylinder_fit : tuple or None, keyword-only, default=None
        Optional precomputed result from :func:`enclosing_cylinder`.

    Returns
    -------
    float
        Distance to the enclosing cylinder's lateral surface, in mesh units.

    Raises
    ------
    ValueError
        If the point, mesh, or supplied cylinder fit is invalid.
    """
    position = _as_finite_vector(point, 3, "point")
    if cylinder_fit is None:
        vertices = np.asarray(mesh, dtype=float)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or vertices.shape[0] < 2
            or not np.all(np.isfinite(vertices))
        ):
            raise ValueError("mesh must contain at least two finite 3D vertices")
        cylinder_fit = enclosing_cylinder(vertices)
    axis, radius, endpoint_min, endpoint_max = cylinder_fit
    axis = _as_finite_vector(axis, 6, "cylinder_fit axis")
    endpoint_min = _as_finite_vector(endpoint_min, 3, "cylinder_fit endpoint_min")
    endpoint_max = _as_finite_vector(endpoint_max, 3, "cylinder_fit endpoint_max")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("cylinder_fit radius must be positive and finite")
    axis_point, direction = _line_parts(axis)
    endpoints = np.array([endpoint_min, endpoint_max])

    axial_limits = (endpoints - axis_point) @ direction
    axial_position = float((position - axis_point) @ direction)
    clamped_axial_position = float(
        np.clip(axial_position, axial_limits.min(), axial_limits.max())
    )
    axis_projection = axis_point + direction * clamped_axial_position
    radial = position - (axis_point + direction * axial_position)
    radial_distance = float(np.linalg.norm(radial))
    if radial_distance == 0.0:
        return float(
            np.hypot(radius, axial_position - clamped_axial_position)
        )
    intersection = axis_projection + radial * (radius / radial_distance)
    return float(np.linalg.norm(position - intersection))


def get_ls_line(*points: np.ndarray) -> np.ndarray:
    """Fit a 3D line by least squares and return ``[point, direction]``.

    Parameters
    ----------
    points : sequence of array-like
        At least two finite 3D points.

    Returns
    -------
    numpy.ndarray, shape (6,)
        Line coefficients ``[x0, y0, z0, dx, dy, dz]`` with unit direction.

    Raises
    ------
    ValueError
        If fewer than two points are supplied or all points coincide.
    """
    if len(points) < 2:
        raise ValueError("at least two points are required to fit a line")
    positions = np.asarray(
        [_as_finite_vector(point, 3, "point") for point in points],
        dtype=float,
    )
    centroid = positions.mean(axis=0)
    centered = positions - centroid
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    if not np.any(centered):
        raise ValueError("the points must not all coincide")
    direction = right_vectors[0]
    projection = float(direction @ (positions[-1] - positions[0]))
    if projection < 0.0:
        direction = -direction
    elif projection == 0.0:
        dominant_axis = int(np.argmax(np.abs(direction)))
        if direction[dominant_axis] < 0.0:
            direction = -direction
    return np.concatenate((centroid, direction))


def get_angles_between_lines(
    line1: np.ndarray, line2: np.ndarray, order: str = "XYZ"
) -> np.ndarray:
    """Return Euler angles for the shortest rotation from line1 to line2.

    Lines use ``[x0, y0, z0, dx, dy, dz]`` coefficients. Their direction
    vectors are treated as oriented. ``order`` is an intrinsic three-axis
    Euler order such as ``"XYZ"``.

    Parameters
    ----------
    line1 : array-like, shape (6,)
        Starting line coefficients.
    line2 : array-like, shape (6,)
        Target line coefficients.
    order : str, default="XYZ"
        Intrinsic Euler axis order, a permutation of ``"XYZ"``.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Euler rotation angles in degrees.

    Raises
    ------
    ValueError
        If either line is invalid or ``order`` is not an axis permutation.
    """
    if (
        not isinstance(order, str)
        or len(order) != 3
        or any(axis not in "XYZ" for axis in order)
        or len(set(order)) != 3
    ):
        raise ValueError("order must be a permutation of 'XYZ'")
    _, direction1 = _line_parts(line1)
    _, direction2 = _line_parts(line2)
    rotation, _ = Rotation.align_vectors(
        np.array([direction2]),
        np.array([direction1]),
    )
    return rotation.as_euler(order, degrees=True)


def enclosing_capsule(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Enclose a point cloud and its convex hull in one capsule.

    The capsule axis is the first principal component of the points. Its
    centre segment spans the projections of every point onto that axis, and
    its radius is the largest perpendicular distance from any point to the
    axis.

    Parameters
    ----------
    vertices : numpy.ndarray, shape (n, 3)
        Points in a single frame, in metres. Typically scaled mesh vertices.

    Returns
    -------
    start, end : numpy.ndarray, shape (3,)
        Extremes of the point projections onto the principal axis, in metres.
    radius : float
        Maximum point distance to that axis, in metres.

    Raises
    ------
    ValueError
        If fewer than three finite 3D points are given, the first two
        principal axes cannot be distinguished, or the resulting radius is
        not finite and positive.

    Notes
    -----
    A capsule is convex, so enclosing a mesh's vertices also encloses every
    triangle built from them. It can substantially overestimate a non-convex
    or strongly tapered shape; it is a cheap proxy, not an exact surface.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("A capsule requires at least three 3D points")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Nonfinite point in capsule input")
    center = vertices.mean(axis=0)
    _, singular, axes = np.linalg.svd(vertices - center, full_matrices=False)
    if singular[0] <= 0 or np.isclose(singular[0], singular[1], rtol=1e-12, atol=0):
        raise ValueError("The point cloud has no unique principal axis")
    axis = axes[0]
    projection = (vertices - center) @ axis
    start = center + projection.min() * axis
    end = center + projection.max() * axis
    radial = vertices - center - projection[:, None] * axis
    radius = float(np.linalg.norm(radial, axis=1).max())
    if radius <= 0 or not math.isfinite(radius):
        raise ValueError("The point cloud cannot define a positive capsule radius")
    return start, end, radius


def segment_pair_distance(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """Return exact distances between corresponding finite 3D segments.

    Parameters
    ----------
    a, b : numpy.ndarray, shape (n, 3)
        Endpoints of the first segment in each pair, in the same frame and
        in metres.
    c, d : numpy.ndarray, shape (n, 3)
        Endpoints of the second segment in each pair.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Nonnegative distances between each segment pair, in metres.

    Notes
    -----
    Vectorized closed-form solution: checks the four parameter-square
    boundaries plus the unconstrained interior stationary point, and keeps
    the smallest valid distance. Subtracting two radii from this distance
    gives the gap between two capsules sharing these centre segments.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)
    u, v, w = b - a, d - c, a - c
    uu = np.einsum("ij,ij->i", u, u)
    vv = np.einsum("ij,ij->i", v, v)
    uv = np.einsum("ij,ij->i", u, v)
    uw = np.einsum("ij,ij->i", u, w)
    vw = np.einsum("ij,ij->i", v, w)
    # Check the minima on all four boundaries of the parameter square.
    safe_u = np.where(uu > 0, uu, 1.0)
    safe_v = np.where(vv > 0, vv, 1.0)
    candidates = []
    for s, t in (
        (np.zeros_like(uu), np.clip(vw / safe_v, 0, 1)),
        (np.ones_like(uu), np.clip((vw + uv) / safe_v, 0, 1)),
        (np.clip(-uw / safe_u, 0, 1), np.zeros_like(uu)),
        (np.clip((uv - uw) / safe_u, 0, 1), np.ones_like(uu)),
    ):
        candidates.append(np.linalg.norm(w + s[:, None] * u - t[:, None] * v, axis=1))
    # Check the interior stationary point when it exists.
    det = uu * vv - uv * uv
    valid_det = det > np.finfo(float).eps * uu * vv
    safe_det = np.where(valid_det, det, 1.0)
    s = (uv * vw - vv * uw) / safe_det
    t = (uu * vw - uv * uw) / safe_det
    valid = valid_det & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    interior = np.linalg.norm(w + s[:, None] * u - t[:, None] * v, axis=1)
    candidates.append(np.where(valid, interior, np.inf))
    return np.min(np.stack(candidates), axis=0)
