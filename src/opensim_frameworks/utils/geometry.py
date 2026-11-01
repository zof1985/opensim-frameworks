"""Generic point-cloud and segment geometry helpers.

These functions are pure NumPy: they do not depend on OpenSim types and make
no assumption about which body or mesh the points came from. They are shared
by every caller that needs a cheap capsule proxy or a vectorized
segment-to-segment distance (for example self-collision and object-contact
checks used while fitting a hand to a grip).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["enclosing_capsule", "segment_pair_distance"]


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
