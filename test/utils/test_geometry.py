from __future__ import annotations

import numpy as np
import pytest

from opensim_frameworks.utils.geometry import enclosing_capsule, segment_pair_distance


def test_geometry_module_wildcard_exports():
    """Export only the declared geometry helpers."""
    namespace = {}
    exec("from opensim_frameworks.utils.geometry import *", namespace)
    assert set(namespace) == {"__builtins__", "enclosing_capsule", "segment_pair_distance"}


def test_enclosing_capsule_recovers_a_known_cylinder_of_points():
    # Points on the surface of a cylinder of radius 2 centred on the x axis,
    # spanning x in [-5, 5].
    angles = np.linspace(0, 2 * np.pi, 37, endpoint=False)
    xs = np.linspace(-5.0, 5.0, 11)
    points = np.array(
        [[x, 2 * np.cos(a), 2 * np.sin(a)] for x in xs for a in angles]
    )
    start, end, radius = enclosing_capsule(points)
    axis = end - start
    assert np.linalg.norm(axis) == pytest.approx(10.0, abs=1e-6)
    assert abs(axis / np.linalg.norm(axis) @ np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0, abs=1e-6)
    assert radius == pytest.approx(2.0, abs=1e-6)


def test_enclosing_capsule_rejects_degenerate_input():
    with pytest.raises(ValueError, match="at least three"):
        enclosing_capsule(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="Nonfinite"):
        enclosing_capsule(np.array([[0, 0, 0], [1, 0, 0], [float("nan"), 0, 0]]))
    with pytest.raises(ValueError, match="principal axis"):
        # A perfect square has no unique first principal axis (x and y
        # directions carry identical variance).
        enclosing_capsule(np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]))
    with pytest.raises(ValueError, match="positive capsule radius"):
        # Collinear points: every point lies exactly on the principal axis.
        enclosing_capsule(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]))


def test_segment_pair_distance_matches_known_configurations():
    # Parallel segments offset by 3 along z: minimum distance is 3.
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    c = np.array([[0.0, 0.0, 3.0]])
    d = np.array([[1.0, 0.0, 3.0]])
    np.testing.assert_allclose(segment_pair_distance(a, b, c, d), [3.0])

    # Perpendicular, non-intersecting segments ("skew" case): known distance 1.
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    c = np.array([[0.5, 1.0, 1.0]])
    d = np.array([[0.5, -1.0, 1.0]])
    np.testing.assert_allclose(segment_pair_distance(a, b, c, d), [1.0])

    # Intersecting segments: distance 0.
    a = np.array([[-1.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    c = np.array([[0.0, -1.0, 0.0]])
    d = np.array([[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(segment_pair_distance(a, b, c, d), [0.0], atol=1e-12)

    # Degenerate "segments" (zero length): point-to-point distance.
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.0]])
    c = np.array([[3.0, 4.0, 0.0]])
    d = np.array([[3.0, 4.0, 0.0]])
    np.testing.assert_allclose(segment_pair_distance(a, b, c, d), [5.0])


def test_segment_pair_distance_is_vectorized_and_symmetric():
    rng = np.random.default_rng(0)
    a, b, c, d = (rng.normal(size=(50, 3)) for _ in range(4))
    forward = segment_pair_distance(a, b, c, d)
    backward = segment_pair_distance(c, d, a, b)
    assert forward.shape == (50,)
    np.testing.assert_allclose(forward, backward, atol=1e-10)
    assert np.all(forward >= 0)
