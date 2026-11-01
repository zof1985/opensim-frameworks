"""Tests for User.fit_handle and User.hand_handle_intersections.

These use a synthetic cylindrical grip body in a standalone OpenSim model
(not opensim-frameworks' HighRowPlateloaded, which lives outside this repo),
so the contract can be exercised against any generic ``opensim.Body``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from opensim_frameworks import User, numpy_to_vec3, vec3_to_numpy

opensim = pytest.importorskip("opensim")
vtk = pytest.importorskip("vtk")


def _write_cylinder_mesh(path: Path, radius: float, height: float) -> None:
    """Write a triangulated cylinder VTP mesh with its axis along local z.

    Parameters
    ----------
    path : pathlib.Path
        Destination VTP file.
    radius, height : float
        Cylinder dimensions in metres.
    """
    source = vtk.vtkCylinderSource()
    source.SetRadius(radius)
    source.SetHeight(height)
    source.SetResolution(24)
    source.SetCapping(True)
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(source.GetOutputPort())
    # vtkCylinderSource's axis is local y; rotate it to local z.
    rotation = vtk.vtkTransform()
    rotation.RotateX(90)
    rotated = vtk.vtkTransformFilter()
    rotated.SetTransform(rotation)
    rotated.SetInputConnection(triangles.GetOutputPort())
    rotated.Update()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputConnection(rotated.GetOutputPort())
    writer.SetDataModeToAscii()
    writer.Write()


def _make_handle(
    tmp_path: Path,
    center_m: np.ndarray,
    radius: float = 0.015,
    height: float = 0.14,
) -> "opensim.Body":
    """Build a standalone OpenSim model with one cylindrical grip body.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the generated mesh file.
    center_m : numpy.ndarray, shape (3,)
        Ground position of the cylinder's centre, in metres.
    radius, height : float
        Cylinder dimensions in metres.

    Returns
    -------
    opensim.Body
        Grip body, welded to ground at `center_m`, belonging to an
        initialized, position-realized model.
    """
    mesh_path = tmp_path / f"handle_{abs(hash(tuple(center_m.tolist()))) % 10**8}.vtp"
    _write_cylinder_mesh(mesh_path, radius, height)

    model = opensim.Model()
    model.setName("synthetic_equipment")
    body = opensim.Body(
        "handle_body", 0.2, opensim.Vec3(0, 0, 0), opensim.Inertia(1e-4, 1e-4, 1e-4)
    )
    model.addBody(body)
    zero = opensim.Vec3(0, 0, 0)
    model.addJoint(
        opensim.WeldJoint(
            "handle_ground", model.getGround(), numpy_to_vec3(center_m), zero, body, zero, zero
        )
    )
    mesh = opensim.Mesh(str(mesh_path))
    mesh.set_scale_factors(opensim.Vec3(1, 1, 1))
    body.attachGeometry(mesh)
    model.finalizeConnections()
    state = model.initSystem()
    model.realizePosition(state)
    # Keep the owning model alive for as long as the returned body is used:
    # nothing else holds a Python reference to it once this function returns,
    # and OpenSim components do not keep their owning Model alive themselves.
    body._owning_model = model
    return body


@pytest.fixture
def user():
    return User("M", percentile=50)


def _handle_near(user: User, tmp_path: Path, side: str, gap_m: float = 0.02) -> "opensim.Body":
    """Place a synthetic handle just outside the palm along the body's local x axis.

    Parameters
    ----------
    user : User
        Reference-pose user (local hand axes are ground-aligned at this pose).
    tmp_path : pathlib.Path
        Directory for the generated mesh file.
    side : {'right', 'left'}
        Hand to approach.
    gap_m : float, default=0.02
        Offset from the palm marker along local x, towards the volar side.

    Returns
    -------
    opensim.Body
        Grip body positioned close to the requested palm.
    """
    palm = np.array(user.palm_markers[side])
    sign = -1.0 if side == "right" else 1.0
    center = palm + np.array([sign * gap_m, 0.0, 0.0])
    return _make_handle(tmp_path, center)


def test_hand_handle_intersections_rejects_invalid_side(user, tmp_path):
    handle = _handle_near(user, tmp_path, "right")
    with pytest.raises(ValueError, match="side"):
        user.hand_handle_intersections("r", handle)


def test_hand_handle_intersections_rejects_non_body_handle(user):
    with pytest.raises(TypeError, match="opensim.Body"):
        user.hand_handle_intersections("right", object())


def test_hand_handle_intersections_rejects_handle_without_model(user):
    handle = opensim.Body(
        "orphan", 0.1, opensim.Vec3(0, 0, 0), opensim.Inertia(1e-4, 1e-4, 1e-4)
    )
    with pytest.raises(ValueError, match="initialized OpenSim model"):
        user.hand_handle_intersections("right", handle)


def test_hand_handle_intersections_detects_a_known_intersection(user, tmp_path):
    # Centre a wide, long cylinder directly on the right metacarpal-3 capsule
    # midpoint so that it clearly overlaps the actual bone mesh.
    bodies = user.getBodySet()
    hand = bodies.get("hand_r")
    for index in range(hand.getPropertyByName("attached_geometry").size()):
        mesh = opensim.Mesh.safeDownCast(hand.get_attached_geometry(index))
        if mesh is not None and mesh.get_mesh_file() == "metacarpal3_rvs.vtp":
            break
    else:
        raise AssertionError("metacarpal3_rvs.vtp not attached to hand_r")
    transform = mesh.getFrame().getTransformInGround(user.state)
    center = vec3_to_numpy(transform.p())
    handle = _make_handle(tmp_path, center, radius=0.03, height=0.1)

    intersections = user.hand_handle_intersections("right", handle)
    assert intersections
    assert any(first == "metacarpal3_rvs.vtp" for first, _ in intersections)


def test_hand_handle_intersections_reports_no_intersection_when_far_away(user, tmp_path):
    handle = _make_handle(tmp_path, np.array([5.0, 5.0, 5.0]))
    assert user.hand_handle_intersections("right", handle) == ()


@pytest.mark.parametrize("function", ["Push", "both", ""])
def test_fit_handle_rejects_invalid_function(user, tmp_path, function):
    handle = _handle_near(user, tmp_path, "right")
    with pytest.raises(ValueError, match="function"):
        user.fit_handle(handle, function=function)


@pytest.mark.parametrize("distance", [0.0, -0.01, float("nan"), float("inf"), "0.01"])
def test_fit_handle_rejects_invalid_soft_tissue_distance(user, tmp_path, distance):
    handle = _handle_near(user, tmp_path, "right")
    with pytest.raises(ValueError, match="soft_tissues_approximation_distance"):
        user.fit_handle(handle, soft_tissues_approximation_distance=distance)


def test_fit_handle_rejects_non_body_handle(user):
    with pytest.raises(TypeError, match="opensim.Body"):
        user.fit_handle(object())


def test_fit_handle_rejects_handle_without_model(user):
    handle = opensim.Body(
        "orphan", 0.1, opensim.Vec3(0, 0, 0), opensim.Inertia(1e-4, 1e-4, 1e-4)
    )
    with pytest.raises(ValueError, match="initialized OpenSim model"):
        user.fit_handle(handle)


def test_fit_handle_rejects_ambiguous_side(user, tmp_path):
    # Far from both palms: neither hand can be inferred.
    handle = _make_handle(tmp_path, np.array([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="Cannot determine which hand"):
        user.fit_handle(handle)


@pytest.mark.parametrize("side", ["right", "left"])
def test_fit_handle_infers_side_and_leaves_other_side_untouched(user, tmp_path, side):
    other = "left" if side == "right" else "right"
    other_suffix = "l" if other == "left" else "r"
    before_other_fingers = {
        key: user.finger_coordinate(other, *key).getValue(user.state)
        for key in (("index", "mcp", "flexion"), ("thumb", "cmc", "opposition"))
    }
    before_shoulder_elbow = {
        name: user.getCoordinateSet().get(name).getValue(user.state)
        for name in (
            f"arm_flex_{other_suffix}",
            f"elbow_flex_{other_suffix}",
            f"arm_flex_{'l' if side == 'left' else 'r'}",
            f"elbow_flex_{'l' if side == 'left' else 'r'}",
        )
    }

    handle = _handle_near(user, tmp_path, side)
    report = user.fit_handle(handle, function="push")

    assert report["side"] == side
    for key, value in before_other_fingers.items():
        assert user.finger_coordinate(other, *key).getValue(user.state) == pytest.approx(value)
        assert user.finger_coordinate(other, *key).getLocked(user.state)
    for name, value in before_shoulder_elbow.items():
        assert user.getCoordinateSet().get(name).getValue(user.state) == pytest.approx(value)


@pytest.mark.parametrize("side", ["right", "left"])
def test_fit_handle_push_satisfies_contact_and_no_intersections(user, tmp_path, side):
    tolerance = 0.01
    handle = _handle_near(user, tmp_path, side)

    report = user.fit_handle(handle, function="push", soft_tissues_approximation_distance=tolerance)

    assert report["function"] == "push"
    assert report["solver_success"] is True
    assert max(report["palm_handle_gaps_m"].values()) <= tolerance + 1e-6
    assert min(report["palm_handle_gaps_m"].values()) >= -1e-6
    assert user.hand_surface_intersections(side) == ()
    assert user.hand_handle_intersections(side, handle) == ()


def test_fit_handle_pull_does_not_require_palm_contact(user, tmp_path):
    handle = _handle_near(user, tmp_path, "left")

    report = user.fit_handle(handle, function="pull", soft_tissues_approximation_distance=0.01)

    assert report["function"] == "pull"
    assert report["solver_success"] is True
    # Pull only has to avoid interference; it may leave the palm farther than
    # the push contact tolerance.
    assert user.hand_surface_intersections("left") == ()
    assert user.hand_handle_intersections("left", handle) == ()
    # The primary pull objective is finger wrap: expect every finger/thumb
    # capsule within a modest margin of the soft-tissue target gap.
    deviations = [abs(gap - 0.01) for gap in report["finger_handle_gaps_m"].values()]
    assert max(deviations) < 0.015


def test_fit_handle_unlocks_finger_coordinates_for_the_fitted_side(user, tmp_path):
    handle = _handle_near(user, tmp_path, "right")
    report = user.fit_handle(handle, function="push")
    assert report["side"] == "right"
    for finger in ("index", "middle", "ring", "little"):
        coordinate = user.finger_coordinate("right", finger, "mcp", "flexion")
        assert not coordinate.getLocked(user.state)
    assert not user.finger_coordinate("right", "thumb", "cmc", "opposition").getLocked(user.state)
