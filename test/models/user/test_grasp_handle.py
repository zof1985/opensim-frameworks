"""Closing the fingers of a User on a handle with ``User.grasp_handle``."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import opensim_frameworks as osf
from opensim_frameworks.utils.constants import MESHES_DIR

opensim = pytest.importorskip("opensim")
pytest.importorskip("vtk")

FINGER_ORDER = ("middle", "ring", "index", "little")
STOPPED_BY = {"handle", "palm", "range", "bone", "finger", "blocked"}
HANDLE_RADIUS_M = 0.0175
HANDLE_LENGTH_M = 0.2


def _palm_frame(user, side):
    """Return the palm marker, the hand direction and the palmar normal in Ground."""
    suffix = "l" if side == "left" else "r"
    user.realizePosition(user.state)
    centers = user.joint_center
    mcp = np.mean(
        [centers(f"{f}_mcp_{suffix}") for f in ("index", "middle", "ring", "little")],
        axis=0,
    )
    wrist = np.asarray(centers(f"radius_hand_{suffix}"))
    longitudinal = (mcp - wrist) / np.linalg.norm(mcp - wrist)
    radial = np.asarray(centers(f"index_mcp_{suffix}")) - np.asarray(
        centers(f"little_mcp_{suffix}")
    )
    radial /= np.linalg.norm(radial)
    # il pollice sta sul lato radiale: palmare = -(l x r) a destra, +(l x r) a sinistra
    palmar = np.cross(longitudinal, radial)
    palmar = palmar if side == "left" else -palmar
    palmar /= np.linalg.norm(palmar)
    marker = user.getMarkerSet().get("LPALM" if side == "left" else "RPALM")
    palm = osf.vec3_to_numpy(marker.getLocationInGround(user.state))
    return palm, longitudinal, radial, palmar


def _handle_in_front_of_the_palm(user, side):
    """Build a cylindrical handle across the knuckles' direction, in front of the palm.

    Returns
    -------
    tuple
        The model, its handle body and the mesh attached to that body.
    """
    palm, longitudinal, radial, palmar = _palm_frame(user, side)
    center = palm + 0.045 * palmar + 0.03 * longitudinal
    level = opensim.Logger.getLevelString()
    # OpenSim registra un errore innocuo mentre la mesh non e' ancora in un modello
    opensim.Logger.setLevelString("Off")
    try:
        model = opensim.Model()
        body = opensim.Body(
            "handle", 1.0, opensim.Vec3(0), opensim.Inertia(0.01, 0.01, 0.01, 0, 0, 0)
        )
        mesh = opensim.Mesh(str(MESHES_DIR / "unit_cylinder.vtp"))
        mesh.set_scale_factors(
            opensim.Vec3(2 * HANDLE_RADIUS_M, HANDLE_LENGTH_M, 2 * HANDLE_RADIUS_M)
        )
        body.attachGeometry(mesh)
        model.addBody(body)
    finally:
        opensim.Logger.setLevelString(level)
    rotation, _ = Rotation.align_vectors([radial], [[0.0, 1.0, 0.0]])
    model.addJoint(
        opensim.WeldJoint(
            "handle_weld",
            model.getGround(),
            osf.numpy_to_vec3(center),
            osf.numpy_to_vec3(rotation.as_euler("XYZ")),
            body,
            opensim.Vec3(0),
            opensim.Vec3(0),
        )
    )
    model.realizePosition(model.initSystem())
    return model, body, opensim.Mesh.safeDownCast(body.get_attached_geometry(0))


def _coordinate_values(user):
    coordinates = user.getCoordinateSet()
    return {
        coordinates.get(i).getName(): coordinates.get(i).getValue(user.state)
        for i in range(coordinates.getSize())
    }


def _is_finger_coordinate(name):
    return name.startswith(FINGER_ORDER)


@pytest.fixture(scope="module", params=["right", "left"])
def grasp(request):
    """Close the fingers once per side on a handle placed in front of the palm."""
    side = request.param
    user = osf.User("M", height_cm=175)
    model, body, mesh = _handle_in_front_of_the_palm(user, side)
    before = _coordinate_values(user)
    report = user.grasp_handle(side, body, mesh)
    return {
        "user": user,
        "side": side,
        "model": model,
        "body": body,
        "mesh": mesh,
        "report": report,
        "before": before,
        "after": _coordinate_values(user),
    }


@pytest.fixture(scope="module")
def scene():
    """A user and a handle, unused until a test asks for the grasp."""
    user = osf.User("M", height_cm=175)
    model, body, mesh = _handle_in_front_of_the_palm(user, "right")
    return user, model, body, mesh


def test_every_finger_reports_its_three_joints_and_the_abduction(grasp):
    report = grasp["report"]
    assert tuple(report) == FINGER_ORDER
    for finger, joints in report.items():
        for joint in ("mcp", "pip", "dip"):
            assert set(joints[joint]) == {"angle_deg", "stopped_by"}
            assert joints[joint]["stopped_by"] in STOPPED_BY
        assert math.isfinite(joints["abduction_deg"])
    # il medio si flette senza abduzione/adduzione
    assert report["middle"]["abduction_deg"] == 0.0


def test_reported_angles_are_the_coordinate_values_within_their_ranges(grasp):
    user, side = grasp["user"], grasp["side"]
    for finger, joints in grasp["report"].items():
        for joint in ("mcp", "pip", "dip"):
            coordinate = user.finger_coordinate(side, finger, joint, "flexion")
            angle = joints[joint]["angle_deg"]
            assert angle == pytest.approx(math.degrees(coordinate.getValue(user.state)))
            assert math.degrees(coordinate.getRangeMin()) - 1e-6 <= angle
            assert angle <= math.degrees(coordinate.getRangeMax()) + 1e-6


def test_every_finger_comes_to_rest_on_the_handle(grasp):
    for finger, joints in grasp["report"].items():
        stops = [joints[joint]["stopped_by"] for joint in ("mcp", "pip", "dip")]
        assert "handle" in stops, finger


def test_no_finger_mesh_crosses_the_handle(grasp):
    pairs = grasp["user"].hand_handle_intersections(grasp["side"], grasp["body"])
    finger_pairs = [pair for pair in pairs if pair[0].startswith(FINGER_ORDER)]
    assert finger_pairs == []


def test_only_the_finger_coordinates_move_and_the_thumb_stays(grasp):
    for name, value in grasp["before"].items():
        if _is_finger_coordinate(name) and name.endswith(
            "_" + ("l" if grasp["side"] == "left" else "r")
        ):
            continue
        assert grasp["after"][name] == pytest.approx(value), name


def test_finger_coordinates_end_locked(grasp):
    user, side = grasp["user"], grasp["side"]
    for finger in FINGER_ORDER:
        for joint, motion in (
            ("mcp", "flexion"),
            ("pip", "flexion"),
            ("dip", "flexion"),
            ("mcp", "abduction"),
        ):
            coordinate = user.finger_coordinate(side, finger, joint, motion)
            assert coordinate.getLocked(user.state), (finger, joint, motion)


def test_grasp_handle_rejects_an_invalid_hand(scene):
    user, _, body, mesh = scene
    with pytest.raises(ValueError, match="hand"):
        user.grasp_handle("both", body, mesh)


def test_grasp_handle_rejects_wrong_types(scene):
    user, _, body, mesh = scene
    with pytest.raises(TypeError, match="handle_body"):
        user.grasp_handle("right", mesh, mesh)
    with pytest.raises(TypeError, match="handle_mesh"):
        user.grasp_handle("right", body, body)


def test_grasp_handle_rejects_a_mesh_of_another_body(scene):
    user, _, body, _ = scene
    hand = user.getBodySet().get("hand_r")
    other = opensim.Mesh.safeDownCast(hand.get_attached_geometry(0))
    with pytest.raises(ValueError, match="not attached"):
        user.grasp_handle("right", body, other)


def test_grasp_handle_rejects_a_body_outside_any_model(scene):
    user, _, _, mesh = scene
    loose = opensim.Body(
        "loose", 1.0, opensim.Vec3(0), opensim.Inertia(0.01, 0.01, 0.01, 0, 0, 0)
    )
    with pytest.raises(ValueError, match="model"):
        user.grasp_handle("right", loose, mesh)
