import inspect
import logging
import warnings
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import opensim as osim
import pytest
import vtk

from opensim_frameworks.models.equipment.chestpress import handlefit
from opensim_frameworks.models.equipment.chestpress import chestpress
from opensim_frameworks.models.equipment.chestpress.chestpress import ChestPress
from opensim_frameworks import Framework, Model
from opensim_frameworks.utils.geometry import body_transform_array
from opensim_frameworks.utils.kinematics import marker_position_array
from opensim_frameworks.utils.motion_profiles import (
    beta_time_for_position,
    beta_velocity_profile,
    gaussian_time_for_position,
    gaussian_velocity_profile,
)


class Coordinate:
    def __init__(self, name, value=0.0, minimum=-1.0, maximum=1.0):
        self.name = name
        self.value = value
        self.minimum = minimum
        self.maximum = maximum
        self.locked = True

    def getName(self):
        return self.name

    def getValue(self, state):
        return self.value

    def setValue(self, state, value, enforce_constraints):
        self.value = value

    def getRangeMin(self):
        return self.minimum

    def getRangeMax(self):
        return self.maximum

    def get_locked(self):
        return self.locked

    def getLocked(self, state):
        return self.locked

    def set_locked(self, locked):
        self.locked = locked

    def setLocked(self, state, locked):
        self.locked = locked


class CoordinateSet:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def getSize(self):
        return len(self.coordinates)

    def get(self, key):
        if isinstance(key, int):
            return self.coordinates[key]
        return next(coordinate for coordinate in self.coordinates if coordinate.name == key)


@pytest.fixture
def fit_setup(tmp_path, monkeypatch):
    source = vtk.vtkCylinderSource()
    source.SetCenter(0.3, 0.4, 0.5)
    source.SetRadius(0.02)
    source.SetHeight(0.24)
    source.SetResolution(64)
    source.Update()
    mesh_path = tmp_path / "handle.vtp"
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(mesh_path))
    writer.SetInputData(source.GetOutput())
    assert writer.Write()

    equipment = osim.Model()
    body = osim.Body("right_handle", 1.0, osim.Vec3(0), osim.Inertia(1))
    body.attachGeometry(osim.Mesh(str(mesh_path)))
    equipment.addBody(body)
    joint = osim.PinJoint(
        "handle_joint", equipment.getGround(), osim.Vec3(0), osim.Vec3(0),
        body, osim.Vec3(0), osim.Vec3(0),
    )
    equipment.addJoint(joint)
    equipment.addMarker(osim.Marker("right_handle", body, osim.Vec3(0.3, 0.4, 0.5)))
    equipment.state = equipment.initSystem()
    equipment.geometry_directories = (tmp_path,)
    equipment.mesh_path = tmp_path

    names = (
        "arm_rot_r", "elbow_flex_r", "pro_sup_r", "wrist_flex_r",
        "wrist_dev_r", "index_mcp_abduction_r", "sc_protraction_r",
        "sc_elevation_r", "ac_protraction_r", "ac_upward_rotation_r",
        "ac_axial_rotation_r",
    )
    coordinates = CoordinateSet([Coordinate(name) for name in names])
    palm = SimpleNamespace(getName=lambda: "RPALM")
    markers = SimpleNamespace(getSize=lambda: 1, get=lambda index: palm)
    centers = {
        "acromial_r": np.array([-0.2, 0.6, 0.3]),
        "elbow_r": np.array([0.0, 0.5, 0.4]),
        "radius_hand_r": np.array([0.25, 0.33, 0.55]),
        **{
            f"{finger}_mcp_r": np.array([0.3, 0.30 + 0.02 * index, 0.55])
            for index, finger in enumerate(("index", "middle", "ring", "little"))
        },
    }
    user = SimpleNamespace(
        state=object(),
        getCoordinateSet=lambda: coordinates,
        getMarkerSet=lambda: markers,
        realizePosition=lambda state: None,
        joint_center=lambda name: centers[name],
        scapula_contact_residual=lambda side: 0.0,
        scapula_contact_report=lambda side: {"gap_m": 0.001, "intersects": False},
        scapula_contact_max_gap_m=0.0015,
        resolve_scapula_intersection=lambda side, reference: False,
    )
    def set_girdle(side, values, *, check_contact=True):
        for name, value in zip(handlefit.SHOULDER_GIRDLE_COORDINATES, values):
            coordinates.get(f"{name}_{side}").value = value

    user.set_scapular_girdle_values = set_girdle
    def set_shoulder(side, protraction, elevation, *, check_contact=True):
        set_girdle(
            side,
            [protraction, elevation, protraction * 0.1, elevation * 0.2, 0.0],
            check_contact=check_contact,
        )

    user.set_shoulder_girdle = set_shoulder
    def scapular_residuals(side, protraction_weight, elevation_weight):
        protraction = coordinates.get(f"sc_protraction_{side}")
        elevation = coordinates.get(f"sc_elevation_{side}")
        return np.array([
            np.sqrt(protraction_weight) * (protraction.value - protraction.minimum),
            np.sqrt(elevation_weight) * max(elevation.value, 0.0),
            0.0, 0.0, 0.0, 0.0,
        ])

    user.scapular_fit_residuals = scapular_residuals
    palm_ground = np.zeros(3)

    def marker_position(marker, state):
        if marker is palm:
            return palm_ground.copy()
        return marker_position_array(marker, state)

    monkeypatch.setattr(handlefit, "marker_position_array", marker_position)
    return user, equipment, coordinates, palm_ground


def run_fit(monkeypatch, user, equipment, *, probes=(), solution=None, **weights):
    costs = []

    def solve(objective, x0, *, bounds, max_nfev, x_scale, ftol, xtol, gtol):
        names = sorted(
            name for name in (c.name for c in user.getCoordinateSet().coordinates)
            if name.startswith(handlefit._FIT_COORDINATE_PREFIXES)
        )
        names += ["sc_protraction_r", "sc_elevation_r"]
        for values in probes:
            candidate = x0.copy()
            for name, value in values.items():
                candidate[names.index(name)] = value
            residuals = objective(candidate)
            assert len(residuals) > 1
            costs.append(float(residuals @ residuals))
        fitted = x0.copy()
        for name, value in (solution or {}).items():
            fitted[names.index(name)] = value
        residuals = objective(fitted)
        return SimpleNamespace(
            success=True, nfev=1, x=fitted, fun=residuals, status=3,
            message="xtol termination", cost=float(residuals @ residuals / 2),
            optimality=0.1,
        )

    monkeypatch.setattr(handlefit, "least_squares", solve)
    zero_weights = {
        "palm_contact_weight": 0.0,
        "knuckle_weight": 0.0,
        "knuckle_end_weight": 0.0,
        "plane_distance_weight": 0.0,
        "pro_sup_weight": 0.0,
        "wrist_dev_weight": 0.0,
        "protraction_weight": 0.0,
        "scapular_elevation_weight": 0.0,
        "wrist_flexion_weight": 0.0,
        "arm_weight": 0.0,
    }
    zero_weights.update(weights)
    result = handlefit.fit_handle(user, equipment, "right_handle", "right", **zero_weights)
    return result, costs


@pytest.mark.parametrize("palm_y", [0.30, 0.40, 0.50])
def test_wrist_axis_target_tracks_palm_and_wrist_sliding_together(
    fit_setup, monkeypatch, palm_y
):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.325, palm_y, 0.5]
    original_joint_center = user.joint_center
    user.joint_center = lambda name: (
        palm_ground.copy()
        if name == "radius_hand_r"
        else original_joint_center(name)
    )
    monkeypatch.setattr(
        handlefit, "enclosing_cylinder",
        lambda vertices: (
            np.array([0.3, 0.4, 0.5, 0.0, 1.0, 0.0]), 0.02,
            np.array([0.3, 0.28, 0.5]), np.array([0.3, 0.52, 0.5]),
        ),
    )
    _, costs = run_fit(
        monkeypatch, user, equipment, probes=[{}],
        palm_contact_weight=2.0,
    )
    assert costs[0] == pytest.approx(2.0 * 0.005**2, abs=1e-10)


@pytest.mark.parametrize("radius", [0.02, 0.025, 0.03])
def test_palm_contact_targets_the_actual_cylinder_radius(fit_setup, monkeypatch, radius):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.33, 0.4, 0.5]
    original_joint_center = user.joint_center
    user.joint_center = lambda name: (
        np.array([0.3, 0.4, 0.5])
        if name == "radius_hand_r"
        else original_joint_center(name)
    )
    monkeypatch.setattr(
        handlefit, "enclosing_cylinder",
        lambda vertices: (
            np.array([0.3, 0.4, 0.5, 0.0, 1.0, 0.0]), radius,
            np.array([0.3, 0.28, 0.5]), np.array([0.3, 0.52, 0.5]),
        ),
    )
    _, costs = run_fit(monkeypatch, user, equipment, probes=[{}], palm_contact_weight=2.0)
    assert costs[0] == pytest.approx(2.0 * (radius + 0.015)**2, abs=1e-10)


def test_wrist_axis_target_includes_half_palm_to_wrist_distance(fit_setup, monkeypatch):
    user, equipment, coordinates, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.4, 0.5]
    original_joint_center = user.joint_center

    def joint_center(name):
        if name == "radius_hand_r":
            return np.array(
                [0.325 - 0.01 * (coordinates.get("pro_sup_r").value > 0.0), 0.4, 0.5]
            )
        return original_joint_center(name)

    user.joint_center = joint_center
    _, costs = run_fit(
        monkeypatch, user, equipment,
        probes=[{"pro_sup_r": -0.5}, {"pro_sup_r": 0.5}],
        solution={"pro_sup_r": -0.5},
        palm_contact_weight=2.0,
    )
    np.testing.assert_allclose(costs, [2.0 * 0.005**2, 2.0 * 0.01**2], atol=2e-9)


@pytest.mark.parametrize("angle", [0.0, 0.7])
def test_palm_relative_to_marker_and_cylindrical_surface(fit_setup, monkeypatch, angle):
    user, equipment, _, palm_ground = fit_setup
    equipment.updCoordinateSet().get(0).setValue(equipment.state, angle)
    equipment.realizePosition(equipment.state)
    transform = body_transform_array(equipment.getBodySet().get("right_handle"), equipment.state)
    original_joint_center = user.joint_center
    palm_local = np.array([0.32, 0.48, 0.5])
    user.joint_center = lambda name: (
        transform[:3, :3] @ palm_local + transform[:3, 3]
        if name == "radius_hand_r"
        else transform[:3, :3] @ original_joint_center(name) + transform[:3, 3]
    )
    palm_ground[:] = transform[:3, :3] @ palm_local + transform[:3, 3]

    (_, palm, _), costs = run_fit(
        monkeypatch, user, equipment, probes=[{}], palm_contact_weight=1.0,
    )
    np.testing.assert_allclose(palm, [0.02, 0.08, 0.0], atol=1e-7)
    assert costs[0] == pytest.approx(0.0, abs=1e-12)

    palm_local[1] = 0.55
    palm_ground[:] = transform[:3, :3] @ palm_local + transform[:3, 3]
    _, costs = run_fit(
        monkeypatch, user, equipment, probes=[{}],
        palm_contact_weight=1.0,
    )
    assert costs[0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("angle", [0.0, 0.7])
@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_index_knuckle_distance_penalty_guides_least_squares_to_target(
    fit_setup, angle, direction
):
    user, equipment, coordinates, palm_ground = fit_setup
    equipment.updCoordinateSet().get(0).setValue(equipment.state, angle)
    equipment.realizePosition(equipment.state)
    transform = body_transform_array(equipment.getBodySet().get("right_handle"), equipment.state)
    palm_ground[:] = transform[:3, :3] @ [0.325, 0.4, 0.5] + transform[:3, 3]
    original_joint_center = user.joint_center

    def joint_center(name):
        if name == "index_mcp_r":
            y = 0.38 + direction * 0.1 * coordinates.get("wrist_dev_r").value
            local = np.array([0.3, y, 0.5])
            return transform[:3, :3] @ local + transform[:3, 3]
        if name == "little_mcp_r":
            local = np.array([0.3, 0.52, 0.5])
            return transform[:3, :3] @ local + transform[:3, 3]
        return transform[:3, :3] @ original_joint_center(name) + transform[:3, 3]

    user.joint_center = joint_center
    elbow, _, _ = handlefit.fit_handle(
        user, equipment, "right_handle", "right",
        palm_contact_weight=0.0, knuckle_weight=0.0,
        knuckle_end_weight=1e8, plane_distance_weight=0.0, pro_sup_weight=0.0,
        wrist_dev_weight=0.0,
        protraction_weight=0.0, scapular_elevation_weight=0.0,
        wrist_flexion_weight=0.0, arm_weight=0.0,
    )
    assert elbow is not None
    assert coordinates.get("wrist_dev_r").value == pytest.approx(
        -0.2 * direction, abs=1e-4
    )


def test_index_knuckle_target_uses_cylinder_endpoints_and_little_knuckle(
    fit_setup, monkeypatch
):
    user, equipment, coordinates, palm_ground = fit_setup
    equipment.updCoordinateSet().get(0).setValue(equipment.state, 0.0)
    equipment.realizePosition(equipment.state)
    transform = body_transform_array(equipment.getBodySet().get("right_handle"), equipment.state)
    palm_ground[:] = transform[:3, :3] @ [0.325, 0.4, 0.5] + transform[:3, 3]
    original_joint_center = user.joint_center

    def joint_center(name):
        if name == "index_mcp_r":
            y = 0.38 + 0.1 * coordinates.get("wrist_dev_r").value
            local = np.array([0.3, y, 0.5])
            return transform[:3, :3] @ local + transform[:3, 3]
        if name == "little_mcp_r":
            local = np.array([0.3, 0.52, 0.5])
            return transform[:3, :3] @ local + transform[:3, 3]
        return transform[:3, :3] @ original_joint_center(name) + transform[:3, 3]

    user.joint_center = joint_center
    _, costs = run_fit(
        monkeypatch, user, equipment,
        probes=[
            {"wrist_dev_r": 0.0}, {"wrist_dev_r": -0.2},
            {"wrist_dev_r": 0.2},
        ],
        knuckle_end_weight=1.0,
    )
    np.testing.assert_allclose(costs, [0.03**2, 0.0, 0.06**2], atol=1e-8)


def test_forearm_rotation_penalizes_only_angles_outside_75_to_115(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.4, 0.5]
    angles = [65.0, 75.0, 90.0, 115.0, 125.0]
    _, costs = run_fit(
        monkeypatch, user, equipment,
        probes=[{"pro_sup_r": np.deg2rad(angle)} for angle in angles],
        pro_sup_weight=1.0,
    )
    np.testing.assert_allclose(
        costs, [np.deg2rad(10.0) ** 2, 0.0, 0.0, 0.0, np.deg2rad(10.0) ** 2],
        atol=1e-12,
    )


def test_wrist_penalizes_deviation_from_45_degrees_extension(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.4, 0.5]
    target = np.deg2rad(-45.0)
    _, costs = run_fit(
        monkeypatch, user, equipment,
        probes=[
            {"wrist_flex_r": angle}
            for angle in (target - 0.2, target, target + 0.2, 0.0)
        ],
        wrist_flexion_weight=2.0,
    )
    np.testing.assert_allclose(
        costs, [2.0 * 0.2**2, 0.0, 2.0 * 0.2**2, 2.0 * target**2], atol=1e-12
    )


def test_upper_arm_rotation_penalizes_external_rotation_only(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.4, 0.5]
    _, costs = run_fit(
        monkeypatch, user, equipment,
        probes=[{"arm_rot_r": angle} for angle in (-0.5, 0.0, 0.5)],
        arm_weight=1.0,
    )
    np.testing.assert_allclose(costs, [0.25, 0.0, 0.0], atol=1e-12)


def test_fit_log_reports_forearm_and_upper_arm_outside_preferred_ranges(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.4, 0.5]
    (_, _, messages), _ = run_fit(
        monkeypatch, user, equipment,
        solution={"pro_sup_r": np.deg2rad(60.0), "arm_rot_r": np.deg2rad(-10.0)},
        pro_sup_weight=1.0, arm_weight=1.0,
    )
    assert any("forearm rotation 60.00 deg outside 75-115 deg" in m["message"] for m in messages)
    assert any("external upper-arm rotation 10.00 deg" in m["message"] for m in messages)


def test_fit_log_reports_palm_orientation_and_side_of_handle(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.4, 0.5]
    (_, _, messages), _ = run_fit(monkeypatch, user, equipment)
    assert any("palm does not face downward" in m["message"] for m in messages)
    assert any("palm is not above handle axis" in m["message"] for m in messages)


def test_index_knuckle_target_deviation_warns_and_preserves_pose(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.375, 0.5]
    original_joint_center = user.joint_center
    user.joint_center = lambda name: (
        np.array([0.3, 0.25, 0.55])
        if name == "index_mcp_r" else original_joint_center(name)
    )
    monkeypatch.setattr(
        handlefit, "enclosing_cylinder",
        lambda vertices: (
            np.array([0.3, 0.375, 0.5, 0.0, 1.0, 0.0]), 0.02,
            np.array([0.3, 0.25, 0.5]), np.array([0.3, 0.5, 0.5]),
        ),
    )
    (elbow, palm, messages), _ = run_fit(monkeypatch, user, equipment)
    assert any(
        "index knuckle-to-handle-center distance" in m["message"]
        for m in messages
    )
    assert elbow is not None
    assert palm is not None


def test_knuckles_are_checked_on_fitted_scapular_pose(fit_setup, monkeypatch):
    user, equipment, coordinates, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.4, 0.5]
    original_joint_center = user.joint_center

    def joint_center(name):
        point = original_joint_center(name).copy()
        if name.endswith("_mcp_r"):
            point[1] += coordinates.get("sc_elevation_r").value
            point[1] += coordinates.get("wrist_dev_r").value
        return point

    user.joint_center = joint_center
    (elbow, palm, messages), _ = run_fit(
        monkeypatch, user, equipment,
        solution={"sc_elevation_r": 0.2}, wrist_dev_weight=1.0,
    )
    assert any(
        "index knuckle-to-handle-center distance" in m["message"]
        for m in messages
    )
    assert elbow is not None
    assert palm is not None
    assert coordinates.get("sc_elevation_r").value == pytest.approx(0.2)


def test_knuckles_outside_warn_and_keep_fitted_coordinates(fit_setup, monkeypatch):
    user, equipment, coordinates, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.4, 0.5]
    original_joint_center = user.joint_center
    user.joint_center = lambda name: (
        original_joint_center(name) + np.array([0.0, 0.5, 0.0])
        if name.endswith("_mcp_r") else original_joint_center(name)
    )
    coordinates.get("arm_rot_r").value = 0.2
    scapula_checks = []
    user.scapula_contact_report = lambda side: scapula_checks.append(side)
    (elbow, palm, messages), _ = run_fit(
        monkeypatch, user, equipment, solution={"arm_rot_r": 0.6}
    )
    assert any(
        "index knuckle-to-handle-center distance" in m["message"]
        for m in messages
    )
    assert elbow is not None
    assert palm is not None
    assert coordinates.get("arm_rot_r").value == pytest.approx(0.6)
    assert scapula_checks == ["r"]


def test_palm_penetration_warns_and_keeps_fitted_coordinates(fit_setup, monkeypatch):
    user, equipment, coordinates, palm_ground = fit_setup
    palm_ground[:] = [0.3, 0.4, 0.5]
    original_joint_center = user.joint_center
    user.joint_center = lambda name: (
        palm_ground.copy()
        if name == "radius_hand_r"
        else original_joint_center(name)
    )
    coordinates.get("arm_rot_r").value = 0.2
    (elbow, palm, messages), _ = run_fit(
        monkeypatch, user, equipment, solution={"arm_rot_r": 0.6},
        palm_contact_weight=1.0,
    )
    assert any(
        "wrist-to-axis distance 0.000 mm differs from target 20.000 mm"
        in m["message"] for m in messages
    )
    assert elbow is not None
    assert palm is not None
    assert coordinates.get("arm_rot_r").value == pytest.approx(0.6)


def test_fit_log_reports_palm_gap_far_from_target(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.335, 0.4, 0.5]
    original_joint_center = user.joint_center
    user.joint_center = lambda name: (
        palm_ground.copy()
        if name == "radius_hand_r"
        else original_joint_center(name)
    )
    (_, _, messages), _ = run_fit(
        monkeypatch, user, equipment, palm_contact_weight=1.0,
    )
    assert any(
        "wrist-to-axis distance 35.000 mm differs from target 20.000 mm"
        in m["message"] for m in messages
    )


def test_warning_reports_both_palm_and_specific_knuckle(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.3, 0.4, 0.5]
    original_joint_center = user.joint_center
    def joint_center(name):
        if name == "radius_hand_r":
            return palm_ground.copy()
        if name == "index_mcp_r":
            return np.array([0.3, 0.25, 0.55])
        if name == "little_mcp_r":
            return np.array([0.3, 0.32, 0.55])
        return original_joint_center(name)

    user.joint_center = joint_center
    (elbow, palm, recorded), _ = run_fit(
        monkeypatch, user, equipment, palm_contact_weight=1.0
    )
    assert elbow is not None
    assert palm is not None
    assert len(recorded) == 1
    warning = recorded[0]["message"]
    assert "right handle fit" in warning
    assert "wrist-to-axis distance 0.000 mm differs from target 20.000 mm" in warning
    assert (
        "index knuckle-to-handle-center distance 158.114 mm differs from target "
        "85.000 mm"
    ) in warning
    assert "little knuckle-to-handle-center distance" not in warning


@pytest.mark.parametrize("weight", [-1.0, np.nan, np.inf])
def test_invalid_knuckle_end_weight_is_rejected(fit_setup, weight):
    user, equipment, _, _ = fit_setup
    with pytest.raises(ValueError, match="knuckle_end_weight"):
        handlefit.fit_handle(user, equipment, "right_handle", "right", knuckle_end_weight=weight)


def test_knuckles_outside_on_fitted_scapula_preserve_pose(fit_setup, monkeypatch):
    user, equipment, coordinates, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.4, 0.5]
    original_joint_center = user.joint_center
    user.joint_center = lambda name: (
        original_joint_center(name) + np.array([0.0, coordinates.get("sc_elevation_r").value, 0.0])
        if name.endswith("_mcp_r") else original_joint_center(name)
    )

    (elbow, palm, messages), _ = run_fit(
        monkeypatch, user, equipment, solution={"sc_elevation_r": 0.5}
    )
    assert any(
        "index knuckle-to-handle-center distance" in m["message"]
        for m in messages
    )
    assert elbow is not None
    assert palm is not None
    assert coordinates.get("sc_elevation_r").value == pytest.approx(0.5)


def test_shoulder_weights_penalize_protraction_and_positive_elevation(fit_setup, monkeypatch):
    user, equipment, _, _ = fit_setup
    probes = [
        {"sc_protraction_r": -0.5, "sc_elevation_r": -0.2},
        {"sc_protraction_r": 0.5, "sc_elevation_r": -0.2},
        {"sc_protraction_r": 0.5, "sc_elevation_r": 0.2},
    ]
    _, costs = run_fit(
        monkeypatch, user, equipment, probes=probes,
        protraction_weight=2.0, scapular_elevation_weight=3.0,
    )
    np.testing.assert_allclose(costs, [0.5, 4.5, 4.62])


def test_fit_optimizes_shoulder_angles_and_user_derives_scapula(fit_setup, monkeypatch):
    user, equipment, coordinates, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.4, 0.5]
    coordinates.get("ac_protraction_r").value = 0.7

    (elbow, _, _), _ = run_fit(
        monkeypatch, user, equipment,
        solution={"sc_protraction_r": 0.4, "sc_elevation_r": 0.3},
    )

    assert elbow is not None
    assert coordinates.get("ac_protraction_r").value == pytest.approx(0.04)
    assert coordinates.get("ac_upward_rotation_r").value == pytest.approx(0.06)


def test_plane_and_knuckle_alignment_use_joint_geometry(fit_setup, monkeypatch):
    user, equipment, _, _ = fit_setup
    points = {
        "acromial_r": np.array([0.0, 0.0, 0.0]),
        "elbow_r": np.array([0.5, 0.5, 0.0]),
        "radius_hand_r": np.array([1.0, 0.0, 0.0]),
        **{
            f"{finger}_mcp_r": np.array([0.3, 0.30 + 0.02 * index, 0.55])
            for index, finger in enumerate(("index", "middle", "ring", "little"))
        },
    }
    user.joint_center = lambda name: points[name]
    _, aligned = run_fit(
        monkeypatch, user, equipment, probes=[{}], plane_distance_weight=1.0,
        knuckle_weight=1.0,
    )
    assert aligned[0] == pytest.approx(0.0, abs=1e-12)
    points["elbow_r"][2] += 0.02
    _, displaced = run_fit(
        monkeypatch, user, equipment, probes=[{}], plane_distance_weight=1.0
    )
    assert displaced[0] == pytest.approx(0.02**2, abs=1e-8)
    for index, finger in enumerate(("index", "middle", "ring", "little")):
        points[f"{finger}_mcp_r"] = np.array([0.25 + 0.02 * index, 0.3, 0.55])
    _, crossed = run_fit(
        monkeypatch, user, equipment, probes=[{}], knuckle_weight=1.0
    )
    assert crossed[0] == pytest.approx(1.0, abs=1e-8)


def test_handle_bodies_must_have_registered_meshes(fit_setup):
    _, equipment, _, _ = fit_setup
    for name in ("left_handle", "back_pad", "seat_pad"):
        equipment.addBody(osim.Body(name, 1.0, osim.Vec3(0), osim.Inertia(1)))
    equipment.handle_meshes = lambda side: ChestPress.handle_meshes(equipment, side)
    assert len(equipment.handle_meshes("right")) == 1
    with pytest.raises(RuntimeError, match="No meshes attached to left_handle"):
        ChestPress._verify_required_bodies(equipment)
    left = equipment.getBodySet().get("left_handle")
    right = equipment.getBodySet().get("right_handle")
    left.attachGeometry(osim.Mesh(osim.Mesh.safeDownCast(right.get_attached_geometry(0)).get_mesh_file()))
    ChestPress._verify_required_bodies(equipment)


@pytest.mark.parametrize(("side", "source", "target"), [
    ("right", "r", "l"), ("left", "l", "r"),
])
def test_mirror_arm_copies_all_upper_limb_coordinates(fit_setup, side, source, target):
    user, _, coordinates, _ = fit_setup
    bases = [
        *handlefit.SHOULDER_GIRDLE_COORDINATES,
        "arm_flex", "arm_add", "arm_rot", "elbow_flex", "pro_sup",
        "wrist_flex", "wrist_dev",
        *(
            f"{finger}_{joint}_{motion}"
            for finger in ("index", "middle", "ring", "little")
            for joint, motion in (
                ("mcp", "flexion"), ("mcp", "abduction"),
                ("pip", "flexion"), ("dip", "flexion"),
            )
        ),
        "thumb_cmc_opposition", "thumb_cmc_flexion",
        "thumb_mcp_flexion", "thumb_ip_flexion",
    ]
    names = {coordinate.name for coordinate in coordinates.coordinates}
    for base in bases:
        for suffix in ("r", "l"):
            name = f"{base}_{suffix}"
            if name not in names:
                coordinates.coordinates.append(Coordinate(name))
                names.add(name)
    coordinates.coordinates.extend([
        Coordinate("hip_flexion_r", 0.1), Coordinate("hip_flexion_l", -0.4)
    ])
    target_locks = {}
    for index, base in enumerate(bases):
        coordinates.get(f"{base}_{source}").value = 0.01 * (index + 1)
        destination = coordinates.get(f"{base}_{target}")
        destination.value = -0.5
        destination.locked = index % 2 == 0
        target_locks[base] = destination.locked
    reports = []
    user.scapula_contact_report = reports.append
    user.set_shoulder_girdle = lambda *args, **kwargs: pytest.fail("shoulder map must not refit")

    handlefit.mirror_arm(user, side)

    for base in bases:
        destination = coordinates.get(f"{base}_{target}")
        assert destination.value == pytest.approx(coordinates.get(f"{base}_{source}").value)
        assert destination.locked == target_locks[base]
    assert coordinates.get("hip_flexion_r").value == 0.1
    assert coordinates.get("hip_flexion_l").value == -0.4
    assert reports == [target]


def test_mirror_arm_rejects_missing_target_before_modifying_pose(fit_setup):
    user, _, coordinates, _ = fit_setup
    names = [coordinate.name for coordinate in coordinates.coordinates]
    coordinates.coordinates.extend(
        Coordinate(name[:-2] + "_l", value=-0.5)
        for name in names if name != "wrist_flex_r"
    )
    with pytest.raises(RuntimeError, match="wrist_flex_l"):
        handlefit.mirror_arm(user, "right")
    assert coordinates.get("arm_rot_l").value == -0.5


def test_simulation_passes_fitting_weights_to_initial_fit():
    captured = {}

    def fit_user(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("initial fit reached")

    equipment = SimpleNamespace(fit_user=fit_user)
    with pytest.raises(RuntimeError, match="initial fit reached"):
        ChestPress.simulate_kinematics(
            equipment, user=object(), protraction_weight=42.0,
            scapular_elevation_weight=23.0,
            knuckle_end_weight=12345.0, palm_contact_weight=2345.0,
            wrist_flexion_weight=3456.0,
        )
    assert captured["protraction_weight"] == 42.0
    assert captured["scapular_elevation_weight"] == 23.0
    assert captured["knuckle_end_weight"] == 12345.0
    assert captured["palm_contact_weight"] == 2345.0
    assert captured["wrist_flexion_weight"] == 3456.0


def test_fit_and_simulation_share_default_weights():
    fit = inspect.signature(ChestPress.fit_user)
    simulation = inspect.signature(ChestPress.simulate_kinematics)
    fit_weights = {
        name: parameter.default for name, parameter in fit.parameters.items()
        if name.endswith("_weight")
    }
    simulation_weights = {
        name: parameter.default for name, parameter in simulation.parameters.items()
        if name.endswith("_weight")
    }
    assert fit_weights == simulation_weights


def test_kinematic_simulation_uses_only_new_public_name():
    assert callable(ChestPress.simulate_kinematics)
    assert not hasattr(ChestPress, "simulate_on_user")


def test_fit_user_returns_messages_for_each_fit(monkeypatch):
    class FakeUser:
        def __init__(self):
            self.state = object()
            self.coordinates = CoordinateSet([
                Coordinate("arm_rot_r"), Coordinate("index_mcp_flexion_r"),
                Coordinate("index_mcp_flexion_l"),
            ])
            self.grasp_calls = []
            self.finger_locks = []

        def getCoordinateSet(self):
            return self.coordinates

        def realizePosition(self, state):
            pass

        def grasp_handle(self, side, body, meshes):
            self.grasp_calls.append(side)
            self.coordinates.get("index_mcp_flexion_r").value = 0.7
            return {"fitted": side}

        def set_finger_coordinate_locked(self, side, finger, joint, motion, locked):
            self.finger_locks.append((side, finger, joint, motion, locked))

    coordinates = CoordinateSet([
        Coordinate("right_lever-angle", minimum=0.0, maximum=np.deg2rad(2)),
        Coordinate("left_lever-angle", minimum=0.0, maximum=np.deg2rad(2)),
    ])
    equipment = SimpleNamespace(
        state=object(),
        getCoordinateSet=lambda: coordinates,
        realizePosition=lambda state: None,
        getBodySet=lambda: SimpleNamespace(get=lambda name: name),
        handle_meshes=lambda side: [],
        _backrest_angle_deg=0.0,
    )
    user = FakeUser()
    monkeypatch.setattr(chestpress, "User", FakeUser)
    framework = SimpleNamespace(user=user, equipment=equipment)
    monkeypatch.setattr(chestpress, "Framework", lambda **kwargs: framework)
    monkeypatch.setattr(chestpress, "position_user", lambda *args, **kwargs: {})
    logger = logging.getLogger("test.chestpress.mirror")
    monkeypatch.setattr(logger, "propagate", False)
    monkeypatch.setattr(logger, "level", logging.INFO)
    monkeypatch.setattr(logger, "handlers", [logging.NullHandler()])

    def mirror_arm(user, to_mirror):
        assert user.grasp_calls == ["right"]
        user.coordinates.get("index_mcp_flexion_l").value = (
            user.coordinates.get("index_mcp_flexion_r").value
        )
        warnings.warn("l shoulder map: optimization did not converge", RuntimeWarning)
        warnings.warn("l scapula: intersection with ribs or spine: True", RuntimeWarning)
        logger.info("left arm mirrored")

    monkeypatch.setattr(handlefit, "mirror_arm", mirror_arm)
    calls = []

    def fit_handle(**kwargs):
        calls.append(kwargs)
        arm_rotation = user.coordinates.get("arm_rot_r")
        if len(calls) == 1:
            arm_rotation.value = 0.3
        elif len(calls) == 2:
            arm_rotation.value = 0.8
        else:
            assert arm_rotation.value == pytest.approx(0.3)
        return (90.0 if len(calls) == 1 else 0.0), np.zeros(3), [{"message": str(len(calls))}]

    monkeypatch.setattr(chestpress, "fit_handle", fit_handle)
    with warnings.catch_warnings(record=True) as visible:
        result, report = ChestPress.fit_user(
            equipment, user, step_deg=1.0, palm_contact_weight=2345.0
        )
    assert not visible
    assert result is framework
    assert user.grasp_calls == ["right"]
    assert user.coordinates.get("index_mcp_flexion_l").value == pytest.approx(0.7)
    assert len(user.finger_locks) == 16
    assert all(side == "left" and locked for side, _, _, _, locked in user.finger_locks)
    assert report["grip"]["fingers"] == {"right": {"fitted": "right"}, "left": None}
    assert all(call["palm_contact_weight"] == 2345.0 for call in calls)
    assert all("palm_position_handle" not in call and "palm_position_weight" not in call for call in calls)
    entries = report["grip"]["handle_fit_log"]
    assert [entry["phase"] for entry in entries] == ["sweep", "sweep", "final"]
    assert [entry["lever_angle_deg"] for entry in entries] == pytest.approx([0, 1, 0])
    assert [entry["messages"][0]["message"] for entry in entries] == ["1", "2", "3"]
    mirrored = entries[-1]["messages"][1:]
    assert [message["message"] for message in mirrored] == [
        "l shoulder map: optimization did not converge",
        "l scapula: intersection with ribs or spine: True",
        "left arm mirrored",
    ]
    assert [message["kind"] for message in mirrored] == ["warning", "warning", "log"]
    assert all(message["function"] == "mirror_arm" for message in mirrored)
    assert all(message["filename"] == __file__ for message in mirrored)
    assert all(isinstance(message["line"], int) for message in mirrored)
    assert all(message["source"] for message in mirrored)
    monkeypatch.setattr(
        chestpress, "fit_handle", lambda **kwargs: (None, None, [{"message": "failed"}])
    )
    with pytest.raises(ValueError, match="No valid handle fits") as raised:
        ChestPress.fit_user(equipment, user, step_deg=1.0)
    assert raised.value.handle_fit_log == [
        {"phase": "sweep", "lever_angle_deg": 0.0, "messages": [{"message": "failed"}]}
    ]


def test_failed_mirror_preserves_accumulated_log_and_restores_hooks(monkeypatch):
    log = [{"phase": "final", "lever_angle_deg": 0.0, "messages": [{"message": "fit"}]}]
    original_factory = logging.getLogRecordFactory()
    original_showwarning = warnings.showwarning

    def failed_mirror(user, to_mirror):
        warnings.warn("left scapula warning", RuntimeWarning)
        raise ValueError("mirror failed")

    monkeypatch.setattr(handlefit, "mirror_arm", failed_mirror)
    with warnings.catch_warnings(record=True) as visible:
        with pytest.raises(ValueError, match="mirror failed") as raised:
            handlefit.mirror_arm_with_log(object(), log)
    assert not visible
    assert raised.value.handle_fit_log is log
    assert [message["message"] for message in log[-1]["messages"]] == [
        "fit", "left scapula warning"
    ]
    assert logging.getLogRecordFactory() is original_factory
    assert warnings.showwarning is original_showwarning


def _simulation_model(names):
    model = Model()
    for name in names:
        body = osim.Body(f"{name}_body", 1.0, osim.Vec3(0), osim.Inertia(1))
        model.addBody(body)
        joint = osim.PinJoint(
            f"{name}_joint", model.getGround(), osim.Vec3(0), osim.Vec3(0),
            body, osim.Vec3(0), osim.Vec3(0),
        )
        joint.updCoordinate().setName(name)
        joint.updCoordinate().setRangeMin(0.0)
        joint.updCoordinate().setRangeMax(np.deg2rad(2.0))
        model.addJoint(joint)
    model.state = model.initSystem()
    return model


@pytest.mark.parametrize("fail_at_end", [False, True])
def test_simulation_records_synchronized_step_trajectories(monkeypatch, fail_at_end):
    user = _simulation_model(["arm_r", "arm_l"])
    equipment = _simulation_model(["right_lever-angle", "left_lever-angle"])
    framework = Framework(user=user, equipment=equipment)
    wrapper = SimpleNamespace(
        fit_user=lambda **kwargs: (
            framework, {"grip": {
                "handle_fit_log": [{"phase": "final", "lever_angle_deg": 0.0, "messages": []}],
            }}
        )
    )
    fitting_weights = []
    def fit_handle(**kwargs):
        fitting_weights.append(kwargs["palm_contact_weight"])
        assert "palm_position_handle" not in kwargs
        assert "palm_position_weight" not in kwargs
        angle = equipment.getCoordinateSet().get("right_lever-angle").getValue(equipment.state)
        if fail_at_end and angle == pytest.approx(np.deg2rad(2.0)):
            return None, None, [{
                "message": (
                    "right handle fit did not converge: evaluation limit exceeded; "
                    "nfev=1000, cost=42, optimality=3"
                )
            }]
        user.getCoordinateSet().get("arm_r").setValue(user.state, angle * 0.5, False)
        return (0.5 if angle == pytest.approx(np.deg2rad(2.0)) else 90.0), None, []

    def mirror_arm(user, to_mirror):
        warnings.warn("l scapula: intersection", RuntimeWarning)
        coords = user.getCoordinateSet()
        coords.get("arm_l").setValue(user.state, coords.get("arm_r").getValue(user.state), False)

    monkeypatch.setattr(chestpress, "fit_handle", fit_handle)
    monkeypatch.setattr(handlefit, "mirror_arm", mirror_arm)
    with warnings.catch_warnings(record=True) as visible:
        if fail_at_end:
            with pytest.warns(
                RuntimeWarning,
                match="concentric press \\(outward\\) phase at 2.00 degrees",
            ) as outward_warning:
                result, fit_log = ChestPress.simulate_kinematics(
                    wrapper, user, step_deg=1.0,
                    concentric_duration_s=1.5, eccentric_duration_s=2.5,
                    palm_contact_weight=4321.0,
                )
            assert "evaluation limit exceeded" in str(outward_warning[0].message)
            assert "nfev=1000, cost=42, optimality=3" in str(
                outward_warning[0].message
            )
        else:
            result, fit_log = ChestPress.simulate_kinematics(
                wrapper, user, step_deg=1.0,
                concentric_duration_s=1.5, eccentric_duration_s=2.5,
                palm_contact_weight=4321.0,
            )
    assert not visible

    expected_degrees = [0, 1, 0] if fail_at_end else [0, 1, 2, 1, 0]
    assert fitting_weights and all(weight == 4321.0 for weight in fitting_weights)
    assert [entry["phase"] for entry in fit_log] == (
        ["final", "outward", "outward", "outward", "return"]
        if fail_at_end else ["final", "outward", "outward", "outward", "return", "return"]
    )
    assert [entry["lever_angle_deg"] for entry in fit_log] == pytest.approx(
        [0, 0, 1, 2, 0] if fail_at_end else [0, 0, 1, 2, 1, 0]
    )
    if fail_at_end:
        assert fit_log[3]["messages"] == [{
            "message": (
                "right handle fit did not converge: evaluation limit exceeded; "
                "nfev=1000, cost=42, optimality=3"
            )
        }]
    for entry in fit_log[1:]:
        if (
            fail_at_end
            and entry["phase"] == "outward"
            and entry["lever_angle_deg"] == pytest.approx(2.0)
        ):
            continue
        assert [message["message"] for message in entry["messages"]] == [
            "l scapula: intersection"
        ]
        assert entry["messages"][0]["function"] == "mirror_arm"
    assert result is framework
    if fail_at_end:
        expected_times = [0.0, 1.5, 4.0]
    else:
        expected_times = [
            0.0,
            1.5 * beta_time_for_position(0.5),
            1.5,
            1.5 + 2.5 * gaussian_time_for_position(0.5),
            4.0,
        ]
    assert result.state_times == pytest.approx(expected_times)
    assert result.states_trajectory.getSize() == len(expected_degrees)
    assert all(t.getSize() == len(expected_degrees) for t in result.states_trajectories.values())
    for index, degrees in enumerate(expected_degrees):
        angle = np.deg2rad(degrees)
        equipment_state = result.states_trajectories["equipment"].get(index)
        user_state = result.states_trajectories["user"].get(index)
        for side in ("left", "right"):
            coordinate = equipment.getCoordinateSet().get(f"{side}_lever-angle")
            assert coordinate.getValue(equipment_state) == pytest.approx(angle)
        for side in ("l", "r"):
            coordinate = user.getCoordinateSet().get(f"arm_{side}")
            assert coordinate.getValue(user_state) == pytest.approx(angle * 0.5)
        if fail_at_end:
            expected_speed = 0.0
        elif index == 1:
            phase_time = beta_time_for_position(0.5)
            expected_speed = (
                np.deg2rad(2.0) / 1.5
                * beta_velocity_profile(phase_time)
            )
        elif index == 3:
            phase_time = gaussian_time_for_position(0.5)
            expected_speed = (
                -np.deg2rad(2.0) / 2.5
                * gaussian_velocity_profile(phase_time)
            )
        else:
            expected_speed = 0.0
        assert equipment.getCoordinateSet().get(
            "right_lever-angle"
        ).getSpeedValue(equipment_state) == pytest.approx(expected_speed)
        assert user.getCoordinateSet().get("arm_r").getSpeedValue(
            user_state
        ) == pytest.approx(expected_speed * 0.5)
        composite_state = result.states_trajectory.get(index)
        assert composite_state.getTime() == pytest.approx(result.state_times[index])
        for member, model, name, value in (
            ("equipment", equipment, "right_lever-angle", angle),
            ("equipment", equipment, "left_lever-angle", angle),
            ("user", user, "arm_r", angle * 0.5),
            ("user", user, "arm_l", angle * 0.5),
        ):
            coordinate = model.getCoordinateSet().get(name)
            path = result.path_map[(member, coordinate.getAbsolutePathString())]
            assert result.composite.getStateVariableValue(
                composite_state, f"{path}/value"
            ) == pytest.approx(value)
    assert result.get_motion_table(1.0).getNumRows() == 5


def test_simulation_rejects_fewer_than_two_states(monkeypatch):
    user = _simulation_model(["arm_r", "arm_l"])
    equipment = _simulation_model(["right_lever-angle", "left_lever-angle"])
    framework = Framework(user=user, equipment=equipment)
    wrapper = SimpleNamespace(
        fit_user=lambda **kwargs: (
            framework, {"grip": {
                "handle_fit_log": [],
            }}
        )
    )
    monkeypatch.setattr(chestpress, "fit_handle", lambda **kwargs: (None, None, []))
    with pytest.warns(
        RuntimeWarning,
        match="concentric press \\(outward\\) phase at 0.00 degrees",
    ) as outward_warning:
        with pytest.raises(ValueError, match="fewer than two") as raised:
            ChestPress.simulate_kinematics(wrapper, user)
    assert "optimizer failure details are missing from the fit log" in str(
        outward_warning[0].message
    )
    assert raised.value.handle_fit_log == [
        {"phase": "outward", "lever_angle_deg": 0.0, "messages": []}
    ]
    assert framework.states_trajectory.getSize() == 0


def test_failed_solver_restores_initial_coordinates(fit_setup, monkeypatch):
    user, equipment, coordinates, _ = fit_setup
    coordinates.get("sc_protraction_r").value = 0.2
    coordinates.get("ac_protraction_r").value = 0.4

    def failed_solve(objective, x0, *, bounds, max_nfev, x_scale, ftol, xtol, gtol):
        objective(x0 + 0.1)
        return SimpleNamespace(
            success=False, nfev=1000, x=x0 + 0.1,
            message="evaluation limit exceeded", cost=42.0, optimality=3.0,
        )

    monkeypatch.setattr(handlefit, "least_squares", failed_solve)
    elbow, palm, messages = handlefit.fit_handle(user, equipment, "right_handle", "right")
    assert elbow is None and palm is None
    assert any(
        "evaluation limit exceeded; nfev=1000, cost=42, optimality=3" in m["message"]
        for m in messages
    )
    assert coordinates.get("sc_protraction_r").value == pytest.approx(0.2)
    assert coordinates.get("ac_protraction_r").value == pytest.approx(0.4)


def test_solver_settings_apply_to_fit(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.4, 0.5]
    calls = []

    def solve(objective, x0, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(success=True, nfev=1, x=x0.copy())

    monkeypatch.setattr(handlefit, "least_squares", solve)
    elbow, _, _ = handlefit.fit_handle(user, equipment, "right_handle", "right")
    assert elbow is not None
    assert len(calls) == 1
    for options in calls:
        assert options["x_scale"] == "jac"
        assert options["ftol"] == options["xtol"] == options["gtol"] == 1e-5
        assert "diff_step" not in options


def test_scapula_intersection_warns_without_changing_fitted_pose(fit_setup, monkeypatch):
    user, equipment, coordinates, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.4, 0.5]
    def report(side):
        warnings.warn(f"{side} scapula intersects ribs", RuntimeWarning)
        return {"gap_m": 0.001, "intersects": True}

    user.scapula_contact_report = report
    (elbow, palm, messages), _ = run_fit(
        monkeypatch, user, equipment, solution={"sc_elevation_r": 0.3}
    )
    assert any("scapula intersects ribs" in m["message"] for m in messages)
    assert elbow is not None
    assert palm is not None
    assert coordinates.get("sc_elevation_r").value == pytest.approx(0.3)


def test_fit_log_captures_warning_and_nonpropagating_logger_with_origin(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.4, 0.5]
    logger = logging.getLogger("test.handlefit.nonpropagating")
    monkeypatch.setattr(logger, "propagate", False)
    monkeypatch.setattr(logger, "level", logging.INFO)
    monkeypatch.setattr(logger, "handlers", [logging.NullHandler()])

    def report(side):
        warnings.warn("contact warning", UserWarning, stacklevel=2)
        logger.info("contact side %s", side)

    user.scapula_contact_report = report
    with warnings.catch_warnings(record=True) as visible:
        (_, _, messages), _ = run_fit(monkeypatch, user, equipment)
    assert not visible
    assert [message["kind"] for message in messages[:2]] == ["warning", "log"]
    assert messages[0]["message"] == "contact warning"
    assert messages[0]["category"] == "UserWarning"
    assert messages[0]["function"] == "report"
    assert "warnings.warn(" in messages[0]["source"]
    assert messages[1]["message"] == "contact side r"
    assert messages[1]["level"] == "INFO"
    assert messages[1]["logger"] == logger.name
    assert messages[1]["function"] == "report"
    assert "logger.info(" in messages[1]["source"]
    assert all(m["filename"] == __file__ and isinstance(m["line"], int) for m in messages[:2])


def test_fit_restores_capture_hooks_after_exception(fit_setup, monkeypatch):
    user, equipment, _, _ = fit_setup
    original_factory = logging.getLogRecordFactory()
    original_showwarning = warnings.showwarning

    def report(side):
        warnings.warn("before failure", RuntimeWarning)
        raise ValueError("contact check failed")

    user.scapula_contact_report = report
    with pytest.raises(ValueError, match="contact check failed") as raised:
        run_fit(monkeypatch, user, equipment)
    assert raised.value.handle_fit_log[0]["message"] == "before failure"
    assert logging.getLogRecordFactory() is original_factory
    assert warnings.showwarning is original_showwarning


def test_fit_signature_describes_the_returned_log():
    assert inspect.signature(handlefit.fit_handle).return_annotation == (
        get_type_hints(handlefit.fit_handle)["return"]
    )
    assert len(inspect.signature(handlefit.fit_handle).return_annotation.__args__) == 3
