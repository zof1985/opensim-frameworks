import inspect
import logging
import re
import warnings
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import opensim as osim
import pytest
import vtk

from opensim_frameworks.models.equipment.chestpress import (
    __all__ as chestpress_exports,
    chestpress,
    handlefit,
)
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
            f"{finger}_mcp_r": (
                np.array([0.32, 0.30 + 0.02 * index, 0.5])
                if finger == "middle"
                else np.array([0.3, 0.30 + 0.02 * index, 0.55])
            )
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
    palm_ground = np.array([0.325, 0.4, 0.5])

    def marker_position(marker, state):
        if marker is palm:
            return palm_ground.copy()
        return marker_position_array(marker, state)

    monkeypatch.setattr(handlefit, "marker_position_array", marker_position)
    return user, equipment, coordinates, palm_ground


def run_fit(
    monkeypatch,
    user,
    equipment,
    *,
    probes=(),
    solution=None,
    **weights,
):
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
        "shoulder_wrist_axis_distance_weight": 0.0,
        "pro_sup_weight": 0.0,
        "wrist_flex_weight": 0.0,
        "wrist_dev_weight": 0.0,
        "protraction_weight": 0.0,
        "scapular_elevation_weight": 0.0,
        "arm_weight": 0.0,
    }
    zero_weights.update(weights)
    result = handlefit.fit_handle(user, equipment, "right_handle", "right", **zero_weights)
    return result, costs


def test_loss_penalties_are_public_and_exported():
    names = (
        "palm_axis_distance_penalty",
        "wrist_flexion_penalty",
        "wrist_deviation_penalty",
        "forearm_supination_penalty",
        "external_shoulder_rotation_penalty",
        "scapular_retraction_penalty",
        "scapular_depression_penalty",
        "knuckle_alignment_penalty",
        "index_knuckle_endpoint_penalty",
        "little_knuckle_endpoint_penalty",
        "shoulder_palm_plane_penalty",
        "shoulder_wrist_axis_distance_penalty",
    )

    for name in names:
        assert name in handlefit.__all__
        assert name in chestpress_exports
        assert callable(getattr(handlefit, name))
    for name in (
        "wrist_axis_distance_penalty",
        "palm_surface_penalty",
        "palm_height_penalty",
        "scapulothoracic_contact_penalty",
        "scapular_collision_penalty",
        "shoulder_wrist_plane_penalty",
    ):
        assert name not in handlefit.__all__
        assert not hasattr(handlefit, name)


def test_public_loss_penalties_return_their_expected_residuals():
    frame = np.eye(4)
    origin = np.zeros(3)
    assert handlefit.palm_axis_distance_penalty(
        [0.025, 0.0, 0.0], frame, origin, [0.0, 1.0, 0.0], 0.02, 4.0
    ) == pytest.approx(10.0)
    assert handlefit.wrist_deviation_penalty(np.deg2rad(2.0), 1.0) == pytest.approx(2.0)
    assert handlefit.wrist_flexion_penalty(np.deg2rad(5.0), 4.0) == pytest.approx(70.0)
    assert handlefit.wrist_flexion_penalty(0.0, 4.0) == pytest.approx(60.0)
    assert handlefit.wrist_flexion_penalty(np.deg2rad(-5.0), 4.0) == pytest.approx(50.0)
    assert handlefit.wrist_flexion_penalty(np.deg2rad(-30.0), 1.0) == 0.0
    assert handlefit.wrist_flexion_penalty(np.deg2rad(-40.0), 1.0) == 0.0
    assert handlefit.forearm_supination_penalty(np.deg2rad(85.0), 1.0) == pytest.approx(-1.0)
    assert handlefit.external_shoulder_rotation_penalty(np.deg2rad(-5.0), 1.0) == pytest.approx(-1.0)
    assert handlefit.scapular_retraction_penalty(2.0, 1.0, 4.0) == pytest.approx(2.0)
    assert handlefit.scapular_depression_penalty(0.2, 4.0) == pytest.approx(0.4)
    np.testing.assert_allclose(
        handlefit.knuckle_alignment_penalty(
            [0, 0, 1], [0, 0, 0], [0, 0, 1], 1.0
        ),
        np.zeros(4),
        atol=1e-12,
    )
    assert handlefit.index_knuckle_endpoint_penalty(0.002, 0.001, 1.0) == pytest.approx(1.0)
    assert handlefit.little_knuckle_endpoint_penalty(0.001, 0.001, 1.0) == 0.0
    one_degree = np.deg2rad(1.0)
    assert handlefit.shoulder_palm_plane_penalty(
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [np.cos(one_degree), 0, np.sin(one_degree)],
        4.0,
    ) == pytest.approx(2.0)
    assert handlefit.shoulder_wrist_axis_distance_penalty(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        1.0,
    ) == pytest.approx(0.0)
    assert handlefit.shoulder_wrist_axis_distance_penalty(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, -1.0, 0.002],
        [0.0, 1.0, 0.0],
        4.0,
    ) == pytest.approx(4.0)
    assert handlefit.shoulder_wrist_axis_distance_penalty(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.002],
        [1.0, 0.0, 0.0],
        4.0,
    ) == pytest.approx(4.0)


def test_scapular_fit_residuals_are_added_directly_to_the_objective(
    fit_setup, monkeypatch
):
    user, equipment, _, _ = fit_setup
    _, baseline_costs = run_fit(monkeypatch, user, equipment, probes=[{}])
    user.scapular_fit_residuals = lambda side, protraction, elevation: np.array(
        [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    )
    _, contact_costs = run_fit(monkeypatch, user, equipment, probes=[{}])
    assert contact_costs[0] - baseline_costs[0] == pytest.approx(30.0)


@pytest.mark.parametrize("palm_y", [0.30, 0.40, 0.50])
def test_palm_axis_target_is_invariant_to_sliding_along_the_axis(
    fit_setup, monkeypatch, palm_y
):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.325, palm_y, 0.5]
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
    assert costs[0] == pytest.approx(
        2.0 * (0.005 / handlefit._POSITION_TOLERANCE_M) ** 2,
        abs=1e-8,
    )


@pytest.mark.parametrize("radius", [0.02, 0.025, 0.03])
def test_palm_contact_targets_the_actual_cylinder_radius(fit_setup, monkeypatch, radius):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.33, 0.4, 0.5]
    monkeypatch.setattr(
        handlefit, "enclosing_cylinder",
        lambda vertices: (
            np.array([0.3, 0.4, 0.5, 0.0, 1.0, 0.0]), radius,
            np.array([0.3, 0.28, 0.5]), np.array([0.3, 0.52, 0.5]),
        ),
    )
    original_joint_center = user.joint_center
    user.joint_center = lambda name: (
        np.array([0.3 + radius, 0.4, 0.5])
        if name == "middle_mcp_r"
        else original_joint_center(name)
    )
    _, costs = run_fit(monkeypatch, user, equipment, probes=[{}], palm_contact_weight=2.0)
    expected = 2.0 * ((0.03 - radius) / handlefit._POSITION_TOLERANCE_M) ** 2
    assert costs[0] == pytest.approx(expected, abs=1e-8)


def test_palm_axis_target_is_independent_of_wrist_position(
    fit_setup, monkeypatch
):
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
    np.testing.assert_allclose(
        costs,
        [50.0, 50.0],
        atol=0.002,
    )


def test_middle_knuckle_targets_the_handle_radius(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.32, 0.4, 0.5]
    monkeypatch.setattr(
        handlefit,
        "enclosing_cylinder",
        lambda vertices: (
            np.array([0.3, 0.4, 0.5, 0.0, 1.0, 0.0]),
            0.02,
            np.array([0.3, 0.28, 0.5]),
            np.array([0.3, 0.52, 0.5]),
        ),
    )
    points = {"middle_mcp_r": np.array([0.33, 0.4, 0.5])}
    original_joint_center = user.joint_center
    user.joint_center = lambda name: points.get(name, original_joint_center(name))

    _, costs = run_fit(
        monkeypatch,
        user,
        equipment,
        probes=[{}],
        palm_contact_weight=2.0,
    )

    assert costs[0] == pytest.approx(200.0, abs=1e-8)


@pytest.mark.parametrize("angle", [0.0, 0.7])
def test_palm_axis_distance_uses_rotated_handle_frame(fit_setup, monkeypatch, angle):
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
    assert costs[0] == pytest.approx(0.0, abs=1e-8)

    palm_local[1] = 0.55
    palm_ground[:] = transform[:3, :3] @ palm_local + transform[:3, 3]
    _, costs = run_fit(
        monkeypatch, user, equipment, probes=[{}],
        palm_contact_weight=1.0,
    )
    assert costs[0] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize(("index_axis", "little_axis", "expected_cost"), [
    (0.55, 0.45, 0.0),
    (0.45, 0.55, 4.0),
])
def test_knuckle_line_must_match_handle_axis_direction(
    fit_setup, monkeypatch, index_axis, little_axis, expected_cost
):
    user, _, _, palm_ground = fit_setup
    palm_ground[:] = [0.32, 0.4, 0.5]
    points = {
        "index_mcp_r": np.array([0.3, index_axis, 0.5]),
        "middle_mcp_r": np.array([0.3, 0.5, 0.5]),
        "ring_mcp_r": np.array([0.3, 0.5, 0.5]),
        "little_mcp_r": np.array([0.3, little_axis, 0.5]),
    }
    original_joint_center = user.joint_center
    user.joint_center = lambda name: points.get(name, original_joint_center(name))

    _, costs = run_fit(
        monkeypatch, user, fit_setup[1], probes=[{}], knuckle_weight=1.0
    )
    assert costs[0] == pytest.approx(expected_cost, abs=1e-7)


def test_knuckle_alignment_uses_lower_to_higher_ground_z_endpoint_direction(
    fit_setup, monkeypatch
):
    user, equipment, _, _ = fit_setup
    monkeypatch.setattr(
        handlefit,
        "enclosing_cylinder",
        lambda vertices: (
            np.array([0.3, 0.4, 0.5, 0.0, 0.0, -1.0]),
            0.02,
            np.array([0.3, 0.4, 0.62]),
            np.array([0.3, 0.4, 0.38]),
        ),
    )
    points = {
        "index_mcp_r": np.array([0.3, 0.4, 0.55]),
        "little_mcp_r": np.array([0.3, 0.4, 0.45]),
    }
    original_joint_center = user.joint_center
    user.joint_center = lambda name: points.get(name, original_joint_center(name))

    _, forward = run_fit(
        monkeypatch, user, equipment, probes=[{}], knuckle_weight=1.0
    )
    points["index_mcp_r"], points["little_mcp_r"] = (
        points["little_mcp_r"],
        points["index_mcp_r"],
    )
    _, reversed_direction = run_fit(
        monkeypatch, user, equipment, probes=[{}], knuckle_weight=1.0
    )

    assert forward[0] == pytest.approx(0.0, abs=1e-8)
    assert reversed_direction[0] == pytest.approx(4.0, abs=1e-8)


def test_knuckle_projections_are_one_sided_endpoint_limits(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.32, 0.4, 0.5]
    points = {
        "index_mcp_r": np.array([0.3, 0.4, 0.52]),
        "middle_mcp_r": np.array([0.3, 0.4, 0.5]),
        "ring_mcp_r": np.array([0.3, 0.4, 0.5]),
        "little_mcp_r": np.array([0.3, 0.4, 0.48]),
    }
    original_joint_center = user.joint_center
    user.joint_center = lambda name: points.get(name, original_joint_center(name))

    _, costs = run_fit(
        monkeypatch,
        user,
        equipment,
        probes=[{}],
        knuckle_end_weight=1.0,
    )
    assert costs[0] == pytest.approx(0.0, abs=1e-8)

    points["index_mcp_r"] = np.array([0.3, 0.4, 0.7])
    points["little_mcp_r"] = np.array([0.3, 0.4, 0.3])
    _, outside_costs = run_fit(
        monkeypatch, user, equipment, probes=[{}], knuckle_end_weight=1.0
    )
    assert outside_costs[0] > 0.0


def test_forearm_rotation_is_fitted_toward_90_degrees(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.4, 0.5]
    angles = [65.0, 75.0, 90.0, 115.0, 125.0]
    _, costs = run_fit(
        monkeypatch, user, equipment,
        probes=[{"pro_sup_r": np.deg2rad(angle)} for angle in angles],
        pro_sup_weight=1.0,
    )
    np.testing.assert_allclose(
        costs,
        [25.0, 9.0, 0.0, 25.0, 49.0],
        atol=1e-12,
    )


def test_fit_handle_favors_at_least_30_degrees_of_wrist_extension(
    fit_setup, monkeypatch
):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.4, 0.5]
    _, costs = run_fit(
        monkeypatch, user, equipment,
        probes=[
            {"wrist_flex_r": np.deg2rad(angle)} for angle in (-40.0, -30.0, -20.0, 0.0)
        ],
        wrist_flex_weight=1.0,
    )
    np.testing.assert_allclose(
        costs,
        [0.0, 0.0, 10.0**2, 30.0**2],
        atol=1e-12,
    )
    assert "wrist_extension_penalty" not in handlefit.__all__
    assert not hasattr(handlefit, "wrist_extension_penalty")


def test_upper_arm_rotation_penalizes_external_rotation_only(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.4, 0.5]
    _, costs = run_fit(
        monkeypatch, user, equipment,
        probes=[{"arm_rot_r": angle} for angle in (-0.5, 0.0, 0.5)],
        arm_weight=1.0,
    )
    np.testing.assert_allclose(
        costs,
        [(-0.5 / np.deg2rad(5.0)) ** 2, 0.0, 0.0],
        atol=1e-12,
    )


def test_fit_log_reports_forearm_and_upper_arm_outside_preferred_ranges(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.4, 0.5]
    (_, _, messages), _ = run_fit(
        monkeypatch, user, equipment,
        solution={"pro_sup_r": np.deg2rad(60.0), "arm_rot_r": np.deg2rad(-10.0)},
        pro_sup_weight=1.0, arm_weight=1.0,
    )
    assert any("forearm_supination: residual=-6, loss=36" in m["message"] for m in messages)
    assert any("external_shoulder_rotation: residual=-2, loss=4" in m["message"] for m in messages)


def test_palm_height_has_no_preferred_side_of_the_handle_axis(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    original_joint_center = user.joint_center
    user.joint_center = lambda name: (
        np.array([0.32, 0.4, 0.5])
        if name == "middle_mcp_r"
        else original_joint_center(name)
    )
    axis = np.array([0.3, 0.4, 0.5, 1.0, 0.0, 0.0])
    monkeypatch.setattr(
        handlefit,
        "enclosing_cylinder",
        lambda vertices: (
            axis,
            0.02,
            np.array([0.3, 0.4, 0.38]),
            np.array([0.3, 0.4, 0.62]),
        ),
    )
    costs = []
    for palm_y in (0.42, 0.38):
        palm_ground[:] = [0.3, palm_y, 0.5]
        fit_result, fit_costs = run_fit(
            monkeypatch,
            user,
            equipment,
            probes=[{}],
            palm_contact_weight=1.0,
        )
        assert fit_result[0] is not None
        costs.extend(fit_costs)
        assert not any(
            "palm_below_axis" in message["message"]
            for message in fit_result[2]
        )
    np.testing.assert_allclose(costs, [0.0, 0.0], atol=1e-12)


def test_knuckle_endpoint_violation_is_reported(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.375, 0.5]
    original_joint_center = user.joint_center
    user.joint_center = lambda name: (
        np.array([0.3, 0.35, 0.5])
        if name == "little_mcp_r"
        else (
            np.array([0.3, 0.49, 0.5])
            if name == "index_mcp_r"
            else original_joint_center(name)
        )
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
        "index_endpoint_projection" in m["message"]
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
            point[2] += coordinates.get("sc_elevation_r").value
            point[2] += coordinates.get("wrist_dev_r").value
        return point

    user.joint_center = joint_center
    (elbow, palm, messages), _ = run_fit(
        monkeypatch, user, equipment,
        solution={"sc_elevation_r": 0.2}, wrist_dev_weight=1.0,
    )
    assert any(
        "index_endpoint_projection" in m["message"]
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
        original_joint_center(name) + np.array([0.0, 0.0, 0.5])
        if name.endswith("_mcp_r") else original_joint_center(name)
    )
    coordinates.get("arm_rot_r").value = 0.2
    scapula_checks = []
    user.scapula_contact_report = lambda side: scapula_checks.append(side)
    (elbow, palm, messages), _ = run_fit(
        monkeypatch, user, equipment, solution={"arm_rot_r": 0.6}
    )
    assert any(
        "index_endpoint_projection" in m["message"]
        for m in messages
    )
    assert elbow is not None
    assert palm is not None
    assert coordinates.get("arm_rot_r").value == pytest.approx(0.6)
    assert scapula_checks == ["r"]


def test_palm_inside_handle_is_logged_without_rejection(fit_setup, monkeypatch):
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
    assert elbow is not None
    assert palm is not None
    assert any("palm_axis_distance: residual=-20" in m["message"] for m in messages)
    assert coordinates.get("arm_rot_r").value == pytest.approx(0.6)


def test_fit_log_reports_palm_gap_far_from_target(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.335, 0.4, 0.5]
    (_, _, messages), _ = run_fit(
        monkeypatch, user, equipment, palm_contact_weight=1.0,
    )
    message = next(
        m["message"] for m in messages if "palm_axis_distance: residual=" in m["message"]
    )
    match = re.search(
        r"palm_axis_distance: residual=([-+0-9.e]+), loss=([-+0-9.e]+)",
        message,
    )
    assert match is not None
    residual, loss = map(float, match.groups())
    assert residual != 0.0
    assert loss == pytest.approx(residual**2, rel=1e-5)


def test_fit_loss_report_contains_each_component(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.4, 0.5]
    original_joint_center = user.joint_center
    def joint_center(name):
        if name == "radius_hand_r":
            return palm_ground.copy()
        if name == "index_mcp_r":
            return np.array([0.3, 0.8, 0.5])
        if name == "little_mcp_r":
            return np.array([0.3, 0.3, 0.5])
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
    assert "palm_axis_distance" in warning
    assert "index_endpoint_projection" in warning
    assert "little_endpoint_projection" in warning
    assert "shoulder_wrist_plane" in warning


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
        original_joint_center(name)
        + np.array([0.0, 0.0, coordinates.get("sc_elevation_r").value])
        if name.endswith("_mcp_r") else original_joint_center(name)
    )

    (elbow, palm, messages), _ = run_fit(
        monkeypatch, user, equipment, solution={"sc_elevation_r": 0.5}
    )
    assert any(
        "index_endpoint_projection" in m["message"]
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
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [1.0, 0.0, 1.0]
    points = {
        "acromial_r": np.array([0.0, 0.0, 0.0]),
        "elbow_r": np.array([0.5, 0.5, 0.0]),
        "radius_hand_r": np.array([1.0, 0.0, 0.0]),
        **{
            f"{finger}_mcp_r": np.array([0.3, 0.36 - 0.02 * index, 0.55])
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
    plane_normal = np.cross(points["elbow_r"], points["radius_hand_r"])
    expected_deviation = abs(np.dot(plane_normal, [0.0, 1.0, 0.0])) / np.linalg.norm(
        plane_normal
    )
    assert displaced[0] == pytest.approx(
        (
            expected_deviation / np.sin(handlefit._ANGLE_TOLERANCE_RAD)
        ) ** 2,
        abs=1e-8,
    )
    for index, finger in enumerate(("index", "middle", "ring", "little")):
        points[f"{finger}_mcp_r"] = np.array([0.3, 0.30 + 0.02 * index, 0.55])
    _, crossed = run_fit(
        monkeypatch, user, equipment, probes=[{}], knuckle_weight=1.0
    )
    assert crossed[0] == pytest.approx(4.0, abs=1e-8)


def test_fit_penalizes_distance_between_shoulder_wrist_and_handle_axis_lines(
    fit_setup, monkeypatch
):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.32, 0.4, 0.5]
    points = {
        "acromial_r": np.array([0.0, 0.0, 0.0]),
        "elbow_r": np.array([0.0, 0.5, 0.0]),
        "radius_hand_r": np.array([1.0, 0.0, 0.0]),
    }
    original_joint_center = user.joint_center
    user.joint_center = lambda name: points.get(name, original_joint_center(name))
    monkeypatch.setattr(
        handlefit,
        "enclosing_cylinder",
        lambda vertices: (
            np.array([0.5, 0.0, 0.002, 0.0, 1.0, 0.0]),
            0.02,
            np.array([0.5, -0.5, 0.002]),
            np.array([0.5, 0.5, 0.002]),
        ),
    )

    _, costs = run_fit(
        monkeypatch,
        user,
        equipment,
        probes=[{}],
        shoulder_wrist_axis_distance_weight=2.0,
    )

    assert costs[0] == pytest.approx(8.0, abs=1e-8)


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


def test_fit_does_not_reject_scapula_contact_warning(fit_setup, monkeypatch):
    user, equipment, coordinates, _ = fit_setup
    user.scapula_contact_report = lambda side: warnings.warn(
        "scapula intersects ribs", RuntimeWarning
    )
    coordinates.get("arm_rot_r").value = 0.2

    (elbow, palm, messages), _ = run_fit(
        monkeypatch, user, equipment, solution={"arm_rot_r": 0.6}
    )

    assert elbow is not None and palm is not None
    assert coordinates.get("arm_rot_r").value == pytest.approx(0.6)
    assert any("scapula intersects ribs" in message["message"] for message in messages)


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
            shoulder_wrist_axis_distance_weight=4321.0,
        )
    assert captured["protraction_weight"] == 42.0
    assert captured["scapular_elevation_weight"] == 23.0
    assert captured["knuckle_end_weight"] == 12345.0
    assert captured["palm_contact_weight"] == 2345.0
    assert captured["shoulder_wrist_axis_distance_weight"] == 4321.0


def test_fit_and_simulation_share_default_weights():
    fit = inspect.signature(ChestPress.fit_user)
    simulation = inspect.signature(ChestPress.simulate_kinematics)
    handle_fit = inspect.signature(handlefit.fit_handle)
    fit_weights = {
        name: parameter.default for name, parameter in fit.parameters.items()
        if name.endswith("_weight")
    }
    simulation_weights = {
        name: parameter.default for name, parameter in simulation.parameters.items()
        if name.endswith("_weight")
    }
    assert fit_weights == simulation_weights
    assert fit_weights["knuckle_weight"] == 10.0
    assert fit_weights["shoulder_wrist_axis_distance_weight"] == 10.0
    assert handle_fit.parameters["knuckle_weight"].default == 10.0
    assert handle_fit.parameters["shoulder_wrist_axis_distance_weight"].default == 10.0


def test_kinematic_simulation_uses_only_new_public_name():
    assert callable(ChestPress.simulate_kinematics)
    assert not hasattr(ChestPress, "simulate_on_user")


def test_fit_user_returns_messages_for_each_fit(monkeypatch):
    class FakeUser:
        def __init__(self):
            self.state = object()
            self.coordinates = CoordinateSet([
                Coordinate("right_lever-angle"),
                Coordinate("left_lever-angle"),
                Coordinate("arm_rot_r"),
                Coordinate("index_mcp_flexion_r"),
                Coordinate("arm_rot_l"),
                Coordinate("index_mcp_flexion_l"),
            ])
            self.grasp_calls = []

        def getCoordinateSet(self):
            return self.coordinates

        def grasp_handle(self, side, body, meshes):
            self.grasp_calls.append(side)
            self.coordinates.get("index_mcp_flexion_r").value = 0.7
            return {"fitted": side}

    user = FakeUser()
    coordinates = user.coordinates
    equipment = SimpleNamespace(
        state=object(),
        getCoordinateSet=lambda: coordinates,
        realizePosition=lambda state: None,
        getBodySet=lambda: SimpleNamespace(get=lambda name: name),
        handle_meshes=lambda side: [],
        _backrest_angle_deg=0.0,
    )
    monkeypatch.setattr(chestpress, "User", FakeUser)
    framework = SimpleNamespace(user=user, equipment=equipment)
    monkeypatch.setattr(chestpress, "Framework", lambda **kwargs: framework)
    calls = []
    monkeypatch.setattr(
        chestpress,
        "position_user",
        lambda *args, **kwargs: calls.append("position") or {},
    )
    monkeypatch.setattr(
        chestpress,
        "_palm_position_in_handle",
        lambda *args, **kwargs: np.zeros(3),
    )
    lock_options = []

    def lock_hands(**kwargs):
        assert kwargs.pop("framework") is framework
        calls.append("lock")
        lock_options.append(kwargs)
        return True

    fitting_distance_weights = []

    def fit_handle(**kwargs):
        calls.append("fit")
        fitting_distance_weights.append(kwargs["shoulder_wrist_axis_distance_weight"])
        return 20.0, np.zeros(3), [{"message": "fit message"}]

    monkeypatch.setattr(chestpress, "lock_hands_to_handles", lock_hands)
    monkeypatch.setattr(chestpress, "fit_handle", fit_handle)
    result, report = ChestPress.fit_user(
        equipment,
        user,
        palm_contact_weight=2345.0,
        shoulder_wrist_axis_distance_weight=4321.0,
    )
    assert result is framework
    assert calls == ["position", "fit", "lock"]
    assert fitting_distance_weights == [4321.0]
    assert user.grasp_calls == ["right"]
    assert lock_options == [{"mirror_left_from_right": True, "fit_log": report["grip"]["handle_fit_log"]}]
    assert report["grip"]["fingers"] == {
        "right": {"fitted": "right"}, "left": {"mirrored_from_right": True}
    }
    assert report["grip"]["left_palm_handle_position_m"] == pytest.approx(
        np.zeros(3)
    )
    entries = report["grip"]["handle_fit_log"]
    assert entries == [
        {"phase": "initial", "lever_angle_deg": 0.0, "messages": [
            {"message": "fit message"}
        ]}
    ]
    monkeypatch.setattr(
        chestpress, "fit_handle", lambda **kwargs: (None, None, [{"message": "failed"}])
    )
    with pytest.raises(RuntimeError, match="initial right arm fit") as raised:
        ChestPress.fit_user(equipment, user)
    assert raised.value.handle_fit_log == [
        {"phase": "initial", "lever_angle_deg": 0.0, "messages": [{"message": "failed"}]}
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
    if "right_lever-angle" in names:
        handle = osim.Body("right_handle", 1.0, osim.Vec3(0), osim.Inertia(1))
        model.addBody(handle)
        handle_joint = osim.PinJoint(
            "right_handle_joint",
            model.getGround(),
            osim.Vec3(0),
            osim.Vec3(0),
            handle,
            osim.Vec3(0),
            osim.Vec3(0),
        )
        model.addJoint(handle_joint)
        model.handle_meshes = lambda side: []
    else:
        model.grasp_handle = lambda *args, **kwargs: {}
    for name in names:
        body = osim.Body(f"{name}_body", 1.0, osim.Vec3(0), osim.Inertia(1))
        model.addBody(body)
        joint = osim.PinJoint(
            f"{name}_joint", model.getGround(), osim.Vec3(0), osim.Vec3(0),
            body, osim.Vec3(0), osim.Vec3(0),
        )
        joint.updCoordinate().setName(name)
        joint.updCoordinate().setRangeMin(0.0)
        joint.updCoordinate().setRangeMax(
            np.deg2rad(90.0 if name.startswith("elbow_flex_") else 2.0)
        )
        model.addJoint(joint)
    model.state = model.initSystem()
    return model


def test_palm_anchor_fit_applies_scapular_optimizer_values(monkeypatch):
    class Coordinate:
        def __init__(self, name, maximum):
            self.name = name
            self.value = 0.0
            self.maximum = maximum

        def getName(self):
            return self.name

        def getLocked(self, _state):
            return False

        def getRangeMin(self):
            return 0.0

        def getRangeMax(self):
            return self.maximum

        def getValue(self, _state):
            return self.value

        def setValue(self, _state, value, _enforce_constraints):
            self.value = value

    girdle_names = [
        f"{name}_r" for name in handlefit.SHOULDER_GIRDLE_COORDINATES
    ]
    active_names = {
        "sc_protraction_r",
        "sc_elevation_r",
        "ac_protraction_r",
    }
    coordinates = {
        name: Coordinate(name, 1.0 if name in active_names else 0.0)
        for name in girdle_names
    }

    class CoordinateSet:
        def getSize(self):
            return len(coordinates)

        def get(self, index_or_name):
            if isinstance(index_or_name, int):
                return list(coordinates.values())[index_or_name]
            return coordinates[index_or_name]

    class MarkerSet:
        def getIndex(self, _name):
            return 0

        def get(self, name):
            return name

    user = SimpleNamespace(state=object())
    user.getCoordinateSet = lambda: CoordinateSet()
    user.getMarkerSet = lambda: MarkerSet()
    user.realizePosition = lambda _state: None

    def set_girdle_values(suffix, values, *, check_contact):
        assert suffix == "r"
        assert not check_contact
        for name, value in zip(girdle_names, values, strict=True):
            coordinates[name].value = value

    user.set_scapular_girdle_values = set_girdle_values
    equipment = SimpleNamespace(state=object())
    target = np.array([0.2, 0.3, 0.4])

    def least_squares(residual, initial_values, **_kwargs):
        assert initial_values.shape == (3,)
        values = np.array([0.4, 0.3, 0.2])
        return SimpleNamespace(
            x=values,
            success=True,
            message="test solution",
        )

    monkeypatch.setattr(
        handlefit,
        "body_transform_array",
        lambda _body, _state: np.eye(4),
    )
    monkeypatch.setattr(handlefit, "least_squares", least_squares)
    monkeypatch.setattr(
        handlefit,
        "marker_position_array",
        lambda _marker, _state: np.array(
            [
                coordinates["sc_protraction_r"].value,
                coordinates["sc_elevation_r"].value,
                coordinates["ac_protraction_r"].value,
            ]
        ),
    )

    error, converged, message = handlefit._fit_palm_marker_to_handle(
        user,
        equipment,
        object(),
        "right",
        tuple(target),
    )

    assert [
        coordinates[name].value
        for name in ("sc_protraction_r", "sc_elevation_r", "ac_protraction_r")
    ] == pytest.approx(target), [
        message,
        [(name, coordinates[name].value) for name in girdle_names],
    ]
    assert converged
    assert error <= handlefit._PALM_POSITION_TOLERANCE_M


def test_lock_palm_to_handles_leaves_other_motion_free(monkeypatch):
    user = Model()
    for side, suffix in (("right", "r"), ("left", "l")):
        hand = osim.Body(f"hand_{suffix}", 1.0, osim.Vec3(0), osim.Inertia(1))
        user.addBody(hand)
        wrist = osim.PinJoint(
            f"radius_hand_{suffix}",
            user.getGround(),
            osim.Vec3(0),
            osim.Vec3(0),
            hand,
            osim.Vec3(0),
            osim.Vec3(0),
        )
        wrist.updCoordinate().setName(f"wrist_flexion_{suffix}")
        user.addJoint(wrist)
        user.addMarker(
            osim.Marker(
                "RPALM" if suffix == "r" else "LPALM",
                hand,
                osim.Vec3(0.1, 0.0, 0.0),
            )
        )

        for finger_name in ("index", "middle", "ring", "little"):
            finger_body = osim.Body(
                f"{finger_name}_{suffix}", 1.0, osim.Vec3(0), osim.Inertia(1)
            )
            user.addBody(finger_body)
            finger = osim.PinJoint(
                f"{finger_name}_mcp_{suffix}",
                hand,
                osim.Vec3(0),
                osim.Vec3(0),
                finger_body,
                osim.Vec3(0),
                osim.Vec3(0),
            )
            finger.updCoordinate().setName(
                f"{finger_name}_mcp_flexion_{suffix}"
            )
            user.addJoint(finger)

    equipment = Model()
    for side in ("right", "left"):
        handle = osim.Body(
            f"{side}_handle", 1.0, osim.Vec3(0), osim.Inertia(1)
        )
        equipment.addBody(handle)
        lever = osim.PinJoint(
            f"{side}_lever",
            equipment.getGround(),
            osim.Vec3(0),
            osim.Vec3(0),
            handle,
            osim.Vec3(0),
            osim.Vec3(0),
        )
        lever_coordinate = lever.updCoordinate()
        lever_coordinate.setName(f"{side}_lever-angle")
        lever_coordinate.setRangeMin(-1.0)
        lever_coordinate.setRangeMax(1.0)
        equipment.addJoint(lever)

    user.state = user.initSystem()
    equipment.state = equipment.initSystem()
    framework = Framework(user=user, equipment=equipment)
    framework.user.joint_center = lambda name: body_transform_array(
        framework.user.getBodySet().get(f"hand_{name[-1]}"),
        framework.user.state,
    )[:3, 3]
    framework.user.scapular_fit_residuals = lambda side, protraction, elevation: (
        np.zeros(6)
    )
    for suffix in ("r", "l"):
        framework.user.getCoordinateSet().get(
            f"index_mcp_flexion_{suffix}"
        ).setValue(framework.user.state, 0.25, False)
        finger = framework.user.getCoordinateSet().get(
            f"index_mcp_flexion_{suffix}"
        )
        finger.set_locked(False)
        finger.setLocked(framework.user.state, False)
        framework.user.getCoordinateSet().get(
            f"wrist_flexion_{suffix}"
        ).setValue(framework.user.state, -0.1, False)
    for side, angle in (("right", 0.35), ("left", -0.2)):
        framework.equipment.getCoordinateSet().get(
            f"{side}_lever-angle"
        ).setValue(framework.equipment.state, angle, False)
    framework.user.realizePosition(framework.user.state)
    framework.equipment.realizePosition(framework.equipment.state)
    initial_hand_positions = {}
    for side, suffix in (("right", "r"), ("left", "l")):
        handle_transform = body_transform_array(
            framework.equipment.getBodySet().get(f"{side}_handle"),
            framework.equipment.state,
        )
        rotation = handle_transform[:3, :3]
        origin = handle_transform[:3, 3]
        marker = framework.user.getMarkerSet().get(
            "RPALM" if suffix == "r" else "LPALM"
        )
        initial_hand_positions[side] = {
            "palm": rotation.T
            @ (marker_position_array(marker, framework.user.state) - origin),
        }

    handlefit.lock_hands_to_handles(framework)
    assert not framework.couplings
    assert framework.composite.getConstraintSet().getSize() == 0

    for side, angle in (("right", 0.5), ("left", -0.3)):
        framework.equipment.getCoordinateSet().get(
            f"{side}_lever-angle"
        ).setValue(framework.equipment.state, angle, False)
        suffix = "r" if side == "right" else "l"
        framework.user.getCoordinateSet().get(
            f"wrist_flexion_{suffix}"
        ).setValue(framework.user.state, 0.0, False)
        framework.user.getCoordinateSet().get(
            f"index_mcp_flexion_{suffix}"
        ).setValue(framework.user.state, 0.5, False)
    framework.user.realizePosition(framework.user.state)
    framework.equipment.realizePosition(framework.equipment.state)
    handlefit.lock_hands_to_handles(framework)

    for side, suffix in (("right", "r"), ("left", "l")):
        handle_transform = body_transform_array(
            framework.equipment.getBodySet().get(f"{side}_handle"),
            framework.equipment.state,
        )
        marker = framework.user.getMarkerSet().get(
            "RPALM" if suffix == "r" else "LPALM"
        )
        palm_position = np.linalg.solve(
            handle_transform[:3, :3],
            marker_position_array(marker, framework.user.state)
            - handle_transform[:3, 3],
        )
        assert np.linalg.norm(
            palm_position - initial_hand_positions[side]["palm"]
        ) <= handlefit._PALM_POSITION_TOLERANCE_M
        lever = framework.equipment.getCoordinateSet().get(
            f"{side}_lever-angle"
        )
        wrist = framework.user.getCoordinateSet().get(
            f"wrist_flexion_{suffix}"
        )
        assert lever.getValue(framework.equipment.state) == pytest.approx(
            0.5 if side == "right" else -0.3
        )
        assert wrist.getLocked(framework.user.state) is False
        finger = framework.user.getCoordinateSet().get(
            f"index_mcp_flexion_{suffix}"
        )
        assert finger.getLocked(framework.user.state) is False
        assert finger.getValue(framework.user.state) == pytest.approx(0.5)


def test_symmetric_handle_lock_fits_right_and_mirrors_left(monkeypatch):
    class Frame:
        def __init__(self, name):
            self.name = name

    bodies = {
        f"{side}_handle": Frame(f"{side}_handle") for side in ("right", "left")
    }
    positions = {
        "RPALM": np.array([0.0, 0.02, 0.0]),
        "LPALM": np.array([0.0, 0.02, 0.0]),
    }
    for suffix in ("r", "l"):
        for index, finger in enumerate(("index", "middle", "ring", "little")):
            positions[f"{finger}_mcp_{suffix}"] = np.array(
                [0.01 * index, 0.03, 0.01]
            )

    class ComponentSet:
        def get(self, name):
            return bodies[name]

    class MarkerSet:
        def get(self, name):
            return name

        def getIndex(self, name):
            return 0

    user = SimpleNamespace(
        state=object(),
        realizePosition=lambda state: None,
        getBodySet=lambda: ComponentSet(),
        getMarkerSet=lambda: MarkerSet(),
    )
    equipment = SimpleNamespace(
        state=object(),
        realizePosition=lambda state: None,
        getBodySet=lambda: ComponentSet(),
    )
    framework = SimpleNamespace(user=user, equipment=equipment)
    fits = []
    mirrors = []
    monkeypatch.setattr(
        handlefit, "body_transform_array", lambda body, state: np.eye(4)
    )
    monkeypatch.setattr(
        handlefit,
        "marker_position_array",
        lambda marker, state: positions[marker].copy(),
    )

    def fit(
        user,
        equipment,
        handle,
        side,
        palm,
        coordinate_prefixes=handlefit._FIT_COORDINATE_PREFIXES,
        starting_values=None,
    ):
        fits.append((side, coordinate_prefixes))
        return 0.0234, False, "maximum evaluations reached"

    def mirror(user, to_mirror):
        mirrors.append(to_mirror)
        positions["LPALM"] = positions["RPALM"].copy()

    monkeypatch.setattr(handlefit, "_fit_palm_marker_to_handle", fit)
    monkeypatch.setattr(handlefit, "mirror_arm", mirror)
    fit_log = [{"messages": []}]
    contact_maintained = handlefit.lock_hands_to_handles(
        framework, mirror_left_from_right=True, fit_log=fit_log
    )

    assert not contact_maintained
    assert fits == [
        ("right", handlefit._FIT_COORDINATE_PREFIXES),
        ("left", handlefit._MIRRORED_CONTACT_COORDINATE_PREFIXES),
    ]
    assert mirrors == ["right"]
    assert np.array_equal(
        positions["LPALM"], positions["RPALM"]
    )
    anchor_messages = [
        message
        for message in fit_log[-1]["messages"]
        if message["kind"] == "contact_fit"
    ]
    assert len(anchor_messages) == 2
    assert all(message["level"] == "WARNING" for message in anchor_messages)
    assert all(
        "residual=23.400 mm" in message["message"]
        for message in anchor_messages
    )
    assert all(
        "maximum evaluations reached" in message["message"]
        for message in anchor_messages
    )


def test_upper_limb_trajectory_filter_suppresses_high_frequency_motion():
    upper_limb_names = [
        "arm_r",
        "elbow_flex_r",
        "wrist_flex_r",
        "index_mcp_flexion_r",
        "sc_protraction_r",
    ]
    user = _simulation_model([*upper_limb_names, "pelvis_tilt"])
    normalized_times = np.linspace(0.0, 1.0, 121)
    times = 4.0 * normalized_times**1.05
    low_frequency_motion = 0.017 + 0.005 * np.sin(2.0 * np.pi * 0.25 * times)
    high_frequency_motion = 0.004 * np.sin(2.0 * np.pi * 8.0 * times)
    raw_motion = low_frequency_motion + high_frequency_motion
    states = []
    for value in raw_motion:
        for name in upper_limb_names:
            user.getCoordinateSet().get(name).setValue(user.state, value, False)
        user.getCoordinateSet().get("pelvis_tilt").setValue(
            user.state, 0.01, False
        )
        states.append((osim.State(user.state), osim.State(user.state)))

    chestpress._filter_upper_limb_trajectories(user, states, times)

    interior = slice(15, -15)
    filtered_arm = np.asarray([
        user.getCoordinateSet().get("arm_r").getValue(user_state)
        for user_state, _ in states
    ])
    raw_error = np.sqrt(np.mean(high_frequency_motion[interior] ** 2))
    filtered_error = np.sqrt(
        np.mean((filtered_arm[interior] - low_frequency_motion[interior]) ** 2)
    )
    assert filtered_error < raw_error * 0.25
    for name in upper_limb_names:
        filtered_values = np.asarray([
            user.getCoordinateSet().get(name).getValue(user_state)
            for user_state, _ in states
        ])
        assert np.allclose(filtered_values, filtered_arm)
    assert all(
        user.getCoordinateSet().get("pelvis_tilt").getValue(user_state)
        == pytest.approx(0.01)
        for user_state, _ in states
    )


def test_filtered_palm_contact_is_refitted_for_every_recorded_state(monkeypatch):
    framework = SimpleNamespace(
        user=SimpleNamespace(state=None),
        equipment=SimpleNamespace(state=None),
    )
    states = [
        ({"arm": -1.0}, {"angle": 0.0}),
        ({"arm": -1.0}, {"angle": 1.0}),
        ({"arm": -1.0}, {"angle": 0.0}),
    ]
    calls = []

    def lock_hands(_framework, *, mirror_left_from_right, fit_log):
        assert mirror_left_from_right
        angle = framework.equipment.state["angle"]
        framework.user.state["arm"] = angle
        calls.append(angle)
        fit_log[-1]["messages"].append(
            {"kind": "contact_fit", "message": f"angle={angle}"}
        )
        return True

    monkeypatch.setattr(chestpress, "lock_hands_to_handles", lock_hands)
    fit_log = []

    chestpress._restore_filtered_palm_contact(
        framework,
        states,
        [0.0, 1.0, 0.0],
        ["concentric", "concentric", "eccentric"],
        [{"arm": -1.0}, {"arm": -1.0}, {"arm": -1.0}],
        fit_log,
    )

    assert calls == [0.0, 1.0, 0.0]
    assert [user_state["arm"] for user_state, _ in states] == [0.0, 1.0, 0.0]
    assert fit_log[0]["phase"] == "post_filter_contact"
    assert [sample["lever_angle_deg"] for sample in fit_log[0]["samples"]] == [
        0.0, pytest.approx(np.rad2deg(1.0)), 0.0
    ]
    assert all(sample["messages"] for sample in fit_log[0]["samples"])


def test_filtered_palm_contact_failure_is_reported(monkeypatch):
    coordinates = SimpleNamespace(
        get=lambda name: SimpleNamespace(
            setValue=lambda state, value, _preserve_velocity: state.__setitem__(
                name, value
            )
        )
    )
    framework = SimpleNamespace(
        user=SimpleNamespace(
            state=None,
            getCoordinateSet=lambda: coordinates,
            realizePosition=lambda _state: None,
        ),
        equipment=SimpleNamespace(state=None),
    )

    def lose_contact(_framework, *, mirror_left_from_right, fit_log):
        fit_log[-1]["messages"].append(
            {"kind": "contact_fit", "level": "WARNING", "message": "residual=2 mm"}
        )
        return False

    monkeypatch.setattr(chestpress, "lock_hands_to_handles", lose_contact)
    fit_log = []

    with pytest.raises(RuntimeError, match="could not be restored after filtering") as raised:
        chestpress._restore_filtered_palm_contact(
            framework,
            [({"arm": 0.0}, {"angle": 1.0})],
            [np.deg2rad(1.0)],
            ["concentric"],
            [{"arm": 0.0}],
            fit_log,
        )

    assert raised.value.handle_fit_log is fit_log
    assert fit_log[-1]["samples"][0]["messages"][0]["message"] == "residual=2 mm"


def test_filtered_contact_falls_back_to_original_pose(monkeypatch):
    coordinates = SimpleNamespace(
        get=lambda name: SimpleNamespace(
            setValue=lambda state, value, _preserve_velocity: state.__setitem__(
                name, value
            )
        )
    )
    framework = SimpleNamespace(
        user=SimpleNamespace(
            state=None,
            getCoordinateSet=lambda: coordinates,
            realizePosition=lambda _state: None,
        ),
        equipment=SimpleNamespace(state=None),
    )
    attempts = 0

    def lock_with_filtered_failure(_framework, *, mirror_left_from_right, fit_log):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            framework.user.state["arm"] = -10.0
            return False
        assert framework.user.state["arm"] == pytest.approx(0.25)
        fit_log[-1]["messages"].append(
            {"kind": "contact_fit", "message": "restored contact"}
        )
        return True

    monkeypatch.setattr(
        chestpress, "lock_hands_to_handles", lock_with_filtered_failure
    )
    fit_log = []
    states = [({"arm": 0.25}, {"angle": 0.0})]

    chestpress._restore_filtered_palm_contact(
        framework,
        states,
        [0.0],
        ["concentric"],
        [{"arm": 0.25}],
        fit_log,
    )

    assert attempts == 2
    assert states[0][0]["arm"] == pytest.approx(0.25)
    sample = fit_log[0]["samples"][0]
    assert sample["fallback"] == "unfiltered_upper_limb_pose"
    assert sample["messages"][0]["level"] == "WARNING"
    assert sample["messages"][1]["message"] == "restored contact"


@pytest.mark.parametrize("fail_at_end", [False, True])
def test_simulation_records_synchronized_step_trajectories(monkeypatch, fail_at_end):
    user = _simulation_model(["arm_r", "arm_l", "elbow_flex_r"])
    equipment = _simulation_model(["right_lever-angle", "left_lever-angle"])
    framework = Framework(user=user, equipment=equipment)
    wrapper = SimpleNamespace(
        fit_user=lambda **kwargs: (
            framework, {"grip": {
                "handle_fit_log": [{"phase": "final", "lever_angle_deg": 0.0, "messages": []}],
            }}
        )
    )
    lock_modes = []

    def lock_hands(_framework, *, mirror_left_from_right=False, fit_log=None):
        lock_modes.append(mirror_left_from_right)
        assert mirror_left_from_right
        user.getCoordinateSet().get("arm_l").setValue(
            user.state, user.getCoordinateSet().get("arm_r").getValue(user.state), False
        )
        return True

    monkeypatch.setattr(chestpress, "lock_hands_to_handles", lock_hands)
    fitting_weights = []
    fitting_distance_weights = []
    fitted_sides = []

    def fit_handle(**kwargs):
        fitting_weights.append(kwargs["palm_contact_weight"])
        fitting_distance_weights.append(kwargs["shoulder_wrist_axis_distance_weight"])
        fitted_sides.append(kwargs["side"])
        assert "palm_position_handle" not in kwargs
        assert "palm_position_weight" not in kwargs
        angle = equipment.getCoordinateSet().get("right_lever-angle").getValue(equipment.state)
        if (
            kwargs["side"] == "right"
            and fail_at_end
            and angle == pytest.approx(np.deg2rad(2.0))
        ):
            return None, None, [{
                "message": (
                    "right handle fit did not converge: evaluation limit exceeded; "
                    "nfev=1000, cost=42, optimality=3"
                )
            }]
        user.getCoordinateSet().get("arm_r").setValue(
            user.state, angle * 0.5, False
        )
        elbow_flexion = (
            0.5 if angle == pytest.approx(np.deg2rad(2.0)) else 90.0
        )
        user.getCoordinateSet().get("elbow_flex_r").setValue(
            user.state, np.deg2rad(elbow_flexion), False
        )
        return elbow_flexion, None, []

    monkeypatch.setattr(chestpress, "fit_handle", fit_handle)
    if fail_at_end:
        with pytest.raises(RuntimeError, match="concentric stroke phase at 2.00 degrees") as raised:
            ChestPress.simulate_kinematics(
                wrapper, user, step_deg=1.0,
                concentric_duration_s=1.5, eccentric_duration_s=2.5,
                palm_contact_weight=4321.0,
                shoulder_wrist_axis_distance_weight=5432.0,
            )
        fit_log = raised.value.handle_fit_log
        assert "evaluation limit exceeded" in str(raised.value)
        assert "nfev=1000, cost=42, optimality=3" in str(raised.value)
        assert [entry["phase"] for entry in fit_log] == [
            "final", "concentric", "concentric"
        ]
        assert [entry["lever_angle_deg"] for entry in fit_log] == pytest.approx(
            [0, 1, 2]
        )
        assert fit_log[2]["messages"] == [{
            "message": (
                "right handle fit did not converge: evaluation limit exceeded; "
                "nfev=1000, cost=42, optimality=3"
            )
        }, {
            "message": "retried with the previous safe shoulder-girdle pose"
        }, {
            "message": (
                "right handle fit did not converge: evaluation limit exceeded; "
                "nfev=1000, cost=42, optimality=3"
            )
        }]
        assert framework.states_trajectory.getSize() == 0
        assert fitting_weights and all(weight == 4321.0 for weight in fitting_weights)
        assert fitting_distance_weights and all(
            weight == 5432.0 for weight in fitting_distance_weights
        )
        assert fitted_sides and set(fitted_sides) == {"right"}
        assert lock_modes and all(lock_modes)
        return

    result, fit_log = ChestPress.simulate_kinematics(
        wrapper, user, step_deg=1.0,
        concentric_duration_s=1.5, eccentric_duration_s=2.5,
        palm_contact_weight=4321.0,
        shoulder_wrist_axis_distance_weight=5432.0,
    )
    expected_degrees = [0, 1, 2, 1, 0]
    assert fitting_weights and all(weight == 4321.0 for weight in fitting_weights)
    assert fitting_distance_weights and all(
        weight == 5432.0 for weight in fitting_distance_weights
    )
    assert fitted_sides and set(fitted_sides) == {"right"}
    assert lock_modes and all(lock_modes)
    assert [entry["phase"] for entry in fit_log] == [
        "final", "concentric", "concentric", "eccentric", "eccentric",
        "post_filter_contact",
    ]
    assert [entry["lever_angle_deg"] for entry in fit_log[:-1]] == pytest.approx(
        [0, 1, 2, 1, 0]
    )
    for entry in fit_log[1:-1]:
        assert entry["messages"] == []
    assert [
        sample["lever_angle_deg"]
        for sample in fit_log[-1]["samples"]
    ] == pytest.approx(expected_degrees)
    assert lock_modes == [True] * (len(expected_degrees) * 2 - 1)
    assert result is framework
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
    raw_arm_values = np.deg2rad(expected_degrees) * 0.5
    filtered_arm_values = np.asarray([
        user.getCoordinateSet().get("arm_r").getValue(
            result.states_trajectories["user"].get(index)
        )
        for index in range(len(expected_degrees))
    ])
    assert np.all(np.isfinite(filtered_arm_values))
    assert not np.allclose(filtered_arm_values, raw_arm_values)
    assert np.sum(np.diff(filtered_arm_values, n=2) ** 2) < np.sum(
        np.diff(raw_arm_values, n=2) ** 2
    )
    for index, degrees in enumerate(expected_degrees):
        angle = np.deg2rad(degrees)
        equipment_state = result.states_trajectories["equipment"].get(index)
        user_state = result.states_trajectories["user"].get(index)
        for side in ("left", "right"):
            coordinate = equipment.getCoordinateSet().get(f"{side}_lever-angle")
            assert coordinate.getValue(equipment_state) == pytest.approx(angle)
        for side in ("l", "r"):
            coordinate = user.getCoordinateSet().get(f"arm_{side}")
            assert coordinate.getValue(user_state) == pytest.approx(
                filtered_arm_values[index]
            )
        if index == 1:
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
        assert np.isfinite(
            user.getCoordinateSet().get("arm_r").getSpeedValue(user_state)
        )
        composite_state = result.states_trajectory.get(index)
        assert composite_state.getTime() == pytest.approx(result.state_times[index])
        for member, model, name, value in (
            ("equipment", equipment, "right_lever-angle", angle),
            ("equipment", equipment, "left_lever-angle", angle),
            ("user", user, "arm_r", filtered_arm_values[index]),
            ("user", user, "arm_l", filtered_arm_values[index]),
        ):
            coordinate = model.getCoordinateSet().get(name)
            path = result.path_map[(member, coordinate.getAbsolutePathString())]
            assert result.composite.getStateVariableValue(
                composite_state, f"{path}/value"
            ) == pytest.approx(value)
    assert result.get_motion_table(1.0).getNumRows() == 5


def test_return_phase_clamps_floating_point_lower_endpoint(monkeypatch):
    user = _simulation_model(["arm_r", "arm_l", "elbow_flex_r"])
    equipment = _simulation_model(["right_lever-angle", "left_lever-angle"])
    equipment.getCoordinateSet().get("right_lever-angle").setRangeMax(
        np.deg2rad(20.0)
    )
    equipment.getCoordinateSet().get("left_lever-angle").setRangeMax(
        np.deg2rad(20.0)
    )
    framework = Framework(user=user, equipment=equipment)
    wrapper = SimpleNamespace(
        fit_user=lambda **kwargs: (
            framework,
            {
                "grip": {
                    "handle_fit_log": [],
                }
            },
        )
    )
    monkeypatch.setattr(
        chestpress, "lock_hands_to_handles", lambda *args, **kwargs: True
    )

    def fit_handle(**kwargs):
        angle = equipment.getCoordinateSet().get("right_lever-angle").getValue(
            equipment.state
        )
        suffix = "r" if kwargs["side"] == "right" else "l"
        user.getCoordinateSet().get(f"arm_{suffix}").setValue(
            user.state, angle * 0.5, False
        )
        elbow_flexion = 0.5 if angle >= np.deg2rad(19.0) else 90.0
        user.getCoordinateSet().get("elbow_flex_r").setValue(
            user.state, np.deg2rad(elbow_flexion), False
        )
        return elbow_flexion, None, []

    monkeypatch.setattr(chestpress, "fit_handle", fit_handle)
    result, fit_log = ChestPress.simulate_kinematics(
        wrapper, user, step_deg=1.0,
        concentric_duration_s=1.5, eccentric_duration_s=2.5,
    )

    assert result.states_trajectory.getSize() == 41
    assert np.all(np.diff(result.state_times) > 0.0)
    final_equipment_state = result.states_trajectories["equipment"].get(40)
    assert equipment.getCoordinateSet().get(
        "right_lever-angle"
    ).getValue(final_equipment_state) == pytest.approx(0.0)


def test_simulation_starts_eccentric_phase_when_palm_contact_breaks(monkeypatch):
    user = _simulation_model(["arm_r", "arm_l", "elbow_flex_r"])
    equipment = _simulation_model(["right_lever-angle", "left_lever-angle"])
    framework = Framework(user=user, equipment=equipment)
    wrapper = SimpleNamespace(
        fit_user=lambda **kwargs: (
            framework,
            {
                "grip": {
                    "handle_fit_log": [],
                }
            },
        )
    )

    def lock_hands(_framework, *, mirror_left_from_right=False, fit_log=None):
        assert mirror_left_from_right
        right_arm = user.getCoordinateSet().get("arm_r").getValue(user.state)
        user.getCoordinateSet().get("arm_l").setValue(user.state, right_arm, False)
        angle = equipment.getCoordinateSet().get("right_lever-angle").getValue(
            equipment.state
        )
        if angle == pytest.approx(np.deg2rad(1.0)):
            user.getCoordinateSet().get("elbow_flex_r").setValue(
                user.state, np.deg2rad(10.0), False
            )
            return True
        return angle != pytest.approx(np.deg2rad(2.0))

    def fit_handle(**kwargs):
        angle = equipment.getCoordinateSet().get("right_lever-angle").getValue(
            equipment.state
        )
        user.getCoordinateSet().get("arm_r").setValue(
            user.state, angle * 0.5, False
        )
        user.getCoordinateSet().get("elbow_flex_r").setValue(
            user.state, np.deg2rad(0.5), False
        )
        return 0.5, None, []

    monkeypatch.setattr(chestpress, "lock_hands_to_handles", lock_hands)
    monkeypatch.setattr(chestpress, "fit_handle", fit_handle)
    monkeypatch.setattr(
        chestpress, "_filter_upper_limb_trajectories", lambda *args: None
    )
    result, fit_log = ChestPress.simulate_kinematics(
        wrapper,
        user,
        step_deg=1.0,
        concentric_duration_s=1.5,
        eccentric_duration_s=2.5,
    )

    user_states = result.states_trajectories["user"]
    elbow = user.getCoordinateSet().get("elbow_flex_r")
    assert np.rad2deg(elbow.getValue(user_states.get(0))) == pytest.approx(0.0)
    assert np.rad2deg(elbow.getValue(user_states.get(1))) == pytest.approx(10.0)
    assert np.rad2deg(elbow.getValue(user_states.get(2))) == pytest.approx(0.5)
    assert [entry["phase"] for entry in fit_log] == [
        "concentric", "concentric", "eccentric", "post_filter_contact"
    ]
    assert [entry["lever_angle_deg"] for entry in fit_log[:-1]] == pytest.approx(
        [1.0, 2.0, 0.0]
    )
    assert [
        sample["lever_angle_deg"] for sample in fit_log[-1]["samples"]
    ] == pytest.approx([0.0, 1.0, 0.0])


def test_simulation_reverses_at_first_unreachable_palm_residual(monkeypatch):
    user = _simulation_model(["arm_r", "arm_l", "elbow_flex_r"])
    equipment = _simulation_model(["right_lever-angle", "left_lever-angle"])
    framework = Framework(user=user, equipment=equipment)
    wrapper = SimpleNamespace(
        fit_user=lambda **kwargs: (
            framework,
            {
                "grip": {
                    "handle_fit_log": [],
                }
            },
        )
    )
    lock_attempts = 0
    lock_logs = []

    def lock_palm(_framework, **kwargs):
        nonlocal lock_attempts
        lock_attempts += 1
        lock_logs.append(kwargs["fit_log"])
        if lock_attempts == 2:
            kwargs["fit_log"][-1]["messages"].append(
                {
                    "kind": "contact_fit",
                    "level": "WARNING",
                    "message": "right RPALM handle-anchor residual=23.400 mm",
                }
            )
            return False
        return True

    def fit_handle(**kwargs):
        angle = equipment.getCoordinateSet().get("right_lever-angle").getValue(
            equipment.state
        )
        user.getCoordinateSet().get("arm_r").setValue(
            user.state, angle * 0.5, False
        )
        elbow_flexion = (
            0.5 if angle == pytest.approx(np.deg2rad(2.0)) else 90.0
        )
        user.getCoordinateSet().get("elbow_flex_r").setValue(
            user.state, np.deg2rad(elbow_flexion), False
        )
        return elbow_flexion, None, []

    def mirror_arm(user, to_mirror):
        coordinates = user.getCoordinateSet()
        coordinates.get("arm_l").setValue(
            user.state, coordinates.get("arm_r").getValue(user.state), False
        )

    monkeypatch.setattr(chestpress, "lock_hands_to_handles", lock_palm)
    monkeypatch.setattr(chestpress, "fit_handle", fit_handle)
    monkeypatch.setattr(handlefit, "mirror_arm", mirror_arm)
    trajectory, simulation_log = ChestPress.simulate_kinematics(
        wrapper, user, step_deg=1.0
    )
    assert trajectory is framework
    assert trajectory.states_trajectory.getSize() == 3
    assert lock_attempts == 6
    recorded_angles = [
        equipment.getCoordinateSet().get("right_lever-angle").getValue(
            trajectory.states_trajectories["equipment"].get(index)
        )
        for index in range(trajectory.states_trajectory.getSize())
    ]
    assert np.rad2deg(recorded_angles) == pytest.approx([0.0, 1.0, 0.0])
    assert any(
        "residual=23.400 mm" in message["message"]
        for entry in lock_logs[1]
        for message in entry["messages"]
    )
    assert [entry["phase"] for entry in simulation_log] == [
        "concentric", "concentric", "eccentric", "post_filter_contact"
    ]


def test_simulation_rejects_fewer_than_two_states(monkeypatch):
    user = _simulation_model(["arm_r", "arm_l", "elbow_flex_r"])
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
    monkeypatch.setattr(chestpress, "lock_hands_to_handles", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="concentric stroke phase at 1.00 degrees") as raised:
        ChestPress.simulate_kinematics(wrapper, user)
    assert raised.value.handle_fit_log == [
        {
            "phase": "concentric",
            "lever_angle_deg": 1.0,
            "messages": [{
                "message": "retried with the previous safe shoulder-girdle pose"
            }],
        }
    ]
    assert framework.states_trajectory.getSize() == 0


def test_failed_solver_keeps_best_coordinates_and_warns(fit_setup, monkeypatch):
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
    assert elbow == pytest.approx(np.rad2deg(0.1))
    assert palm is not None
    assert any(
        "evaluation limit exceeded; nfev=1000, cost=42, optimality=3" in m["message"]
        for m in messages
    )
    assert coordinates.get("sc_protraction_r").value == pytest.approx(0.3)
    assert coordinates.get("ac_protraction_r").value == pytest.approx(0.03)


def test_fit_can_keep_scapula_fixed(fit_setup, monkeypatch):
    user, equipment, coordinates, _ = fit_setup
    calls = []
    shoulder_values = {
        name: coordinates.get(f"{name}_r").value
        for name in handlefit.SHOULDER_GIRDLE_COORDINATES
    }

    def solve(objective, x0, **kwargs):
        calls.append(x0.copy())
        return SimpleNamespace(success=True, nfev=1, x=x0.copy())

    monkeypatch.setattr(handlefit, "least_squares", solve)
    elbow, palm, _ = handlefit.fit_handle(
        user,
        equipment,
        "right_handle",
        "right",
        fit_shoulder_girdle=False,
        palm_contact_weight=0.0,
        knuckle_weight=0.0,
        knuckle_end_weight=0.0,
        plane_distance_weight=0.0,
        pro_sup_weight=0.0,
        wrist_dev_weight=0.0,
        protraction_weight=0.0,
        scapular_elevation_weight=0.0,
        arm_weight=0.0,
    )

    assert elbow is not None
    assert palm is not None
    assert len(calls) == 1
    assert len(calls[0]) == len(
        [
            coordinate
            for coordinate in coordinates.coordinates
            if coordinate.name.endswith("_r")
            and coordinate.name.startswith(handlefit._FIT_COORDINATE_PREFIXES)
        ]
    )
    assert {
        name: coordinates.get(f"{name}_r").value
        for name in handlefit.SHOULDER_GIRDLE_COORDINATES
    } == shoulder_values


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
    warning = next(message for message in messages if message["message"] == "contact warning")
    log = next(message for message in messages if message["message"] == "contact side r")
    assert warning["kind"] == "warning"
    assert warning["category"] == "UserWarning"
    assert warning["function"] == "report"
    assert "warnings.warn(" in warning["source"]
    assert log["kind"] == "log"
    assert log["level"] == "INFO"
    assert log["logger"] == logger.name
    assert log["function"] == "report"
    assert "logger.info(" in log["source"]
    assert all(
        message["filename"] == __file__ and isinstance(message["line"], int)
        for message in (warning, log)
    )


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
    assert any(
        message["message"] == "before failure"
        for message in raised.value.handle_fit_log
    )
    assert logging.getLogRecordFactory() is original_factory
    assert warnings.showwarning is original_showwarning


def test_fit_signature_describes_the_returned_log():
    assert inspect.signature(handlefit.fit_handle).return_annotation == (
        get_type_hints(handlefit.fit_handle)["return"]
    )
    assert len(inspect.signature(handlefit.fit_handle).return_annotation.__args__) == 3
    assert (
        inspect.signature(handlefit.fit_handle)
        .parameters["wrist_flex_weight"]
        .default
        == 10.0
    )
    assert (
        inspect.signature(ChestPress.fit_user)
        .parameters["wrist_flex_weight"]
        .default
        == 10.0
    )
    assert (
        inspect.signature(ChestPress.simulate_kinematics)
        .parameters["wrist_flex_weight"]
        .default
        == 10.0
    )
