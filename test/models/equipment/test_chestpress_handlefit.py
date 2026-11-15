from types import SimpleNamespace

import numpy as np
import opensim as osim
import pytest
import vtk

from opensim_frameworks.models.equipment.chestpress import handlefit
from opensim_frameworks.models.equipment.chestpress.chestpress import ChestPress
from opensim_frameworks.utils.geometry import body_transform_array
from opensim_frameworks.utils.kinematics import marker_position_array


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
        "sc_elevation_r",
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
    )
    palm_ground = np.zeros(3)

    def marker_position(marker, state):
        if marker is palm:
            return palm_ground.copy()
        return marker_position_array(marker, state)

    monkeypatch.setattr(handlefit, "marker_position_array", marker_position)
    return user, equipment, coordinates, palm_ground


def run_fit(monkeypatch, user, equipment, *, probes=(), **weights):
    costs = []

    def solve(objective, x0, *, bounds, max_nfev):
        names = sorted(
            name for name in (c.name for c in user.getCoordinateSet().coordinates)
            if name.startswith(handlefit._FIT_COORDINATE_PREFIXES)
        )
        names += sorted(("sc_elevation_r", "sc_protraction_r"))
        for values in probes:
            candidate = x0.copy()
            for name, value in values.items():
                candidate[names.index(name)] = value
            residuals = objective(candidate)
            assert len(residuals) > 1
            costs.append(float(residuals @ residuals))
        return SimpleNamespace(success=True, nfev=1, x=x0)

    monkeypatch.setattr(handlefit, "least_squares", solve)
    zero_weights = {
        "palm_contact_weight": 0.0,
        "palm_facing_weight": 0.0,
        "knuckle_weight": 0.0,
        "plane_distance_weight": 0.0,
        "handle_end_weight": 0.0,
        "pro_sup_weight": 0.0,
        "wrist_dev_weight": 0.0,
        "wrist_extension_weight": 0.0,
        "protraction_weight": 0.0,
        "scapular_elevation_weight": 0.0,
        "wrist_flexion_weight": 0.0,
        "arm_weight": 0.0,
    }
    zero_weights.update(weights)
    result = handlefit.fit_handle(user, equipment, "right_handle", "right", **zero_weights)
    return result, costs


@pytest.mark.parametrize("angle", [0.0, 0.7])
def test_palm_relative_to_marker_and_cylindrical_surface(fit_setup, monkeypatch, angle):
    user, equipment, _, palm_ground = fit_setup
    equipment.updCoordinateSet().get(0).setValue(equipment.state, angle)
    equipment.realizePosition(equipment.state)
    transform = body_transform_array(equipment.getBodySet().get("right_handle"), equipment.state)
    palm_local = np.array([0.325, 0.48, 0.5])
    palm_ground[:] = transform[:3, :3] @ palm_local + transform[:3, 3]

    (_, palm), costs = run_fit(
        monkeypatch, user, equipment, probes=[{}], palm_contact_weight=1.0,
        palm_handle_distance_m=0.005, handle_end_weight=1.0,
    )
    np.testing.assert_allclose(palm, [0.025, 0.08, 0.0], atol=1e-7)
    assert costs[0] == pytest.approx(0.0, abs=1e-12)

    palm_local[1] = 0.55
    palm_ground[:] = transform[:3, :3] @ palm_local + transform[:3, 3]
    _, costs = run_fit(
        monkeypatch, user, equipment, probes=[{}], handle_end_weight=1.0
    )
    assert costs[0] == pytest.approx(0.03**2, abs=1e-8)


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


def test_mirror_arm_includes_shoulder_girdle(fit_setup):
    user, _, coordinates, _ = fit_setup
    names = [coordinate.name for coordinate in coordinates.coordinates]
    coordinates.coordinates.extend(Coordinate(name[:-2] + "_l") for name in names)
    coordinates.get("sc_protraction_r").value = -0.3
    coordinates.get("sc_elevation_r").value = 0.2
    handlefit.mirror_arm(user, "right")
    assert coordinates.get("sc_protraction_l").value == pytest.approx(-0.3)
    assert coordinates.get("sc_elevation_l").value == pytest.approx(0.2)
    assert coordinates.get("sc_protraction_l").locked


def test_simulation_passes_fitting_weights_to_initial_fit():
    captured = {}

    def fit_user(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("initial fit reached")

    equipment = SimpleNamespace(fit_user=fit_user)
    with pytest.raises(RuntimeError, match="initial fit reached"):
        ChestPress.simulate_on_user(
            equipment, user=object(), protraction_weight=42.0,
            scapular_elevation_weight=23.0, palm_handle_distance_m=0.01,
        )
    assert captured["protraction_weight"] == 42.0
    assert captured["scapular_elevation_weight"] == 23.0
    assert captured["palm_handle_distance_m"] == 0.01


def test_failed_solver_restores_initial_coordinates(fit_setup, monkeypatch):
    user, equipment, coordinates, _ = fit_setup
    coordinates.get("sc_protraction_r").value = 0.2

    def failed_solve(objective, x0, *, bounds, max_nfev):
        objective(x0 + 0.1)
        return SimpleNamespace(success=False, nfev=1000, x=x0 + 0.1)

    monkeypatch.setattr(handlefit, "least_squares", failed_solve)
    assert handlefit.fit_handle(user, equipment, "right_handle", "right") == (None, None)
    assert coordinates.get("sc_protraction_r").value == pytest.approx(0.2)


@pytest.mark.parametrize("target", [[0.0, 0.0], [0.0, np.nan, 0.0]])
def test_invalid_palm_target_is_rejected(fit_setup, target):
    user, equipment, _, _ = fit_setup
    with pytest.raises(ValueError, match="palm_position_handle"):
        handlefit.fit_handle(user, equipment, "right_handle", "right", palm_position_handle=target)
