from types import SimpleNamespace

import numpy as np
import opensim as osim
import pytest
import vtk

from opensim_frameworks.models.equipment.chestpress import handlefit
from opensim_frameworks.models.equipment.chestpress import chestpress
from opensim_frameworks.models.equipment.chestpress.chestpress import ChestPress
from opensim_frameworks import Framework, FrameworkModel
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
    def scapular_residuals(side, protraction_weight, elevation_weight):
        protraction = coordinates.get(f"sc_protraction_{side}")
        elevation = coordinates.get(f"sc_elevation_{side}")
        return np.array([
            np.sqrt(protraction_weight) * (protraction.value - protraction.minimum),
            np.sqrt(elevation_weight) * max(elevation.value, 0.0),
            0.0,
        ])

    user.scapular_fit_residuals = scapular_residuals
    palm_ground = np.zeros(3)

    def marker_position(marker, state):
        if marker is palm:
            return palm_ground.copy()
        return marker_position_array(marker, state)

    monkeypatch.setattr(handlefit, "marker_position_array", marker_position)
    return user, equipment, coordinates, palm_ground


def run_fit(monkeypatch, user, equipment, *, probes=(), **weights):
    costs = []

    def solve(objective, x0, *, bounds, max_nfev, diff_step):
        names = sorted(
            name for name in (c.name for c in user.getCoordinateSet().coordinates)
            if name.startswith(handlefit._FIT_COORDINATE_PREFIXES)
        )
        names += [name + "_r" for name in handlefit.SHOULDER_GIRDLE_COORDINATES]
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


def test_plane_and_knuckle_alignment_use_joint_geometry(fit_setup, monkeypatch):
    user, equipment, _, palm_ground = fit_setup
    palm_ground[:] = [0.325, 0.48, 0.5]
    points = {
        "acromial_r": np.array([0.325, 0.2, 0.0]),
        "elbow_r": np.array([0.325, 0.35, 0.25]),
        "radius_hand_r": np.array([0.2, 0.35, 0.5]),
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
    points["elbow_r"][0] += 0.02
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
    equipment._handle_meshes = lambda side: ChestPress._handle_meshes(equipment, side)
    with pytest.raises(RuntimeError, match="No meshes attached to left_handle"):
        ChestPress._verify_required_bodies(equipment)
    left = equipment.getBodySet().get("left_handle")
    right = equipment.getBodySet().get("right_handle")
    left.attachGeometry(osim.Mesh(osim.Mesh.safeDownCast(right.get_attached_geometry(0)).get_mesh_file()))
    ChestPress._verify_required_bodies(equipment)


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


def _simulation_model(names):
    model = FrameworkModel()
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
            framework, {"grip": {"palm_handle_position_m": [0.0, 0.0, 0.0]}}
        )
    )
    def fit_handle(**kwargs):
        angle = equipment.getCoordinateSet().get("right_lever-angle").getValue(equipment.state)
        if fail_at_end and angle == pytest.approx(np.deg2rad(2.0)):
            return None, None
        user.getCoordinateSet().get("arm_r").setValue(user.state, angle * 0.5, False)
        return (0.5 if angle == pytest.approx(np.deg2rad(2.0)) else 90.0), None

    def mirror_arm(user, side):
        coords = user.getCoordinateSet()
        coords.get("arm_l").setValue(user.state, coords.get("arm_r").getValue(user.state), False)

    monkeypatch.setattr(chestpress, "fit_handle", fit_handle)
    monkeypatch.setattr(chestpress, "mirror_arm", mirror_arm)
    if fail_at_end:
        with pytest.warns(RuntimeWarning, match="Outward handle fit failed"):
            result = ChestPress.simulate_on_user(wrapper, user, step_deg=1.0, duration_s=4.0)
    else:
        result = ChestPress.simulate_on_user(wrapper, user, step_deg=1.0, duration_s=4.0)

    expected_degrees = [0, 1, 0] if fail_at_end else [0, 1, 2, 1, 0]
    assert result is framework
    assert result.state_times == pytest.approx(np.linspace(0, 4, len(expected_degrees)))
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
            framework, {"grip": {"palm_handle_position_m": [0.0, 0.0, 0.0]}}
        )
    )
    monkeypatch.setattr(chestpress, "fit_handle", lambda **kwargs: (None, None))
    with pytest.warns(RuntimeWarning, match="Outward handle fit failed"):
        with pytest.raises(ValueError, match="fewer than two"):
            ChestPress.simulate_on_user(wrapper, user)
    assert framework.states_trajectory.getSize() == 0


def test_failed_solver_restores_initial_coordinates(fit_setup, monkeypatch):
    user, equipment, coordinates, _ = fit_setup
    coordinates.get("sc_protraction_r").value = 0.2

    def failed_solve(objective, x0, *, bounds, max_nfev, diff_step):
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
