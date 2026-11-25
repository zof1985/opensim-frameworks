from __future__ import annotations

import numpy as np
import pytest

opensim = pytest.importorskip("opensim")

from opensim_frameworks import User, get_referenced_meshes
from opensim_frameworks.models.user.data import resolve_reference
from opensim_frameworks.models.user.mapping import segment_scale_factors
from opensim_frameworks.models.user.shoulder import (
    CLAVICLE_TO_BIACROMIAL_RATIO,
    SHOULDER_GIRDLE_COORDINATES,
)


@pytest.fixture(scope="module")
def user() -> User:
    return User("M", height_cm=175)


def _position(user: User, joint: str) -> np.ndarray:
    frame = user.getJointSet().get(joint).getChildFrame()
    position = frame.getPositionInGround(user.state)
    return np.array([position.get(i) for i in range(3)])


def test_torso_scale_uses_chest_depth_antero_posteriorly_and_breadth_laterally():
    # x is the antero-posterior and z the lateral axis of the torso frame.
    baseline = resolve_reference("M", percentile=50.0)
    target = resolve_reference("M", percentile=95.0)

    factors = segment_scale_factors(target, baseline)["torso"]

    assert factors[0] == pytest.approx(
        target.values["chestdepth"] / baseline.values["chestdepth"]
    )
    assert factors[2] == pytest.approx(
        target.values["chestbreadth"] / baseline.values["chestbreadth"]
    )


def test_clavicle_length_is_the_documented_fraction_of_biacromial_breadth(user):
    user.realizePosition(user.state)
    biacromial = user.biacromial_breadth

    for side in ("r", "l"):
        vector = _position(user, f"acromioclavicular_{side}") - _position(
            user, f"sternoclavicular_{side}"
        )
        # Lateral axis dominates, the x and y torso scaling add a small error.
        assert np.linalg.norm(vector) == pytest.approx(
            CLAVICLE_TO_BIACROMIAL_RATIO * biacromial, rel=0.03
        )


# -- Scapulothoracic contact residuals and reference pose.


@pytest.mark.slow
@pytest.mark.parametrize("gender, height_cm", [("M", 150), ("F", 195), ("M", 205)])
def test_contact_residuals_vanish_at_zero_girdle_pose_for_any_size(gender, height_cm):
    # The contact radii are calibrated on each user at the zero pose.
    sized = User(gender, height_cm=height_cm)

    for side in ("r", "l"):
        residuals = sized.scapulothoracic_residuals(side)

        assert set(residuals) == {"AI", "TS", "AA", "MB1", "MB2", "SA"}
        for value in residuals.values():
            assert value == pytest.approx(0.0, abs=1e-4)


def test_contact_residuals_vanish_at_zero_girdle_pose(user):
    for side in ("r", "l"):
        for value in user.scapulothoracic_residuals(side).values():
            assert value == pytest.approx(0.0, abs=1e-4)


def test_acromion_angle_location_tracks_its_scapula(user):
    right_before = np.asarray(user.acromion_angle_location("r"))
    left_before = np.asarray(user.acromion_angle_location("l"))
    coordinate = user.getCoordinateSet().get("sc_protraction_r")
    coordinate.setValue(
        user.state,
        coordinate.getValue(user.state) + np.deg2rad(1.0),
        False,
    )
    user.realizePosition(user.state)

    right_after = np.asarray(user.acromion_angle_location("r"))
    left_after = np.asarray(user.acromion_angle_location("l"))

    assert not np.allclose(right_after, right_before)
    assert left_after == pytest.approx(left_before)
    with pytest.raises(ValueError, match="side must be 'r' or 'l'"):
        user.acromion_angle_location("right")


def test_set_girdle_pose_applies_reference_to_both_sides(user, monkeypatch):
    pose = {
        name: float(index + 1)
        for index, name in enumerate(SHOULDER_GIRDLE_COORDINATES)
    }
    reference_calls = []
    set_calls = []
    monkeypatch.setattr(
        user,
        "shoulder_girdle_reference_pose",
        lambda contact=True: reference_calls.append(contact) or pose,
    )
    monkeypatch.setattr(
        user,
        "set_scapular_girdle_values",
        lambda side, values, *, check_contact=True: set_calls.append(
            (side, values.copy(), check_contact)
        ),
    )

    user.set_girdle_pose()

    expected = np.deg2rad(list(pose.values()))
    assert reference_calls == [True]
    assert [call[0] for call in set_calls] == ["r", "l"]
    for _, values, check_contact in set_calls:
        np.testing.assert_allclose(values, expected)
        assert not check_contact


def test_residual_vector_matches_the_dictionary_variants(user):
    pose = {"sc_protraction": 10.0, "ac_upward_rotation": 15.0}
    values = [pose.get(name, 0.0) for name in SHOULDER_GIRDLE_COORDINATES]

    vector = user.scapulothoracic_residual_vector("r", values)
    residuals = user.scapulothoracic_residuals_for("r", pose)

    assert vector.shape == (6,)
    assert vector == pytest.approx(list(residuals.values()), abs=1e-12)
    with pytest.raises(ValueError):
        user.scapulothoracic_residual_vector("r", values[:-1])


def test_residuals_are_mirrored_between_sides(user):
    pose = {"sc_protraction": 10.0, "ac_upward_rotation": 15.0}

    right = user.scapulothoracic_residuals_for("r", pose)
    left = user.scapulothoracic_residuals_for("l", pose)

    for name, value in right.items():
        assert left[name] == pytest.approx(value, abs=1e-9)


def test_residuals_grow_when_the_scapula_leaves_the_thorax(user):
    residuals = user.scapulothoracic_residuals_for("r", {"sc_elevation": 35.0})

    assert abs(residuals["TS"]) > 0.01
    assert abs(residuals["AI"]) > 0.01


def test_residuals_for_coordinates_do_not_modify_the_state(user):
    before = user.getCoordinateSet().get("sc_elevation_r").getValue(user.state)

    user.scapulothoracic_residuals_for("r", {"sc_elevation": 20.0})

    after = user.getCoordinateSet().get("sc_elevation_r").getValue(user.state)
    assert after == before


def test_scapulothoracic_api_rejects_invalid_arguments(user):
    with pytest.raises(ValueError):
        user.scapulothoracic_residuals("x")
    with pytest.raises(ValueError):
        user.scapulothoracic_residuals_for("r", {"arm_flex": 10.0})


@pytest.mark.parametrize("side", ["r", "l"])
def test_costal_point_distances_follow_the_scapula_and_match_rib_mesh(user, side):
    import vtk

    from opensim_frameworks.utils.collision import mesh_matrix, mesh_polydata

    contact = user._scapula_mesh_contact(side)
    user.realizePosition(user.state)
    rib_transform = vtk.vtkTransform()
    rib_transform.SetMatrix(mesh_matrix(contact.ribs, user.state))
    ribs_ground = vtk.vtkTransformFilter()
    ribs_ground.SetTransform(rib_transform)
    ribs_ground.SetInputData(mesh_polydata(contact.ribs, user.geometry_directories))
    ribs_ground.Update()
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(ribs_ground.GetOutput())
    locator.BuildLocator()
    scapula_matrix = mesh_matrix(contact.scapula, user.state)
    points = np.array([
        scapula_matrix.MultiplyPoint(vertex.tolist())[:3]
        for vertex in contact.costal_vertices
    ])
    assert np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])) > 1e-3
    expected = []
    for point in points:
        closest = [0.0, 0.0, 0.0]
        cell, sub, distance = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
        locator.FindClosestPoint(point, closest, cell, sub, distance)
        expected.append(np.sqrt(float(distance)))

    np.testing.assert_allclose(user.scapula_costal_distances(side), expected, atol=1e-7)
    gaps = user.scapula_costal_distances(side)
    coordinate = user.getCoordinateSet().get(f"sc_protraction_{side}")
    original = coordinate.getValue(user.state)
    try:
        coordinate.setValue(user.state, original + np.radians(10.0), False)
        moved = user.scapula_costal_distances(side)
        assert np.all(np.abs(moved - gaps) > 1e-5)
        assert contact is user._scapula_mesh_contact(side)
    finally:
        coordinate.setValue(user.state, original, False)
        user.realizePosition(user.state)


def test_scapula_vertex_distances_use_rib_mesh_vertex_centroid(user):
    from opensim_frameworks.utils.collision import mesh_matrix

    user.realizePosition(user.state)
    contact = user._scapula_mesh_contact("r")
    scapula_matrix = mesh_matrix(contact.scapula, user.state)
    ribs_matrix = mesh_matrix(contact.ribs, user.state)
    scapula_points = np.array([
        scapula_matrix.MultiplyPoint(vertex.tolist())[:3]
        for vertex in contact.vertices
    ])
    ribs_points = np.array([
        ribs_matrix.MultiplyPoint(vertex.tolist())[:3]
        for vertex in contact.ribs_vertices
    ])
    expected = np.linalg.norm(
        scapula_points - np.mean(ribs_points, axis=0), axis=1
    )

    actual = contact.distances_to_rib_center()

    assert actual.shape == (len(contact.vertices),)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert np.sum(actual) == pytest.approx(np.sum(expected))


def test_scapular_fit_uses_all_three_costal_distances(user, monkeypatch):
    from opensim_frameworks.models.user import scapulothoracic as contact

    gaps = np.array([0.001, 0.003, 0.006])
    monkeypatch.setattr(user, "scapula_costal_distances", lambda side: gaps)

    class IntersectingContact:
        def intersects_ribs(self):
            return True

    monkeypatch.setattr(user, "_scapula_mesh_contact", lambda side: IntersectingContact())

    residuals = user.scapular_fit_residuals("r", 5.0, 1000.0)

    assert residuals.shape == (6,)
    np.testing.assert_allclose(residuals[2:5], 1000.0 / np.sqrt(3) * (gaps - 0.005))
    assert residuals[5] == pytest.approx(
        contact._SHOULDER_COLLISION_PENALTY
    )


def test_contact_report_warns_when_one_costal_point_is_far(user, monkeypatch):
    class Contact:
        def distance(self):
            return 0.001

        def costal_distances(self):
            return np.array([0.005, 0.005, 0.008])

        def intersects(self):
            return False

    monkeypatch.setattr(user, "_scapula_mesh_contact", lambda side: Contact())
    with pytest.warns(RuntimeWarning, match="costal point gaps.*8.0"):
        report = user.scapula_contact_report("r")
    assert report["gap_m"] == pytest.approx(0.001)
    assert report["max_costal_gap_m"] == pytest.approx(0.008)
    assert not report["intersects"]


def test_shoulder_map_is_gender_specific_and_height_independent():
    from opensim_frameworks.models.user.scapular_map import load_scapular_map

    short = User("M", percentile=10.0)
    tall = User("M", percentile=90.0)
    short_before = np.array(
        [
            short.getCoordinateSet().get(f"{name}_r").getValue(short.state)
            for name in SHOULDER_GIRDLE_COORDINATES
        ]
    )
    short_map = short._shoulder_map("r")
    tall_map = tall._shoulder_map("r")
    shoulder = np.radians([5.0, -5.0])

    np.testing.assert_array_equal(
        short_map.model.coefficients_deg, tall_map.model.coefficients_deg
    )
    np.testing.assert_allclose(short_map.predict(*shoulder), tall_map.predict(*shoulder))
    np.testing.assert_allclose(
        [
            short.getCoordinateSet().get(f"{name}_r").getValue(short.state)
            for name in SHOULDER_GIRDLE_COORDINATES
        ],
        short_before,
    )
    assert not np.allclose(
        load_scapular_map("M").coefficients_deg,
        load_scapular_map("F").coefficients_deg,
    )


def test_shoulder_map_extrapolates_outside_fitted_domain(user):
    coordinates = [
        user.getCoordinateSet().get(f"{name}_r")
        for name in SHOULDER_GIRDLE_COORDINATES
    ]
    previous = np.array([coordinate.getValue(user.state) for coordinate in coordinates])
    shoulder = np.radians([-20.0, -10.0])
    try:
        with pytest.warns(RuntimeWarning, match="extrapolating"):
            user.set_shoulder_girdle("r", *shoulder)
        actual = np.array([coordinate.getValue(user.state) for coordinate in coordinates])
        np.testing.assert_allclose(actual[:2], shoulder)
        assert np.all(np.isfinite(actual))
    finally:
        user.set_scapular_girdle_values("r", previous)


@pytest.mark.parametrize("side", ["r", "l"])
def test_shoulder_map_applies_bilinear_pose_without_mesh_intersection(user, side):
    coordinates = [
        user.getCoordinateSet().get(f"{name}_{side}")
        for name in SHOULDER_GIRDLE_COORDINATES
    ]
    previous = np.array([coordinate.getValue(user.state) for coordinate in coordinates])
    shoulder = np.radians([5.0, -5.0])
    predicted = user._shoulder_map(side).predict(*shoulder)
    try:
        user.set_shoulder_girdle(side, *shoulder)
        actual = np.array([coordinate.getValue(user.state) for coordinate in coordinates])
        np.testing.assert_allclose(actual, predicted)
        assert not user._scapula_mesh_contact(side).intersects_ribs()
    finally:
        user.set_scapular_girdle_values(side, previous)


def test_shoulder_map_rejects_invalid_inputs_before_building(user, monkeypatch):
    def unexpected_map(side):
        raise AssertionError("map must not be built for invalid input")

    monkeypatch.setattr(user, "_shoulder_map", unexpected_map)
    with pytest.raises(ValueError):
        user.set_shoulder_girdle("x", 0.0, 0.0)
    with pytest.raises(ValueError):
        user.set_shoulder_girdle("r", np.nan, 0.0)
    with pytest.raises(ValueError):
        user.set_shoulder_girdle("r", np.pi, 0.0)


def test_shoulder_angle_setters_preserve_the_other_angle_and_use_the_map(
    user, monkeypatch
):
    coordinates = user.getCoordinateSet()
    coordinate_names = (
        "sc_protraction_r",
        "sc_elevation_r",
        "sc_protraction_l",
        "sc_elevation_l",
    )
    initial_angles = {
        name: coordinates.get(name).getValue(user.state)
        for name in coordinate_names
    }
    coordinates.get("sc_protraction_r").setValue(user.state, np.radians(5.0), False)
    coordinates.get("sc_elevation_r").setValue(user.state, np.radians(8.0), False)
    left_protraction = coordinates.get("sc_protraction_l").getValue(user.state)
    calls = []

    def capture(side, protraction, elevation, *, check_contact=False):
        calls.append((side, protraction, elevation, check_contact))
        coordinates.get(f"sc_protraction_{side}").setValue(
            user.state, protraction, False
        )
        coordinates.get(f"sc_elevation_{side}").setValue(
            user.state, elevation, False
        )

    monkeypatch.setattr(user, "set_shoulder_girdle", capture)

    user.set_right_shoulder_elevation(
        12.0, -5.0, 20.0, check_contact=False
    )
    user.set_left_shoulder_elevation(-3.0)
    user.set_right_shoulder_protraction(20.0, -10.0, 30.0)
    user.set_left_shoulder_protraction(-7.0)

    assert calls[0][0] == "r"
    assert calls[0][1:3] == pytest.approx((np.radians(5.0), np.radians(12.0)))
    assert calls[0][3] is False
    assert calls[1][0] == "l"
    assert calls[1][1:3] == pytest.approx(
        (left_protraction, np.radians(-3.0))
    )
    assert calls[1][3] is False
    assert calls[2][0] == "r"
    assert calls[2][1:3] == pytest.approx(
        (np.radians(20.0), np.radians(12.0))
    )
    assert calls[2][3] is False
    assert calls[3][0] == "l"
    assert calls[3][1:3] == pytest.approx(
        (np.radians(-7.0), np.radians(-3.0))
    )
    assert calls[3][3] is False
    assert coordinates.get("sc_elevation_r").getRangeMin() == pytest.approx(
        np.radians(-5.0)
    )
    assert coordinates.get("sc_elevation_r").getRangeMax() == pytest.approx(
        np.radians(20.0)
    )
    assert coordinates.get("sc_protraction_r").getRangeMin() == pytest.approx(
        np.radians(-10.0)
    )
    for name, value in initial_angles.items():
        coordinates.get(name).setValue(user.state, value, False)


def test_shoulder_protraction_and_elevation_properties_read_degrees(user):
    coordinates = user.getCoordinateSet()
    initial_values = {
        name: coordinates.get(name).getValue(user.state)
        for name in (
            "sc_protraction_l",
            "sc_protraction_r",
            "sc_elevation_l",
            "sc_elevation_r",
        )
    }
    try:
        for side, suffix in (("left", "l"), ("right", "r")):
            coordinates.get(f"sc_protraction_{suffix}").setValue(
                user.state, np.radians(12.5), False
            )
            coordinates.get(f"sc_elevation_{suffix}").setValue(
                user.state, np.radians(-7.5), False
            )
            assert getattr(user, f"{side}_shoulder_protraction") == pytest.approx(
                12.5
            )
            assert getattr(user, f"{side}_shoulder_elevation") == pytest.approx(
                -7.5
            )
    finally:
        for name, value in initial_values.items():
            coordinates.get(name).setValue(user.state, value, False)
    assert coordinates.get("sc_protraction_r").getRangeMax() == pytest.approx(
        np.radians(30.0)
    )


def test_shoulder_map_warns_with_contact_count_and_retains_pose(user, monkeypatch):
    from types import SimpleNamespace

    class IntersectingContact:
        def intersects_ribs(self):
            return True

        def rib_intersection_contact_count(self):
            return 17

    monkeypatch.setattr(user, "_scapula_mesh_contact", lambda side: IntersectingContact())
    monkeypatch.setattr(
        user,
        "_shoulder_map",
        lambda side: SimpleNamespace(
            model=SimpleNamespace(contains=lambda *angles: True),
            predict=lambda protraction, elevation: np.radians(
                [0.0, 0.0, 10.0, 20.0, 10.0]
            )
        ),
    )
    predicted = np.radians([0.0, 0.0, 10.0, 20.0, 10.0])
    coordinates = [
        user.getCoordinateSet().get(f"{name}_r")
        for name in SHOULDER_GIRDLE_COORDINATES
    ]
    previous = np.array([coordinate.getValue(user.state) for coordinate in coordinates])

    try:
        with pytest.warns(RuntimeWarning, match="17 detected mesh contacts"):
            user.set_shoulder_girdle("r", 0.0, 0.0, check_contact=False)
        actual = np.array([coordinate.getValue(user.state) for coordinate in coordinates])
        np.testing.assert_allclose(actual, predicted)
    finally:
        user.set_scapular_girdle_values("r", previous)


def test_shoulder_map_clips_out_of_range_ac_predictions_with_warning(user, monkeypatch):
    from types import SimpleNamespace

    class ClearContact:
        def intersects_ribs(self):
            return False

    monkeypatch.setattr(user, "_scapula_mesh_contact", lambda side: ClearContact())
    monkeypatch.setattr(
        user,
        "_shoulder_map",
        lambda side: SimpleNamespace(
            model=SimpleNamespace(contains=lambda *angles: True),
            predict=lambda protraction, elevation: np.radians(
                [0.0, 0.0, 40.0, -30.0, 40.0]
            ),
        ),
    )
    coordinates = [
        user.getCoordinateSet().get(f"{name}_r")
        for name in SHOULDER_GIRDLE_COORDINATES
    ]
    previous = np.array([coordinate.getValue(user.state) for coordinate in coordinates])
    try:
        with pytest.warns(RuntimeWarning, match="was clipped"):
            user.set_shoulder_girdle("r", 0.0, 0.0)
        actual = np.array([coordinate.getValue(user.state) for coordinate in coordinates])
        assert np.degrees(actual) == pytest.approx([0.0, 0.0, 25.0, -15.0, 30.0])
    finally:
        user.set_scapular_girdle_values("r", previous)


def test_reference_poses_are_within_ranges_and_contact_pose_touches(user):
    raw = user.shoulder_girdle_reference_pose(contact=False)
    contact = user.shoulder_girdle_reference_pose()

    for pose in (raw, contact):
        assert set(pose) == set(SHOULDER_GIRDLE_COORDINATES)
        for name, value in pose.items():
            low, high = SHOULDER_GIRDLE_COORDINATES[name]
            assert low - 1e-9 <= value <= high + 1e-9
    # The raw hypothesis does not satisfy the contact, the other one does.
    assert abs(user.scapulothoracic_residuals_for("r", raw)["AI"]) > 0.01
    residuals = user.scapulothoracic_residuals_for("r", contact)
    assert residuals["AI"] == pytest.approx(0.0, abs=1e-4)
    assert residuals["TS"] == pytest.approx(0.0, abs=1e-4)
    for name in ("AA", "MB1", "MB2", "SA"):
        assert residuals[name] == pytest.approx(0.0, abs=1e-4)


def test_ellipsoid_radius_is_scale_invariant_and_gradient_is_analytic():
    from opensim_frameworks.models.user import scapulothoracic as st

    points = np.array([0.0, 0.0, 0.17]) + st._POINT_OFFSETS + [0.0, 0.0, 0.02]
    contact = np.array([1.158, 1.239])
    scale = np.array([1.05, 0.96, 1.12])
    unit = st._residual_array(st._radii(points, np.ones(3), "r"), contact, 1.0)
    scaled = st._residual_array(st._radii(points * scale, scale, "r"), contact, 1.0)
    # Radii are independent of the torso scale when points scale with it.
    assert scaled == pytest.approx(unit)

    point = np.array([-0.03, 0.30, 0.10])
    _, gradient = st._ellipsoid_radius(point)
    step = 1e-7
    for axis in range(3):
        shift = np.zeros(3)
        shift[axis] = step
        numeric = (
            st._ellipsoid_radius(point + shift)[0]
            - st._ellipsoid_radius(point - shift)[0]
        ) / (2 * step)
        assert gradient[axis] == pytest.approx(numeric, rel=1e-5)


def test_residual_array_matches_the_vector_variant(user):
    rng = np.random.default_rng(2)
    lower, upper = np.array(list(SHOULDER_GIRDLE_COORDINATES.values())).T
    values = rng.uniform(lower, upper, (6, len(lower)))

    for side in ("r", "l"):
        batch = user.scapulothoracic_residual_array(side, values)
        assert batch.shape == (6, 6)
        for row, expected in zip(values, batch):
            np.testing.assert_allclose(
                user.scapulothoracic_residual_vector(side, row), expected, atol=1e-12
            )
    with pytest.raises(ValueError):
        user.scapulothoracic_residual_array("r", values[0])


def test_unilateral_residual_is_zero_up_to_the_allowed_depth_and_smooth_beyond():
    from opensim_frameworks.models.user import scapulothoracic as contact

    length = 0.12
    depth = np.linspace(-5e-3, 1.5e-3, 6501)
    radii = np.tile([1.2, 1.1, 1.0, 1.0, 1.0, 1.0], (len(depth), 1))
    radii[:, 2] = contact._CLEARANCE_RADIUS - depth / length

    residual = contact._residual_array(radii, np.array([1.2, 1.1]), length)[:, 2]

    # no floor: exactly zero outside and up to the allowed depth
    assert np.all(residual[depth <= contact._PENETRATION_ALLOWANCE] == 0.0)
    # continuous, with a continuous derivative, and increasing beyond
    slope = np.diff(residual) / np.diff(depth)
    assert np.abs(np.diff(residual)).max() <= np.diff(depth).max() * (1 + 1e-9)
    assert np.abs(np.diff(slope)).max() < 0.1
    assert np.all(np.diff(residual) >= 0.0)
    # far beyond the blend it is the depth minus half the blend and the allowance
    expected = 1.5e-3 - contact._PENETRATION_ALLOWANCE - 0.5 * contact._PENETRATION_BLEND
    assert residual[-1] == pytest.approx(expected, abs=1e-9)


# -- Meshes of the clavicles and the scapulae.


def test_split_meshes_are_the_original_thorax_mesh_distributed():
    from opensim_frameworks import read_vtp_vertices
    from opensim_frameworks.models.user import girdle_meshes as meshes

    original = read_vtp_vertices(meshes.MESHES_DIR / meshes.SOURCE_MESH_FILE)
    parts = [
        meshes.THORAX_MESH_FILE,
        *meshes.CLAVICLE_MESH_FILES.values(),
        *meshes.SCAPULA_MESH_FILES.values(),
    ]
    union = np.vstack([read_vtp_vertices(meshes.MESHES_DIR / name) for name in parts])

    assert len(union) == len(original)
    assert np.array_equal(
        np.unique(np.round(union, 9), axis=0), np.unique(np.round(original, 9), axis=0)
    )


def test_bones_of_the_girdle_are_drawn_on_their_bodies(user):
    files = {}
    for mesh, _ in get_referenced_meshes(user):
        files[mesh.getName()] = mesh.get_mesh_file()

    assert files["clavicle_r_geom"] == "hat_clavicle_r.vtp"
    assert files["scapula_l_geom"] == "hat_scapula_l.vtp"
    # the torso no longer draws the bones that the girdle moves
    assert "hat_ribs_torso.vtp" in files.values()
    assert "hat_ribs_scap.vtp" not in files.values()


@pytest.mark.parametrize("side", ["r", "l"])
def test_girdle_meshes_keep_their_place_at_rest(user, side):
    from opensim_frameworks import read_vtp_vertices
    from opensim_frameworks.models.user import girdle_meshes as meshes
    from opensim_frameworks.utils.opensim import rotation_to_numpy, vec3_to_numpy

    state = user.state
    user.realizePosition(state)
    torso = user.getBodySet().get("torso").getTransformInGround(state)
    thorax = user.getBodySet().get("torso")
    scale = None
    for index in range(thorax.getPropertyByName("attached_geometry").size()):
        mesh = opensim.Mesh.safeDownCast(thorax.get_attached_geometry(index))
        if mesh is not None and mesh.get_mesh_file() == meshes.THORAX_MESH_FILE:
            scale = np.array([mesh.get_scale_factors().get(i) for i in range(3)])
    original = (
        read_vtp_vertices(meshes.MESHES_DIR / meshes.SCAPULA_MESH_FILES[side]) * scale
    ) @ rotation_to_numpy(torso.R()).T + vec3_to_numpy(torso.p())
    scapula = [
        mesh
        for mesh, _ in get_referenced_meshes(user)
        if mesh.getName() == f"scapula_{side}_geom"
    ][0]
    frame = scapula.getFrame().getTransformInGround(state)
    factors = np.array([scapula.get_scale_factors().get(i) for i in range(3)])
    placed = (
        read_vtp_vertices(meshes.MESHES_DIR / scapula.get_mesh_file()) * factors
    ) @ rotation_to_numpy(frame.R()).T + vec3_to_numpy(frame.p())

    assert placed == pytest.approx(original, abs=1e-9)


@pytest.mark.parametrize("side", ["r", "l"])
def test_clavicle_mesh_ends_on_the_sternoclavicular_and_acromioclavicular_centres(
    user, side
):
    from opensim_frameworks import read_vtp_vertices
    from opensim_frameworks.models.user import girdle_meshes as meshes
    from opensim_frameworks.utils.opensim import rotation_to_numpy, vec3_to_numpy

    state = user.state
    user.realizePosition(state)
    sc = vec3_to_numpy(
        user.getJointSet().get(f"sternoclavicular_{side}").getChildFrame().getPositionInGround(state)
    )
    ac = vec3_to_numpy(
        user.getJointSet().get(f"acromioclavicular_{side}").getChildFrame().getPositionInGround(state)
    )
    mesh = [
        mesh
        for mesh, _ in get_referenced_meshes(user)
        if mesh.getName() == f"clavicle_{side}_geom"
    ][0]
    frame = mesh.getFrame().getTransformInGround(state)
    factors = np.array([mesh.get_scale_factors().get(i) for i in range(3)])
    points = (
        read_vtp_vertices(meshes.MESHES_DIR / mesh.get_mesh_file()) * factors
    ) @ rotation_to_numpy(frame.R()).T + vec3_to_numpy(frame.p())
    projection = (points - sc) @ ((ac - sc) / np.linalg.norm(ac - sc))

    assert np.linalg.norm(points[projection.argmin()] - sc) < 1e-6
    assert np.linalg.norm(points[projection.argmax()] - ac) < 1e-6


def test_to_osim_exports_the_girdle_meshes(user, tmp_path):
    destination = user.to_osim(tmp_path / "user.osim")

    for name in (
        "hat_ribs_torso.vtp",
        "hat_clavicle_r.vtp",
        "hat_clavicle_l.vtp",
        "hat_scapula_r.vtp",
        "hat_scapula_l.vtp",
    ):
        assert (tmp_path / "Geometry" / name).is_file()
    exported = opensim.Model(str(destination))
    exported.initSystem()
    assert exported.getBodySet().get("scapula_r").getPropertyByName("components").size() >= 1


# -- Scapular protraction as a single angle.


def test_protraction_range_contains_the_rest_pose_and_is_cached(user):
    low, high = user.right_scapular_protraction_range

    assert low <= 0.0 <= high
    assert high <= SHOULDER_GIRDLE_COORDINATES["sc_protraction"][1]
    assert user.right_scapular_protraction_range == (low, high)
    # the left side is the mirror of the right one: same values, same interval
    assert user.left_scapular_protraction_range == (low, high)


def test_set_protraction_writes_the_five_coordinates_and_keeps_the_gh_height(user):
    from opensim_frameworks.utils.opensim import vec3_to_numpy

    def gh_height_in_torso():
        user.realizePosition(user.state)
        torso = user.getBodySet().get("torso").getTransformInGround(user.state)
        joint = user.getJointSet().get("acromial_r").getParentFrame()
        point = vec3_to_numpy(joint.getPositionInGround(user.state))
        rotation = np.array([[torso.R().get(i, j) for j in range(3)] for i in range(3)])
        return (rotation.T @ (point - vec3_to_numpy(torso.p())))[1]

    rest = gh_height_in_torso()
    _, high = user.right_scapular_protraction_range
    user.set_right_scapular_protraction(high)
    try:
        assert user.right_scapular_protraction == pytest.approx(high)
        assert gh_height_in_torso() == pytest.approx(rest, abs=1e-5)
        others = user.scapular_girdle_from_protraction("r", high)
        names = list(SHOULDER_GIRDLE_COORDINATES)[1:]
        for name, value in zip(names, others):
            assert user._coordinate_degrees(f"{name}_r") == pytest.approx(value)
    finally:
        user.set_right_scapular_protraction(0.0)


def test_protraction_outside_the_range_is_rejected_without_clamping(user):
    low, high = user.right_scapular_protraction_range

    with pytest.raises(ValueError):
        user.set_right_scapular_protraction(high + 1.0)
    with pytest.raises(ValueError):
        user.set_left_scapular_protraction(low - 1.0)
    with pytest.raises(ValueError):
        user.set_right_scapular_protraction(float("nan"))
    with pytest.raises(ValueError):
        user.scapular_girdle_from_protraction("x", 0.0)


def test_protraction_map_has_no_penetration_in_the_admissible_range(user):
    built = user._protraction_map("r")

    inside = (built.grid >= built.low) & (built.grid <= built.high)
    assert np.all(built.contacts[inside] <= built.rest_contacts)


# -- Widening of the range of a coordinate.


def test_extend_coordinate_range_widens_only_and_keeps_the_value(user):
    coordinate = user.getCoordinateSet().get("pro_sup_r")
    low, high = coordinate.getRangeMin(), coordinate.getRangeMax()

    assert not user.extend_coordinate_range("pro_sup_r", 10.0, 100.0)
    assert (coordinate.getRangeMin(), coordinate.getRangeMax()) == (low, high)
    assert user.extend_coordinate_range("pro_sup_r", None, np.degrees(high) + 30.0)
    try:
        assert coordinate.getRangeMax() == pytest.approx(high + np.radians(30.0))
        assert coordinate.getRangeMin() == low
        with pytest.raises(ValueError):
            user.extend_coordinate_range("pro_sup_r", float("nan"), None)
    finally:
        coordinate.setRangeMax(high)


def test_framework_merges_a_model_with_a_value_outside_the_original_range():
    from opensim_frameworks import Framework

    fitted = User("M", height_cm=175)
    maximum = fitted.getCoordinateSet().get("pro_sup_r").getRangeMax()
    fitted._set_coordinate_degrees("pro_sup_r", np.degrees(maximum) + 20.0)
    fitted.extend_coordinate_range("pro_sup_r", None, np.degrees(maximum) + 25.0)
    fitted.reinitialize()

    merged = Framework(user=fitted).merge_to_model()

    assert merged.initSystem() is not None
