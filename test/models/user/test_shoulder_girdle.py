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
    assert np.abs(np.diff(residual)).max() < 1e-6
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
