"""Regression checks for the hand geometry used by User."""

import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from opensim_frameworks import Framework, User
from opensim_frameworks.models.user.hand import (
    FINGERS,
    JOINT_SEPARATION_MM,
    MOVING_HAND_SEGMENTS,
    add_articulated_hands,
)
from opensim_frameworks.models.user.user import DEFAULT_MESHES_DIR, DEFAULT_MODEL_PATH
from opensim_frameworks.utils.opensim import read_vtp_vertices

opensim = pytest.importorskip("opensim")
vtk = pytest.importorskip("vtk")


def test_hand_module_wildcard_exports():
    """Export only the declared hand constants and construction function."""
    namespace = {}
    exec("from opensim_frameworks.models.user.hand import *", namespace)
    assert set(namespace) == {
        "__builtins__",
        "FINGERS",
        "FINGER_SEGMENTS",
        "MOVING_HAND_SEGMENTS",
        "JOINT_SEPARATION_MM",
        "add_articulated_hands",
    }


def _finger_mesh_names(side):
    """Return the moving finger mesh filenames for one side.

    Parameters
    ----------
    side : {'r', 'l'}
        Hand side.

    Returns
    -------
    set of str
        Names of the moving bone mesh files.
    """
    names = ["metacarpal1", "thumb_proximal", "thumb_distal"]
    for finger in ("index", "middle", "ring", "little"):
        names.extend(
            f"{finger}_{segment}" for segment in ("proximal", "medial", "distal")
        )
    return {f"{name}_{side}vs.vtp" for name in names}


def _rotation_matrix(rotation):
    """Convert an OpenSim rotation to a NumPy matrix.

    Parameters
    ----------
    rotation : opensim.Rotation
        Rotation to convert.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Rotation matrix.
    """
    return np.array(
        [[rotation.get(row, column) for column in range(3)] for row in range(3)]
    )


def _vtk_mesh(mesh):
    """Load an attached bone mesh as VTK polygonal data.

    Parameters
    ----------
    mesh : opensim.Mesh
        Mesh whose file is in ``DEFAULT_MESHES_DIR``.

    Returns
    -------
    vtk.vtkPolyData
        Polygonal mesh.
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(DEFAULT_MESHES_DIR / mesh.get_mesh_file()))
    reader.Update()
    return reader.GetOutput()


def _vtk_matrix(user, mesh):
    """Build the scaled mesh-to-ground transform for the current user state.

    Parameters
    ----------
    user : opensim.Model
        Initialized user model with a ``state`` attribute.
    mesh : opensim.Mesh
        Mesh attached to a frame in the model.

    Returns
    -------
    vtk.vtkMatrix4x4
        Transform from unscaled mesh vertices to ground.
    """
    transform = mesh.getFrame().getTransformInGround(user.state)
    scale = mesh.get_scale_factors()
    matrix = vtk.vtkMatrix4x4()
    for row in range(3):
        for column in range(3):
            matrix.SetElement(row, column, transform.R().get(row, column) * scale.get(column))
        matrix.SetElement(row, 3, transform.p().get(row))
    return matrix


def _finger_vertices_in_hand(user, side, mesh_directory=DEFAULT_MESHES_DIR):
    """Return each finger mesh in the hand frame, regardless of its owning body.

    Parameters
    ----------
    user : opensim.Model
        Initialized user model with a ``state`` attribute.
    side : {'r', 'l'}
        Hand side to inspect.
    mesh_directory : pathlib.Path
        Directory containing the bone meshes.

    Returns
    -------
    dict of str to numpy.ndarray
        Scaled vertices in the hand frame, keyed by mesh filename.
    """
    # Resolve the current hand frame and identify the expected finger meshes.
    user.realizePosition(user.state)
    hand = user.getBodySet().get(f"hand_{side}")
    hand_transform = hand.getTransformInGround(user.state)
    hand_rotation = _rotation_matrix(hand_transform.R())
    hand_translation = np.array([hand_transform.p().get(i) for i in range(3)])
    wanted = _finger_mesh_names(side)
    found = {}

    # Transform each finger mesh through ground into the hand frame.
    bodies = user.getBodySet()
    for body_index in range(bodies.getSize()):
        body = bodies.get(body_index)
        for geometry_index in range(body.getPropertyByName("attached_geometry").size()):
            mesh = opensim.Mesh.safeDownCast(body.get_attached_geometry(geometry_index))
            if mesh is None or Path(mesh.get_mesh_file()).name not in wanted:
                continue
            name = Path(mesh.get_mesh_file()).name
            assert name not in found, f"{name} is attached more than once"
            vertices = read_vtp_vertices(mesh_directory / mesh.get_mesh_file())
            scale = np.array([mesh.get_scale_factors().get(i) for i in range(3)])
            transform = mesh.getFrame().getTransformInGround(user.state)
            rotation = _rotation_matrix(transform.R())
            translation = np.array([transform.p().get(i) for i in range(3)])
            in_ground = (vertices * scale) @ rotation.T + translation
            found[name] = (in_ground - hand_translation) @ hand_rotation

    assert set(found) == wanted
    return found


@pytest.mark.parametrize("side", ("r", "l"))
def test_finger_meshes_match_reference_with_documented_separations(side):
    """Compare finger meshes with the reference and documented corrections."""
    user = User("M", percentile=50)
    actual = _finger_vertices_in_hand(user, side)
    reference = opensim.Model(str(DEFAULT_MODEL_PATH))
    hand = reference.getBodySet().get(f"hand_{side}")
    expected_names = _finger_mesh_names(side)
    reference_names = set()
    displacements = {}

    for index in range(hand.getPropertyByName("attached_geometry").size()):
        mesh = opensim.Mesh.safeDownCast(hand.get_attached_geometry(index))
        if mesh is None or mesh.get_mesh_file() not in expected_names:
            continue
        name = mesh.get_mesh_file()
        assert name not in reference_names
        reference_names.add(name)
        vertices = read_vtp_vertices(DEFAULT_MESHES_DIR / name)
        scale = np.array([mesh.get_scale_factors().get(i) for i in range(3)])
        delta = actual[name] - vertices * scale
        displacements[name.removesuffix(f"_{side}vs.vtp")] = delta.mean(axis=0)
        np.testing.assert_allclose(
            delta, np.broadcast_to(delta.mean(axis=0), delta.shape), rtol=0, atol=1e-8
        )
    assert reference_names == expected_names
    for finger in (*FINGERS, "thumb"):
        segments = (
            ("metacarpal1", "thumb_proximal", "thumb_distal")
            if finger == "thumb"
            else tuple(f"{finger}_{part}" for part in ("proximal", "medial", "distal"))
        )
        previous = np.zeros(3)
        for segment, distance_mm in zip(segments, JOINT_SEPARATION_MM[finger]):
            offset = displacements[segment]
            assert np.linalg.norm(offset - previous) == pytest.approx(
                distance_mm / 1000, abs=1e-8
            )
            previous = offset


@pytest.mark.parametrize(("gender", "height_cm"), (("M", 155), ("F", 180)))
@pytest.mark.parametrize("side", ("r", "l"))
def test_fingers_follow_anthropometric_hand_scaling(gender, height_cm, side):
    """Check finger scaling for two anthropometries on both sides."""
    user = User(gender, height_cm=height_cm)
    actual = _finger_vertices_in_hand(user, side)
    baseline = User(gender, percentile=50)
    baseline_vertices = _finger_vertices_in_hand(baseline, side)
    hand = user.getBodySet().get(f"hand_{side}")
    reference_mesh = opensim.Mesh.safeDownCast(hand.get_attached_geometry(0))
    scale = np.array([reference_mesh.get_scale_factors().get(i) for i in range(3)])
    for name, vertices in actual.items():
        original = read_vtp_vertices(DEFAULT_MESHES_DIR / name)
        correction = (baseline_vertices[name] - original * 0.85).mean(axis=0)
        np.testing.assert_allclose(
            vertices, (original * 0.85 + correction) * scale / 0.85, rtol=0, atol=1e-8
        )


@pytest.mark.parametrize(("side", "other"), (("r", "l"), ("l", "r")))
def test_palm_length_scaling_preserves_bilateral_finger_geometry(side, other):
    """Check that changing one palm length scales only its own fingers."""
    user = User("M", percentile=50)
    before = _finger_vertices_in_hand(user, side)
    opposite_before = _finger_vertices_in_hand(user, other)
    name = "right" if side == "r" else "left"
    getattr(user, f"set_{name}_palm_length")(getattr(user, f"{name}_palm_length") * 1.1)
    after = _finger_vertices_in_hand(user, side)
    opposite_after = _finger_vertices_in_hand(user, other)

    for mesh_name, vertices in before.items():
        np.testing.assert_allclose(
            after[mesh_name], vertices * np.array([1.0, 1.1, 1.0]), rtol=0, atol=1e-8
        )
    for mesh_name, vertices in opposite_before.items():
        np.testing.assert_allclose(opposite_after[mesh_name], vertices, rtol=0, atol=1e-8)


@pytest.mark.parametrize(("side", "suffix"), (("right", "r"), ("left", "l")))
def test_finger_bodies_and_coordinates_start_locked(side, suffix):
    """Check finger masses, inertias, meshes and initial coordinate locks."""
    user = User("M", percentile=50)
    bodies = user.getBodySet()
    for segment in MOVING_HAND_SEGMENTS:
        body = bodies.get(f"{segment}_{suffix}")
        assert body.getMass() > 0
        assert all(body.getInertia().getMoments().get(i) > 0 for i in range(3))
        assert body.getPropertyByName("attached_geometry").size() == 1

    coordinates = [
        (finger, joint, motion)
        for finger in FINGERS
        for joint, motion in (
            ("mcp", "flexion"),
            ("mcp", "abduction"),
            ("pip", "flexion"),
            ("dip", "flexion"),
        )
    ]
    coordinates.extend(
        ("thumb", joint, motion)
        for joint, motion in (
            ("cmc", "opposition"),
            ("cmc", "flexion"),
            ("mcp", "flexion"),
            ("ip", "flexion"),
        )
    )
    for finger, joint, motion in coordinates:
        coordinate = user.finger_coordinate(side, finger, joint, motion)
        assert coordinate.getLocked(user.state)
        assert coordinate.getRangeMin() <= 0 <= coordinate.getRangeMax()
        assert user.finger_angle(side, finger, joint, motion) == pytest.approx(0)
        with pytest.raises(ValueError, match="locked"):
            user.set_finger_angle(side, finger, joint, motion, 10)


@pytest.mark.parametrize(("side", "suffix"), (("right", "r"), ("left", "l")))
@pytest.mark.parametrize("finger", FINGERS)
def test_mcp_flexion_moves_only_the_selected_finger(side, suffix, finger):
    """Check that MCP flexion moves only the selected finger."""
    user = User("M", percentile=50)
    before = _finger_vertices_in_hand(user, suffix)
    user.set_finger_coordinate_locked(side, finger, "mcp", "flexion", False)
    user.set_finger_angle(side, finger, "mcp", "flexion", 20)
    after = _finger_vertices_in_hand(user, suffix)

    for name in before:
        if name.startswith(f"{finger}_"):
            assert not np.allclose(after[name], before[name], rtol=0, atol=1e-6)
            assert np.mean(after[name][:, 0] - before[name][:, 0]) > 0
        else:
            np.testing.assert_allclose(after[name], before[name], rtol=0, atol=1e-8)
    assert user.finger_angle(side, finger, "mcp", "flexion") == pytest.approx(20)


@pytest.mark.parametrize(("side", "suffix", "sign"), (("right", "r", -1), ("left", "l", 1)))
def test_thumb_opposition_moves_towards_palm_midline(side, suffix, sign):
    """Check mirrored thumb opposition towards the palm midline."""
    user = User("M", percentile=50)
    name = f"thumb_distal_{suffix}vs.vtp"
    before = _finger_vertices_in_hand(user, suffix)[name]
    user.set_finger_coordinate_locked(side, "thumb", "cmc", "opposition", False)
    user.set_finger_angle(side, "thumb", "cmc", "opposition", 20)
    after = _finger_vertices_in_hand(user, suffix)[name]
    assert sign * np.mean(after[:, 2] - before[:, 2]) > 0


@pytest.mark.parametrize(("side", "suffix"), (("right", "r"), ("left", "l")))
@pytest.mark.parametrize("finger", FINGERS)
@pytest.mark.parametrize("joint", ("pip", "dip"))
def test_interphalangeal_flexion_moves_distal_segments_only(side, suffix, finger, joint):
    """Check that PIP and DIP flexion move only downstream segments."""
    user = User("M", percentile=50)
    before = _finger_vertices_in_hand(user, suffix)
    user.set_finger_coordinate_locked(side, finger, joint, "flexion", False)
    user.set_finger_angle(side, finger, joint, "flexion", 20)
    after = _finger_vertices_in_hand(user, suffix)
    moved = {f"{finger}_distal"}
    if joint == "pip":
        moved.add(f"{finger}_medial")
    for name in before:
        if any(name.startswith(f"{segment}_") for segment in moved):
            assert not np.allclose(after[name], before[name], rtol=0, atol=1e-6)
            assert np.mean(after[name][:, 0] - before[name][:, 0]) > 0
        else:
            np.testing.assert_allclose(after[name], before[name], rtol=0, atol=1e-8)


@pytest.mark.parametrize(("side", "suffix"), (("right", "r"), ("left", "l")))
@pytest.mark.parametrize("joint", ("cmc", "mcp", "ip"))
def test_thumb_flexion_moves_distal_segments_only(side, suffix, joint):
    """Check that thumb flexion moves only downstream segments."""
    user = User("M", percentile=50)
    before = _finger_vertices_in_hand(user, suffix)
    user.set_finger_coordinate_locked(side, "thumb", joint, "flexion", False)
    user.set_finger_angle(side, "thumb", joint, "flexion", 20)
    after = _finger_vertices_in_hand(user, suffix)
    moved = {"thumb_distal"}
    if joint in ("cmc", "mcp"):
        moved.add("thumb_proximal")
    if joint == "cmc":
        moved.add("metacarpal1")
    for name in before:
        if any(name.startswith(f"{segment}_") for segment in moved):
            assert not np.allclose(after[name], before[name], rtol=0, atol=1e-6)
            assert np.mean(after[name][:, 0] - before[name][:, 0]) > 0
        else:
            np.testing.assert_allclose(after[name], before[name], rtol=0, atol=1e-8)


@pytest.mark.parametrize(("side", "suffix", "sign"), (("right", "r", 1), ("left", "l", -1)))
@pytest.mark.parametrize(
    ("finger", "direction"), (("index", 1), ("middle", 1), ("ring", -1), ("little", -1))
)
def test_mcp_abduction_has_mirrored_directions(side, suffix, sign, finger, direction):
    """Check the side-dependent direction of finger abduction."""
    user = User("M", percentile=50)
    name = f"{finger}_proximal_{suffix}vs.vtp"
    before = _finger_vertices_in_hand(user, suffix)[name]
    user.set_finger_coordinate_locked(side, finger, "mcp", "abduction", False)
    user.set_finger_angle(side, finger, "mcp", "abduction", 10)
    after = _finger_vertices_in_hand(user, suffix)[name]
    assert sign * direction * np.mean(after[:, 2] - before[:, 2]) > 0


def test_finger_state_survives_reinitialization_and_export(tmp_path):
    """Check that reinitialization and export preserve finger pose and locks."""
    # Set a finger pose and verify its state after reinitialization.
    user = User("M", percentile=50)
    user.set_finger_coordinate_locked("right", "index", "pip", "flexion", False)
    user.set_finger_angle("right", "index", "pip", "flexion", 20)
    user.reinitialize()
    coordinate = user.finger_coordinate("right", "index", "pip", "flexion")
    assert not coordinate.getLocked(user.state)
    assert user.finger_angle("right", "index", "pip", "flexion") == pytest.approx(20)
    user.set_finger_angle("right", "index", "pip", "flexion", 30)
    expected = _finger_vertices_in_hand(user, "r")

    # Export and reload the model, then compare its pose and finger meshes.
    filename = tmp_path / "user.osim"
    user.to_osim(filename)
    restored = opensim.Model(str(filename))
    restored.finalizeConnections()
    state = restored.initSystem()
    restored.realizePosition(state)
    restored_coordinate = restored.getCoordinateSet().get("index_pip_flexion_r")
    assert restored_coordinate.getValue(state) == pytest.approx(math.radians(30))
    assert not restored_coordinate.getLocked(state)
    restored.state = state
    actual = _finger_vertices_in_hand(restored, "r", tmp_path)
    for segment in MOVING_HAND_SEGMENTS:
        assert restored.getBodySet().contains(f"{segment}_r")
    for name, vertices in expected.items():
        np.testing.assert_allclose(actual[name], vertices, rtol=0, atol=1e-8)


def test_framework_composition_preserves_finger_pose_and_lock():
    """Check that framework composition preserves finger pose and lock state."""
    user = User("M", percentile=50)
    user.set_finger_coordinate_locked("right", "index", "pip", "flexion", False)
    user.set_finger_angle("right", "index", "pip", "flexion", 20)
    expected = _finger_vertices_in_hand(user, "r")

    merged = Framework({"user": user}).merge_to_model()
    merged.state = merged.initSystem()
    coordinate = merged.getCoordinateSet().get("index_pip_flexion_r")
    assert coordinate.getValue(merged.state) == pytest.approx(math.radians(20))
    assert not coordinate.getLocked(merged.state)
    actual = _finger_vertices_in_hand(merged, "r")
    for name, vertices in expected.items():
        np.testing.assert_allclose(actual[name], vertices, rtol=0, atol=1e-8)


def test_finger_coordinate_rejects_invalid_values_and_survives_scaling():
    """Check coordinate validation, exact bounds and pose retention after scaling."""
    user = User("M", percentile=50)
    with pytest.raises(ValueError, match="side"):
        user.finger_coordinate("r", "index", "mcp", "flexion")
    with pytest.raises(ValueError, match="Unknown"):
        user.finger_coordinate("right", "thumb", "pip", "flexion")
    with pytest.raises(ValueError, match="bool"):
        user.set_finger_coordinate_locked("right", "index", "mcp", "flexion", 1)

    user.set_finger_coordinate_locked("right", "index", "mcp", "flexion", False)
    with pytest.raises(ValueError, match="finite"):
        user.set_finger_angle("right", "index", "mcp", "flexion", float("nan"))
    with pytest.raises(ValueError, match="range"):
        user.set_finger_angle("right", "index", "mcp", "flexion", 180)
    coordinate = user.finger_coordinate("right", "index", "mcp", "flexion")
    for bound in (coordinate.getRangeMin(), coordinate.getRangeMax()):
        user.set_finger_angle("right", "index", "mcp", "flexion", math.degrees(bound))
        assert coordinate.getValue(user.state) == bound
    user.set_finger_angle("right", "index", "mcp", "flexion", 20)
    user.set_right_palm_length(user.right_palm_length * 1.1)
    coordinate = user.finger_coordinate("right", "index", "mcp", "flexion")
    assert not coordinate.getLocked(user.state)
    assert user.finger_angle("right", "index", "mcp", "flexion") == pytest.approx(20)


def test_hand_split_preserves_mass_and_first_moment():
    """Check the original hand mass and first moment after articulation."""
    # Save the mass and centre of each hand before adding the articulated bodies.
    model = opensim.Model(str(DEFAULT_MODEL_PATH))
    reference = {}
    for side in ("r", "l"):
        hand = model.getBodySet().get(f"hand_{side}")
        reference[side] = (
            hand.getMass(),
            np.array([hand.getMassCenter().get(i) for i in range(3)]),
        )
    # Articulate the hands and compare the combined bodies with the originals.
    add_articulated_hands(model, opensim, DEFAULT_MESHES_DIR)
    model.finalizeConnections()
    state = model.initSystem()
    model.realizePosition(state)
    for side in ("r", "l"):
        mass, center = reference[side]
        bodies = [
            model.getBodySet().get(name)
            for name in (f"hand_{side}", *(f"{part}_{side}" for part in MOVING_HAND_SEGMENTS))
        ]
        actual_mass = sum(body.getMass() for body in bodies)
        transform = bodies[0].getTransformInGround(state)
        rotation = _rotation_matrix(transform.R())
        translation = np.array([transform.p().get(i) for i in range(3)])
        moment = sum(
            body.getMass()
            * (
                np.array(
                    [
                        body.findStationLocationInGround(state, body.getMassCenter()).get(i)
                        for i in range(3)
                    ]
                )
                - translation
            )
            @ rotation
            for body in bodies
        )
        assert actual_mass == pytest.approx(mass)
        np.testing.assert_allclose(moment, mass * center, rtol=0, atol=1e-12)


@pytest.mark.parametrize(("side", "suffix"), (("right", "r"), ("left", "l")))
@pytest.mark.parametrize(("gender", "height"), (("M", 175), ("F", 155)))
def test_adjacent_bone_surfaces_do_not_intersect_at_sampled_joint_poses(
    side, suffix, gender, height
):
    """Check adjacent bone intersections at sampled poses and anthropometries."""
    user = User(gender, height_cm=height)

    def mesh_for(body_name, mesh_name):
        """Find a named mesh attached to the specified body.

        Parameters
        ----------
        body_name, mesh_name : str
            Body name and mesh stem to find.

        Returns
        -------
        opensim.Mesh
            Matching attached mesh.

        Raises
        ------
        AssertionError
            If the mesh is not attached to the body.
        """
        body = user.getBodySet().get(body_name)
        for index in range(body.getPropertyByName("attached_geometry").size()):
            mesh = opensim.Mesh.safeDownCast(body.get_attached_geometry(index))
            if mesh is not None and mesh.get_mesh_file() == f"{mesh_name}_{suffix}vs.vtp":
                return mesh
        raise AssertionError(f"Missing mesh {mesh_name!r} in {body_name!r}")

    def sample_degrees(coordinate):
        """Sample the ends, midpoints and zero of a coordinate's range.

        Parameters
        ----------
        coordinate : opensim.Coordinate
            Coordinate whose range is sampled.

        Returns
        -------
        list of float
            Sampled coordinate angles in degrees.
        """
        low = math.degrees(coordinate.getRangeMin())
        high = math.degrees(coordinate.getRangeMax())
        return [low, low / 2, 0, high / 2, high]

    # Enumerate adjacent bone pairs for each finger and for the thumb.
    pairs = []
    for finger, number in zip(FINGERS, (2, 3, 4, 5)):
        pairs.extend(
            (
                (finger, "mcp", f"hand_{suffix}", f"metacarpal{number}", f"{finger}_proximal"),
                (finger, "pip", f"{finger}_proximal_{suffix}", f"{finger}_proximal", f"{finger}_medial"),
                (finger, "dip", f"{finger}_medial_{suffix}", f"{finger}_medial", f"{finger}_distal"),
            )
        )
    pairs.extend(
        (
            ("thumb", "cmc", f"hand_{suffix}", "trapezium", "metacarpal1"),
            ("thumb", "mcp", f"metacarpal1_{suffix}", "metacarpal1", "thumb_proximal"),
            ("thumb", "ip", f"thumb_proximal_{suffix}", "thumb_proximal", "thumb_distal"),
        )
    )
    # Check each pair across the sampled joint coordinate combinations.
    for finger, joint, parent_body, parent_name, child_name in pairs:
        parent = mesh_for(parent_body, parent_name)
        child = mesh_for(f"{child_name}_{suffix}", child_name)
        collision = vtk.vtkCollisionDetectionFilter()
        for index, mesh in enumerate((parent, child)):
            collision.SetInputData(index, _vtk_mesh(mesh))
        collision.SetCollisionModeToFirstContact()
        collision.SetBoxTolerance(0)
        collision.SetCellTolerance(0)

        primary = "opposition" if joint == "cmc" else "flexion"
        secondary = "flexion" if joint == "cmc" else (
            "abduction" if joint == "mcp" and finger != "thumb" else None
        )
        user.set_finger_coordinate_locked(side, finger, joint, primary, False)
        primary_coordinate = user.finger_coordinate(side, finger, joint, primary)
        if secondary is not None:
            user.set_finger_coordinate_locked(side, finger, joint, secondary, False)
            secondary_angles = sample_degrees(user.finger_coordinate(side, finger, joint, secondary))
        else:
            secondary_angles = [0]
        for angle in sample_degrees(primary_coordinate):
            user.set_finger_angle(side, finger, joint, primary, angle)
            for secondary_angle in secondary_angles:
                if secondary is not None:
                    user.set_finger_angle(side, finger, joint, secondary, secondary_angle)
                user.realizePosition(user.state)
                collision.SetMatrix(0, _vtk_matrix(user, parent))
                collision.SetMatrix(1, _vtk_matrix(user, child))
                collision.Update()
                assert collision.GetNumberOfContacts() == 0, (
                    f"{side} {finger} {joint}: intersection at "
                    f"{primary}={angle:g}, {secondary}={secondary_angle:g} degrees"
                )
        # Restore the tested coordinates to the locked reference pose.
        user.set_finger_angle(side, finger, joint, primary, 0)
        user.set_finger_coordinate_locked(side, finger, joint, primary, True)
        if secondary is not None:
            user.set_finger_angle(side, finger, joint, secondary, 0)
            user.set_finger_coordinate_locked(side, finger, joint, secondary, True)


@pytest.mark.parametrize(("side", "suffix"), (("right", "r"), ("left", "l")))
@pytest.mark.parametrize("flexed", (False, True))
def test_distinct_hand_bodies_do_not_intersect_in_representative_poses(side, suffix, flexed):
    """Check intersections between different hand bodies in two sampled poses."""
    # Optionally set a representative flexed pose before collecting meshes.
    user = User("M", percentile=50)
    if flexed:
        for finger, joint in itertools.product(FINGERS, ("mcp", "pip", "dip")):
            user.set_finger_coordinate_locked(side, finger, joint, "flexion", False)
            user.set_finger_angle(side, finger, joint, "flexion", 25)
        user.set_finger_coordinate_locked(side, "thumb", "cmc", "opposition", False)
        user.set_finger_angle(side, "thumb", "cmc", "opposition", 20)
    user.realizePosition(user.state)

    # Collect meshes and their ground transforms for the selected hand.
    meshes = []
    bodies = user.getBodySet()
    for index in range(bodies.getSize()):
        body = bodies.get(index)
        if not body.getName().endswith(f"_{suffix}"):
            continue
        for geometry_index in range(body.getPropertyByName("attached_geometry").size()):
            mesh = opensim.Mesh.safeDownCast(body.get_attached_geometry(geometry_index))
            if mesh is None or not mesh.get_mesh_file().endswith(f"_{suffix}vs.vtp"):
                continue
            meshes.append((body.getName(), mesh.get_mesh_file(), _vtk_mesh(mesh), _vtk_matrix(user, mesh)))

    # Test each pair belonging to different hand bodies for surface contact.
    assert len(meshes) == 27
    for first, second in itertools.combinations(meshes, 2):
        if first[0] == second[0]:
            continue
        collision = vtk.vtkCollisionDetectionFilter()
        collision.SetInputData(0, first[2])
        collision.SetInputData(1, second[2])
        collision.SetMatrix(0, first[3])
        collision.SetMatrix(1, second[3])
        collision.SetCollisionModeToFirstContact()
        collision.SetBoxTolerance(0)
        collision.SetCellTolerance(0)
        collision.Update()
        assert collision.GetNumberOfContacts() == 0, f"{first[1]} intersects {second[1]}"
