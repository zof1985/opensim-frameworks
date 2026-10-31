from __future__ import annotations

import numpy as np
import pytest

import opensim_frameworks as osf
from opensim_frameworks.utils.opensim import global_to_local, local_to_global
from opensim_frameworks.utils.opensim import (
    body_mass_center_array,
    get_referenced_meshes,
    import_opensim,
    read_vtp_vertices,
)
from opensim_frameworks.utils.step import (
    extract_step_body_and_mesh,
    weld_step_body_mesh_tree,
)

opensim = import_opensim()


def _body(name: str, center: tuple[float, float, float] = (0, 0, 0)):
    return opensim.Body(
        name,
        1.0,
        opensim.Vec3(*center),
        opensim.Inertia(1, 1, 1, 0, 0, 0),
    )


def _posed_body():
    model = opensim.Model()
    body = opensim.Body(
        "test_body",
        1.0,
        opensim.Vec3(0, 0, 0),
        opensim.Inertia(1, 1, 1, 0, 0, 0),
    )
    model.addBody(body)
    model.addJoint(
        opensim.WeldJoint(
            "test_weld",
            model.getGround(),
            opensim.Vec3(1, 2, 3),
            opensim.Vec3(0, 0, np.pi / 2),
            body,
            opensim.Vec3(0, 0, 0),
            opensim.Vec3(0, 0, 0),
        )
    )
    state = model.initSystem()
    model.realizePosition(state)
    return model, body, state


@pytest.mark.parametrize(
    "coordinates",
    [np.array([1.25, -2.0, 3.5]), np.array([1, 2, 3]), [1, -2, 3], (1, -2, 3)],
)
def test_vec3_numpy_conversions_round_trip_without_aliasing(coordinates):
    vector = osf.numpy_to_vec3(coordinates)
    result = osf.vec3_to_numpy(vector)

    assert isinstance(vector, opensim.Vec3)
    assert result.shape == (3,)
    assert result.dtype == np.dtype(float)
    np.testing.assert_array_equal(result, coordinates)
    result[0] = 100.0
    assert vector.get(0) == coordinates[0]
    if isinstance(coordinates, np.ndarray):
        coordinates[1] = 200
        assert vector.get(1) != coordinates[1]


def test_numpy_to_vec3_accepts_and_validates_existing_vec3():
    vector = opensim.Vec3(1, 2, 3)
    assert osf.numpy_to_vec3(vector) is vector


@pytest.mark.parametrize(
    "coordinates",
    [
        [],
        [1, 2],
        [1, 2, 3, 4],
        [[1], [2], [3]],
        1,
        [1, float("nan"), 3],
        [1, float("inf"), 3],
        [1, -float("inf"), 3],
    ],
)
def test_numpy_to_vec3_rejects_invalid_coordinates(coordinates):
    with pytest.raises(ValueError, match="three finite coordinates"):
        osf.numpy_to_vec3(coordinates)


@pytest.mark.parametrize("bad_value", [None, [1, 2, 3], np.array([1, 2, 3])])
def test_vec3_to_numpy_rejects_non_vec3_values(bad_value):
    with pytest.raises(TypeError, match="OpenSim Vec3"):
        osf.vec3_to_numpy(bad_value)


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), -float("inf")])
def test_conversions_reject_nonfinite_vec3(coordinate):
    vector = opensim.Vec3(1, coordinate, 3)
    with pytest.raises(ValueError, match="three finite coordinates"):
        osf.vec3_to_numpy(vector)
    with pytest.raises(ValueError, match="three finite coordinates"):
        osf.numpy_to_vec3(vector)


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(3),
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        np.array([[0.36, -0.48, 0.8], [0.8, 0.6, 0], [-0.48, 0.64, 0.6]]),
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
    ],
)
def test_rotation_numpy_conversions_preserve_entries_and_do_not_alias(matrix):
    expected = np.asarray(matrix, dtype=float).copy()
    rotation = osf.numpy_to_rotation(matrix)
    result = osf.rotation_to_numpy(rotation)

    assert isinstance(rotation, opensim.Rotation)
    assert result.shape == (3, 3)
    assert result.dtype == np.dtype(float)
    np.testing.assert_array_equal(result, expected)
    result[0, 0] = 100.0
    assert rotation.get(0, 0) == expected[0, 0]
    if isinstance(matrix, np.ndarray):
        matrix[0, 1] = 100.0
        assert rotation.get(0, 1) == expected[0, 1]


def test_rotation_conversion_preserves_frame_direction():
    _model, body, state = _posed_body()
    transform = body.getTransformInGround(state)
    matrix = osf.rotation_to_numpy(transform.R())
    local = np.array([1.0, 0.0, 0.0])
    expected_ground = osf.vec3_to_numpy(local_to_global(body, state, local))
    np.testing.assert_allclose(
        matrix @ local + osf.vec3_to_numpy(transform.p()),
        expected_ground,
        atol=1e-12,
    )
    converted = osf.numpy_to_rotation(matrix)
    np.testing.assert_allclose(
        osf.vec3_to_numpy(converted.multiply(osf.numpy_to_vec3(local))),
        matrix @ local,
        atol=1e-12,
    )


def test_numpy_to_rotation_preserves_accepted_entries_without_projection():
    matrix = np.eye(3)
    matrix[0, 1] = 1e-8
    result = osf.rotation_to_numpy(osf.numpy_to_rotation(matrix))
    np.testing.assert_array_equal(result, matrix)


def test_numpy_to_rotation_accepts_existing_rotation():
    rotation = opensim.Rotation()
    assert osf.numpy_to_rotation(rotation) is rotation


@pytest.mark.parametrize(
    "matrix",
    [
        [1, 2, 3],
        np.ones((3, 2)),
        np.zeros((3, 3, 1)),
        [[1, 0, 0], [0, np.nan, 0], [0, 0, 1]],
        [[1, 0, 0], [0, np.inf, 0], [0, 0, 1]],
    ],
)
def test_numpy_to_rotation_rejects_bad_shape_or_nonfinite_values(matrix):
    with pytest.raises(ValueError, match="finite 3x3 matrix"):
        osf.numpy_to_rotation(matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0.01, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 1.1e-6, 0], [0, 1, 0], [0, 0, 1]],
        np.zeros((3, 3)),
    ],
)
def test_numpy_to_rotation_rejects_non_rotations(matrix):
    with pytest.raises(ValueError, match="orthonormal with determinant"):
        osf.numpy_to_rotation(matrix)


@pytest.mark.parametrize("value", [None, np.eye(3), opensim.Mat33(1.0)])
def test_rotation_to_numpy_rejects_non_rotation_values(value):
    with pytest.raises(TypeError, match="OpenSim Rotation"):
        osf.rotation_to_numpy(value)


@pytest.mark.parametrize("entry", [float("nan"), 2.0])
def test_conversions_reject_invalid_opensim_rotation(entry):
    rotation = opensim.Rotation()
    rotation.set(0, 0, entry)
    with pytest.raises(ValueError, match="rotation"):
        osf.rotation_to_numpy(rotation)
    with pytest.raises(ValueError, match="rotation"):
        osf.numpy_to_rotation(rotation)


def test_local_to_global_applies_frame_translation_and_rotation():
    _model, body, state = _posed_body()

    global_point = local_to_global(body, state, (1, 0, 0))

    np.testing.assert_allclose(
        [global_point.get(index) for index in range(3)],
        [1, 3, 3],
        atol=1e-12,
    )


def test_global_to_local_inverts_frame_translation_and_rotation():
    _model, body, state = _posed_body()

    local_point = global_to_local(body, state, opensim.Vec3(1, 3, 3))

    np.testing.assert_allclose(
        [local_point.get(index) for index in range(3)],
        [1, 0, 0],
        atol=1e-12,
    )


def test_read_vtp_vertices_stacks_points(tmp_path):
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter

    mesh_path = tmp_path / "points.vtp"
    points = vtkPoints()
    points.InsertNextPoint(1, 2, 3)
    points.InsertNextPoint(4, 5, 6)
    polydata = vtkPolyData()
    polydata.SetPoints(points)
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(str(mesh_path))
    writer.SetInputData(polydata)
    assert writer.Write() == 1

    np.testing.assert_array_equal(
        read_vtp_vertices(mesh_path),
        [[1, 2, 3], [4, 5, 6]],
    )


def test_body_mass_center_array_returns_numpy_coordinates():
    center = body_mass_center_array(_body("centered", (1.0, 2.0, 3.0)))

    np.testing.assert_array_equal(center, [1.0, 2.0, 3.0])


def test_get_referenced_meshes_returns_attached_mesh_and_filename():
    model = opensim.Model()
    body = _body("body")
    body.attachGeometry(opensim.Mesh("mesh.vtp"))
    model.addBody(body)

    meshes = get_referenced_meshes(model)

    assert len(meshes) == 1
    assert meshes[0][1] == "mesh.vtp"


def test_weld_step_body_mesh_tree_combines_recursive_leaves():
    body_tree = {"assembly": {"part": _body("part")}}
    mesh_tree = {"assembly": {"part": ()}}

    welded_body, mesh_paths = weld_step_body_mesh_tree(body_tree, mesh_tree)

    assert welded_body.getMass() == pytest.approx(1.0)
    assert mesh_paths == ()
    assert "part" in body_tree["assembly"]


def test_extract_step_body_and_mesh_removes_target_and_empty_ancestors():
    body_tree = {"assembly": {"part": _body("part")}}
    mesh_tree = {"assembly": {"part": ()}}

    welded_body, mesh_paths = extract_step_body_and_mesh(
        body_tree,
        mesh_tree,
        "part",
    )

    assert welded_body.getMass() == pytest.approx(1.0)
    assert mesh_paths == ()
    assert body_tree == {}
    assert mesh_tree == {}


def test_extract_step_body_and_mesh_can_preserve_input_trees():
    body = _body("part")
    body_tree = {"part": body}
    mesh_tree = {"part": ()}

    welded_body, mesh_paths = extract_step_body_and_mesh(
        body_tree,
        mesh_tree,
        "part",
        remove=False,
    )

    assert welded_body.getMass() == pytest.approx(1.0)
    assert mesh_paths == ()
    assert body_tree["part"] is body
    assert mesh_tree == {"part": ()}


def test_extract_step_body_and_mesh_raises_for_missing_key():
    with pytest.raises(KeyError, match="not imported"):
        extract_step_body_and_mesh({}, {}, "missing")
