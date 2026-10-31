from __future__ import annotations

import numpy as np
import pytest

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
