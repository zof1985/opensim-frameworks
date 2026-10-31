from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from opensim_frameworks.models.equipment import (
    read_step,
    weld_bodies,
)
from opensim_frameworks.models.equipment.equipment import opensim
from opensim_frameworks.utils.step import StepBodyImport

step_module = importlib.import_module("opensim_frameworks.utils.step")


def _write_box_step(path):
    pytest.importorskip("OCC.Core.STEPControl")
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    writer.Transfer(BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape(), STEPControl_AsIs)
    assert writer.Write(str(path)) == IFSelect_RetDone


def test_read_step_requires_density_for_unassigned_solids(tmp_path):
    asset_directory = tmp_path / "assets"
    asset_directory.mkdir()
    step_file = asset_directory / "box.step"
    _write_box_step(step_file)

    with pytest.raises(ValueError, match="no material density"):
        read_step(step_file, tmp_path / "mesh_output")

    assert not (tmp_path / "mesh_output").exists()


def test_read_step_returns_body_with_mass_properties_and_mesh(tmp_path):
    asset_directory = tmp_path / "assets"
    asset_directory.mkdir()
    step_file = asset_directory / "box.step"
    _write_box_step(step_file)
    mesh_directory = tmp_path / "custom_meshes"

    bodies, meshes = read_step(step_file, mesh_directory, density_kg_m3=1000.0)

    body = bodies["box"]
    assert body.getName() == "box"
    assert body.getMass() == pytest.approx(0.006)
    np.testing.assert_allclose(
        [body.getMassCenter().get(index) for index in range(3)],
        [0.005, 0.01, 0.015],
    )
    np.testing.assert_allclose(
        [body.getInertia().getMoments().get(index) for index in range(3)],
        [6.5e-7, 5e-7, 2.5e-7],
    )

    mesh_path = mesh_directory / "box.vtp"
    assert mesh_path.is_file()
    assert meshes == {"box": mesh_path.resolve()}
    assert body.getPropertyByName("attached_geometry").size() == 1
    mesh = opensim.Mesh.safeDownCast(body.get_attached_geometry(0))
    assert Path(mesh.get_mesh_file()).resolve() == mesh_path.resolve()


def test_read_step_returns_parallel_body_and_mesh_trees(tmp_path):
    asset_directory = tmp_path / "assets"
    asset_directory.mkdir()
    step_file = asset_directory / "box.step"
    _write_box_step(step_file)
    mesh_directory = tmp_path / "component_meshes"

    bodies, meshes = read_step(
        step_file,
        mesh_directory,
        density_kg_m3=1000.0,
    )

    assert set(bodies) == {"box"}
    assert isinstance(bodies["box"], opensim.Body)
    assert meshes == {"box": (mesh_directory / "box.vtp").resolve()}


def test_read_step_preserves_nested_assembly_paths(tmp_path, monkeypatch):
    frame = opensim.Body(
        "frame",
        1.0,
        opensim.Vec3(0, 0, 0),
        opensim.Inertia(1, 1, 1, 0, 0, 0),
    )
    pad = opensim.Body(
        "pad",
        1.0,
        opensim.Vec3(0, 0, 0),
        opensim.Inertia(1, 1, 1, 0, 0, 0),
    )
    frame_mesh = (tmp_path / "frame.vtp").resolve()
    pad_meshes = (
        (tmp_path / "pad_1.vtp").resolve(),
        (tmp_path / "pad_2.vtp").resolve(),
    )
    monkeypatch.setattr(
        step_module,
        "_read_step_imports",
        lambda *args, **kwargs: {
            "frame": StepBodyImport(frame, (frame_mesh,), ("machine", "assembly")),
            "pad": StepBodyImport(pad, pad_meshes, ("machine", "assembly")),
        },
    )

    bodies, meshes = read_step("ignored.step", tmp_path)

    assert bodies == {
        "machine": {
            "assembly": {
                "frame": frame,
                "pad": pad,
            }
        }
    }
    assert meshes == {
        "machine": {
            "assembly": {
                "frame": frame_mesh,
                "pad": pad_meshes,
            }
        }
    }


def test_weld_bodies_combines_mass_properties_and_fuses_meshes(tmp_path):
    import vtk

    mesh_path = tmp_path / "box.vtp"
    box = vtk.vtkCubeSource()
    box.SetCenter(0.005, 0.01, 0.015)
    box.SetXLength(0.01)
    box.SetYLength(0.02)
    box.SetZLength(0.03)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(mesh_path))
    writer.SetInputConnection(box.GetOutputPort())
    writer.SetDataModeToAscii()
    assert writer.Write()

    body = opensim.Body(
        "first_" + "x" * 160,
        0.006,
        opensim.Vec3(0.005, 0.01, 0.015),
        opensim.Inertia(6.5e-7, 5e-7, 2.5e-7, 0.0, 0.0, 0.0),
    )
    body.attachGeometry(opensim.Mesh(mesh_path.as_posix()))
    second_body = opensim.Body(
        "second_" + "y" * 160,
        0.004,
        opensim.Vec3(0.015, 0.01, 0.015),
        opensim.Inertia(2e-7, 3e-7, 4e-7, 0.0, 0.0, 0.0),
    )
    source_mesh = opensim.Mesh.safeDownCast(body.get_attached_geometry(0))
    second_mesh = opensim.Mesh(mesh_path.as_posix())
    second_mesh.setColor(source_mesh.getColor())
    second_mesh.setOpacity(source_mesh.getOpacity())
    second_mesh.setRepresentation(source_mesh.getRepresentation())
    second_mesh.set_scale_factors(opensim.Vec3(2.0, 1.0, 1.0))
    second_body.attachGeometry(second_mesh)
    combined, fused_mesh_path = weld_bodies(
        [body, second_body],
        [mesh_path, (mesh_path,)],
    )

    assert combined.getMass() == pytest.approx(0.01)
    np.testing.assert_allclose(
        [combined.getMassCenter().get(index) for index in range(3)],
        [0.009, 0.01, 0.015],
    )
    np.testing.assert_allclose(
        [combined.getInertia().getMoments().get(index) for index in range(3)],
        [8.5e-7, 1.04e-6, 8.9e-7],
    )
    assert combined.getPropertyByName("attached_geometry").size() == 1
    fused_mesh = opensim.Mesh.safeDownCast(combined.get_attached_geometry(0))
    assert Path(fused_mesh.get_mesh_file()).resolve() == fused_mesh_path
    assert fused_mesh_path.parent == tmp_path.resolve()
    assert fused_mesh_path.is_file()
    assert len(fused_mesh_path.name) < 255
    from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

    source_reader = vtkXMLPolyDataReader()
    source_reader.SetFileName(str(mesh_path))
    source_reader.Update()
    fused_reader = vtkXMLPolyDataReader()
    fused_reader.SetFileName(str(fused_mesh_path))
    fused_reader.Update()
    assert fused_reader.GetOutput().GetNumberOfPoints() == (
        source_reader.GetOutput().GetNumberOfPoints() * 2
    )
    assert fused_reader.GetOutput().GetBounds()[1] == pytest.approx(0.02)
    assert body.getPropertyByName("attached_geometry").size() == 1
    assert second_body.getPropertyByName("attached_geometry").size() == 1


def test_weld_bodies_keeps_meshes_with_different_colors_separate(tmp_path):
    import vtk

    mesh_path = tmp_path / "box.vtp"
    box = vtk.vtkCubeSource()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(mesh_path))
    writer.SetInputConnection(box.GetOutputPort())
    writer.SetDataModeToAscii()
    assert writer.Write()

    bodies = [
        opensim.Body(
            name,
            1.0,
            opensim.Vec3(0, 0, 0),
            opensim.Inertia(1, 1, 1, 0, 0, 0),
        )
        for name in ("first", "second")
    ]
    first_mesh = opensim.Mesh(mesh_path.as_posix())
    second_mesh = opensim.Mesh(mesh_path.as_posix())
    second_mesh.setColor(opensim.Vec3(1.0, 0.0, 0.0))
    bodies[0].attachGeometry(first_mesh)
    bodies[1].attachGeometry(second_mesh)

    body_only = weld_bodies(bodies)
    assert isinstance(body_only, opensim.Body)
    assert body_only.getPropertyByName("attached_geometry").size() == 2

    unfused, unfused_mesh_paths = weld_bodies(
        bodies,
        [mesh_path, mesh_path],
        fuse_meshes=False,
    )
    assert unfused.getPropertyByName("attached_geometry").size() == 2
    assert unfused_mesh_paths == (mesh_path.resolve(), mesh_path.resolve())

    combined, fused_mesh_paths = weld_bodies(bodies, [mesh_path, mesh_path])

    assert combined.getPropertyByName("attached_geometry").size() == 2
    assert len(fused_mesh_paths) == 2
    assert all(path.is_file() for path in fused_mesh_paths)
    colors = {
        tuple(
            opensim.Mesh.safeDownCast(combined.get_attached_geometry(index))
            .getColor()
            .get(axis)
            for axis in range(3)
        )
        for index in range(2)
    }
    assert colors == {(1.0, 1.0, 1.0), (1.0, 0.0, 0.0)}


def test_weld_bodies_requires_at_least_one_body():
    with pytest.raises(ValueError, match="at least one body"):
        weld_bodies([])

    body = opensim.Body(
        "body",
        1.0,
        opensim.Vec3(0, 0, 0),
        opensim.Inertia(1, 1, 1, 0, 0, 0),
    )
    with pytest.raises(ValueError, match="same length as bodies"):
        weld_bodies([body], [])
