from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from opensim_extensions.models.equipment import read_step
from opensim_extensions.models.equipment.equipment import opensim


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

    body, mesh_paths = read_step(
        step_file, mesh_directory, density_kg_m3=1000.0
    )

    assert body.getName() == "box"
    assert body.getMass() == pytest.approx(0.006)
    np.testing.assert_allclose(
        [body.getMassCenter().get(index) for index in range(3)],
        [0.005, 0.01, 0.015],
    )
    np.testing.assert_allclose(
        [
            body.getInertia().getMoments().get(index)
            for index in range(3)
        ],
        [6.5e-7, 5e-7, 2.5e-7],
    )

    mesh_path = mesh_directory / "box.vtp"
    assert mesh_path.is_file()
    assert mesh_paths == [mesh_path.resolve()]
    assert body.getPropertyByName("attached_geometry").size() == 1
    mesh = opensim.Mesh.safeDownCast(body.get_attached_geometry(0))
    assert Path(mesh.get_mesh_file()).resolve() == mesh_path.resolve()


def test_read_step_can_return_nested_assembly_hierarchy(tmp_path):
    asset_directory = tmp_path / "assets"
    asset_directory.mkdir()
    step_file = asset_directory / "box.step"
    _write_box_step(step_file)

    hierarchy = read_step(
        step_file,
        tmp_path / "hierarchical_meshes",
        density_kg_m3=1000.0,
        return_hierarchy=True,
    )

    assert set(hierarchy) == {"box"}
    assert set(hierarchy["box"]) == {"box"}
    body = hierarchy["box"]["box"]
    assert body.getName() == "box"
    assert body.getMass() == pytest.approx(0.006)
