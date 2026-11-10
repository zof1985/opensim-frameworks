"""Exact mesh collision helpers."""

from __future__ import annotations

import numpy as np
import pytest

import opensim_frameworks as osf
from opensim_frameworks.utils.collision import (
    collision_detector,
    mesh_matrix,
    mesh_polydata,
)
from opensim_frameworks.utils.constants import MESHES_DIR

opensim = pytest.importorskip("opensim")
pytest.importorskip("vtk")


def _sphere_model(centers):
    """Model with one unit-diameter sphere body per center, welded to Ground."""
    level = opensim.Logger.getLevelString()
    # OpenSim registra un errore innocuo mentre la mesh non e' ancora in un modello
    opensim.Logger.setLevelString("Off")
    try:
        model = opensim.Model()
        meshes = []
        for index, center in enumerate(centers):
            body = opensim.Body(
                f"sphere{index}", 1.0, opensim.Vec3(0), opensim.Inertia(1, 1, 1, 0, 0, 0)
            )
            mesh = opensim.Mesh(str(MESHES_DIR / "unit_sphere.vtp"))
            body.attachGeometry(mesh)
            model.addBody(body)
            model.addJoint(
                opensim.WeldJoint(
                    f"weld{index}",
                    model.getGround(),
                    opensim.Vec3(*center),
                    opensim.Vec3(0),
                    body,
                    opensim.Vec3(0),
                    opensim.Vec3(0),
                )
            )
            meshes.append(opensim.Mesh.safeDownCast(body.get_attached_geometry(0)))
    finally:
        opensim.Logger.setLevelString(level)
    state = model.initSystem()
    model.realizePosition(state)
    return model, state, meshes


def _contacts(model, state, first, second):
    detector = collision_detector(mesh_polydata(first), mesh_polydata(second))
    detector.SetMatrix(0, mesh_matrix(first, state))
    detector.SetMatrix(1, mesh_matrix(second, state))
    detector.Update()
    return detector.GetNumberOfContacts()


def test_mesh_matrix_places_the_mesh_in_ground():
    model, state, (mesh,) = _sphere_model([(1.0, 2.0, 3.0)])
    matrix = mesh_matrix(mesh, state)
    translation = [matrix.GetElement(row, 3) for row in range(3)]
    assert translation == pytest.approx([1.0, 2.0, 3.0])


def test_overlapping_meshes_collide_and_separated_ones_do_not():
    model, state, (first, second, far) = _sphere_model(
        [(0.0, 0.0, 0.0), (0.4, 0.0, 0.0), (3.0, 0.0, 0.0)]
    )
    assert _contacts(model, state, first, second) > 0
    assert _contacts(model, state, first, far) == 0


def test_mesh_polydata_resolves_relative_names_in_the_given_directories():
    level = opensim.Logger.getLevelString()
    opensim.Logger.setLevelString("Off")
    try:
        mesh = opensim.Mesh("unit_sphere.vtp")
    finally:
        opensim.Logger.setLevelString(level)
    polydata = mesh_polydata(mesh, [MESHES_DIR])
    assert polydata.GetNumberOfPoints() > 0
    with pytest.raises(FileNotFoundError):
        mesh_polydata(mesh, [])


def test_helpers_are_part_of_the_top_level_api():
    assert osf.mesh_polydata is mesh_polydata
    assert osf.mesh_matrix is mesh_matrix
    assert osf.collision_detector is collision_detector
