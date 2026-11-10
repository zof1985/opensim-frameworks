"""Exact triangle-surface collision checks between attached OpenSim meshes."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

__all__ = ["mesh_polydata", "mesh_matrix", "collision_detector"]


def mesh_polydata(mesh: Any, directories: Iterable[str | Path] = ()) -> Any:
    """Load the VTP file of an attached mesh as VTK polygonal data.

    Parameters
    ----------
    mesh : opensim.Mesh
        Attached mesh. Its file is either an existing absolute path or a
        path relative to one of `directories`.
    directories : iterable of str or pathlib.Path, optional
        Directories searched, in order, when the mesh file path is relative.

    Returns
    -------
    vtk.vtkPolyData
        Unscaled mesh in its own frame.

    Raises
    ------
    FileNotFoundError
        If the mesh file cannot be resolved.
    """
    import vtk

    filename = Path(mesh.get_mesh_file())
    if not (filename.is_absolute() and filename.is_file()):
        existing = [
            Path(directory) / filename
            for directory in directories
            if (Path(directory) / filename).is_file()
        ]
        if not existing:
            raise FileNotFoundError(f"Mesh not found: {filename}")
        filename = existing[0]
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(filename))
    reader.Update()
    return reader.GetOutput()


def mesh_matrix(mesh: Any, state: Any) -> Any:
    """Return the scaled mesh-to-Ground transform of an attached mesh.

    Parameters
    ----------
    mesh : opensim.Mesh
        Attached mesh.
    state : opensim.State
        State of the model owning the mesh, realized to Position.

    Returns
    -------
    vtk.vtkMatrix4x4
        Transform from unscaled mesh vertices to Ground.
    """
    import vtk

    frame = mesh.getFrame().getTransformInGround(state)
    scale = mesh.get_scale_factors()
    matrix = vtk.vtkMatrix4x4()
    for row in range(3):
        for column in range(3):
            matrix.SetElement(
                row, column, frame.R().get(row, column) * scale.get(column)
            )
        matrix.SetElement(row, 3, frame.p().get(row))
    return matrix


def collision_detector(first: Any, second: Any) -> Any:
    """Build a first-contact collision detector between two surfaces.

    Parameters
    ----------
    first, second : vtk.vtkPolyData
        Surfaces, as returned by :func:`mesh_polydata`.

    Returns
    -------
    vtk.vtkCollisionDetectionFilter
        Detector reporting surfaces that touch or cross. Set its two
        matrices with :func:`mesh_matrix`, call ``Update()`` and read
        ``GetNumberOfContacts()``.

    Notes
    -----
    VTK's filter is not symmetric: a contact can be found with one surface as
    the first input and missed with the other, so callers that need a
    reliable answer should test both orders.
    """
    import vtk

    detector = vtk.vtkCollisionDetectionFilter()
    detector.SetInputData(0, first)
    detector.SetInputData(1, second)
    detector.SetCollisionModeToFirstContact()
    detector.SetBoxTolerance(0)
    detector.SetCellTolerance(0)
    return detector
