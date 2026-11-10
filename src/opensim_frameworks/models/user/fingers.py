"""Finger coordinates and exact hand-surface intersection checks of :class:`User`.

Everything here is exposed on :class:`User` through :class:`_FingerMixin`:
reading and writing individual finger coordinates
(``finger_coordinate``/``set_finger_angle``/...) and exact triangle-surface
intersection checks (``hand_surface_intersections`` and
``hand_handle_intersections``). The anatomical construction of the
articulated hand bodies (joints, axes, rest pose) lives in :mod:`.hand`; the
procedure that closes the fingers on a handle lives in :mod:`.grasp`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .hand import MOVING_HAND_SEGMENTS

__all__ = []

# -- Shared mesh-surface loading, used by hand_surface_intersections and
# -- hand_handle_intersections.


def _resolve_body_mesh_path(
    filename: Path, directories: Iterable[Path], label: str
) -> Path:
    """Resolve a mesh filename as an absolute path or within given directories.

    Parameters
    ----------
    filename : pathlib.Path
        Mesh path as stored on an OpenSim geometry component.
    directories : iterable of pathlib.Path
        Directories searched, in order, when `filename` is relative. An empty
        iterable means only a literal absolute path (or one resolvable
        relative to the current working directory) is accepted.
    label : str
        Capitalized noun describing the mesh owner, used in error messages
        (for example ``"Hand"``).

    Returns
    -------
    pathlib.Path
        Existing mesh path.

    Raises
    ------
    FileNotFoundError
        If `filename` cannot be resolved to an existing file. This never
        falls back to OpenSim's native geometry search path, which silently
        fails to resolve some valid absolute paths under the OpenSim 4.6
        Python bindings.
    """
    if not filename.is_absolute():
        source = next(
            (
                directory / filename
                for directory in directories
                if (directory / filename).is_file()
            ),
            None,
        )
        if source is None:
            raise FileNotFoundError(f"{label} mesh not found: {filename}")
        filename = source
    if not filename.is_file():
        raise FileNotFoundError(f"{label} mesh not found: {filename}")
    return filename


def _load_mesh_surface(
    mesh: Any, state: Any, directories: Iterable[Path], label: str
) -> tuple[Any, Any]:
    """Load one attached OpenSim mesh as VTK polydata with its ground transform.

    Parameters
    ----------
    mesh : opensim.Mesh
        Attached mesh geometry.
    state : opensim.State
        State realized through Position, used for the mesh frame transform.
    directories : iterable of pathlib.Path
        Directories searched when the mesh file path is relative.
    label : str
        Capitalized noun describing the mesh owner, used in error messages.

    Returns
    -------
    tuple of (vtk.vtkPolyData, vtk.vtkMatrix4x4)
        Loaded triangular surface and its scaled mesh-to-ground transform.

    Raises
    ------
    FileNotFoundError
        If the mesh file cannot be resolved.
    ValueError
        If the mesh is not a triangulated VTP surface with finite geometry
        and a valid scale.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    filename = _resolve_body_mesh_path(Path(mesh.get_mesh_file()), directories, label)
    if filename.suffix.lower() != ".vtp":
        raise ValueError(f"{label} mesh must be a VTP file: {filename}")
    scale = np.array([mesh.get_scale_factors().get(i) for i in range(3)])
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError(f"Invalid scale factors for {label.lower()} mesh: {filename}")
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(filename))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata.GetNumberOfPoints() == 0 or polydata.GetNumberOfPolys() == 0:
        raise ValueError(f"{label} mesh has no polygonal surface: {filename}")
    if not np.all(np.isfinite(vtk_to_numpy(polydata.GetPoints().GetData()))):
        raise ValueError(f"{label} mesh contains nonfinite vertices: {filename}")
    offsets = vtk_to_numpy(polydata.GetPolys().GetOffsetsArray())
    if np.any(np.diff(offsets) != 3):
        raise ValueError(f"{label} mesh contains non-triangular polygons: {filename}")
    transform = mesh.getFrame().getTransformInGround(state)
    matrix = vtk.vtkMatrix4x4()
    for row in range(3):
        for column in range(3):
            matrix.SetElement(
                row, column, float(transform.R().get(row, column) * scale[column])
            )
        matrix.SetElement(row, 3, float(transform.p().get(row)))
    return polydata, matrix


def _surfaces_intersecting_pairs(
    first_surfaces: list[tuple[str, str, Any, Any]],
    second_surfaces: list[tuple[str, str, Any, Any]] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return mesh filename pairs whose VTK surfaces touch or cross.

    Parameters
    ----------
    first_surfaces : list of (body name, mesh filename, vtkPolyData, vtkMatrix4x4)
        Surfaces to compare.
    second_surfaces : list of the same shape, or None, default=None
        When given, every surface in `first_surfaces` is compared against
        every surface in `second_surfaces` (a cross-model check). When
        omitted, every pair within `first_surfaces` belonging to different
        bodies is compared instead (a self-check).

    Returns
    -------
    tuple of tuple of str
        Pairs of mesh filenames whose surfaces touch or cross, in the order
        checked.
    """
    import vtk

    if second_surfaces is None:
        pairs = (
            (first, second)
            for index, first in enumerate(first_surfaces)
            for second in first_surfaces[index + 1 :]
            if first[0] != second[0]
        )
    else:
        pairs = ((first, second) for first in first_surfaces for second in second_surfaces)

    intersections = []
    for first, second in pairs:
        collision = vtk.vtkCollisionDetectionFilter()
        collision.SetInputData(0, first[2])
        collision.SetInputData(1, second[2])
        collision.SetMatrix(0, first[3])
        collision.SetMatrix(1, second[3])
        collision.SetCollisionModeToFirstContact()
        collision.SetBoxTolerance(0)
        collision.SetCellTolerance(0)
        collision.Update()
        if collision.GetNumberOfContacts():
            intersections.append((first[1], second[1]))
    return tuple(intersections)


class _FingerMixin:
    """Finger coordinates and exact hand-surface intersection checks."""

    def finger_coordinate(
        self, side: str, finger: str, joint: str, motion: str
    ) -> Any:
        """Return a finger coordinate for explicit state and lock control.

        Parameters
        ----------
        side : {'left', 'right'}
            Anatomical side.
        finger : {'thumb', 'index', 'middle', 'ring', 'little'}
            Digit containing the joint.
        joint : {'cmc', 'mcp', 'pip', 'dip', 'ip'}
            Joint name. Available joints depend on the digit.
        motion : {'flexion', 'abduction', 'opposition'}
            Coordinate name. Available motions depend on the joint.

        Returns
        -------
        opensim.Coordinate
            The coordinate belonging to this user.

        Raises
        ------
        ValueError
            If the side or the requested coordinate is not available.
        """
        # Resolve the requested anatomical side and validate the coordinate name.
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        name = f"{finger}_{joint}_{motion}_{'l' if side == 'left' else 'r'}"
        coordinates = self.getCoordinateSet()
        if not coordinates.contains(name):
            raise ValueError(f"Unknown finger coordinate {name!r}")
        return coordinates.get(name)

    def set_finger_coordinate_locked(
        self, side: str, finger: str, joint: str, motion: str, locked: bool
    ) -> None:
        """Set a finger coordinate's current and default lock state.

        Parameters
        ----------
        side, finger, joint, motion : str
            Identify a coordinate as described by :meth:`finger_coordinate`.
        locked : bool
            Whether the coordinate should be locked.

        Raises
        ------
        ValueError
            If the coordinate is unavailable or ``locked`` is not a boolean.
        """
        # Validate the lock request and apply it to both the model and state.
        if not isinstance(locked, bool):
            raise ValueError("locked must be a bool")
        coordinate = self.finger_coordinate(side, finger, joint, motion)
        coordinate.set_locked(locked)
        coordinate.setLocked(self.state, locked)

    def set_finger_angle(
        self, side: str, finger: str, joint: str, motion: str, degrees: float
    ) -> None:
        """Set an unlocked finger coordinate relative to the corrected reference pose.

        Parameters
        ----------
        side, finger, joint, motion : str
            Identify a coordinate as described by :meth:`finger_coordinate`.
        degrees : float
            Coordinate value in degrees relative to the corrected reference pose.

        Raises
        ------
        ValueError
            If the coordinate is locked or the value is nonfinite or outside
            the approximate coordinate range.
        """
        # Resolve the coordinate and validate its lock state and requested angle.
        coordinate = self.finger_coordinate(side, finger, joint, motion)
        if not math.isfinite(degrees):
            raise ValueError("finger angle must be finite")
        if coordinate.getLocked(self.state):
            raise ValueError(f"Finger coordinate {coordinate.getName()!r} is locked")
        lower = coordinate.getRangeMin()
        upper = coordinate.getRangeMax()
        lower_degrees = math.degrees(lower)
        upper_degrees = math.degrees(upper)
        if not lower_degrees <= degrees <= upper_degrees:
            raise ValueError(f"Finger angle is outside the range of {coordinate.getName()!r}")
        # Preserve exact bounds when converting endpoint values back to radians.
        if degrees == lower_degrees:
            angle = lower
        elif degrees == upper_degrees:
            angle = upper
        else:
            angle = math.radians(degrees)
        coordinate.setValue(self.state, angle, False)

    def finger_angle(self, side: str, finger: str, joint: str, motion: str) -> float:
        """Return a finger coordinate in degrees relative to the reference pose.

        Parameters
        ----------
        side, finger, joint, motion : str
            Identify a coordinate as described by :meth:`finger_coordinate`.

        Returns
        -------
        float
            Coordinate value in degrees relative to the corrected reference pose.

        Raises
        ------
        ValueError
            If the requested coordinate is unavailable.
        """
        coordinate = self.finger_coordinate(side, finger, joint, motion)
        return math.degrees(coordinate.getValue(self.state))

    def hand_surface_intersections(
        self, side: str
    ) -> tuple[tuple[str, str], ...]:
        """Find intersections between bone surfaces on different hand bodies.

        Parameters
        ----------
        side : {'left', 'right'}
            Hand to inspect in the current user state.

        Returns
        -------
        tuple of tuple of str
            Pairs of intersecting mesh filenames. A pair is also reported
            when its surfaces touch.

        Raises
        ------
        ValueError
            If the side, mesh, or mesh scale is invalid.
        FileNotFoundError
            If a referenced bone mesh cannot be found in the registered
            geometry directories.

        Notes
        -----
        This checks triangle-surface intersections only. It does not detect
        complete containment without surface intersection, distinguish
        tangency from penetration, or model skin and soft tissue. The meshes
        belonging to the same rigid body are not compared. It neither changes
        the pose nor enforces a collision-free trajectory.
        """
        # Resolve the selected side and realize the current body transforms.
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        suffix = "l" if side == "left" else "r"
        self.realizePosition(self.state)
        surfaces = []

        # Load the palm and finger meshes and build their transforms to ground.
        for name in ("hand", *MOVING_HAND_SEGMENTS):
            body = self.getBodySet().get(f"{name}_{suffix}")
            count = body.getPropertyByName("attached_geometry").size()
            if count == 0:
                raise ValueError(f"Body {body.getName()!r} has no attached mesh")
            for index in range(count):
                mesh = self.opensim.Mesh.safeDownCast(body.get_attached_geometry(index))
                if mesh is None:
                    raise ValueError(f"Non-mesh geometry on body {body.getName()!r}")
                polydata, matrix = _load_mesh_surface(
                    mesh, self.state, self.geometry_directories, "Hand"
                )
                surfaces.append((body.getName(), mesh.get_mesh_file(), polydata, matrix))

        # Compare different bodies and report the mesh pairs that touch or cross.
        return _surfaces_intersecting_pairs(surfaces)

    def hand_handle_intersections(
        self, side: str, handle: Any
    ) -> tuple[tuple[str, str], ...]:
        """Find exact triangle-surface intersections between a hand and a grip body.

        Parameters
        ----------
        side : {'left', 'right'}
            Hand to inspect in the current user state.
        handle : opensim.Body
            Grip body belonging to a separately initialized OpenSim model
            (for example a piece of equipment). Its pose is read from that
            model's own working state (``handle.getModel().getWorkingState()``),
            not from this user's state, so `handle` may come from a Python
            wrapper distinct from whichever object last moved its model.

        Returns
        -------
        tuple of tuple of str
            Pairs of ``(hand mesh filename, handle mesh filename)`` whose
            surfaces touch or cross.

        Raises
        ------
        ValueError
            If `side` is invalid, `handle` is not connected to an
            initialized model, or a mesh/scale is invalid.
        TypeError
            If `handle` is not an ``opensim.Body``.
        FileNotFoundError
            If a referenced mesh cannot be resolved. Hand meshes are searched
            in :attr:`geometry_directories`; the handle's mesh file must be
            an absolute path or resolvable relative to the current working
            directory -- this does not search directories registered for the
            handle's own model.

        Notes
        -----
        Same limitations as :meth:`hand_surface_intersections`: triangle
        surface intersections only, no skin/soft-tissue model, no check of
        intermediate poses along a trajectory.
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        suffix = "l" if side == "left" else "r"
        if not isinstance(handle, self.opensim.Body):
            raise TypeError("handle must be an opensim.Body")
        if not handle.hasModel():
            raise ValueError("handle is not connected to an initialized OpenSim model")
        handle_model = handle.getModel()
        handle_state = handle_model.getWorkingState()
        handle_model.realizePosition(handle_state)

        self.realizePosition(self.state)
        hand_surfaces = []
        for name in ("hand", *MOVING_HAND_SEGMENTS):
            body = self.getBodySet().get(f"{name}_{suffix}")
            count = body.getPropertyByName("attached_geometry").size()
            if count == 0:
                raise ValueError(f"Body {body.getName()!r} has no attached mesh")
            for index in range(count):
                mesh = self.opensim.Mesh.safeDownCast(body.get_attached_geometry(index))
                if mesh is None:
                    raise ValueError(f"Non-mesh geometry on body {body.getName()!r}")
                polydata, matrix = _load_mesh_surface(
                    mesh, self.state, self.geometry_directories, "Hand"
                )
                hand_surfaces.append((body.getName(), mesh.get_mesh_file(), polydata, matrix))

        handle_count = handle.getPropertyByName("attached_geometry").size()
        if handle_count == 0:
            raise ValueError(f"Body {handle.getName()!r} has no attached mesh")
        handle_surfaces = []
        for index in range(handle_count):
            mesh = self.opensim.Mesh.safeDownCast(handle.get_attached_geometry(index))
            if mesh is None:
                raise ValueError(f"Non-mesh geometry on body {handle.getName()!r}")
            polydata, matrix = _load_mesh_surface(mesh, handle_state, (), "Handle")
            handle_surfaces.append((handle.getName(), mesh.get_mesh_file(), polydata, matrix))

        return _surfaces_intersecting_pairs(hand_surfaces, handle_surfaces)
