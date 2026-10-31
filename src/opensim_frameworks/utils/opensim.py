"""OpenSim binding, frame, and geometry utilities."""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

__all__ = [
    "import_opensim",
    "register_geometry_search_path",
    "resolve_geometry_file",
    "local_to_global",
    "global_to_local",
    "weld_bodies",
    "read_vtp_vertices",
    "body_mass_center_array",
    "get_referenced_meshes",
]
def _ensure_visualizer_dll_path() -> None:
    if not sys.platform.startswith("win"):
        return
    library_bin = Path(sys.prefix) / "Library" / "bin"
    if not library_bin.is_dir():
        return
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if str(library_bin) not in path_entries:
        os.environ["PATH"] = str(library_bin) + os.pathsep + os.environ.get("PATH", "")


def register_geometry_search_path(opensim: Any, directory: str | Path) -> None:
    """Register a directory with OpenSim's native mesh search paths.

    Parameters
    ----------
    opensim : module
        Imported OpenSim Python bindings exposing ``ModelVisualizer``.
    directory : str or pathlib.Path
        Directory containing geometry files. User-home markers are expanded
        and the path is resolved before registration.
    """
    opensim.ModelVisualizer.addDirToGeometrySearchPaths(
        str(Path(directory).expanduser().resolve())
    )


def resolve_geometry_file(
    opensim: Any, model: Any, filename: str, directories: Any = ()
) -> Path | None:
    """Resolve a mesh file against local and OpenSim geometry search paths.

    Parameters
    ----------
    opensim : module
        Imported OpenSim Python bindings.
    model : opensim.Model
        Model used by OpenSim's native geometry resolver.
    filename : str
        Mesh filename or path stored in a model geometry component.
    directories : iterable of str or pathlib.Path, optional
        Directories searched in order before consulting OpenSim's registered
        geometry paths.

    Returns
    -------
    pathlib.Path or None
        First matching local path, or the native resolved path. Returns
        ``None`` if the file cannot be resolved.
    """
    for directory in directories:
        candidate = Path(directory) / filename
        if candidate.is_file():
            return candidate
    try:
        resolved = opensim.ModelVisualizer.findGeometryFile(
            model, filename, False, opensim.ArrayStr()
        )
    except (RuntimeError, TypeError):
        return None
    return Path(str(resolved)) if resolved else None


def import_opensim() -> Any:
    """Import and return the OpenSim Python bindings.

    On Windows, makes the active environment's native libraries available
    before importing OpenSim. Routine OpenSim informational logs are
    suppressed while warnings and errors remain visible.

    Returns
    -------
    module
        The imported ``opensim`` module.

    Raises
    ------
    RuntimeError
        If the OpenSim Python bindings are unavailable.
    """
    _ensure_visualizer_dll_path()
    try:
        import opensim
    except ImportError as error:
        raise RuntimeError(
            "OpenSim Python bindings are required. "
            "Install a compatible OpenSim/Conda environment first."
        ) from error
    opensim.Logger.removeFileSink()
    opensim.Logger.setLevelString("Warn")
    return opensim


def _as_vec3(point: Any, opensim: Any) -> Any:
    is_vec3 = isinstance(point, opensim.Vec3)
    coordinates = np.asarray(
        [point.get(index) for index in range(3)] if is_vec3 else point,
        dtype=float,
    )
    if coordinates.shape != (3,) or not np.all(np.isfinite(coordinates)):
        raise ValueError("point must contain three finite coordinates")
    return point if is_vec3 else opensim.Vec3(*map(float, coordinates))


def local_to_global(frame: Any, state: Any, point: Any) -> Any:
    """Express a point from a frame's local coordinates in OpenSim Ground.

    Parameters
    ----------
    frame
        OpenSim physical frame in which ``point`` is expressed.
    state
        State belonging to the frame's model, realized to the required pose.
    point
        Local point as an OpenSim ``Vec3`` or a sequence of three coordinates,
        in metres.

    Returns
    -------
    opensim.Vec3
        Point coordinates expressed in Ground, in metres.
    """
    opensim = import_opensim()
    return frame.findStationLocationInGround(state, _as_vec3(point, opensim))


def global_to_local(frame: Any, state: Any, point: Any) -> Any:
    """Express a point from OpenSim Ground in a frame's local coordinates.

    Parameters
    ----------
    frame
        OpenSim physical frame in which to express the result.
    state
        State belonging to the frame's model, realized to the required pose.
    point
        Ground point as an OpenSim ``Vec3`` or a sequence of three
        coordinates, in metres.

    Returns
    -------
    opensim.Vec3
        Point coordinates expressed in ``frame``, in metres.
    """
    opensim = import_opensim()
    ground_point = _as_vec3(point, opensim)
    return frame.getTransformInGround(state).shiftBaseStationToFrame(ground_point)


def _step_name(value: str, used: set[str]) -> str:
    base = re.sub(r"\W", "_", value).strip("_") or "body"
    name, index = base, 1
    while name in used:
        index += 1
        name = f"{base}_{index}"
    used.add(name)
    return name


def weld_bodies(
    bodies: list[Any],
    meshes: list[str | Path | list[str | Path] | tuple[str | Path, ...]] | None = None,
    fuse_meshes: bool = True,
) -> Any:
    """Combine bodies and optionally fuse their corresponding mesh files.

    All bodies and mesh coordinates must use the same reference frame. When
    ``meshes`` is omitted, attached OpenSim mesh geometry is copied. When
    supplied, each entry contains the mesh path(s) corresponding to the body
    at the same index. Compatible meshes are fused by visual style unless
    ``fuse_meshes`` is ``False``, in which case each mesh is attached separately
    and the original resolved paths are returned as a tuple.

    Parameters
    ----------
    bodies
        OpenSim bodies to combine.
    meshes
        Optional list with one entry per body. An entry can be a single path
        or a list/tuple of paths.
    fuse_meshes
        Whether to fuse explicitly supplied meshes. This option has no effect
        when ``meshes`` is omitted.

    Returns
    -------
    opensim.Body or tuple[opensim.Body, Path | list[Path] | tuple[Path, ...]]
        The combined body alone when ``meshes`` is ``None``; otherwise a pair
        containing the body and either fused mesh path(s) or, when
        ``fuse_meshes`` is ``False``, the tuple of original mesh paths.

    Raises
    ------
    ValueError
        If no bodies are provided, mesh counts differ from the body count, or
        the combined mass is not positive.
    TypeError
        If bodies or mesh paths have unsupported types, or body geometry is
        not mesh geometry when ``meshes`` is omitted.
    FileNotFoundError
        If a source mesh cannot be found.
    RuntimeError
        If VTK fails to read or write a fused mesh.
    """
    opensim = import_opensim()
    if not isinstance(fuse_meshes, bool):
        raise TypeError("fuse_meshes must be a bool")
    if not isinstance(bodies, list):
        raise TypeError("bodies must be a list of opensim.Body instances")
    if not bodies:
        raise ValueError("at least one body is required")
    if any(not isinstance(body, opensim.Body) for body in bodies):
        raise TypeError("all bodies must be opensim.Body instances")

    mesh_paths_by_body: list[tuple[Path, ...]] | None = None
    if meshes is not None:
        if not isinstance(meshes, list):
            raise TypeError("meshes must be a list or None")
        if len(meshes) != len(bodies):
            raise ValueError("meshes must have the same length as bodies")
        mesh_paths_by_body = []
        for body_meshes in meshes:
            if isinstance(body_meshes, (str, Path)):
                paths = (body_meshes,)
            elif isinstance(body_meshes, (list, tuple)):
                paths = body_meshes
            else:
                raise TypeError(
                    "each meshes entry must be a path or a list/tuple of paths"
                )
            if any(not isinstance(path, (str, Path)) for path in paths):
                raise TypeError("mesh paths must be strings or pathlib.Path")
            mesh_paths_by_body.append(
                tuple(Path(path).expanduser().resolve() for path in paths)
            )

    masses = np.array([body.getMass() for body in bodies], dtype=float)
    centers = np.array(
        [[body.getMassCenter().get(index) for index in range(3)] for body in bodies],
        dtype=float,
    )
    total_mass = float(masses.sum())
    if (
        not math.isfinite(total_mass)
        or total_mass <= 0
        or not np.all(np.isfinite(masses))
        or not np.all(np.isfinite(centers))
    ):
        raise ValueError("body mass properties must be finite with positive mass")

    center = np.average(centers, axis=0, weights=masses)
    inertia = np.zeros((3, 3), dtype=float)
    for mass, body, body_center in zip(masses, bodies, centers):
        source_inertia = body.getInertia()
        moments = source_inertia.getMoments()
        products = source_inertia.getProducts()
        tensor = np.array(
            [
                [moments.get(0), products.get(0), products.get(1)],
                [products.get(0), moments.get(1), products.get(2)],
                [products.get(1), products.get(2), moments.get(2)],
            ],
            dtype=float,
        )
        offset = body_center - center
        inertia += tensor + mass * (
            np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
        )

    name = _step_name("welded_" + "_".join(body.getName() for body in bodies), set())
    result = opensim.Body(
        name,
        total_mass,
        opensim.Vec3(*map(float, center)),
        opensim.Inertia(
            float(inertia[0, 0]),
            float(inertia[1, 1]),
            float(inertia[2, 2]),
            float(inertia[0, 1]),
            float(inertia[0, 2]),
            float(inertia[1, 2]),
        ),
    )
    used_geometry_names: set[str] = set()

    def copy_mesh(source: Any) -> Any:
        mesh = opensim.Mesh(source.get_mesh_file())
        mesh.setName(
            _step_name(
                f"{source.getName()}_{source.get_mesh_file()}", used_geometry_names
            )
        )
        mesh.set_scale_factors(source.get_scale_factors())
        mesh.setColor(source.getColor())
        mesh.setOpacity(source.getOpacity())
        mesh.setRepresentation(source.getRepresentation())
        return mesh

    def mesh_style(source: Any) -> tuple[tuple[float, ...], float, int]:
        color = source.getColor()
        return (
            tuple(float(color.get(index)) for index in range(3)),
            float(source.getOpacity()),
            int(source.getRepresentation()),
        )

    def source_mesh_for_path(body: Any, mesh_path: Path) -> Any:
        candidates = []
        geometry_property = body.getPropertyByName("attached_geometry")
        for index in range(geometry_property.size()):
            candidates.append(body.get_attached_geometry(index))
        for component in body.getComponentsList():
            frame = opensim.PhysicalOffsetFrame.safeDownCast(component)
            if frame is None:
                continue
            geometry = frame.getPropertyByName("attached_geometry")
            candidates.extend(
                frame.get_attached_geometry(index) for index in range(geometry.size())
            )
        for candidate in candidates:
            source = opensim.Mesh.safeDownCast(candidate)
            if source is None:
                continue
            if Path(source.get_mesh_file()).expanduser().resolve() == mesh_path:
                return source
        return opensim.Mesh(mesh_path.as_posix())

    def fuse_mesh_sources(
        sources: list[tuple[Path, Any]],
    ) -> tuple[Any, Path]:
        import vtk

        reader_types = {
            ".obj": vtk.vtkOBJReader,
            ".ply": vtk.vtkPLYReader,
            ".stl": vtk.vtkSTLReader,
            ".vtp": vtk.vtkXMLPolyDataReader,
        }
        source_paths = [path for path, _ in sources]
        append = vtk.vtkAppendPolyData()
        filters = []
        for source_path, source in sources:
            if not source_path.is_file():
                raise FileNotFoundError(f"Mesh file was not found: {source_path}")
            reader_type = reader_types.get(source_path.suffix.lower())
            if reader_type is None:
                raise ValueError(
                    f"Unsupported mesh format for fusion: {source_path.suffix}"
                )
            reader = reader_type()
            reader.SetFileName(str(source_path))
            reader.Update()
            if reader.GetErrorCode() or reader.GetOutput().GetNumberOfPoints() == 0:
                raise RuntimeError(f"Unable to read mesh: {source_path}")

            scale = source.get_scale_factors()
            transform = vtk.vtkTransform()
            transform.Scale(*(float(scale.get(index)) for index in range(3)))
            transform_filter = vtk.vtkTransformFilter()
            transform_filter.SetTransform(transform)
            transform_filter.SetInputConnection(reader.GetOutputPort())
            transform_filter.Update()
            append.AddInputData(transform_filter.GetOutput())
            filters.append((reader, transform_filter))

        append.Update()
        style = mesh_style(sources[0][1])
        signature = tuple(
            (
                source_path.as_posix(),
                tuple(float(source.get_scale_factors().get(axis)) for axis in range(3)),
            )
            for source_path, source in sources
        )
        digest = hashlib.sha256(repr((signature, style)).encode("utf-8")).hexdigest()[
            :12
        ]
        output_path = source_paths[0].parent / f"welded_mesh_{digest}.vtp"

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(output_path))
        writer.SetInputConnection(append.GetOutputPort())
        writer.SetDataModeToAscii()
        if not writer.Write():
            raise RuntimeError(f"Unable to write fused VTP mesh: {output_path}")

        mesh = opensim.Mesh(output_path.as_posix())
        mesh.setName(_step_name(f"{name}_mesh_{digest}", used_geometry_names))
        mesh.set_scale_factors(opensim.Vec3(1.0, 1.0, 1.0))
        mesh.setColor(sources[0][1].getColor())
        mesh.setOpacity(sources[0][1].getOpacity())
        mesh.setRepresentation(sources[0][1].getRepresentation())
        return mesh, output_path

    previous_log_level = opensim.Logger.getLevelString()
    opensim.Logger.setLevelString("Off")
    output_mesh_paths: list[Path] = []
    try:
        if mesh_paths_by_body is None:
            for body in bodies:
                geometry_property = body.getPropertyByName("attached_geometry")
                for index in range(geometry_property.size()):
                    source = body.get_attached_geometry(index)
                    mesh = opensim.Mesh.safeDownCast(source)
                    if mesh is None:
                        raise TypeError(
                            "weld_bodies only supports mesh geometry; "
                            f"{source.getName()!r} is {source.getConcreteClassName()}"
                        )
                    result.attachGeometry(copy_mesh(mesh))

                for component in body.getComponentsList():
                    source_frame = opensim.PhysicalOffsetFrame.safeDownCast(component)
                    if source_frame is None:
                        continue
                    frame_geometry = source_frame.getPropertyByName("attached_geometry")
                    if not frame_geometry.size():
                        continue
                    parent = source_frame.getParentFrame()
                    if parent.getName() != body.getName():
                        raise ValueError(
                            "weld_bodies cannot merge geometry attached to "
                            "nested offset frames"
                        )
                    frame = opensim.PhysicalOffsetFrame()
                    frame.setName(
                        _step_name(
                            f"{body.getName()}_{source_frame.getName()}",
                            used_geometry_names,
                        )
                    )
                    frame.setParentFrame(result)
                    frame.set_translation(source_frame.get_translation())
                    frame.set_orientation(source_frame.get_orientation())
                    for index in range(frame_geometry.size()):
                        source = source_frame.get_attached_geometry(index)
                        mesh = opensim.Mesh.safeDownCast(source)
                        if mesh is None:
                            raise TypeError(
                                "weld_bodies only supports mesh geometry; "
                                f"{source.getName()!r} is "
                                f"{source.getConcreteClassName()}"
                            )
                        frame.attachGeometry(copy_mesh(mesh))
                    result.addComponent(frame)
        else:
            styled_meshes: dict[
                tuple[tuple[float, ...], float, int], list[tuple[Path, Any]]
            ] = {}
            for body, body_paths in zip(bodies, mesh_paths_by_body):
                for mesh_path in body_paths:
                    if not mesh_path.is_file():
                        raise FileNotFoundError(f"Mesh file was not found: {mesh_path}")
                    source = source_mesh_for_path(body, mesh_path)
                    if fuse_meshes:
                        styled_meshes.setdefault(mesh_style(source), []).append(
                            (mesh_path, source)
                        )
                    else:
                        result.attachGeometry(copy_mesh(source))
                        output_mesh_paths.append(mesh_path)
            for sources in styled_meshes.values():
                fused_mesh, output_path = fuse_mesh_sources(sources)
                result.attachGeometry(fused_mesh)
                output_mesh_paths.append(output_path)
    finally:
        opensim.Logger.setLevelString(previous_log_level)

    if mesh_paths_by_body is None:
        return result
    if not fuse_meshes:
        return result, tuple(output_mesh_paths)
    if len(output_mesh_paths) == 1:
        return result, output_mesh_paths[0]
    return result, output_mesh_paths


def read_vtp_vertices(paths: str | Path | Iterable[str | Path]) -> np.ndarray:
    """Read and stack the vertices from one or more VTP files.

    Parameters
    ----------
    paths
        A VTP path or an iterable of VTP paths.

    Returns
    -------
    numpy.ndarray
        Vertex coordinates with shape ``(n, 3)``.

    Raises
    ------
    FileNotFoundError
        If any supplied path does not exist.
    RuntimeError
        If VTK cannot read a VTP file.
    ValueError
        If no vertices are present in the supplied files.
    TypeError
        If a path is not a string or ``pathlib.Path``.
    """
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

    if isinstance(paths, (str, Path)):
        mesh_paths = (Path(paths).expanduser(),)
    else:
        try:
            path_values = tuple(paths)
        except TypeError as error:
            raise TypeError("paths must be a VTP path or an iterable of paths") from error
        if any(not isinstance(path, (str, Path)) for path in path_values):
            raise TypeError("all VTP paths must be strings or pathlib.Path instances")
        mesh_paths = tuple(Path(path).expanduser() for path in path_values)
    if not mesh_paths:
        raise ValueError("at least one VTP path is required")

    vertex_arrays = []
    for path in mesh_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        reader = vtkXMLPolyDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        if reader.GetErrorCode():
            raise RuntimeError(f"Unable to read VTP mesh: {path}")
        points = reader.GetOutput().GetPoints()
        if points is not None and points.GetNumberOfPoints():
            vertex_arrays.append(vtk_to_numpy(points.GetData()))
    if not vertex_arrays:
        raise ValueError(f"No vertices found in VTP meshes: {mesh_paths}")
    return np.vstack(vertex_arrays)


def body_mass_center_array(body: Any) -> np.ndarray:
    """Return an OpenSim body's mass-centre coordinates as a NumPy array.

    Parameters
    ----------
    body : opensim.Body
        Body whose mass centre is queried.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Mass-centre coordinates in metres.
    """
    center = body.getMassCenter()
    return np.array([center.get(index) for index in range(3)], dtype=float)


def get_referenced_meshes(model: Any) -> list[tuple[Any, str]]:
    """Return attached OpenSim meshes and their referenced file names.

    Parameters
    ----------
    model : opensim.Model
        Model whose components are searched for attached meshes.

    Returns
    -------
    list of tuple
        ``(mesh, file_name)`` pairs for every attached ``opensim.Mesh``.
    """
    opensim = import_opensim()
    meshes: list[tuple[Any, str]] = []
    for component in model.getComponentsList():
        frame = opensim.PhysicalFrame.safeDownCast(component)
        if frame is None:
            continue
        geometry_property = frame.getPropertyByName("attached_geometry")
        for geometry_index in range(geometry_property.size()):
            mesh = opensim.Mesh.safeDownCast(
                frame.get_attached_geometry(geometry_index)
            )
            if mesh is not None:
                meshes.append((mesh, mesh.get_mesh_file()))
    return meshes
