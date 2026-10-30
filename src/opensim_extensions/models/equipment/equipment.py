"""Equipment models, screen geometry, and STEP import helpers."""

from __future__ import annotations

import copy
import hashlib
import math
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping, Self

import numpy as np

from ...utils.constants import MM_TO_M
from ...utils.functions import import_opensim

opensim = import_opensim()

__all__ = [
    "Equipment",
    "Screen",
    "StepBodyImport",
    "read_step",
    "read_step_components",
    "weld_bodies",
]

_STEP_DEFAULT_COLOR = (0.7, 0.7, 0.7)


@dataclass(frozen=True)
class Screen:
    """Screen dimensions, mounting offsets, and world-frame pose.

    The screen is modelled in the package world convention: ``x`` is forward,
    ``y`` is up, and ``z`` is lateral. The lower-centre mounting point is
    calculated from ``position_anchor_m`` and the two offsets.

    Parameters
    ----------
    width_mm : float
        Screen width across the lateral axis, in millimetres.
    height_mm : float
        Screen height along the panel, in millimetres.
    inclination_deg : float
        Directed angle of the panel's upward direction from ``+x`` toward
        ``+y``, in degrees.
    position_anchor_m : sequence of float
        World-frame reference point for the screen offsets, in metres.
    horizontal_offset_mm : float
        Forward offset of the lower-centre mounting point from the reference,
        in millimetres.
    vertical_offset_mm : float
        Upward offset of the lower-centre mounting point from the reference,
        in millimetres.
    horizontal_offset_min_mm, vertical_offset_min_mm : float or None
        Inclusive lower limits on the corresponding offsets. ``None`` means
        the screen implementation does not impose a separate lower limit.
    """

    width_mm: float
    height_mm: float
    inclination_deg: float
    position_anchor_m: tuple[float, float, float]
    horizontal_offset_mm: float
    vertical_offset_mm: float
    horizontal_offset_min_mm: float | None = None
    vertical_offset_min_mm: float | None = None

    def __post_init__(self) -> None:
        for name in ("width_mm", "height_mm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite value")
            object.__setattr__(self, name, value)
        for name in (
            "inclination_deg",
            "horizontal_offset_mm",
            "vertical_offset_mm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

        anchor = np.asarray(self.position_anchor_m, dtype=float)
        if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
            raise ValueError("position_anchor_m must contain three finite values")
        object.__setattr__(self, "position_anchor_m", tuple(map(float, anchor)))
        for name in ("horizontal_offset_min_mm", "vertical_offset_min_mm"):
            minimum = getattr(self, name)
            if minimum is None:
                continue
            minimum = float(minimum)
            if not math.isfinite(minimum):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, minimum)

    @property
    def base_position_m(self) -> np.ndarray:
        """Return the lower-centre mounting point in world coordinates.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Lower-centre screen position in metres.
        """
        return np.asarray(self.position_anchor_m) + MM_TO_M * np.array(
            [self.horizontal_offset_mm, self.vertical_offset_mm, 0.0]
        )

    @property
    def center_position_m(self) -> np.ndarray:
        """Return the geometric centre of the panel in world coordinates.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Screen-centre position in metres.
        """
        return self.base_position_m + (self.height_mm * MM_TO_M / 2) * self._up_direction(
            self.inclination_deg
        )

    @staticmethod
    def _up_direction(inclination_deg: float) -> np.ndarray:
        angle = math.radians(inclination_deg)
        return np.array([math.cos(angle), math.sin(angle), 0.0])

    def corners(self, position: tuple[float, float, float] | None = None) -> np.ndarray:
        """Return the four panel corners for the current or candidate pose.

        Parameters
        ----------
        position : tuple of float or None, optional
            Candidate ``(horizontal_offset_mm, vertical_offset_mm,
            inclination_deg)``. ``None`` uses the configured pose.

        Returns
        -------
        numpy.ndarray, shape (4, 3)
            Panel corners in metres in the world frame.
        """
        horizontal, vertical, inclination = (
            (
                self.horizontal_offset_mm,
                self.vertical_offset_mm,
                self.inclination_deg,
            )
            if position is None
            else position
        )
        base = np.asarray(self.position_anchor_m) + MM_TO_M * np.array(
            [horizontal, vertical, 0.0]
        )
        up = self._up_direction(inclination)
        half_width = self.width_mm * MM_TO_M / 2
        height = self.height_mm * MM_TO_M
        return np.asarray(
            [
                base + side * np.array([0.0, 0.0, 1.0]) + rise * up
                for rise in (0.0, height)
                for side in (-half_width, half_width)
            ]
        )

    def samples(
        self,
        grid_size: int,
        position: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        """Sample a uniform rectangular grid over the current or candidate pose.

        Parameters
        ----------
        grid_size : int
            Number of evenly spaced cell centres along each panel dimension.
        position : tuple of float or None, optional
            Candidate ``(horizontal_offset_mm, vertical_offset_mm,
            inclination_deg)``. ``None`` uses the configured pose.

        Returns
        -------
        numpy.ndarray, shape (grid_size**2, 3)
            Sample points in metres in the world frame.
        """
        if (
            not isinstance(grid_size, int)
            or isinstance(grid_size, bool)
            or grid_size < 1
        ):
            raise ValueError("grid_size must be a positive integer")
        horizontal, vertical, inclination = (
            (
                self.horizontal_offset_mm,
                self.vertical_offset_mm,
                self.inclination_deg,
            )
            if position is None
            else position
        )
        base = np.asarray(self.position_anchor_m) + MM_TO_M * np.array(
            [horizontal, vertical, 0.0]
        )
        up = self._up_direction(inclination)
        values = (np.arange(grid_size) + 0.5) / grid_size
        return np.asarray(
            [
                base
                + self.height_mm * MM_TO_M * rise * up
                + self.width_mm * MM_TO_M * (2 * side - 1) / 2 * np.array([0.0, 0.0, 1.0])
                for rise in values
                for side in values
            ]
        )


class Equipment(opensim.Model, ABC):
    """Base class for user-facing OpenSim equipment.

    Concrete equipment provides its own user-fitting procedure. If it has a
    screen, :attr:`screen` returns the complete screen dimensions and pose;
    otherwise it returns ``None``. The default initializer loads a serialized
    ``.osim`` model from the subclass's ``_model_path`` asset directory,
    calling ``_build`` first when that file is missing. Subclasses that build
    parameter-specific models in memory can provide their own initializer.
    """

    _model_path: ClassVar[Path]
    _model_filename: ClassVar[str] = "model.osim"
    _rebuild_model_on_init: ClassVar[bool] = False

    def __init__(self) -> None:
        self.opensim = opensim
        model_directory = getattr(type(self), "_model_path", None)
        if not isinstance(model_directory, Path):
            raise NotImplementedError(
                f"{type(self).__name__} must define its _model_path"
            )
        model_path = model_directory / self._model_filename
        if not model_path.is_file() or self._rebuild_model_on_init:
            model_directory.mkdir(parents=True, exist_ok=True)
            self._build()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{type(self).__name__}._build did not create {model_path}"
            )
        opensim.Model.__init__(self, str(model_path))
        self.finalizeConnections()
        self.state = self.initSystem()
        self.realizePosition(self.state)

    @property
    def screen(self) -> Screen | None:
        """Return this equipment's screen description, if present.

        Returns
        -------
        Screen or None
            Current screen dimensions and world-frame pose, or ``None`` when
            this equipment has no screen.
        """
        return None

    def has_screen(self) -> bool:
        """Report whether this equipment provides a screen.

        Returns
        -------
        bool
            ``True`` when :attr:`screen` contains a screen description.
        """
        return self.screen is not None

    def set_screen_position(
        self,
        horizontal_offset_mm: float,
        vertical_offset_mm: float,
        inclination_deg: float,
    ) -> None:
        """Apply a candidate pose to the equipment screen.

        Screen-bearing equipment must override this method so optimizers can
        apply the selected screen pose to its native model.

        Parameters
        ----------
        horizontal_offset_mm, vertical_offset_mm : float
            Screen mounting offsets from its equipment-specific reference, in
            millimetres.
        inclination_deg : float
            Directed panel inclination in degrees.

        Raises
        ------
        NotImplementedError
            If the equipment has a screen but does not support repositioning.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement screen repositioning"
        )

    def expand_screen_offset_limits(
        self,
        horizontal_min_mm: float | None = None,
        vertical_min_mm: float | None = None,
    ) -> None:
        """Expand equipment-specific lower bounds for screen offsets.

        Screen-bearing equipment with configurable offset limits should
        override this method. The default is suitable for equipment whose
        screen offsets do not have separate lower limits.
        """
        if self.screen is not None and (
            (
                horizontal_min_mm is not None
                and self.screen.horizontal_offset_min_mm is not None
            )
            or (
                vertical_min_mm is not None
                and self.screen.vertical_offset_min_mm is not None
            )
        ):
            raise NotImplementedError(
                f"{type(self).__name__} does not support expanding screen offset limits"
            )

    def screen_position_constraints(
        self, position: tuple[float, float, float]
    ) -> np.ndarray:
        """Return feasibility margins for a candidate screen pose.

        Positive values indicate satisfied constraints. Equipment without
        special placement constraints inherits an always-feasible margin.

        Parameters
        ----------
        position : tuple of float
            Candidate ``(horizontal_offset_mm, vertical_offset_mm,
            inclination_deg)``.

        Returns
        -------
        numpy.ndarray
            Non-negative feasibility margins, or negative values for violated
            constraints.
        """
        return np.ones(1)

    def clone(self) -> Self:
        """Create an independent native and Python-level copy of this equipment.

        Concrete subclasses with non-copyable Python state may override this
        method. The copied OpenSim model is reinitialized so its working state
        belongs to the clone.

        Returns
        -------
        Equipment
            Independent copy of this equipment instance.
        """
        result = type(self).__new__(type(self))
        opensim.Model.__init__(result, self)
        for name, value in vars(self).items():
            if name == "opensim":
                setattr(result, name, value)
            elif name == "_visualizer":
                setattr(result, name, None)
            elif name == "state":
                continue
            else:
                setattr(result, name, copy.deepcopy(value))
        result.state = result.initSystem()
        result.realizePosition(result.state)
        return result

    @abstractmethod
    def _build(self) -> None:
        """Construct this equipment's OpenSim model.

        Subclasses using the default initializer must also save the model and
        any display meshes under ``_model_path``.
        """
        raise NotImplementedError

    @abstractmethod
    def fit_user(self, *args: Any, **kwargs: Any) -> Any:
        """Fit a user to this equipment using a subclass-specific procedure."""
        raise NotImplementedError


def _step_name(value: str, used: set[str]) -> str:
    base = re.sub(r"\W", "_", value).strip("_") or "body"
    name, index = base, 1
    while name in used:
        index += 1
        name = f"{base}_{index}"
    used.add(name)
    return name


def weld_bodies(
    bodies: list[opensim.Body],
    meshes: list[str | Path | list[str | Path] | tuple[str | Path, ...]]
    | None = None,
) -> opensim.Body | tuple[opensim.Body, Path | list[Path]]:
    """Combine bodies and optionally fuse their corresponding mesh files.

    All bodies and mesh coordinates must use the same reference frame. When
    ``meshes`` is omitted, attached OpenSim mesh geometry is copied. When
    supplied, each entry contains the mesh path(s) corresponding to the body
    at the same index; compatible meshes are fused by visual style.

    Parameters
    ----------
    bodies
        OpenSim bodies to combine.
    meshes
        Optional list with one entry per body. An entry can be a single path
        or a list/tuple of paths.

    Returns
    -------
    opensim.Body or tuple[opensim.Body, Path | list[Path]]
        The combined body alone when ``meshes`` is ``None``; otherwise a pair
        containing the body and its fused mesh path(s).

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
        [
            [body.getMassCenter().get(index) for index in range(3)]
            for body in bodies
        ],
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

    name = _step_name(
        "welded_" + "_".join(body.getName() for body in bodies), set()
    )
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
                frame.get_attached_geometry(index)
                for index in range(geometry.size())
            )
        for candidate in candidates:
            source = opensim.Mesh.safeDownCast(candidate)
            if source is None:
                continue
            if Path(source.get_mesh_file()).expanduser().resolve() == mesh_path:
                return source
        return opensim.Mesh(mesh_path.as_posix())

    def fuse_meshes(
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
            transform.Scale(
                *(float(scale.get(index)) for index in range(3))
            )
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
                tuple(
                    float(source.get_scale_factors().get(axis))
                    for axis in range(3)
                ),
            )
            for source_path, source in sources
        )
        digest = hashlib.sha256(
            repr((signature, style)).encode("utf-8")
        ).hexdigest()[:12]
        output_path = source_paths[0].parent / f"{name}_mesh_{digest}.vtp"

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(output_path))
        writer.SetInputConnection(append.GetOutputPort())
        writer.SetDataModeToAscii()
        if not writer.Write():
            raise RuntimeError(f"Unable to write fused VTP mesh: {output_path}")

        mesh = opensim.Mesh(output_path.as_posix())
        mesh.setName(
            _step_name(f"{name}_mesh_{digest}", used_geometry_names)
        )
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
                    source_frame = opensim.PhysicalOffsetFrame.safeDownCast(
                        component
                    )
                    if source_frame is None:
                        continue
                    frame_geometry = source_frame.getPropertyByName(
                        "attached_geometry"
                    )
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
                        raise FileNotFoundError(
                            f"Mesh file was not found: {mesh_path}"
                        )
                    source = source_mesh_for_path(body, mesh_path)
                    styled_meshes.setdefault(mesh_style(source), []).append(
                        (mesh_path, source)
                    )
            for sources in styled_meshes.values():
                fused_mesh, output_path = fuse_meshes(sources)
                result.attachGeometry(fused_mesh)
                output_mesh_paths.append(output_path)
    finally:
        opensim.Logger.setLevelString(previous_log_level)

    if mesh_paths_by_body is None:
        return result
    if len(output_mesh_paths) == 1:
        return result, output_mesh_paths[0]
    return result, output_mesh_paths


def _step_triangulate(shape, linear_mm: float, angular_rad: float):
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods

    BRepMesh_IncrementalMesh(shape, linear_mm, False, angular_rad, True)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is not None:
            transformation = location.Transformation()
            offset = len(vertices)
            for index in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(index).Transformed(transformation)
                vertices.append([point.X(), point.Y(), point.Z()])
            reverse = face.Orientation() == TopAbs_REVERSED
            for index in range(1, triangulation.NbTriangles() + 1):
                a, b, c = triangulation.Triangle(index).Get()
                if reverse:
                    b, c = c, b
                faces.append([offset + a - 1, offset + b - 1, offset + c - 1])
        explorer.Next()
    return np.asarray(vertices, dtype=float).reshape(-1, 3), np.asarray(
        faces, dtype=np.int64
    ).reshape(-1, 3)


def _step_write_vtp(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    import vtk
    from vtk.util import numpy_support

    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(vertices, deep=True))
    cells = vtk.vtkCellArray()
    for a, b, c in faces:
        cells.InsertNextCell(3, (int(a), int(b), int(c)))
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(polydata)
    clean.ConvertLinesToPointsOff()
    clean.ConvertPolysToLinesOff()
    clean.ConvertStripsToPolysOff()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(clean.GetOutputPort())
    normals.SetFeatureAngle(40.0)
    normals.Update()
    result = normals.GetOutput()
    xyz = numpy_support.vtk_to_numpy(result.GetPoints().GetData())
    unit = numpy_support.vtk_to_numpy(result.GetPointData().GetNormals())
    polys = result.GetPolys()
    connectivity = numpy_support.vtk_to_numpy(polys.GetConnectivityArray())
    offsets = numpy_support.vtk_to_numpy(polys.GetOffsetsArray())[1:]
    if not len(connectivity):
        raise RuntimeError(f"Empty mesh for {path.name}")

    def block(values: np.ndarray, fmt: str) -> str:
        return "\n".join(
            " ".join(fmt % v for v in row) for row in np.atleast_2d(values)
        )

    # Simbody only reads the plain ASCII layout, without verts/lines/strips arrays.
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
        "<PolyData>\n"
        f'<Piece NumberOfPoints="{len(xyz)}" NumberOfVerts="0" NumberOfLines="0" '
        f'NumberOfStrips="0" NumberOfPolys="{len(offsets)}">\n'
        '<PointData Normals="Normals">\n'
        '<DataArray type="Float32" Name="Normals" NumberOfComponents="3" format="ascii">\n'
        f'{block(unit, "%.6f")}\n</DataArray>\n</PointData>\n'
        '<Points>\n<DataArray type="Float32" NumberOfComponents="3" format="ascii">\n'
        f'{block(xyz, "%.6f")}\n</DataArray>\n</Points>\n'
        '<Polys>\n<DataArray type="Int32" Name="connectivity" format="ascii">\n'
        f'{block(connectivity.reshape(-1, 3), "%d")}\n</DataArray>\n'
        '<DataArray type="Int32" Name="offsets" format="ascii">\n'
        f'{block(offsets, "%d")}\n</DataArray>\n</Polys>\n'
        "</Piece>\n</PolyData>\n</VTKFile>\n",
        encoding="utf-8",
    )


@dataclass
class _RigidStepPart:
    mass_kg: float
    center_m: np.ndarray
    inertia_kg_m2: np.ndarray
    vertices_m: np.ndarray
    faces: np.ndarray
    color: tuple[float, float, float]


@dataclass(frozen=True)
class StepBodyImport:
    """OpenSim body and geometry data imported from one STEP component.

    Attributes
    ----------
    body
        OpenSim body with mass properties and attached colored display meshes.
    mesh_paths
        Paths to the generated VTP display meshes.
    vertices_m
        Transformed CAD vertices in metres, or an empty array when vertex
        collection was not requested.
    assembly_path
        Names of the parent STEP assemblies, from outermost to innermost.
    """

    body: Any
    mesh_paths: tuple[Path, ...]
    vertices_m: np.ndarray
    assembly_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class _NamedStepSolid:
    name: str
    volume_mm3: float
    center_mm: np.ndarray


def _read_step_imports(
    file: str | Path,
    meshes_dir: Path,
    *,
    density_kg_m3: float | Mapping[str, float] | None = None,
    linear_deflection_mm: float = 0.5,
    angular_deflection_deg: float = 20.0,
    component_names: Iterable[str] = (),
    solid_names: Iterable[str] = (),
    root_body_name: str | None = None,
    component_prefix: str | None = None,
    include_vertices: bool = False,
    split_all_components: bool = False,
    preserve_hierarchy: bool = False,
) -> dict[str, StepBodyImport]:
    """Build rigid bodies for selected STEP components and residual geometry."""
    try:
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_sRGB
        from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.TDF import TDF_Label, TDF_LabelSequence
        from OCC.Core.TDocStd import TDocStd_Document
        from OCC.Core.StepShape import StepShape_ManifoldSolidBrep
        from OCC.Core.TopAbs import TopAbs_SOLID
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.XCAFApp import XCAFApp_Application
        from OCC.Core.XCAFDoc import (
            XCAFDoc_ColorGen,
            XCAFDoc_ColorSurf,
            XCAFDoc_DocumentTool,
        )
    except ImportError as error:
        raise ImportError(
            "read_step requires pythonocc-core (pip install opensim-extensions[step])"
        ) from error

    step_path = Path(file).expanduser().resolve()
    if not step_path.is_file():
        raise FileNotFoundError(step_path)
    if step_path.suffix.lower() not in {".step", ".stp"}:
        raise ValueError("file must have a .step or .stp extension")
    if (
        not math.isfinite(linear_deflection_mm)
        or linear_deflection_mm <= 0
        or not math.isfinite(angular_deflection_deg)
        or angular_deflection_deg <= 0
    ):
        raise ValueError("mesh deflections must be positive finite values")

    requested_components = tuple(component_names)
    requested_solids = tuple(solid_names)
    requested_names = (*requested_components, *requested_solids)
    if any(not isinstance(name, str) or not name for name in requested_names):
        raise ValueError(
            "component_names and solid_names must contain non-empty strings"
        )
    if len({name.casefold() for name in requested_components}) != len(
        requested_components
    ):
        raise ValueError("component_names must not contain duplicates")
    if len({name.casefold() for name in requested_solids}) != len(
        requested_solids
    ):
        raise ValueError("solid_names must not contain duplicates")

    fallback_densities: dict[str, float] | None
    fallback_density: float | None
    if isinstance(density_kg_m3, Mapping):
        fallback_densities = {}
        fallback_density = None
        for part_name, value in density_kg_m3.items():
            density = float(value)
            if (
                not isinstance(part_name, str)
                or not part_name
                or isinstance(value, bool)
                or not math.isfinite(density)
                or density <= 0
            ):
                raise ValueError(
                    "density_kg_m3 mapping must contain non-empty names and "
                    "positive finite densities"
                )
            fallback_densities[part_name] = density
    elif density_kg_m3 is None:
        fallback_densities = None
        fallback_density = None
    else:
        if isinstance(density_kg_m3, bool):
            raise ValueError("density_kg_m3 must be positive and finite")
        fallback_density = float(density_kg_m3)
        fallback_densities = None
        if not math.isfinite(fallback_density) or fallback_density <= 0:
            raise ValueError("density_kg_m3 must be positive and finite")

    document = TDocStd_Document("MDTV-XCAF")
    XCAFApp_Application.GetApplication().NewDocument("MDTV-XCAF", document)
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetNameMode(True)
    reader.SetMatMode(True)
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone or not reader.Transfer(
        document
    ):
        raise ValueError(f"Could not read STEP file {step_path}")

    shape_tool = XCAFDoc_DocumentTool.ShapeTool(document.Main())
    material_tool = XCAFDoc_DocumentTool.MaterialTool(document.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool(document.Main())
    root_name = _step_name(root_body_name or step_path.stem, set())
    used_body_names = {root_name}
    parts_by_body: dict[str, list[_RigidStepPart]] = {root_name: []}
    vertices_by_body: dict[str, list[np.ndarray]] = {root_name: []}
    assembly_paths: dict[str, tuple[str, ...]] = {root_name: ()}
    requested_by_casefold = {name.casefold(): name for name in requested_components}
    seen_components: set[str] = set()
    seen_solids: set[str] = set()
    component_counts: dict[str, int] = {}
    missing_density: set[str] = set()
    named_solids: list[_NamedStepSolid] = []

    if requested_solids:
        solid_reader = STEPControl_Reader()
        if solid_reader.ReadFile(str(step_path)) != IFSelect_RetDone:
            raise ValueError(f"Could not read STEP file {step_path}")
        source_model = solid_reader.WS().Model()
        requested_solid_names = {name.casefold() for name in requested_solids}
        for entity_index in range(1, source_model.NbEntities() + 1):
            entity = source_model.Value(entity_index)
            if "StepShape_ManifoldSolidBrep" not in entity.DynamicType().Name():
                continue
            brep = StepShape_ManifoldSolidBrep.DownCast(entity)
            solid_name = brep.Name().ToCString().strip()
            if solid_name.casefold() not in requested_solid_names:
                continue
            if not solid_reader.TransferEntity(entity):
                raise ValueError(f"Could not transfer STEP solid {solid_name!r}")
            shape = solid_reader.Shape(solid_reader.NbShapes())
            properties = GProp_GProps()
            brepgprop.VolumeProperties(shape, properties, True)
            center = properties.CentreOfMass()
            named_solids.append(
                _NamedStepSolid(
                    solid_name,
                    float(properties.Mass()),
                    np.array([center.X(), center.Y(), center.Z()], dtype=float),
                )
            )

    def new_body_group(name: str, assembly_path: tuple[str, ...] = ()) -> str:
        count = component_counts.get(name.casefold(), 0) + 1
        component_counts[name.casefold()] = count
        component_name = name if count == 1 else f"{name}_{count}"
        candidate = (
            f"{component_prefix}_{component_name}"
            if component_prefix
            else component_name
        )
        body_name = _step_name(candidate, used_body_names)
        parts_by_body[body_name] = []
        vertices_by_body[body_name] = []
        assembly_paths[body_name] = assembly_path
        return body_name

    def matching_component(label: Any, reference: Any) -> str | None:
        for value in (label.GetLabelName(), reference.GetLabelName()):
            if not value:
                continue
            candidates = {value.casefold()}
            if ":" in value:
                candidates.add(value.rsplit(":", 1)[0].casefold())
            for candidate in candidates:
                if candidate in requested_by_casefold:
                    return requested_by_casefold[candidate]
        return None

    def component_matrix(label: Any) -> np.ndarray:
        transformation = shape_tool.GetLocation(label).Transformation()
        matrix = np.eye(4)
        for row in range(3):
            for column in range(4):
                matrix[row, column] = transformation.Value(row + 1, column + 1)
        matrix[:3, 3] *= MM_TO_M
        return matrix

    def make_part(
        shape: Any,
        reference: Any,
        world: np.ndarray,
        part_name: str,
    ) -> _RigidStepPart | None:
        properties = GProp_GProps()
        brepgprop.VolumeProperties(shape, properties, True)
        volume_mm3 = float(properties.Mass())
        density_kg_mm3 = float(material_tool.GetDensityForShape(reference))
        if volume_mm3 > 0:
            if not math.isfinite(density_kg_mm3) or density_kg_mm3 <= 0:
                part_density = (
                    fallback_densities.get(part_name)
                    if fallback_densities is not None
                    else fallback_density
                )
                if part_density is None:
                    missing_density.add(part_name)
                    return None
                density_kg_mm3 = part_density * 1e-9
            mass = volume_mm3 * density_kg_mm3
            center = properties.CentreOfMass()
            center_local_m = np.array([center.X(), center.Y(), center.Z()]) * MM_TO_M
            center_world_m = world[:3, :3] @ center_local_m + world[:3, 3]
            matrix = properties.MatrixOfInertia()
            inertia_local = (
                np.array(
                    [
                        [matrix.Value(row, column) for column in range(1, 4)]
                        for row in range(1, 4)
                    ]
                )
                * density_kg_mm3
                * 1e-6
            )
            inertia_world = world[:3, :3] @ inertia_local @ world[:3, :3].T
        else:
            mass = 0.0
            center_world_m = np.zeros(3)
            inertia_world = np.zeros((3, 3))

        vertices_mm, faces = _step_triangulate(
            shape, linear_deflection_mm, math.radians(angular_deflection_deg)
        )
        if not len(faces):
            return None
        vertices_m = vertices_mm * MM_TO_M @ world[:3, :3].T + world[:3, 3]

        color = Quantity_Color()
        has_color = color_tool.GetColor(shape, XCAFDoc_ColorSurf, color)
        if not has_color:
            has_color = color_tool.GetColor(shape, XCAFDoc_ColorGen, color)
        if not has_color:
            reference_shape = shape_tool.GetShape(reference)
            has_color = color_tool.GetColor(
                reference_shape, XCAFDoc_ColorSurf, color
            )
            if not has_color:
                has_color = color_tool.GetColor(
                    reference_shape, XCAFDoc_ColorGen, color
                )
        rgb = (
            tuple(float(value) for value in color.Values(Quantity_TOC_sRGB))
            if has_color
            else _STEP_DEFAULT_COLOR
        )
        return _RigidStepPart(
            mass_kg=mass,
            center_m=center_world_m,
            inertia_kg_m2=inertia_world,
            vertices_m=vertices_m,
            faces=faces,
            color=tuple(round(value, 4) for value in rgb),
        )

    def matching_named_solid(shape: Any) -> str | None:
        if not named_solids:
            return None
        properties = GProp_GProps()
        brepgprop.VolumeProperties(shape, properties, True)
        volume_mm3 = float(properties.Mass())
        center = properties.CentreOfMass()
        center_mm = np.array([center.X(), center.Y(), center.Z()], dtype=float)
        for named_solid in named_solids:
            if named_solid.name.casefold() in seen_solids:
                continue
            volume_matches = math.isclose(
                volume_mm3,
                named_solid.volume_mm3,
                rel_tol=1e-7,
                abs_tol=1e-5,
            )
            center_matches = np.allclose(
                center_mm, named_solid.center_mm, rtol=0.0, atol=1e-5
            )
            if volume_matches and center_matches:
                return named_solid.name
        return None

    def visit(
        label: Any,
        parent_world: np.ndarray,
        active_body: str | None = None,
        assembly_path: tuple[str, ...] = (),
    ) -> None:
        reference = TDF_Label()
        is_reference = shape_tool.IsReference(label)
        if is_reference and not shape_tool.GetReferredShape(label, reference):
            raise ValueError(
                f"Could not resolve STEP component {label.GetLabelName()!r}"
            )
        if not is_reference:
            reference = label
        world = parent_world @ component_matrix(label) if is_reference else parent_world

        if (
            (split_all_components or preserve_hierarchy)
            and shape_tool.IsAssembly(reference)
        ):
            assembly_name = (
                label.GetLabelName()
                or reference.GetLabelName()
                or root_name
            )
            assembly_path = (*assembly_path, _step_name(assembly_name, set()))
            if preserve_hierarchy and len(assembly_path) == 1:
                assembly_paths[root_name] = assembly_path

        if active_body is None and not split_all_components:
            component_name = matching_component(label, reference)
            if component_name is not None:
                component_path = assembly_path
                if preserve_hierarchy and shape_tool.IsAssembly(reference):
                    component_path = assembly_path[:-1]
                active_body = new_body_group(component_name, component_path)
                seen_components.add(component_name.casefold())

        if shape_tool.IsAssembly(reference):
            components = TDF_LabelSequence()
            shape_tool.GetComponents(reference, components)
            for index in range(1, components.Length() + 1):
                visit(
                    components.Value(index),
                    world,
                    active_body,
                    assembly_path,
                )
            return

        shape = shape_tool.GetShape(reference)
        if shape.IsNull():
            return
        part_name = reference.GetLabelName() or step_path.stem
        if split_all_components and active_body is None:
            if assembly_path:
                active_body = new_body_group(part_name, assembly_path)
            else:
                assembly_paths[root_name] = (root_name,)
                active_body = root_name
        if active_body is None and named_solids:
            explorer = TopExp_Explorer(shape, TopAbs_SOLID)
            solids: list[Any] = []
            while explorer.More():
                solids.append(explorer.Current())
                explorer.Next()
            if solids:
                matched_solid = False
                for solid in solids:
                    solid_name = matching_named_solid(solid)
                    if solid_name is None:
                        target_body = root_name
                        solid_part_name = part_name
                    else:
                        target_body = new_body_group(solid_name)
                        seen_solids.add(solid_name.casefold())
                        solid_part_name = solid_name
                        matched_solid = True
                    part = make_part(
                        solid, reference, world, solid_part_name
                    )
                    if part is not None:
                        parts_by_body[target_body].append(part)
                        if include_vertices:
                            vertices_by_body[target_body].append(part.vertices_m)
                if matched_solid:
                    return

        target_body = active_body or root_name
        part = make_part(shape, reference, world, part_name)
        if part is not None:
            parts_by_body[target_body].append(part)
            if include_vertices:
                vertices_by_body[target_body].append(part.vertices_m)

    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)
    for index in range(1, roots.Length() + 1):
        visit(roots.Value(index), np.eye(4))

    missing_components = {
        name.casefold() for name in requested_components
    } - seen_components
    missing_solids = {
        name.casefold() for name in requested_solids
    } - seen_solids
    missing_names = sorted(missing_components | missing_solids)
    if missing_names:
        raise ValueError(
            f"STEP components or solids were not found in {step_path}: "
            f"{', '.join(missing_names)}"
        )
    if missing_density:
        names = ", ".join(sorted(missing_density))
        raise ValueError(
            f"STEP solids have no material density: {names}. Pass "
            "density_kg_m3 as a positive value or a mapping by STEP part name."
        )

    mesh_directory = meshes_dir.expanduser().resolve()
    imports: dict[str, StepBodyImport] = {}
    omitted_components: list[str] = []
    for body_name, parts in parts_by_body.items():
        if not parts:
            if split_all_components or preserve_hierarchy:
                if body_name != root_name:
                    omitted_components.append(body_name)
                continue
            if body_name == root_name:
                raise ValueError(f"No closed solid volume found in {step_path}")
            raise ValueError(
                f"STEP component {body_name!r} contains no renderable geometry"
            )
        mass = sum(part.mass_kg for part in parts)
        if mass <= 0:
            if split_all_components or preserve_hierarchy:
                omitted_components.append(body_name)
                continue
            raise ValueError(
                f"STEP component {body_name!r} has no closed solid volume"
            )
        center = (
            sum(
                (part.mass_kg * part.center_m for part in parts),
                start=np.zeros(3),
            )
            / mass
        )
        inertia = np.zeros((3, 3))
        for part in parts:
            offset = part.center_m - center
            inertia += part.inertia_kg_m2 + part.mass_kg * (
                np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
            )
        body = opensim.Body(
            body_name,
            float(mass),
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

        color_groups: dict[tuple[float, float, float], list[_RigidStepPart]] = {}
        for part in parts:
            color_groups.setdefault(part.color, []).append(part)
        mesh_paths: list[Path] = []
        mesh_directory.mkdir(parents=True, exist_ok=True)
        for index, (color, color_parts) in enumerate(color_groups.items(), 1):
            vertices: list[np.ndarray] = []
            faces: list[np.ndarray] = []
            offset = 0
            for part in color_parts:
                vertices.append(part.vertices_m)
                faces.append(part.faces + offset)
                offset += len(part.vertices_m)
            suffix = f"_{index}" if len(color_groups) > 1 else ""
            mesh_name = f"{body_name}{suffix}.vtp"
            mesh_path = mesh_directory / mesh_name
            _step_write_vtp(
                mesh_path, np.vstack(vertices), np.vstack(faces)
            )
            mesh_paths.append(mesh_path.resolve())
            mesh = opensim.Mesh(mesh_path.as_posix())
            mesh.setName(f"{body_name}{suffix}_mesh")
            mesh.setColor(opensim.Vec3(*color))
            previous_log_level = opensim.Logger.getLevelString()
            opensim.Logger.setLevelString("Off")
            try:
                body.attachGeometry(mesh)
            finally:
                opensim.Logger.setLevelString(previous_log_level)
        vertices = (
            np.vstack(vertices_by_body[body_name])
            if include_vertices and vertices_by_body[body_name]
            else np.empty((0, 3), dtype=float)
        )
        imports[body_name] = StepBodyImport(
            body=body,
            mesh_paths=tuple(mesh_paths),
            vertices_m=vertices,
            assembly_path=assembly_paths[body_name],
        )
    if not imports:
        raise ValueError(f"No closed solid volume found in {step_path}")
    if omitted_components:
        names = ", ".join(sorted(omitted_components))
        warnings.warn(
            f"STEP components without renderable closed-solid volume were "
            f"omitted from the body hierarchy: {names}",
            UserWarning,
            stacklevel=2,
        )
    return imports


def read_step(
    file: str | Path,
    meshes_dir: Path,
    *,
    density_kg_m3: float | Mapping[str, float] | None = None,
    linear_deflection_mm: float = 0.5,
    angular_deflection_deg: float = 20.0,
    return_hierarchy: bool = False,
) -> tuple[Any, list[Path]] | dict[str, Any]:
    """Import a STEP model as an OpenSim body and generate VTP display meshes.

    STEP material densities are used to calculate each body's mass and
    inertia, and STEP surface colors are assigned to the generated meshes.
    ``density_kg_m3`` supplies a fallback for solids without usable STEP
    density data. The default result combines the file's geometry into one
    body and returns ``(body, mesh_paths)``. With ``return_hierarchy=True``,
    components are split into bodies and returned as nested assembly and
    component dictionaries. Components without renderable closed-solid
    volume are omitted with a warning in the hierarchy result.

    Parameters
    ----------
    file
        Input file with a ``.step`` or ``.stp`` extension.
    meshes_dir
        Directory where generated VTP display meshes are written.
    density_kg_m3
        Optional positive finite fallback density, in kg/m³, or a mapping
        from exact STEP part names to fallback densities.
    linear_deflection_mm, angular_deflection_deg
        Positive finite linear and angular tessellation tolerances.
    return_hierarchy
        When true, return a nested ``assembly -> component -> Body`` mapping
        instead of the ``(body, mesh_paths)`` tuple.

    Returns
    -------
    tuple or dict
        A pair containing the combined OpenSim body and generated mesh paths,
        or a nested mapping whose leaves are OpenSim bodies.

    Raises
    ------
    ImportError
        If the optional ``pythonocc-core`` dependency is not installed.
    FileNotFoundError
        If ``file`` does not exist.
    ValueError
        If the file type, tessellation tolerances, material densities, or
        solid geometry cannot be imported.
    """
    body_name = _step_name(Path(file).stem, set())
    imports = _read_step_imports(
        file,
        meshes_dir,
        density_kg_m3=density_kg_m3,
        linear_deflection_mm=linear_deflection_mm,
        angular_deflection_deg=angular_deflection_deg,
        root_body_name=body_name,
        split_all_components=return_hierarchy,
    )
    if return_hierarchy:
        hierarchy: dict[str, Any] = {}
        for imported_name, imported in imports.items():
            path = imported.assembly_path or (body_name,)
            node = hierarchy
            for name in path:
                node = node.setdefault(name, {})
            node[imported_name] = imported.body
        return hierarchy
    imported = imports[body_name]
    return imported.body, list(imported.mesh_paths)


def _insert_step_tree_value(tree: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = tree
    for name in path[:-1]:
        existing = node.setdefault(name, {})
        if not isinstance(existing, dict):
            raise ValueError(f"STEP hierarchy name {name!r} is both a body and assembly")
        node = existing
    if path[-1] in node:
        raise ValueError(f"duplicate STEP hierarchy path: {'/'.join(path)}")
    node[path[-1]] = value


def read_step_components(
    file: str | Path,
    meshes_dir: Path,
    *,
    component_names: Iterable[str] = (),
    solid_names: Iterable[str] = (),
    root_body_name: str | None = None,
    component_prefix: str | None = None,
    density_kg_m3: float | Mapping[str, float] | None = None,
    linear_deflection_mm: float = 0.5,
    angular_deflection_deg: float = 20.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Import selected STEP components and solids as rigid OpenSim bodies.

    STEP assemblies and components are represented recursively in two
    parallel dictionaries. The first contains OpenSim bodies at its leaves;
    the second contains absolute paths to the generated VTP files. Unselected
    geometry is grouped into ``root_body_name``. Repeated assembly components
    receive numbered names. STEP material densities and colors are preserved
    when present; ``density_kg_m3`` supplies fallback densities.

    Parameters
    ----------
    file
        Input file with a ``.step`` or ``.stp`` extension.
    meshes_dir
        Directory where generated VTP display meshes are written.
    component_names
        STEP assembly component names to split into separate bodies.
    solid_names
        STEP solid names to split into separate bodies.
    root_body_name
        Name for unselected residual geometry; defaults to the STEP file stem.
    component_prefix
        Optional prefix for the generated component body names.
    density_kg_m3
        Optional positive finite fallback density, in kg/m³, or a mapping
        from exact STEP part names to fallback densities.
    linear_deflection_mm, angular_deflection_deg
        Positive finite linear and angular tessellation tolerances.
    Returns
    -------
    tuple[dict, dict]
        ``(bodies, meshes)`` with matching recursive name structure. Leaves
        of ``bodies`` are ``opensim.Body`` objects. Leaves of ``meshes`` are
        an absolute ``Path`` when a body has one mesh, or a tuple of absolute
        paths when it has multiple color-separated meshes.

    Raises
    ------
    ImportError
        If the optional ``pythonocc-core`` dependency is not installed.
    FileNotFoundError
        If ``file`` does not exist.
    ValueError
        If the file type, requested names, tessellation tolerances, material
        densities, or solid geometry cannot be imported.
    """
    imports = _read_step_imports(
        file,
        meshes_dir,
        density_kg_m3=density_kg_m3,
        linear_deflection_mm=linear_deflection_mm,
        angular_deflection_deg=angular_deflection_deg,
        component_names=component_names,
        solid_names=solid_names,
        root_body_name=root_body_name,
        component_prefix=component_prefix,
        preserve_hierarchy=True,
    )
    bodies: dict[str, Any] = {}
    meshes: dict[str, Any] = {}
    for body_name, imported in imports.items():
        path = (*imported.assembly_path, body_name)
        _insert_step_tree_value(bodies, path, imported.body)
        mesh_value: Path | tuple[Path, ...]
        if len(imported.mesh_paths) == 1:
            mesh_value = imported.mesh_paths[0]
        else:
            mesh_value = imported.mesh_paths
        _insert_step_tree_value(meshes, path, mesh_value)
    return bodies, meshes
