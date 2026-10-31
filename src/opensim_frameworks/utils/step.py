"""STEP parsing, tessellation, and recursive body/mesh tree utilities."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .constants import MM_TO_M
from .opensim import _step_name, import_opensim, weld_bodies

__all__ = [
    "read_step",
    "weld_step_body_mesh_tree",
    "extract_step_body_and_mesh",
]


_STEP_DEFAULT_COLOR = (0.7, 0.7, 0.7)


def weld_step_body_mesh_tree(
    bodies: dict[str, Any],
    meshes: dict[str, Any],
) -> tuple[Any, Any]:
    """Weld matching body and mesh leaves from a recursive ``read_step`` tree.

    The input trees must have matching dictionary structure. They are not
    modified; mesh leaves are returned as the tuple produced by
    ``weld_bodies(..., fuse_meshes=False)``.

    Parameters
    ----------
    bodies : dict
        Recursive body tree returned by :func:`read_step`.
    meshes : dict
        Matching recursive mesh tree returned by :func:`read_step`.

    Returns
    -------
    tuple
        Welded OpenSim body and its mesh paths.

    Raises
    ------
    TypeError
        If a body assembly does not correspond to a mesh assembly.
    ValueError
        If the trees have different keys or contain no bodies.
    """
    if not isinstance(bodies, dict) or not isinstance(meshes, dict):
        raise TypeError("bodies and meshes must be dictionaries")
    if bodies.keys() != meshes.keys():
        raise ValueError("STEP body and mesh trees must have matching keys")

    body_nodes = []
    mesh_nodes = []
    for name, body_node in bodies.items():
        mesh_node = meshes[name]
        if isinstance(body_node, dict):
            if not isinstance(mesh_node, dict):
                raise TypeError(f"STEP mesh subtree for {name!r} is not a dict")
            body_node, mesh_node = weld_step_body_mesh_tree(body_node, mesh_node)
        elif isinstance(mesh_node, dict):
            raise TypeError(f"STEP body subtree for {name!r} is not a dict")
        body_nodes.append(body_node)
        mesh_nodes.append(mesh_node)
    return weld_bodies(body_nodes, mesh_nodes, fuse_meshes=False)


def extract_step_body_and_mesh(
    bodies: dict[str, Any],
    meshes: dict[str, Any],
    key: str,
    remove: bool = True,
) -> tuple[Any, Any]:
    """Find and weld a named STEP body subtree and its corresponding meshes.

    ``bodies`` and ``meshes`` are matching recursive trees returned by
    :func:`read_step`. The selected node and all descendant body/mesh leaves
    are welded into a single body and mesh-path tuple. By default, the selected
    node is removed from both input trees; empty ancestor assemblies are also
    pruned. Set ``remove=False`` to leave both trees unchanged.

    Parameters
    ----------
    bodies, meshes
        Parallel recursive STEP body and mesh trees.
    key
        Name of the body or assembly to extract. The first matching key in
        traversal order is selected.
    remove
        Whether to remove the extracted node from both trees. Defaults to
        ``True``.

    Returns
    -------
    tuple
        The welded OpenSim body and its mesh paths.

    Raises
    ------
    KeyError
        If the key is absent or body and mesh keys do not match during search.
    TypeError
        If the inputs are not dictionaries, ``remove`` is not a bool, or a
        body assembly does not correspond to a mesh assembly.
    ValueError
        If the extracted assembly's body and mesh trees have different keys.
    """
    if not isinstance(bodies, dict) or not isinstance(meshes, dict):
        raise TypeError("bodies and meshes must be dictionaries")
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a non-empty string")
    if not isinstance(remove, bool):
        raise TypeError("remove must be a bool")

    def find_node(
        body_tree: dict[str, Any],
        mesh_tree: dict[str, Any],
        ancestors: list[tuple[dict[str, Any], dict[str, Any], str]],
    ) -> Any:
        if body_tree.keys() != mesh_tree.keys():
            raise KeyError("STEP body and mesh trees do not have matching keys")
        for name, body_node in body_tree.items():
            mesh_node = mesh_tree[name]
            if name == key:
                if isinstance(body_node, dict) != isinstance(mesh_node, dict):
                    raise TypeError(
                        f"STEP body and mesh subtrees for {key!r} do not match"
                    )
                return body_tree, mesh_tree, body_node, mesh_node, ancestors, name
            if isinstance(body_node, dict):
                if not isinstance(mesh_node, dict):
                    raise TypeError(f"STEP mesh subtree for {name!r} is not a dict")
                result = find_node(
                    body_node,
                    mesh_node,
                    [*ancestors, (body_tree, mesh_tree, name)],
                )
                if result is not None:
                    return result
            elif isinstance(mesh_node, dict):
                raise TypeError(f"STEP body subtree for {name!r} is not a dict")
        return None

    result = find_node(bodies, meshes, [])
    if result is None:
        raise KeyError(f"STEP assembly or body {key!r} was not imported")
    parent_bodies, parent_meshes, body_node, mesh_node, ancestors, name = result

    if isinstance(body_node, dict):
        welded = weld_step_body_mesh_tree(body_node, mesh_node)
    else:
        welded = weld_bodies([body_node], [mesh_node], fuse_meshes=False)

    if remove:
        del parent_bodies[name]
        del parent_meshes[name]
        for ancestor_bodies, ancestor_meshes, ancestor_name in reversed(ancestors):
            if ancestor_bodies[ancestor_name]:
                break
            del ancestor_bodies[ancestor_name]
            del ancestor_meshes[ancestor_name]
    return welded


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
    assembly_path
        Names of the parent STEP assemblies, from outermost to innermost.
    """

    body: Any
    mesh_paths: tuple[Path, ...]
    assembly_path: tuple[str, ...] = ()


def _read_step_imports(
    file: str | Path,
    meshes_dir: Path,
    *,
    density_kg_m3: float | Mapping[str, float] | None = None,
    linear_deflection_mm: float = 0.5,
    angular_deflection_deg: float = 20.0,
) -> dict[str, StepBodyImport]:
    """Build rigid bodies for STEP components and preserve their hierarchy.

    Parameters
    ----------
    file : str or pathlib.Path
        STEP file to import.
    meshes_dir : pathlib.Path
        Directory where generated VTP meshes are written.
    density_kg_m3 : float, mapping of str to float, or None, optional
        Fallback density in kg/m³, either for all solids or keyed by STEP part
        name. Material densities embedded in the STEP document take precedence.
    linear_deflection_mm : float, optional
        Maximum linear tessellation deflection in millimetres.
    angular_deflection_deg : float, optional
        Maximum angular tessellation deflection in degrees.

    Returns
    -------
    dict of str to StepBodyImport
        Imported bodies and their corresponding mesh paths and assembly paths.

    Raises
    ------
    ImportError
        If ``pythonocc-core`` is unavailable.
    FileNotFoundError
        If the STEP file does not exist.
    ValueError
        If the file, tessellation settings, material densities, or STEP
        geometry are invalid.
    """
    opensim = import_opensim()
    try:
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_sRGB
        from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
        from OCC.Core.TDF import TDF_Label, TDF_LabelSequence
        from OCC.Core.TDocStd import TDocStd_Document
        from OCC.Core.XCAFApp import XCAFApp_Application
        from OCC.Core.XCAFDoc import (
            XCAFDoc_ColorGen,
            XCAFDoc_ColorSurf,
            XCAFDoc_DocumentTool,
        )
    except ImportError as error:
        raise ImportError(
            "read_step requires pythonocc-core (pip install opensim-frameworks[step])"
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

    root_name = _step_name(step_path.stem, set())
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
    used_body_names = {root_name}
    parts_by_body: dict[str, list[_RigidStepPart]] = {root_name: []}
    assembly_paths: dict[str, tuple[str, ...]] = {root_name: ()}
    component_counts: dict[str, int] = {}
    missing_density: set[str] = set()

    def new_body_group(name: str, assembly_path: tuple[str, ...] = ()) -> str:
        count = component_counts.get(name.casefold(), 0) + 1
        component_counts[name.casefold()] = count
        component_name = name if count == 1 else f"{name}_{count}"
        body_name = _step_name(component_name, used_body_names)
        parts_by_body[body_name] = []
        assembly_paths[body_name] = assembly_path
        return body_name

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
            has_color = color_tool.GetColor(reference_shape, XCAFDoc_ColorSurf, color)
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

        if shape_tool.IsAssembly(reference):
            assembly_name = (
                label.GetLabelName() or reference.GetLabelName() or root_name
            )
            assembly_path = (*assembly_path, _step_name(assembly_name, set()))
            if len(assembly_path) == 1:
                assembly_paths[root_name] = assembly_path

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
        if active_body is None:
            if assembly_path:
                active_body = new_body_group(part_name, assembly_path)
            else:
                assembly_paths[root_name] = (root_name,)
                active_body = root_name

        target_body = active_body or root_name
        part = make_part(shape, reference, world, part_name)
        if part is not None:
            parts_by_body[target_body].append(part)

    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)
    for index in range(1, roots.Length() + 1):
        visit(roots.Value(index), np.eye(4))

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
            if body_name != root_name:
                omitted_components.append(body_name)
            continue
        mass = sum(part.mass_kg for part in parts)
        if mass <= 0:
            omitted_components.append(body_name)
            continue
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
            _step_write_vtp(mesh_path, np.vstack(vertices), np.vstack(faces))
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
        imports[body_name] = StepBodyImport(
            body=body,
            mesh_paths=tuple(mesh_paths),
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


def _insert_step_tree_value(
    tree: dict[str, Any], path: tuple[str, ...], value: Any
) -> None:
    node = tree
    for name in path[:-1]:
        existing = node.setdefault(name, {})
        if not isinstance(existing, dict):
            raise ValueError(
                f"STEP hierarchy name {name!r} is both a body and assembly"
            )
        node = existing
    if path[-1] in node:
        raise ValueError(f"duplicate STEP hierarchy path: {'/'.join(path)}")
    node[path[-1]] = value


def read_step(
    file: str | Path,
    meshes_dir: str | Path,
    density_kg_m3: float | Mapping[str, float] | None = None,
    linear_deflection_mm: float = 0.5,
    angular_deflection_deg: float = 20.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Import a STEP model into parallel body and mesh hierarchies.

    STEP material densities and surface colors are preserved when present;
    ``density_kg_m3`` supplies fallback densities for solids without usable
    STEP density data. The body and mesh dictionaries have matching recursive
    assembly/component structure. Mesh leaves are absolute paths, or tuples of
    paths when a body has multiple color-separated meshes.

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

    Returns
    -------
    tuple[dict, dict]
        ``(bodies, meshes)`` with matching recursive name structure. Leaves
        of ``bodies`` are OpenSim bodies; leaves of ``meshes`` are absolute
        ``Path`` objects, or tuples of paths for multiple meshes.

    Raises
    ------
    ImportError
        If OpenSim or the optional ``pythonocc-core`` dependency is missing.
    FileNotFoundError
        If ``file`` does not exist.
    ValueError
        If the file type, tessellation tolerances, material densities, or
        solid geometry cannot be imported.
    """
    root_name = _step_name(Path(file).stem, set())
    imports = _read_step_imports(
        file,
        Path(meshes_dir),
        density_kg_m3=density_kg_m3,
        linear_deflection_mm=linear_deflection_mm,
        angular_deflection_deg=angular_deflection_deg,
    )
    bodies: dict[str, Any] = {}
    meshes: dict[str, Any] = {}
    for body_name, imported in imports.items():
        path = (*imported.assembly_path, body_name)
        if body_name == root_name and imported.assembly_path == (root_name,):
            path = (body_name,)
        _insert_step_tree_value(bodies, path, imported.body)
        mesh_value: Path | tuple[Path, ...]
        if len(imported.mesh_paths) == 1:
            mesh_value = imported.mesh_paths[0]
        else:
            mesh_value = imported.mesh_paths
        _insert_step_tree_value(meshes, path, mesh_value)
    return bodies, meshes
