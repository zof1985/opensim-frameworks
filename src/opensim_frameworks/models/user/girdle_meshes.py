"""Meshes of the clavicles and the scapulae of :class:`User`.

The bundled model draws the thorax, the sternum, the clavicles and the
scapulae with one mesh, ``hat_ribs_scap.vtp``, attached to the torso. The
shoulder girdle bodies (see :mod:`.shoulder`) move the humerus and its
glenohumeral centre with respect to the torso, so the bones of the girdle must
be attached to ``clavicle_<side>`` and ``scapula_<side>`` or the humeral head
leaves the drawn glenoid. This module splits that mesh, once, into five files
(written next to it, the original is kept):

``hat_ribs_torso.vtp``
    The thorax without the four bones below.
``hat_clavicle_r.vtp``, ``hat_clavicle_l.vtp``, ``hat_scapula_r.vtp``, ``hat_scapula_l.vtp``
    The connected components of the clavicles and of the scapulae.

The vertices keep their original coordinates (torso frame of the unscaled
model): the split only distributes the triangles, so that the union of the
five meshes is the original mesh. :class:`User` attaches each bone mesh to a
frame offset so that, with all the girdle coordinates at zero, every vertex is
exactly where it was before the split (see ``User._update_girdle_meshes``).

Regenerate the files with ``python -m opensim_frameworks.models.user.girdle_meshes``
or with :func:`split_thorax_mesh`.

Placement of the bone meshes (:func:`attach_girdle_meshes`,
:func:`update_girdle_meshes`)
-----------------------------------------------------------------
Each bone mesh hangs on its own offset frame of ``clavicle_<side>`` or
``scapula_<side>``. With all the girdle coordinates at zero:

* the scapulae are exactly where the original mesh had them: the mesh takes
  the scale factors of the thorax mesh and the frame is translated by minus the
  position of the body origin (the acromioclavicular centre) in the torso
  frame, so every vertex is in the Ground where it was before the split;
* the clavicles are adapted only visually: the clavicle mesh of the bundled
  model is shorter than the clavicle of the kinematic model (12.9-13.5 cm
  against 15.4-16.5 cm between the sternoclavicular and acromioclavicular
  centres at 175-205 cm, measured by the biomechanics specialist, who advises
  against moving the centres). The mesh is scaled (mostly along the lateral
  axis, which is the axis of the bone), rotated about its medial end and
  translated so that its medial and lateral ends coincide with the two centres.
  No kinematic quantity depends on it.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ...utils.opensim import read_vtp_vertices

__all__ = [
    "CLAVICLE_MESH_FILES",
    "SCAPULA_MESH_FILES",
    "SOURCE_MESH_FILE",
    "THORAX_MESH_FILE",
    "attach_girdle_meshes",
    "split_thorax_mesh",
    "update_girdle_meshes",
]

MESHES_DIR = Path(__file__).resolve().parent / "assets" / "meshes"
SOURCE_MESH_FILE = "hat_ribs_scap.vtp"
THORAX_MESH_FILE = "hat_ribs_torso.vtp"
CLAVICLE_MESH_FILES = {"r": "hat_clavicle_r.vtp", "l": "hat_clavicle_l.vtp"}
SCAPULA_MESH_FILES = {"r": "hat_scapula_r.vtp", "l": "hat_scapula_l.vtp"}

# Components of the bundled mesh (measured on ``hat_ribs_scap.vtp``, 29
# connected components): each scapula has 798 points and each
# clavicle 264 (after removing the points of the other components);
# the side is the sign of the lateral coordinate z of the centroid. These
# numbers are modelling data of the bundled mesh, not general constants.
_POINTS_PER_BONE = {"scapula": 798, "clavicle": 264}


@lru_cache(maxsize=None)
def _vertices(filename: str) -> np.ndarray:
    """Vertices of a mesh of the package, in its own (unscaled torso) frame."""
    return read_vtp_vertices(MESHES_DIR / filename)


def split_thorax_mesh(
    directory: str | Path = MESHES_DIR, overwrite: bool = False
) -> dict[str, Path]:
    """Split ``hat_ribs_scap.vtp`` into thorax, clavicles and scapulae.

    Parameters
    ----------
    directory : str or pathlib.Path, optional
        Directory with ``hat_ribs_scap.vtp``; the five files are written
        there. Defaults to the meshes of the package.
    overwrite : bool, default=False
        Replace the files if they exist.

    Returns
    -------
    dict of str to pathlib.Path
        Keys ``thorax``, ``clavicle_r``, ``clavicle_l``, ``scapula_r`` and
        ``scapula_l``.

    Raises
    ------
    FileNotFoundError
        If the source mesh is missing.
    FileExistsError
        If a file exists and `overwrite` is False.
    RuntimeError
        If the mesh does not have exactly one component of the expected size
        for each bone and side.
    """
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    from vtkmodules.vtkFiltersCore import (
        vtkAppendPolyData,
        vtkCleanPolyData,
        vtkPolyDataConnectivityFilter,
    )
    from vtkmodules.vtkIOXML import vtkXMLPolyDataReader, vtkXMLPolyDataWriter

    directory = Path(directory)
    source = directory / SOURCE_MESH_FILE
    if not source.is_file():
        raise FileNotFoundError(source)
    names = {
        "thorax": directory / THORAX_MESH_FILE,
        **{f"clavicle_{s}": directory / f for s, f in CLAVICLE_MESH_FILES.items()},
        **{f"scapula_{s}": directory / f for s, f in SCAPULA_MESH_FILES.items()},
    }
    existing = [path for path in names.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(existing[0])

    reader = vtkXMLPolyDataReader()
    reader.SetFileName(str(source))
    reader.Update()
    mesh = reader.GetOutput()

    # one polydata per connected component, in the order of the filter
    connectivity = vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(mesh)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()
    components = []
    for region in range(connectivity.GetNumberOfExtractedRegions()):
        connectivity.SetExtractionModeToSpecifiedRegions()
        connectivity.InitializeSpecifiedRegionList()
        connectivity.AddSpecifiedRegion(region)
        connectivity.Update()
        # the filter keeps every point of the mesh: drop the unused ones
        clean = vtkCleanPolyData()
        clean.SetInputData(connectivity.GetOutput())
        clean.Update()
        part = vtkPolyData()
        part.DeepCopy(clean.GetOutput())
        components.append(part)

    # bones of the girdle: size and side of the centroid
    chosen: dict[str, int] = {}
    for index, part in enumerate(components):
        points = vtk_to_numpy(part.GetPoints().GetData())
        distinct = len(points)
        for bone, count in _POINTS_PER_BONE.items():
            if distinct == count:
                side = "r" if points[:, 2].mean() > 0.0 else "l"
                key = f"{bone}_{side}"
                if key in chosen:
                    raise RuntimeError(f"more than one component for {key}")
                chosen[key] = index
    missing = [key for key in names if key != "thorax" and key not in chosen]
    if missing:
        raise RuntimeError(f"components not found: {missing}")

    thorax = vtkAppendPolyData()
    for index, part in enumerate(components):
        if index not in chosen.values():
            thorax.AddInputData(part)
    thorax.Update()
    outputs = {"thorax": thorax.GetOutput()}
    outputs.update({key: components[index] for key, index in chosen.items()})
    for key, path in names.items():
        writer = vtkXMLPolyDataWriter()
        writer.SetFileName(str(path))
        writer.SetInputData(outputs[key])
        # OpenSim reads only ASCII VTP files, like the original mesh
        writer.SetDataModeToAscii()
        if writer.Write() != 1:
            raise RuntimeError(f"cannot write {path}")
        # the loader of OpenSim does not accept the information keys that
        # recent VTK versions write inside the arrays
        text = path.read_text(encoding="utf-8")
        path.write_text(
            re.sub(r"\s*<InformationKey.*?</InformationKey>", "", text, flags=re.S),
            encoding="utf-8",
        )
    return names


def _translation(frame: Any) -> np.ndarray:
    """Translation of a ``PhysicalOffsetFrame`` as a vector."""
    vector = frame.get_translation()
    return np.array([vector.get(i) for i in range(3)])


def _thorax_mesh(model: Any, opensim: Any) -> Any:
    """Mesh of the torso with the thorax (with or without the girdle bones)."""
    torso = model.getBodySet().get("torso")
    for index in range(torso.getPropertyByName("attached_geometry").size()):
        mesh = opensim.Mesh.safeDownCast(torso.get_attached_geometry(index))
        if mesh is not None and mesh.get_mesh_file() in {
            SOURCE_MESH_FILE,
            THORAX_MESH_FILE,
        }:
            return mesh
    raise RuntimeError("The torso has no thorax mesh")


def attach_girdle_meshes(model: Any, opensim: Any) -> None:
    """Draw the clavicles and the scapulae on their own bodies.

    Parameters
    ----------
    model : opensim.Model
        Model with the shoulder girdle (:func:`.shoulder.add_shoulder_girdle`)
        and the torso mesh ``hat_ribs_scap.vtp``, before initialisation of the
        system.
    opensim : module
        The ``opensim`` module.

    Raises
    ------
    RuntimeError
        If the torso has no thorax mesh.

    Notes
    -----
    The torso mesh is replaced by the thorax without the four bones
    (``hat_ribs_torso.vtp``). For each side and bone an offset frame
    ``<bone>_<side>_mesh_frame`` is added to the body and the bone mesh is
    attached to it with the colour of the thorax; the meshes are placed by
    :func:`update_girdle_meshes` (called here and by ``User`` after every
    scaling).
    """
    thorax = _thorax_mesh(model, opensim)
    thorax.set_mesh_file(THORAX_MESH_FILE)
    for side in ("r", "l"):
        for bone, files in (
            ("clavicle", CLAVICLE_MESH_FILES),
            ("scapula", SCAPULA_MESH_FILES),
        ):
            body = model.getBodySet().get(f"{bone}_{side}")
            frame = opensim.PhysicalOffsetFrame(
                f"{bone}_{side}_mesh_frame", body, opensim.Transform()
            )
            # the frame joins the body first: a mesh attached to a frame that
            # is not yet connected to the model makes OpenSim log an error
            body.addComponent(frame)
            mesh = opensim.Mesh(files[side])
            mesh.setName(f"{bone}_{side}_geom")
            mesh.setColor(thorax.getColor())
            frame.attachGeometry(mesh)
    update_girdle_meshes(model, opensim)


def update_girdle_meshes(model: Any, opensim: Any) -> None:
    """Place the meshes of the clavicles and the scapulae after a scaling.

    The scale factors of the meshes and the offset frames are set from the
    scale of the thorax mesh and from the (scaled) positions of the
    sternoclavicular and acromioclavicular centres, so the call is idempotent.

    Parameters
    ----------
    model : opensim.Model
        Model prepared by :func:`attach_girdle_meshes`; the system is to be
        initialised again afterwards.
    opensim : module
        The ``opensim`` module.

    Raises
    ------
    RuntimeError
        If the clavicle mesh is longer than the clavicle of the model.
    """
    thorax = _thorax_mesh(model, opensim)
    scale = np.array([thorax.get_scale_factors().get(i) for i in range(3)])
    for side in ("r", "l"):
        lateral = 1.0 if side == "r" else -1.0
        sc = _translation(
            opensim.PhysicalOffsetFrame.safeDownCast(
                model.getJointSet().get(f"sternoclavicular_{side}").getParentFrame()
            )
        )
        to_ac = _translation(
            opensim.PhysicalOffsetFrame.safeDownCast(
                model.getJointSet().get(f"acromioclavicular_{side}").getParentFrame()
            )
        )

        # scapula: the vertices are where the original mesh had them
        frame = opensim.PhysicalOffsetFrame.safeDownCast(
            model.getBodySet()
            .get(f"scapula_{side}")
            .getComponent(f"scapula_{side}_mesh_frame")
        )
        frame.set_translation(opensim.Vec3(*(-(sc + to_ac))))
        frame.set_orientation(opensim.Vec3(0.0))
        mesh = opensim.Mesh.safeDownCast(frame.get_attached_geometry(0))
        mesh.set_scale_factors(opensim.Vec3(*scale))

        # clavicle: medial end on the SC centre, lateral end on the AC centre
        frame = opensim.PhysicalOffsetFrame.safeDownCast(
            model.getBodySet()
            .get(f"clavicle_{side}")
            .getComponent(f"clavicle_{side}_mesh_frame")
        )
        vertices = _vertices(CLAVICLE_MESH_FILES[side])
        axis = np.linalg.svd(vertices - vertices.mean(axis=0))[2][0]
        axis = axis if axis[2] * lateral >= 0.0 else -axis
        projection = vertices @ axis
        medial, outer = vertices[projection.argmin()], vertices[projection.argmax()]
        delta = outer - medial
        length = float(np.linalg.norm(to_ac))
        # scale the lateral coordinate (the axis of the bone) to the length of
        # the model, then rotate the mesh axis onto the SC-AC direction
        other = float(np.sum((scale[:2] * delta[:2]) ** 2))
        if length**2 <= other:
            raise RuntimeError("The clavicle mesh is longer than the clavicle")
        clavicle_scale = scale.copy()
        clavicle_scale[2] *= np.sqrt(length**2 - other) / (scale[2] * abs(delta[2]))
        mesh_axis = clavicle_scale * delta
        mesh_axis /= np.linalg.norm(mesh_axis)
        target = to_ac / length
        cross = np.cross(mesh_axis, target)
        norm = float(np.linalg.norm(cross))
        rotation = (
            Rotation.from_rotvec(cross / norm * np.arctan2(norm, mesh_axis @ target))
            if norm > 1e-12
            else Rotation.identity()
        )
        frame.set_translation(
            opensim.Vec3(*(-rotation.apply(clavicle_scale * medial)))
        )
        frame.set_orientation(opensim.Vec3(*rotation.as_euler("XYZ")))
        mesh = opensim.Mesh.safeDownCast(frame.get_attached_geometry(0))
        mesh.set_scale_factors(opensim.Vec3(*clavicle_scale))


if __name__ == "__main__":
    for key, path in split_thorax_mesh(overwrite=True).items():
        print(key, path)
