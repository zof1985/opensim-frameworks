"""Save the player's current 3D view as an image, or a loaded motion as a video.

Used by the "Export" controls in :mod:`opensim_frameworks.viewer.player` -- kept
separate from that module since these are pure render/encode operations
(no Tk widgets), independently testable/reusable without a live window.
``render_motion_animation`` reuses the player's existing visualizer; the
off-screen :func:`opensim_frameworks.viewer.animation.save_animation` creates
its own visualizer for a model and motion table.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from ..utils.opensim import import_opensim

__all__ = [
    "save_image",
    "render_motion_animation",
    "export_tables_to_xlsx",
    "xlsx_export_paths",
]

# opensim.Coordinate.Rotational (shared with MotionData in player.py)
_ROTATIONAL = 1


def save_image(visualizer: Any, path: str | Path, *, dpi: int = 300) -> None:
    """Save ``visualizer``'s current frame as a PNG.

    Parameters
    ----------
    visualizer : opensim_frameworks.viewer.visualizer.VTKVisualizer
        The visualizer to capture (see :meth:`~opensim_frameworks.viewer.visualizer.VTKVisualizer.capture_frame`).
    path : str or pathlib.Path
        Destination file. Any existing file at ``path`` is overwritten.
    dpi : int, optional
        Resolution metadata embedded in the PNG (does not change the pixel
        dimensions, which always match the visualizer's current window
        size -- this only affects the physical size the image prints/
        displays at in tools that honour it). Defaults to ``300``.
    """
    from PIL import Image

    Image.fromarray(visualizer.capture_frame()).save(str(path), dpi=(dpi, dpi))


def render_motion_animation(
    model: Any,
    data: "MotionData",
    visualizer: Any,
    apply_time: Callable[[Any, "MotionData", float], None],
    path: str | Path,
    *,
    fps: float = 30.0,
) -> None:
    """Render ``data`` from start to end at ``fps`` and encode it as an MP4.

    Drives ``model`` through the motion exactly like live playback does
    (via ``apply_time``), capturing one frame per sample -- the caller is
    responsible for restoring whatever posture/playback state ``model``
    had before this call once it returns (this function always leaves
    ``model`` sitting at the motion's last sampled frame).

    Parameters
    ----------
    model : opensim.Model
        Model the motion drives.
    data : opensim_frameworks.viewer.player.MotionData
        The motion to render, start to end.
    visualizer : opensim_frameworks.viewer.visualizer.VTKVisualizer
        The visualizer to capture each frame from.
    apply_time : callable
        ``(model, data, time) -> None``, applying ``data``'s coordinate
        values at ``time`` to ``model`` and redrawing ``visualizer`` --
        pass :func:`opensim_frameworks.viewer.player._apply_time`.
    path : str or pathlib.Path
        Destination ``.mp4`` file. Any existing file at ``path`` is
        overwritten.
    fps : float, optional
        Frames per second, both for sampling ``data`` and for the encoded
        video's own playback rate. Defaults to ``30.0``.
    """
    import imageio

    duration = data.end_time - data.start_time
    frame_count = max(1, round(duration * fps))
    # macro_block_size=1 disables imageio's default macroblock-alignment
    # padding (which would otherwise silently resize an odd frame size up
    # to the next multiple of 16): libx264 handles arbitrary even
    # dimensions directly, and the visualizer's own render window size is
    # already controllable (DEFAULT_SIZE/resize()), so the saved video
    # should match it exactly rather than being padded.
    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    try:
        for index in range(frame_count + 1):
            time = data.start_time + duration * index / frame_count
            apply_time(model, data, time)
            writer.append_data(visualizer.capture_frame())
    finally:
        writer.close()


def _set_pose(model: Any, data: Any, time: float) -> None:
    """Pose ``model`` at ``time`` of ``data`` without redrawing anything.

    Locked coordinates keep their value and the non-motion coordinates keep
    the value they already have in the state, exactly as the player does.

    Parameters
    ----------
    model : opensim.Model or opensim_frameworks.viewer.animation._ViewerModel
        Model to pose; a ``_ViewerModel`` supplies its own ``state``.
    data : opensim_frameworks.viewer.player.MotionData
        Motion whose columns drive the coordinates (radians/metres).
    time : float
        Query time in seconds, clamped to the motion interval.
    """
    native_model = getattr(model, "model", model)
    state = getattr(model, "state", None)
    if state is None:
        state = native_model.getWorkingState()
    coordinate_set = native_model.getCoordinateSet()
    # Apply every motion column that names an unlocked coordinate of the model.
    for name, value in data.values_at(time).items():
        if not coordinate_set.contains(name):
            continue
        coordinate = coordinate_set.get(name)
        if coordinate.get_locked():
            continue
        coordinate.setValue(state, value, False)
    # A single position realization makes every frame/marker query valid.
    native_model.realizePosition(state)


def _safe_file_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "model"


def _table_groups(model: Any, path_map: Mapping[tuple[str, str], str] | None) -> list:
    """Return ``(file_stem, [(marker_name, Marker)], [(coord_name, Coordinate)])``.

    Without ``path_map`` the whole model is one group; with it, one group per
    framework member, named with the member's original component names.
    """
    opensim = import_opensim()
    native_model = getattr(model, "model", model)
    if path_map is None:
        markers = native_model.getMarkerSet()
        coordinates = native_model.getCoordinateSet()
        return [
            (
                _safe_file_stem(native_model.getName()),
                [(markers.get(i).getName(), markers.get(i)) for i in range(markers.getSize())],
                [
                    (coordinates.get(i).getName(), coordinates.get(i))
                    for i in range(coordinates.getSize())
                ],
            )
        ]
    groups: dict[str, tuple[list, list]] = {}
    for (member, source_path), composite_path in path_map.items():
        parts = source_path.strip("/").split("/")
        markers, coordinates = groups.setdefault(member, ([], []))
        # Markers are "/markerset/<name>"; coordinates "/jointset/<joint>/<name>".
        if len(parts) == 2 and parts[0] == "markerset":
            item = opensim.Marker.safeDownCast(native_model.getComponent(composite_path))
            if item is not None:
                markers.append((parts[1], item))
        elif len(parts) == 3 and parts[0] == "jointset":
            item = opensim.Coordinate.safeDownCast(
                native_model.getComponent(composite_path)
            )
            if item is not None:
                coordinates.append((parts[2], item))
    return [(_safe_file_stem(member), *items) for member, items in groups.items()]


def xlsx_export_paths(
    model: Any,
    directory: str | Path,
    *,
    path_map: Mapping[tuple[str, str], str] | None = None,
) -> list[Path]:
    """Return the ``.xlsx`` files :func:`export_tables_to_xlsx` would write.

    Parameters
    ----------
    model : opensim.Model or opensim_frameworks.viewer.animation._ViewerModel
        Model (or framework composite) that would be exported.
    directory : str or pathlib.Path
        Destination directory (it need not exist yet).
    path_map : mapping or None, optional
        ``(member, source_path) -> composite_path`` of the framework composite
        (``Framework.merge_to_model(return_mapping=True)``). ``None`` exports
        the model as a single file named after it.

    Returns
    -------
    list[pathlib.Path]
        One path per file, ``<directory>/<member>.xlsx``.

    Raises
    ------
    ValueError
        If two members map to the same file name.
    """
    stems = [group[0] for group in _table_groups(model, path_map)]
    if len(set(stems)) != len(stems):
        raise ValueError(f"member names give duplicate file names: {stems}")
    return [Path(directory) / f"{stem}.xlsx" for stem in stems]


def export_tables_to_xlsx(
    model: Any,
    data: Any,
    directory: str | Path,
    *,
    path_map: Mapping[tuple[str, str], str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Write marker and joint tables of a (motion of a) model to ``.xlsx`` files.

    One file per framework member (or one for a plain model), each with two
    sheets with one row per row of the motion table (a single row at time 0
    when there is no motion):

    - ``markers``: column ``time [s]`` then ``<marker>.X [m]``, ``.Y``, ``.Z``,
      the marker position in the Ground frame, in metres.
    - ``joints``: column ``time [s]`` then one column per coordinate named
      ``<coordinate> [deg]`` (rotational) or ``<coordinate> [m]``
      (translational), locked coordinates included.

    The motion is applied as the viewer does (``MotionData`` columns set on the
    coordinates, locked ones skipped, other coordinates left at their current
    state value) with one position realization per row for the whole model.

    Parameters
    ----------
    model : opensim.Model or opensim_frameworks.viewer.animation._ViewerModel
        Model, or framework composite, to pose; it is posed in place, so pass
        a copy of the state if it must not change.
    data : opensim_frameworks.viewer.player.MotionData or None
        Motion to export; ``None`` exports the current pose at time 0.
    directory : str or pathlib.Path
        Destination directory, created when missing.
    path_map : mapping or None, optional
        ``(member, source_path) -> composite_path`` of the composite
        (``Framework.merge_to_model(return_mapping=True)``); restores the
        original names and the per-member split. ``None`` treats ``model`` as
        a single model.
    overwrite : bool, optional
        Replace existing files when ``True``; when ``False`` (default) raise
        before writing anything if one exists.

    Returns
    -------
    list[pathlib.Path]
        The files written.

    Raises
    ------
    ImportError
        If ``openpyxl`` is not installed.
    FileExistsError
        If a target file exists and ``overwrite`` is ``False``.
    ValueError
        If two members map to the same file name.
    """
    try:
        from openpyxl import Workbook
    except ImportError as error:
        raise ImportError("xlsx export requires openpyxl>=3.1") from error

    groups = _table_groups(model, path_map)
    paths = xlsx_export_paths(model, directory, path_map=path_map)
    if not overwrite:
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise FileExistsError(f"files already exist: {existing}")

    native_model = getattr(model, "model", model)
    state = getattr(model, "state", None)
    if state is None:
        state = native_model.getWorkingState()
    times = [0.0] if data is None else [float(time) for time in data.times]

    # Header of every sheet: time, then the columns of the group.
    workbooks = []
    for _, markers, coordinates in groups:
        workbook = Workbook()
        marker_sheet = workbook.active
        marker_sheet.title = "markers"
        joint_sheet = workbook.create_sheet("joints")
        marker_sheet.append(
            ["time [s]"]
            + [f"{name}.{axis} [m]" for name, _ in markers for axis in "XYZ"]
        )
        joint_sheet.append(
            ["time [s]"]
            + [
                f"{name} [{'deg' if item.getMotionType() == _ROTATIONAL else 'm'}]"
                for name, item in coordinates
            ]
        )
        workbooks.append((workbook, marker_sheet, joint_sheet))

    # One pose + realization per row, shared by all members of the composite.
    for time in times:
        if data is not None:
            _set_pose(model, data, time)
        else:
            native_model.realizePosition(state)
        for (_, markers, coordinates), (_, marker_sheet, joint_sheet) in zip(
            groups, workbooks
        ):
            row = [time]
            for _, marker in markers:
                position = marker.getLocationInGround(state)
                row.extend(position.get(axis) for axis in range(3))
            marker_sheet.append(row)
            joint_sheet.append(
                [time]
                + [
                    math.degrees(item.getValue(state))
                    if item.getMotionType() == _ROTATIONAL
                    else item.getValue(state)
                    for _, item in coordinates
                ]
            )

    # Write the files only after every row has been collected successfully.
    Path(directory).mkdir(parents=True, exist_ok=True)
    for path, (workbook, _, _) in zip(paths, workbooks):
        workbook.save(path)
    return paths
