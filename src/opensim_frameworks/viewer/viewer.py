"""Public viewer entry point for native OpenSim models and frameworks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ..framework import Framework
from ..utils.constants import DEFAULT_SIZE
from ..utils.opensim import import_opensim
from .animation import _ViewerModel, save_animation as _render_animation
from .export import export_tables_to_xlsx as _export_tables, save_image as _save_image
from .player import MotionData, PlayerWindow, _apply_time, start_player
from .standalone import (
    _player_runtime as _prepare_player_runtime,
    save_player_executable as _save_player_executable,
)
from .visualizer import VTKVisualizer

__all__ = ["VTKViewer"]


def _camera_vector(
    name: str, value: tuple[float, float, float] | None
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        vector = tuple(float(component) for component in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain three finite numbers") from error
    if len(vector) != 3 or not all(math.isfinite(component) for component in vector):
        raise ValueError(f"{name} must contain three finite numbers")
    return vector


class VTKViewer:
    """Display and export native OpenSim models and frameworks with VTK.

    Examples
    --------
    >>> viewer = VTKViewer()
    >>> viewer.show(framework)  # doctest: +SKIP
    >>> viewer.save_snapshot(framework, path="pose.svg")  # doctest: +SKIP
    >>> viewer.save_animation(model, "motion.mp4", motion="motion.sto")  # doctest: +SKIP
    """

    def __init__(self) -> None:
        self._opensim = import_opensim()

    def _native_model_and_map(self, obj: Any) -> tuple[Any, Any]:
        if isinstance(obj, Framework):
            return obj.merge_to_model(return_mapping=True)
        return self._native_model(obj), None

    def _native_model(self, obj: Any) -> Any:
        if isinstance(obj, Framework):
            return obj.merge_to_model()
        if isinstance(obj, self._opensim.Model):
            return obj
        raise TypeError("object must be an opensim.Model or Framework")

    def _table(self, motion: Any) -> Any:
        if isinstance(motion, (str, Path)):
            return self._opensim.TimeSeriesTable(str(motion))
        if isinstance(motion, self._opensim.TimeSeriesTable):
            return motion
        raise TypeError("motion must be a path or opensim.TimeSeriesTable")

    def show(
        self,
        object: Framework | Any,
        motion: Any = None,
        *,
        loop: bool = False,
        fps: float = 30.0,
    ) -> PlayerWindow:
        """Open the interactive viewer.

        Parameters
        ----------
        object : opensim.Model or Framework
            Model to display. A framework is displayed as a merged snapshot;
            its member models remain unchanged.
        motion : str, pathlib.Path, opensim.TimeSeriesTable, or None, default=None
            Optional motion table or path to a ``.mot``/``.sto`` file.
        loop : bool, default=False
            Whether playback repeats after the final sample.
        fps : float, default=30.0
            Playback refresh and animation-export frame rate.

        Returns
        -------
        PlayerWindow
            Handle used to inspect playback state or close the viewer.

        Raises
        ------
        TypeError
            If ``object`` is neither an OpenSim model nor a ``Framework``.
        """
        model, path_map = self._native_model_and_map(object)
        adapter = _ViewerModel(model)
        # Lets the player's xlsx export restore the members' original names.
        adapter.path_map = path_map
        return start_player(adapter, motion, loop=loop, fps=fps)

    def save_player(
        self,
        object: Framework | Any,
        path: str | Path,
        motion: Any = None,
        *,
        fps: float = 30.0,
        clean: bool = False,
        overwrite: bool = True,
    ) -> Path:
        """Build a self-contained Windows player executable.

        The model is sampled and its VTK geometry is packaged frame by frame.
        The generated executable keeps the interactive camera, visibility,
        playback, and export controls, but does not include or load OpenSim.

        Parameters
        ----------
        object : opensim.Model or Framework
            Model or framework to package.
        path : str or pathlib.Path
            Destination ``.exe`` file.
        motion : str, pathlib.Path, opensim.TimeSeriesTable, or None
            Optional motion to precalculate into the player.
        fps : float, default=30.0
            Frame rate used for the precalculated motion and MP4 export.
        clean : bool, default=False
            Rebuild the model-independent PyInstaller runtime instead of
            reusing a compatible cached build.
        overwrite : bool, default=True
            Replace an existing executable at ``path``.

        Returns
        -------
        pathlib.Path
            The generated executable path.

        Notes
        -----
        The exported player starts with standard size, camera, visibility, and
        loop settings. If ``motion`` is provided, its animation is included;
        the current player window's view and playback-control settings are not
        copied.

        The model-independent PyInstaller runtime is cached in AppData and
        reused by later exports. If that location is not writable, the runtime
        is built in a temporary directory, with a warning, and removed after
        the executable is created.

        Raises
        ------
        ImportError
            If PyInstaller is not installed.
        RuntimeError
            If the current platform is not Windows.
        FileExistsError
            If the destination exists and ``overwrite`` is false.
        """
        model, path_map = self._native_model_and_map(object)
        adapter = _ViewerModel(model)
        try:
            adapter.state = self._opensim.State(model.getWorkingState())
        except RuntimeError:
            adapter.state = self._opensim.State(adapter.state)
        adapter.path_map = path_map
        data = None if motion is None else MotionData(adapter, self._table(motion))
        return _save_player_executable(
            adapter,
            data,
            path,
            fps=fps,
            clean=clean,
            overwrite=overwrite,
            path_map=path_map,
        )

    def prepare_player_export(self) -> None:
        """Prebuild the reusable runtime for future :meth:`save_player` calls.

        If the persistent cache is unavailable, the temporary build is removed
        after this call and will need to be rebuilt during export.
        """
        with _prepare_player_runtime():
            pass

    def save_snapshot(
        self,
        object: Framework | Any,
        frame: int | None = None,
        time_s: float | None = 0.0,
        path: str | Path = "snapshot.svg",
        *,
        motion: Any = None,
        size: tuple[int, int] = DEFAULT_SIZE,
        dpi: int = 300,
        camera_position: tuple[float, float, float] | None = None,
        camera_focal_point: tuple[float, float, float] | None = None,
        camera_view_up: tuple[float, float, float] | None = None,
        show_ground: bool = True,
        show_muscles: bool = True,
        show_markers: bool = True,
        show_axes: bool = True,
        show_reachability: bool = True,
        show_vision_cone: bool = True,
    ) -> Path:
        """Save the posed model as an SVG or PNG image, without opening a window.

        Parameters
        ----------
        object : opensim.Model or Framework
            Model to render. It is not modified.
        frame : int or None, default=None
            Row of ``motion`` to render. Takes precedence over ``time_s``.
        time_s : float or None, default=0.0
            Time in seconds on ``motion`` to render (linearly interpolated).
        path : str or pathlib.Path
            Destination; the extension must be ``.svg`` or ``.png``.
        motion : str, pathlib.Path, opensim.TimeSeriesTable, or None, default=None
            Motion that ``frame`` and ``time_s`` refer to. Without a motion
            the model is rendered in its current posture, and ``frame`` and
            ``time_s`` must be ``None``, ``0`` or ``0.0``.
        size : tuple[int, int], default=DEFAULT_SIZE
            Render size in pixels; sets PNG dimensions and SVG layout.
        dpi : int, default=300
            Resolution metadata for PNG output. Ignored for SVG output.
        camera_position : tuple of float or None, default=None
            Camera position ``(x, y, z)`` in ground-frame metres. ``None``
            uses the automatically framed position.
        camera_focal_point : tuple of float or None, default=None
            Ground-frame point in metres for the camera to look at. ``None``
            uses the automatically framed target.
        camera_view_up : tuple of float or None, default=None
            Ground-frame camera up direction. ``None`` keeps the automatic
            direction unless another camera vector is supplied.
        show_ground : bool, default=True
            Show the ground reference grid.
        show_muscles : bool, default=True
            Show muscle paths.
        show_markers : bool, default=True
            Show model markers.
        show_axes : bool, default=True
            Show the ground-frame XYZ coordinate axes.
        show_reachability : bool, default=True
            Show palm reachability regions when present.
        show_vision_cone : bool, default=True
            Show the visual cone and its centre line when present.

        Returns
        -------
        pathlib.Path
            The destination path.

        Raises
        ------
        ValueError
            If the extension is not ``.svg`` or ``.png``, PNG ``dpi`` is not
            a positive integer, ``frame`` is out of range, or ``frame``/
            ``time_s`` is requested without a motion.
        TypeError
            If ``object`` or ``motion`` has an unsupported type.
        ValueError
            If a camera vector is malformed or the requested camera vectors
            do not define a valid view.
        """
        destination = Path(path)
        extension = destination.suffix.lower()
        if extension not in {".svg", ".png"}:
            raise ValueError("path must have a .svg or .png extension")
        if extension == ".png" and (
            not isinstance(dpi, int) or isinstance(dpi, bool) or dpi <= 0
        ):
            raise ValueError("dpi must be a positive integer for PNG output")
        visibility_options = {
            "show_ground": show_ground,
            "show_muscles": show_muscles,
            "show_markers": show_markers,
            "show_axes": show_axes,
            "show_reachability": show_reachability,
            "show_vision_cone": show_vision_cone,
        }
        for name, value in visibility_options.items():
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be a boolean")
        camera_vectors = {
            "camera_position": _camera_vector("camera_position", camera_position),
            "camera_focal_point": _camera_vector(
                "camera_focal_point", camera_focal_point
            ),
            "camera_view_up": _camera_vector("camera_view_up", camera_view_up),
        }
        model = self._native_model(object)
        meta = _ViewerModel(model)
        try:
            state = self._opensim.State(model.getWorkingState())
        except RuntimeError:
            state = self._opensim.State(meta.state)
        meta.state = state

        if motion is None:
            if frame not in (None, 0) or time_s not in (None, 0, 0.0):
                raise ValueError("frame and time_s require a motion")
        else:
            data = MotionData(meta, self._table(motion))
            if frame is not None:
                if not -len(data.times) <= frame < len(data.times):
                    raise ValueError("frame is out of range")
                time = float(data.times[frame])
            else:
                time = data.start_time if time_s is None else float(time_s)
            _apply_time(meta, data, time)

        destination.parent.mkdir(parents=True, exist_ok=True)
        visualizer = VTKVisualizer(meta, offscreen=True, size=size)
        try:
            visualizer.set_ground_visible(show_ground)
            visualizer.set_muscles_visible(show_muscles)
            visualizer.set_markers_visible(show_markers)
            visualizer.set_axes_visible(show_axes)
            visualizer.set_reachability_visible(show_reachability)
            visualizer.set_vision_cone_visible(show_vision_cone)
            if any(value is not None for value in camera_vectors.values()):
                visualizer.set_camera(
                    position=camera_vectors["camera_position"],
                    focal_point=camera_vectors["camera_focal_point"],
                    view_up=camera_vectors["camera_view_up"],
                )
            if extension == ".svg":
                visualizer.save_svg(destination)
            else:
                _save_image(visualizer, destination, dpi=dpi)
        finally:
            visualizer.close()
        return destination

    def save_animation(
        self,
        object: Framework | Any,
        file: str | Path,
        *,
        motion: Any,
        size: tuple[int, int] = DEFAULT_SIZE,
        camera_position: tuple[float, float, float] | None = None,
        camera_focal_point: tuple[float, float, float] | None = None,
        camera_view_up: tuple[float, float, float] | None = None,
        show_ground: bool = True,
        show_muscles: bool = True,
        show_markers: bool = True,
        show_axes: bool = True,
        show_reachability: bool = True,
        show_vision_cone: bool = True,
    ) -> Path:
        """Render a motion of the model to an MP4 without opening a window.

        Parameters
        ----------
        object : opensim.Model or Framework
            Model driven by ``motion``. It is not modified.
        file : str or pathlib.Path
            Destination ``.mp4`` path.
        motion : str, pathlib.Path or opensim.TimeSeriesTable
            Motion whose coordinate columns drive the model.
        size : tuple[int, int], default=DEFAULT_SIZE
            Output frame dimensions in pixels.
        camera_position : tuple of float or None, default=None
            Camera position ``(x, y, z)`` in ground-frame metres. ``None``
            uses the automatically framed position.
        camera_focal_point : tuple of float or None, default=None
            Ground-frame point in metres for the camera to look at. ``None``
            uses the automatically framed target.
        camera_view_up : tuple of float or None, default=None
            Ground-frame camera up direction.
        show_ground : bool, default=True
            Show the ground reference grid.
        show_muscles : bool, default=True
            Show muscle paths.
        show_markers : bool, default=True
            Show model markers.
        show_axes : bool, default=True
            Show the ground-frame XYZ coordinate axes.
        show_reachability : bool, default=True
            Show palm reachability regions when present.
        show_vision_cone : bool, default=True
            Show the visual cone and its centre line when present.

        Returns
        -------
        pathlib.Path
            The destination path.
        """
        model = self._native_model(object)
        return _render_animation(
            model,
            self._table(motion),
            file,
            size=size,
            camera_position=camera_position,
            camera_focal_point=camera_focal_point,
            camera_view_up=camera_view_up,
            show_ground=show_ground,
            show_muscles=show_muscles,
            show_markers=show_markers,
            show_axes=show_axes,
            show_reachability=show_reachability,
            show_vision_cone=show_vision_cone,
        )

    def save_tables(
        self,
        object: Framework | Any,
        directory: str | Path,
        motion: Any = None,
        *,
        overwrite: bool = False,
    ) -> list[Path]:
        """Export marker and joint tables to ``.xlsx`` files without a window.

        Same export as the viewer's Export button: one file per framework
        member (``<member>.xlsx``) or a single file for a model, each with a
        ``markers`` sheet (Ground-frame ``X``/``Y``/``Z`` per marker, metres)
        and a ``joints`` sheet (one column per coordinate, degrees for
        rotational and metres for translational, locked ones included), with
        a ``time [s]`` column and one row per row of ``motion`` (a single row
        at time 0 without a motion).

        Parameters
        ----------
        object : opensim.Model or Framework
            Model or framework to export. It is not modified.
        directory : str or pathlib.Path
            Destination directory, created when missing.
        motion : str, pathlib.Path, opensim.TimeSeriesTable, or None, default=None
            Motion whose coordinate columns drive the model; ``None`` exports
            the current pose.
        overwrite : bool, default=False
            Replace existing files. When ``False``, ``FileExistsError`` is
            raised before anything is written if a target file exists (the
            safe default, since several files are produced at once).

        Returns
        -------
        list[pathlib.Path]
            The files written.

        Raises
        ------
        FileExistsError
            If a target file exists and ``overwrite`` is ``False``.
        ImportError
            If ``openpyxl`` is not installed.
        ValueError
            If ``motion`` has fewer than 2 rows or two members map to the
            same file name.
        """
        model, path_map = self._native_model_and_map(object)
        meta = _ViewerModel(model)
        # Work on a copy of the state so the given model keeps its pose.
        try:
            meta.state = self._opensim.State(model.getWorkingState())
        except RuntimeError:
            meta.state = self._opensim.State(meta.state)
        data = None if motion is None else MotionData(meta, self._table(motion))
        return _export_tables(meta, data, directory, path_map=path_map, overwrite=overwrite)
