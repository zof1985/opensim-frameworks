"""Framework-centered viewer and export operations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..framework import Framework
from ..models.model import Model
from ..utils.constants import DEFAULT_SIZE
from ..utils.opensim import import_opensim
from .animation import _ViewerModel, save_animation as _render_animation
from .export import save_image as _save_image
from .player import PlayerWindow, start_player
from .standalone import (
    _player_runtime as _prepare_player_runtime,
    save_player_executable as _save_player_executable,
)
from .visualizer import VTKVisualizer

__all__ = ["FrameworkViewer"]

_TIME_TOLERANCE = 1e-10


def _camera_vector(
    name: str, value: Sequence[float] | None
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


class FrameworkViewer:
    """View and export a framework, normalizing every input to ``Framework``.

    Parameters
    ----------
    framework : Framework, Model, or opensim.Model
        Object to display. OpenSim models are copied into a temporary
        ``Model`` and then wrapped in a ``Framework``.
    trajectories : opensim.StatesTrajectory or None, optional
        States used only when ``framework`` is a plain ``opensim.Model``.
    sampling_frequency_hz : float, default=30.0
        Sampling frequency used to interpolate recorded framework states.
    """

    def __init__(
        self,
        framework: Framework | Model | Any,
        trajectories: Any = None,
        sampling_frequency_hz: float = 30.0,
    ) -> None:
        self._opensim = import_opensim()
        self._framework: Framework
        self._source_model: Any = None
        self._trajectories: Any = None
        self._sampling_frequency_hz = 30.0
        self.set_sampling_frequency_hz(sampling_frequency_hz)
        self.set_framework(framework, trajectories=trajectories)

    @property
    def framework(self) -> Framework:
        """Return the normalized framework used by this viewer."""
        return self._framework

    @framework.setter
    def framework(self, value: Framework | Model | Any) -> None:
        self.set_framework(value)

    def set_framework(
        self,
        value: Framework | Model | Any,
        *,
        trajectories: Any = None,
    ) -> None:
        """Replace the viewed object and normalize it to a Framework."""
        if isinstance(value, Framework):
            if trajectories is not None:
                raise ValueError("trajectories may only be supplied with opensim.Model")
            framework = value
            source_model = None
            stored_trajectories = None
        elif isinstance(value, Model):
            if trajectories is not None and not isinstance(
                trajectories, self._opensim.StatesTrajectory
            ):
                raise ValueError(
                    "trajectories must be an opensim.StatesTrajectory or None"
                )
            model = (
                Model.from_model(value, trajectories)
                if trajectories is not None
                else value
            )
            framework = Framework(model=model)
            source_model = value if trajectories is not None else None
            stored_trajectories = trajectories
            converted = getattr(model, "_osf_trajectory", None)
            if converted is not None:
                for index in range(converted.getSize()):
                    state = converted.get(index)
                    model.state = self._opensim.State(state)
                    framework.record_state(time=state.getTime())
        elif isinstance(value, self._opensim.Model):
            if trajectories is not None and not isinstance(
                trajectories, self._opensim.StatesTrajectory
            ):
                raise TypeError(
                    "trajectories must be an opensim.StatesTrajectory or None"
                )
            model = Model.from_model(value, trajectories)
            framework = Framework(model=model)
            source_model = value
            stored_trajectories = trajectories
            converted = getattr(model, "_osf_trajectory", None)
            if converted is not None:
                for index in range(converted.getSize()):
                    state = converted.get(index)
                    model.state = self._opensim.State(state)
                    framework.record_state(time=state.getTime())
        else:
            raise TypeError(
                "framework must be a Framework, Model, or opensim.Model"
            )
        self._framework = framework
        self._source_model = source_model
        self._trajectories = stored_trajectories

    @property
    def trajectories(self) -> Any:
        """Return the StatesTrajectory supplied for a plain OpenSim model."""
        return self._trajectories

    @trajectories.setter
    def trajectories(self, value: Any) -> None:
        self.set_trajectories(value)

    def set_trajectories(self, value: Any) -> None:
        """Replace the source trajectory and rebuild the normalized framework."""
        if self._source_model is None:
            if value is not None:
                raise ValueError(
                    "trajectories can only be set when the source is opensim.Model"
                )
            self._trajectories = None
            return
        self.set_framework(self._source_model, trajectories=value)

    @property
    def sampling_frequency_hz(self) -> float:
        """Return the interpolation and export sampling frequency."""
        return self._sampling_frequency_hz

    @sampling_frequency_hz.setter
    def sampling_frequency_hz(self, value: float) -> None:
        self.set_sampling_frequency_hz(value)

    def set_sampling_frequency_hz(self, value: float) -> None:
        """Set a finite, positive sampling frequency in hertz."""
        self._sampling_frequency_hz = self._validate_sampling_frequency(value)

    @staticmethod
    def _validate_sampling_frequency(value: float) -> float:
        if isinstance(value, bool):
            raise ValueError("sampling_frequency_hz must be finite and positive")
        try:
            frequency = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "sampling_frequency_hz must be finite and positive"
            ) from error
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("sampling_frequency_hz must be finite and positive")
        return frequency

    def _motion_table(self, sampling_frequency_hz: float | None = None) -> Any | None:
        if self.framework.states_trajectory.getSize() < 2:
            return None
        frequency = (
            self.sampling_frequency_hz
            if sampling_frequency_hz is None
            else self._validate_sampling_frequency(sampling_frequency_hz)
        )
        table = self.framework.get_motion_table(frequency)
        return table if table.getNumRows() >= 2 else None

    def show(
        self,
        sampling_frequency_hz: float | None = None,
        *,
        motion: Any = None,
        loop: bool = False,
    ) -> PlayerWindow:
        """Open the interactive viewer for this framework."""
        frequency = (
            self.sampling_frequency_hz
            if sampling_frequency_hz is None
            else self._validate_sampling_frequency(sampling_frequency_hz)
        )
        model = self.framework.merge_to_model()
        adapter = _ViewerModel(model)
        table = (
            self._motion_table(frequency)
            if motion is None
            else motion
        )
        return start_player(
            adapter,
            table,
            loop=loop,
            fps=frequency,
        )

    def save_player(
        self,
        path: str | Path,
        *,
        clean: bool = False,
        overwrite: bool = True,
    ) -> Path:
        """Build a self-contained player executable for this framework."""
        destination = Path(path)
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a bool")
        if not isinstance(clean, bool):
            raise TypeError("clean must be a bool")
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        model = self.framework.merge_to_model()
        adapter = _ViewerModel(model)
        table = self._motion_table()
        from .player import MotionData

        data = None if table is None else MotionData(adapter, table)
        return _save_player_executable(
            adapter,
            data,
            destination,
            fps=self.sampling_frequency_hz,
            clean=clean,
            overwrite=overwrite,
        )

    def _snapshot_state(
        self,
        frame: int | None,
        time_s: float | None,
    ) -> Any:
        framework = self.framework
        model = framework.composite
        current_state = framework.state
        trajectory_size = framework.states_trajectory.getSize()

        if frame is not None and time_s is not None:
            raise ValueError("specify either frame or time_s, not both")

        if trajectory_size < 2:
            if frame not in (None, 0):
                raise ValueError("frame is out of range")
            state_time = float(current_state.getTime())
            requested_time = state_time if time_s is None else float(time_s)
            if not math.isfinite(requested_time) or not math.isclose(
                requested_time,
                state_time,
                rel_tol=0.0,
                abs_tol=_TIME_TOLERANCE,
            ):
                raise ValueError("the requested time is unavailable in this framework")
            return self._opensim.State(current_state)

        table = self._motion_table()
        if table is None:
            raise ValueError("the framework has no interpolated motion samples")
        times = [float(value) for value in table.getIndependentColumn()]
        if frame is not None:
            if isinstance(frame, bool) or not isinstance(frame, int):
                raise TypeError("frame must be an integer or None")
            if not -len(times) <= frame < len(times):
                raise ValueError("frame is out of range")
            row_index = frame % len(times)
            selected_time = times[row_index]
            row = table.getRowAtIndex(row_index)
            labels = list(table.getColumnLabels())
        else:
            recorded_times = framework.state_times
            requested_time = (
                float(recorded_times[0])
                if time_s is None
                else float(time_s)
            )
            if not math.isfinite(requested_time):
                raise ValueError("time_s must be finite")
            available_times = (*recorded_times, *times)
            selected_time = next(
                (
                    value
                    for value in available_times
                    if math.isclose(
                        requested_time,
                        value,
                        rel_tol=0.0,
                        abs_tol=_TIME_TOLERANCE,
                    )
                ),
                None,
            )
            if selected_time is None:
                raise ValueError(
                    "time_s must match a recorded state or a sampled time"
                )
            source_table = framework._motion_source_table()
            one_row_table = framework._interpolate_motion_table(
                source_table, [requested_time]
            )
            row = one_row_table.getRowAtIndex(0)
            labels = list(one_row_table.getColumnLabels())

        label_values = {
            str(labels[index]): float(row[index])
            for index in range(len(labels))
        }
        state = self._opensim.State(current_state)
        state_names = model.getStateVariableNames()
        for index in range(state_names.getSize()):
            name = state_names.get(index)
            if name not in label_values:
                raise ValueError(f"motion table is missing state variable {name!r}")
            model.setStateVariableValue(state, name, label_values[name])
        state.setTime(float(selected_time))
        model.realizePosition(state)
        model.realizeVelocity(state)
        return state

    def save_snapshot(
        self,
        path: str | Path,
        *,
        time_s: float | None = None,
        frame: int | None = None,
        overwrite: bool = True,
        size: tuple[int, int] = DEFAULT_SIZE,
        dpi: int = 300,
        camera_position: Sequence[float] | None = None,
        camera_focal_point: Sequence[float] | None = None,
        camera_view_up: Sequence[float] | None = None,
        show_ground: bool = True,
        show_muscles: bool = True,
        show_markers: bool = True,
        show_axes: bool = True,
        show_reachability: bool = True,
        show_vision_cone: bool = True,
    ) -> Path:
        """Save the framework at a recorded or sampled time as PNG or SVG."""
        destination = Path(path)
        extension = destination.suffix.lower()
        if extension not in {".svg", ".png"}:
            raise ValueError("path must have a .svg or .png extension")
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a bool")
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        if extension == ".png" and (
            isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0
        ):
            raise ValueError("dpi must be a positive integer for PNG output")
        visibility = {
            "show_ground": show_ground,
            "show_muscles": show_muscles,
            "show_markers": show_markers,
            "show_axes": show_axes,
            "show_reachability": show_reachability,
            "show_vision_cone": show_vision_cone,
        }
        for name, enabled in visibility.items():
            if not isinstance(enabled, bool):
                raise TypeError(f"{name} must be a bool")
        camera = {
            "camera_position": _camera_vector("camera_position", camera_position),
            "camera_focal_point": _camera_vector(
                "camera_focal_point", camera_focal_point
            ),
            "camera_view_up": _camera_vector("camera_view_up", camera_view_up),
        }
        model = self.framework.composite
        state = self._snapshot_state(frame, time_s)
        adapter = _ViewerModel(model, state)
        destination.parent.mkdir(parents=True, exist_ok=True)
        visualizer = VTKVisualizer(adapter, offscreen=True, size=size)
        try:
            visualizer.set_ground_visible(show_ground)
            visualizer.set_muscles_visible(show_muscles)
            visualizer.set_markers_visible(show_markers)
            visualizer.set_axes_visible(show_axes)
            visualizer.set_reachability_visible(show_reachability)
            visualizer.set_vision_cone_visible(show_vision_cone)
            if any(value is not None for value in camera.values()):
                visualizer.set_camera(
                    position=camera["camera_position"],
                    focal_point=camera["camera_focal_point"],
                    view_up=camera["camera_view_up"],
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
        path: str | Path,
        *,
        sampling_frequency_hz: float | None = None,
        overwrite: bool = True,
        size: tuple[int, int] = DEFAULT_SIZE,
        camera_position: Sequence[float] | None = None,
        camera_focal_point: Sequence[float] | None = None,
        camera_view_up: Sequence[float] | None = None,
        show_ground: bool = True,
        show_muscles: bool = True,
        show_markers: bool = True,
        show_axes: bool = True,
        show_reachability: bool = True,
        show_vision_cone: bool = True,
    ) -> Path:
        """Render the framework trajectory as an MP4."""
        destination = Path(path)
        if destination.suffix.lower() != ".mp4":
            raise ValueError("path must have an .mp4 extension")
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a bool")
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        table = self._motion_table(sampling_frequency_hz)
        if table is None:
            raise ValueError("animation requires at least two recorded states")
        camera = {
            "camera_position": _camera_vector("camera_position", camera_position),
            "camera_focal_point": _camera_vector(
                "camera_focal_point", camera_focal_point
            ),
            "camera_view_up": _camera_vector("camera_view_up", camera_view_up),
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        return _render_animation(
            self.framework.merge_to_model(),
            table,
            destination,
            size=size,
            camera_position=camera["camera_position"],
            camera_focal_point=camera["camera_focal_point"],
            camera_view_up=camera["camera_view_up"],
            show_ground=show_ground,
            show_muscles=show_muscles,
            show_markers=show_markers,
            show_axes=show_axes,
            show_reachability=show_reachability,
            show_vision_cone=show_vision_cone,
        )

    def save_framework(self, path: str | Path, *, overwrite: bool = True) -> Path:
        """Save this viewer's framework to an .osf archive."""
        return self.framework.to_osf(path, overwrite=overwrite)

    def prepare_player_export(self) -> None:
        """Prebuild the reusable runtime for future :meth:`save_player` calls."""
        with _prepare_player_runtime():
            pass
