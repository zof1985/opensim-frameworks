"""Runtime for the OpenSim-free player executable built by standalone.py."""

from __future__ import annotations

import bisect
import json
import math
import struct
import shutil
import sys
import time
import tkinter as tk
import zipfile
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import imageio_ffmpeg
import numpy as np
from PIL import Image
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkFiltersSources import vtkPlaneSource
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
import vtkmodules.vtkRenderingOpenGL2 as _vtk_rendering_opengl2
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCellPicker,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkWindowToImageFilter,
)
from vtkmodules.util.numpy_support import vtk_to_numpy

if __package__:
    from .svg import write_svg
    from . import win32_embed as _win32_embed
else:
    from svg import write_svg
    import win32_embed as _win32_embed

_VIEW_DIRECTIONS = {
    "front": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "back": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "right": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "left": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "top": ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    "bottom": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
}
_BG = "#2b2b2b"
_FG = "#ffffff"


def _asset_path(relative: str) -> Path:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    return root / "assets" / relative


def _prepare_assets() -> None:
    if not getattr(sys, "frozen", False):
        return
    root = Path(sys._MEIPASS)
    assets = root / "assets"
    if (assets / "manifest.json").is_file():
        return
    trailer_size = struct.calcsize("!8sQ")
    with Path(sys.executable).open("rb") as executable:
        executable.seek(-trailer_size, 2)
        magic, archive_size = struct.unpack("!8sQ", executable.read(trailer_size))
        if magic != b"OSFPLAY1" or archive_size <= 0:
            raise RuntimeError("The executable does not contain a player scene.")
        file_size = executable.seek(0, 2)
        if archive_size > file_size - trailer_size:
            raise RuntimeError("The executable contains an invalid player scene archive.")
        assets.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(sys.executable) as archive:
            root_path = assets.resolve()
            for name in archive.namelist():
                target = (assets / name).resolve()
                if not target.is_relative_to(root_path):
                    raise RuntimeError("The player scene archive contains an invalid path.")
                archive.extract(name, assets)


def _close_startup_splash() -> None:
    if getattr(sys, "frozen", False):
        import pyi_splash

        pyi_splash.close()


class _Tooltips:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.window = tk.Toplevel(root)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.label = tk.Label(
            self.window, bg=_BG, fg=_FG, padx=8, pady=4, relief="solid", borderwidth=1
        )
        self.label.pack()
        self.pending: str | None = None

    def bind(self, widget: Any, text: str) -> None:
        widget.bind("<Enter>", lambda _event: self.schedule(widget, text), add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def schedule(self, widget: Any, text: str) -> None:
        self.hide()
        self.pending = self.root.after(1000, lambda: self.show(widget, text))

    def show(self, widget: Any, text: str) -> None:
        self.pending = None
        if not widget.winfo_exists():
            return
        self.label.config(text=text)
        self.window.geometry(
            f"+{self.root.winfo_pointerx() + 18}+{self.root.winfo_pointery() + 18}"
        )
        self.window.deiconify()

    def hide(self, _event: Any = None) -> None:
        if self.pending is not None:
            self.root.after_cancel(self.pending)
            self.pending = None
        self.window.withdraw()


class _CameraStyle(vtkInteractorStyleTrackballCamera):
    def OnLeftButtonDown(self) -> None:
        interactor = self.GetInteractor()
        ctrl = bool(interactor.GetControlKey())
        shift = bool(interactor.GetShiftKey())
        if ctrl and not shift:
            interactor.SetControlKey(0)
            interactor.SetShiftKey(1)
            try:
                super().OnLeftButtonDown()
            finally:
                interactor.SetControlKey(1)
                interactor.SetShiftKey(0)
        else:
            super().OnLeftButtonDown()


class Player:
    def __init__(self) -> None:
        _prepare_assets()
        self.manifest = json.loads(_asset_path("manifest.json").read_text(encoding="utf-8"))
        self.root = tk.Tk()
        self.root.title(self.manifest["title"])
        self.fps = float(self.manifest["fps"])
        self.frames = self.manifest["frames"]
        self.times = [float(frame["time"]) for frame in self.frames]
        self.start_time = float(self.manifest["start_time"])
        self.end_time = float(self.manifest["end_time"])
        self.has_motion = bool(self.manifest["has_motion"])
        self.current_time = self.start_time
        self.playing = False
        self.speed = 1.0
        self.loop = bool(self.manifest["loop"])
        self.frame_index = -1
        self._last_tick = time.monotonic()
        self._readers: list[Any] = []
        self._actor_info: dict[Any, str] = {}
        self._hover_info: tuple[str, float, float, float] | None = None

        self._build_ui()
        self._build_scene()
        self.root.update_idletasks()
        controls_height = self.controls.winfo_reqheight()
        width, height = self.manifest["size"]
        self.root.geometry(f"{width}x{height + controls_height}")
        self.root.update()
        if hasattr(self, "_hwnd"):
            _win32_embed.resize_embedded_window(
                self._hwnd,
                self.viewer_frame.winfo_width(),
                self.viewer_frame.winfo_height(),
            )
        self._apply_frame(0)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.update()
        _close_startup_splash()
        self._tick()
        self.root.mainloop()

    def _build_ui(self) -> None:
        width, height = self.manifest["size"]
        self.viewer_frame = tk.Frame(
            self.root, bg="black", width=width, height=height
        )
        self.viewer_frame.pack(side="top", fill="both", expand=True)
        self.viewer_frame.pack_propagate(False)
        controls = ttk.Frame(self.root, padding=(8, 8, 8, 0))
        self.controls = controls
        controls.pack(side="bottom", fill="x")
        for column in range(3):
            controls.columnconfigure(column, weight=1)
        self.slider_var = tk.DoubleVar(value=0.0)
        self.time_var = tk.StringVar(value="")
        self.slider = ttk.Scale(
            controls,
            from_=0,
            to=1000,
            orient="horizontal",
            variable=self.slider_var,
            command=self._seek,
            length=360,
        )
        self.slider.grid(row=0, column=0, columnspan=3, padx=4, pady=(0, 2), sticky="ew")
        self.slider.bind("<Button-1>", lambda _event: setattr(self, "_dragging", True))
        self.slider.bind("<ButtonRelease-1>", lambda _event: setattr(self, "_dragging", False))
        self._dragging = False
        ttk.Label(controls, textvariable=self.time_var).grid(
            row=1, column=0, columnspan=3, pady=(0, 8)
        )
        playback = ttk.LabelFrame(controls, text="Playback", padding=(8, 4))
        playback.grid(row=2, column=0, padx=(0, 4), pady=(0, 8), sticky="nsew")
        view = ttk.LabelFrame(controls, text="View", padding=(8, 4))
        view.grid(row=2, column=1, padx=4, pady=(0, 8), sticky="nsew")
        export = ttk.LabelFrame(controls, text="Export", padding=(8, 4))
        export.grid(row=2, column=2, padx=(4, 0), pady=(0, 8), sticky="nsew")
        self.tooltips = _Tooltips(self.root)
        self.hover_window = tk.Toplevel(self.root)
        self.hover_window.withdraw()
        self.hover_window.overrideredirect(True)
        self.hover_window.attributes("-topmost", True)
        self.hover_label = tk.Label(
            self.hover_window, bg=_BG, fg=_FG, padx=8, pady=6, justify="left"
        )
        self.hover_label.pack()

        controls_spec = [
            (playback, "⏪", self._backward, "Play backward faster"),
            (playback, "◀|", lambda: self._step_frame(-1), "Step one frame backward"),
            (playback, "▶", self._play_pause, "Play or pause"),
            (playback, "⏹", self._stop, "Stop and return to the start"),
            (playback, "|▶", lambda: self._step_frame(1), "Step one frame forward"),
            (playback, "⏩", self._forward, "Play forward faster"),
        ]
        self.play_button = None
        for index, (parent, text, callback, tip) in enumerate(controls_spec):
            button = ttk.Button(parent, text=text, width=3, command=callback)
            button.grid(row=0, column=index, padx=2)
            self.tooltips.bind(button, tip)
            if text == "▶":
                self.play_button = button
        self.loop_var = tk.BooleanVar(value=self.loop)
        loop = ttk.Checkbutton(
            playback,
            text="Cycle",
            variable=self.loop_var,
            command=lambda: setattr(self, "loop", self.loop_var.get()),
        )
        loop.grid(row=0, column=6, padx=(10, 2))
        self.tooltips.bind(loop, "Repeat playback at the end")

        self.visibility: dict[str, tk.BooleanVar] = {}
        visibility_options = [
            ("ground", "Ground", 0, 0),
            ("muscles", "Muscles", 0, 1),
            ("markers", "Markers", 0, 2),
            ("axes", "Axes", 0, 3),
            ("vision_cone", "Vision cone + axis", 2, 0),
            ("reachability", "Reachability", 2, 2),
        ]
        for key, label, row, column in visibility_options:
            variable = tk.BooleanVar(value=bool(self.manifest[f"show_{key}"]))
            self.visibility[key] = variable
            checkbox = ttk.Checkbutton(
                view, text=label, variable=variable, command=self._update_visibility
            )
            checkbox.grid(
                row=row,
                column=column,
                columnspan=2 if key in {"vision_cone", "reachability"} else 1,
                padx=2,
                sticky="w",
            )
            self.tooltips.bind(checkbox, f"Show or hide {label.lower()}")
        ttk.Label(view, text="Camera:").grid(row=1, column=0, pady=(6, 0), sticky="w")
        self.view_var = tk.StringVar(value="Front")
        camera = ttk.Combobox(
            view,
            textvariable=self.view_var,
            state="readonly",
            width=9,
            values=[name.capitalize() for name in _VIEW_DIRECTIONS],
        )
        camera.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._set_view(self.view_var.get().lower()),
        )
        camera.grid(row=1, column=1, columnspan=2, pady=(6, 0), sticky="w")
        self.tooltips.bind(camera, "Choose a preset camera direction")

        export_buttons = [
            (export, "📷", self._save_image, "Save the current view as PNG or SVG"),
            (export, "🎬", self._save_animation, "Export the motion as an MP4 video"),
            (export, "📊", self._save_tables, "Export the precomputed marker and joint tables"),
        ]
        for index, (parent, text, callback, tip) in enumerate(export_buttons):
            button = ttk.Button(parent, text=text, width=3, command=callback)
            button.grid(row=0, column=index, padx=2)
            self.tooltips.bind(button, tip)
            if text == "🎬" and not self.has_motion:
                button.config(state="disabled")
        if not self.has_motion:
            for child in playback.winfo_children():
                child.config(state="disabled")
            self.slider.config(state="disabled")
        self.root.bind_all("<space>", self._on_tk_keypress, add="+")
        self.root.bind_all("<Left>", self._on_tk_keypress, add="+")
        self.root.bind_all("<Right>", self._on_tk_keypress, add="+")
        self.slider.bind("<Left>", self._on_tk_keypress, add="+")
        self.slider.bind("<Right>", self._on_tk_keypress, add="+")

    def _build_scene(self) -> None:
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.92, 0.93, 0.95)
        self.window = vtkRenderWindow()
        self.window.AddRenderer(self.renderer)
        self.window.SetSize(*self.manifest["size"])
        self.window.SetWindowName(self.manifest["title"])
        self.interactor = vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)
        self._style = _CameraStyle()
        self.interactor.SetInteractorStyle(self._style)
        self.interactor.Initialize()
        self.interactor.AddObserver("KeyPressEvent", self._on_vtk_keypress)
        self.interactor.AddObserver("MouseMoveEvent", self._on_mouse_move)
        self.interactor.AddObserver("LeaveEvent", self._on_mouse_leave)
        self.picker = vtkCellPicker()
        self.picker.SetTolerance(0.0005)

        self.body_actors = []
        self.muscle_actors = []
        self.marker_actors = []
        self._load_actors(self.manifest["body_actors"], "body")
        self._load_actors(self.manifest["muscle_actors"], "muscle")
        self._load_actors(self.manifest["marker_actors"], "marker")

        self.renderer.ResetCamera()
        self.ground_actor = self._ground_actor()
        self.renderer.AddActor(self.ground_actor)
        self.axes_actor = vtkAxesActor()
        self.axes_actor.SetTotalLength(0.5, 0.5, 0.5)
        self.renderer.AddActor(self.axes_actor)
        self._set_window_parent()
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(*self.manifest["camera"]["position"])
        camera.SetFocalPoint(*self.manifest["camera"]["focal_point"])
        camera.SetViewUp(*self.manifest["camera"]["view_up"])
        self.renderer.ResetCameraClippingRange()
        self.window.Render()

    def _load_polydata(
        self, relative: str, *, retain: bool = True
    ) -> tuple[Any, Any]:
        reader = vtkXMLPolyDataReader()
        reader.SetFileName(str(_asset_path(relative)))
        reader.Update()
        if retain:
            self._readers.append(reader)
        return reader, reader.GetOutput()

    def _load_actors(self, specifications: list[dict[str, Any]], kind: str) -> None:
        target = {
            "body": self.body_actors,
            "muscle": self.muscle_actors,
            "marker": self.marker_actors,
        }[kind]
        for specification in specifications:
            actor = vtkActor()
            mapper = vtkPolyDataMapper()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*specification["color"])
            actor.GetProperty().SetOpacity(float(specification.get("opacity", 1.0)))
            if kind != "muscle":
                reader, _ = self._load_polydata(specification["geometry"])
                mapper.SetInputConnection(reader.GetOutputPort())
            self.renderer.AddActor(actor)
            target.append(
                {
                    "actor": actor,
                    "mapper": mapper,
                    "specification": specification,
                    "reader": None,
                    "geometry": None,
                }
            )
            self._actor_info[actor] = specification["name"]

    def _set_window_parent(self) -> None:
        if not _win32_embed.IS_WINDOWS:
            return
        _win32_embed.set_process_dpi_aware()
        hwnd = _win32_embed.hwnd_from_vtk_window_id(
            self.window.GetGenericWindowId()
        )
        _win32_embed.embed_visualizer_window(hwnd, self.viewer_frame.winfo_id())
        width, height = self.viewer_frame.winfo_width(), self.viewer_frame.winfo_height()
        _win32_embed.resize_embedded_window(hwnd, width, height)
        self._hwnd = hwnd
        self.viewer_frame.bind("<Configure>", self._resize_window)

    def _resize_window(self, event: Any) -> None:
        if event.width > 1 and event.height > 1:
            _win32_embed.resize_embedded_window(
                self._hwnd, event.width, event.height
            )
            self.window.SetSize(event.width, event.height)

    @staticmethod
    def _ground_actor() -> Any:
        plane = vtkPlaneSource()
        plane.SetOrigin(-5, 0, -5)
        plane.SetPoint1(5, 0, -5)
        plane.SetPoint2(-5, 0, 5)
        plane.SetXResolution(10)
        plane.SetYResolution(10)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.85, 0.85, 0.82)
        actor.GetProperty().SetAmbient(0.6)
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(0.6, 0.6, 0.6)
        return actor

    def _apply_frame(self, index: int) -> None:
        index = max(0, min(index, len(self.frames) - 1))
        frame = self.frames[index]
        for item, matrix_values in zip(
            self.body_actors, frame["body_matrices"], strict=True
        ):
            matrix = vtkMatrix4x4()
            for row in range(4):
                for column in range(4):
                    matrix.SetElement(row, column, matrix_values[row * 4 + column])
            item["actor"].SetUserMatrix(matrix)
        for item, path in zip(self.muscle_actors, frame["muscles"], strict=True):
            if path != item["geometry"]:
                item["actor"].SetVisibility(False)
                if path is not None:
                    reader, _ = self._load_polydata(path, retain=False)
                    item["mapper"].SetInputConnection(reader.GetOutputPort())
                    item["reader"] = reader
                    item["geometry"] = path
                else:
                    item["mapper"].SetInputData(vtkPolyData())
                    item["reader"] = None
                    item["geometry"] = None
        for item, position in zip(self.marker_actors, frame["markers"], strict=True):
            item["actor"].SetPosition(*position)
        self.frame_index = index
        self.current_time = float(frame["time"])
        self._update_visibility(render=False)
        self.window.Render()

    def _update_visibility(self, render: bool = True) -> None:
        self.ground_actor.SetVisibility(self.visibility["ground"].get())
        self.axes_actor.SetVisibility(self.visibility["axes"].get())
        for item in self.body_actors:
            category = item["specification"]["category"]
            visible = (
                self.visibility["reachability"].get()
                if category == "reachability"
                else self.visibility["vision_cone"].get()
                if category == "vision_cone"
                else True
            )
            item["actor"].SetVisibility(visible)
        for item in self.muscle_actors:
            has_geometry = item["geometry"] is not None
            item["actor"].SetVisibility(has_geometry and self.visibility["muscles"].get())
        for item in self.marker_actors:
            item["actor"].SetVisibility(self.visibility["markers"].get())
        if render:
            self.window.Render()

    def _set_view(self, name: str, *, render: bool = True) -> None:
        direction, view_up = _VIEW_DIRECTIONS[name]
        bounds = self.renderer.ComputeVisiblePropBounds()
        center = tuple((bounds[i] + bounds[i + 1]) / 2 for i in (0, 2, 4))
        diagonal = math.sqrt(
            (bounds[1] - bounds[0]) ** 2
            + (bounds[3] - bounds[2]) ** 2
            + (bounds[5] - bounds[4]) ** 2
        )
        distance = max(diagonal, 1.0) * 1.5
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*center)
        camera.SetPosition(
            *(center[i] + direction[i] * distance for i in range(3))
        )
        camera.SetViewUp(*view_up)
        self.renderer.ResetCameraClippingRange()
        if render:
            self.window.Render()

    def _on_mouse_move(self, _obj: Any, _event: str) -> None:
        x, y = self.interactor.GetEventPosition()
        self.picker.Pick(x, y, 0, self.renderer)
        actor = self.picker.GetActor()
        if actor in self._actor_info:
            point = self.picker.GetPickPosition()
            self._hover_info = (
                self._actor_info[actor],
                point[0],
                point[1],
                point[2],
            )
        else:
            self._hover_info = None

    def _on_mouse_leave(self, _obj: Any, _event: str) -> None:
        self._hover_info = None

    def _show_hover(self) -> None:
        if self._hover_info is None:
            self.hover_window.withdraw()
            return
        label, x, y, z = self._hover_info
        self.hover_label.config(text=f"{label}\nX: {x:.3f}  Y: {y:.3f}  Z: {z:.3f}")
        self.hover_window.geometry(
            f"+{self.root.winfo_pointerx() + 18}+{self.root.winfo_pointery() + 18}"
        )
        self.hover_window.deiconify()

    def _seek(self, value: str) -> None:
        if not self.has_motion or not self._dragging:
            return
        fraction = float(value) / 1000.0
        self.current_time = self.start_time + fraction * (self.end_time - self.start_time)
        self._apply_frame(self._frame_at_time(self.current_time))

    def _frame_at_time(self, value: float) -> int:
        upper = bisect.bisect_left(self.times, value)
        if upper == 0:
            return 0
        if upper == len(self.times):
            return upper - 1
        lower = upper - 1
        if value - self.times[lower] <= self.times[upper] - value:
            return lower
        return upper

    def _play_pause(self) -> None:
        if not self.has_motion:
            return
        self.playing = not self.playing
        self.play_button.config(text="⏸" if self.playing else "▶")

    def _step_frame(self, direction: int) -> None:
        if not self.has_motion:
            return
        self.playing = False
        self.play_button.config(text="▶")
        index = min(max(self.frame_index + direction, 0), len(self.frames) - 1)
        self._apply_frame(index)
        fraction = (self.current_time - self.start_time) / (
            self.end_time - self.start_time
        )
        self.slider_var.set(fraction * 1000.0)
        self.time_var.set(
            f"{self.current_time - self.start_time:6.2f} / "
            f"{self.end_time - self.start_time:.2f} s  ({self.speed:+.0f}x)"
        )

    def _handle_playback_key(self, key: str) -> bool:
        normalized = key.casefold()
        if normalized == "space":
            self._play_pause()
        elif normalized == "left":
            self._step_frame(-1)
        elif normalized == "right":
            self._step_frame(1)
        else:
            return False
        return True

    def _on_tk_keypress(self, event: Any) -> str | None:
        widget_class = event.widget.winfo_class()
        key = event.keysym.casefold()
        if key == "space" and widget_class in {
            "TButton",
            "TCheckbutton",
            "TCombobox",
            "TEntry",
        }:
            return None
        if key in {"left", "right"} and widget_class in {
            "TCombobox",
            "TEntry",
        }:
            return None
        return "break" if self._handle_playback_key(event.keysym) else None

    def _on_vtk_keypress(self, _interactor: Any, _event: str) -> None:
        self._handle_playback_key(self.interactor.GetKeySym())

    def _stop(self) -> None:
        self.playing = False
        self.speed = 1.0
        self.play_button.config(text="▶")
        self.current_time = self.start_time
        self._apply_frame(0)

    def _forward(self) -> None:
        speeds = (-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0)
        index = speeds.index(self.speed)
        self.speed = speeds[min(index + 1, len(speeds) - 1)]
        self.playing = True
        self.play_button.config(text="⏸")

    def _backward(self) -> None:
        speeds = (-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0)
        index = speeds.index(self.speed)
        self.speed = speeds[max(index - 1, 0)]
        self.playing = True
        self.play_button.config(text="⏸")

    def _tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now
        if self.has_motion and self.playing:
            span = self.end_time - self.start_time
            self.current_time += dt * self.speed
            if self.current_time > self.end_time or self.current_time < self.start_time:
                if self.loop:
                    self.current_time = self.start_time + (
                        self.current_time - self.start_time
                    ) % span
                else:
                    self.current_time = min(
                        max(self.current_time, self.start_time), self.end_time
                    )
                    self.playing = False
                    self.play_button.config(text="▶")
            frame = self._frame_at_time(self.current_time)
            if frame != self.frame_index:
                self._apply_frame(frame)
            self.slider_var.set(
                1000 * (self.current_time - self.start_time) / span
            )
        if self.has_motion:
            self.time_var.set(
                f"{self.current_time - self.start_time:6.2f} / "
                f"{self.end_time - self.start_time:.2f} s  ({self.speed:+.0f}x)"
            )
        self.interactor.ProcessEvents()
        self._show_hover()
        elapsed_ms = (time.monotonic() - now) * 1000
        delay_ms = max(1, round(1000 / self.fps - elapsed_ms))
        self.root.after(delay_ms, self._tick)

    def _capture(self) -> np.ndarray:
        image = vtkWindowToImageFilter()
        image.SetInput(self.window)
        image.SetInputBufferTypeToRGB()
        image.ReadFrontBufferOff()
        image.Update()
        output = image.GetOutput()
        width, height, _ = output.GetDimensions()
        return np.flipud(
            vtk_to_numpy(output.GetPointData().GetScalars()).reshape(height, width, 3)
        )

    def _save_image(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save image",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("SVG image", "*.svg")],
        )
        if not path:
            return
        try:
            if Path(path).suffix.lower() == ".svg":
                write_svg(self.renderer, self.window.GetSize(), path)
            else:
                Image.fromarray(self._capture()).save(path, dpi=(300, 300))
        except Exception as error:
            messagebox.showerror("Export failed", str(error), parent=self.root)

    def _save_animation(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save animation",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4")],
        )
        if not path:
            return
        original_time = self.current_time
        was_playing = self.playing
        self.playing = False
        self.root.config(cursor="wait")
        self.root.update()
        writer = None
        try:
            writer = imageio_ffmpeg.write_frames(
                path,
                self.window.GetSize(),
                fps=self.fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )
            writer.send(None)
            for index in range(len(self.frames)):
                self._apply_frame(index)
                writer.send(np.ascontiguousarray(self._capture()).tobytes())
        except Exception as error:
            messagebox.showerror("Export failed", str(error), parent=self.root)
        finally:
            try:
                if writer is not None:
                    writer.close()
            finally:
                self._apply_frame(self._frame_at_time(original_time))
                self.playing = was_playing
                self.root.config(cursor="")

    def _save_tables(self) -> None:
        directory = filedialog.askdirectory(
            parent=self.root, title="Export marker and joint tables (.xlsx)"
        )
        if not directory:
            return
        targets = [Path(directory) / name for name in self.manifest["tables"]]
        existing = [target.name for target in targets if target.exists()]
        if existing and not messagebox.askyesno(
            "Overwrite files",
            "These files already exist and will be replaced:\n" + "\n".join(existing),
            parent=self.root,
        ):
            return
        try:
            for name, target in zip(self.manifest["tables"], targets, strict=True):
                shutil.copy2(_asset_path(f"tables/{name}"), target)
        except Exception as error:
            messagebox.showerror("Export failed", str(error), parent=self.root)

    def close(self) -> None:
        self.window.Finalize()
        self.root.destroy()


if __name__ == "__main__":
    try:
        Player()
    except Exception as error:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Player failed", str(error), parent=root)
        root.destroy()
