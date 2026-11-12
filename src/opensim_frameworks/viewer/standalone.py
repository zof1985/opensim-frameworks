"""Build a Windows player executable from a precomputed VTK scene."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import uuid
import warnings
import zipfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from ..utils.constants import DEFAULT_SIZE
from . import export as _export
from .animation import _ViewerModel
from .player import MotionData
from .visualizer import VTKVisualizer

_OVERLAY_MAGIC = b"OSFPLAY1"
_DEFAULT_PLAYER_SIZE = (DEFAULT_SIZE[0] // 2, DEFAULT_SIZE[1] // 2)


def _write_splash_image(path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (520, 180), "#202b38")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 10, 180), fill="#00a0df")
    font_path = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "segoeui.ttf"
    try:
        title_font = ImageFont.truetype(str(font_path), 27)
        message_font = ImageFont.truetype(str(font_path), 17)
    except OSError:
        title_font = message_font = ImageFont.load_default()
    draw.text((40, 48), "OpenSim Frameworks", font=title_font, fill="#ffffff")
    draw.text(
        (42, 94),
        "Opening standalone viewer...",
        font=message_font,
        fill="#c6d1de",
    )
    image.save(path, format="PNG")


def _runtime_cache_directory() -> Path:
    digest = hashlib.sha256()
    digest.update(sys.version.encode())
    digest.update(sys.implementation.cache_tag.encode())
    digest.update(sys.platform.encode())
    digest.update(os.environ.get("PROCESSOR_ARCHITECTURE", "").encode())
    for package in (
        "PyInstaller",
        "vtk",
        "imageio-ffmpeg",
        "Pillow",
        "numpy",
    ):
        digest.update(importlib.metadata.version(package).encode())
    for source in (
        Path(__file__),
        Path(__file__).with_name("standalone_player.py"),
        Path(__file__).with_name("svg.py"),
        Path(__file__).with_name("win32_embed.py"),
    ):
        digest.update(source.read_bytes())
    local_app_data = Path(
        os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
    )
    return (
        local_app_data
        / "opensim-frameworks"
        / "player-runtime"
        / digest.hexdigest()[:20]
    )


def _build_runtime(work_dir: Path) -> Path:
    try:
        import PyInstaller.__main__
    except ImportError as error:
        raise ImportError(
            "standalone player export requires PyInstaller; "
            "install or upgrade opensim-frameworks to install its dependencies"
        ) from error

    player_script = Path(__file__).with_name("standalone_player.py")
    if not player_script.is_file():
        raise FileNotFoundError(f"standalone player source is missing: {player_script}")
    splash_image = work_dir / "player-splash.png"
    _write_splash_image(splash_image)
    PyInstaller.__main__.run(
        [
            "--noconfirm",
            "--clean",
            "--onefile",
            "--windowed",
            "--splash",
            str(splash_image),
            "--name",
            "player-runtime",
            "--distpath",
            str(work_dir),
            "--workpath",
            str(work_dir / "build"),
            "--specpath",
            str(work_dir),
            "--paths",
            str(player_script.parent),
            "--collect-binaries",
            "imageio_ffmpeg",
            "--exclude-module",
            "opensim",
            "--exclude-module",
            "opensim_frameworks",
            "--exclude-module",
            "pandas",
            "--exclude-module",
            "scipy",
            str(player_script),
        ]
    )
    built = work_dir / "player-runtime.exe"
    if not built.is_file():
        raise RuntimeError("PyInstaller did not create the generic player runtime")
    return built


def _warn_temporary_runtime(cache: Path, reason: OSError) -> None:
    warnings.warn(
        f"Cannot reuse the player runtime in {cache}: {reason}. "
        "The build will use a temporary directory, which will be removed "
        "after the operation completes; the runtime must be rebuilt next time.",
        RuntimeWarning,
        stacklevel=3,
    )


@contextmanager
def _player_runtime(*, clean: bool = False) -> Iterator[Path]:
    if not sys.platform.startswith("win"):
        raise RuntimeError("standalone .exe export is supported only on Windows")
    if not isinstance(clean, bool):
        raise TypeError("clean must be a boolean")
    cache = _runtime_cache_directory()
    runtime = cache / "player-runtime.exe"
    if not clean and runtime.is_file():
        try:
            with runtime.open("rb") as cached:
                readable = bool(cached.read(1))
        except OSError as error:
            _warn_temporary_runtime(cache, error)
        else:
            if readable:
                yield runtime
                return

    try:
        cache.mkdir(parents=True, exist_ok=True)
        build = tempfile.TemporaryDirectory(prefix="build-", dir=cache)
    except OSError as error:
        _warn_temporary_runtime(cache, error)
        with tempfile.TemporaryDirectory(
            prefix="opensim-frameworks-player-runtime-"
        ) as temporary:
            yield _build_runtime(Path(temporary))
        return

    with build as temporary:
        built = _build_runtime(Path(temporary))
        cached_runtime = cache / f"runtime-{uuid.uuid4().hex}.tmp"
        try:
            shutil.copyfile(built, cached_runtime)
            os.replace(cached_runtime, runtime)
        except OSError as error:
            try:
                cached_runtime.unlink(missing_ok=True)
            except OSError as cleanup_error:
                warnings.warn(
                    f"Could not remove incomplete player runtime cache file "
                    f"{cached_runtime}: {cleanup_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            _warn_temporary_runtime(cache, error)
            yield built
        else:
            yield runtime


def _write_polydata(vtk: Any, polydata: Any, path: Path) -> None:
    if not isinstance(polydata, vtk.vtkPolyData):
        raise TypeError("player export encountered non-polydata VTK geometry")
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.SetCompressorTypeToZLib()
    writer.SetCompressionLevel(9)
    if not writer.Write():
        raise OSError(f"unable to write VTK geometry: {path}")


def _actor_matrix(actor: Any) -> list[float]:
    matrix = actor.GetMatrix()
    return [
        matrix.GetElement(row, column)
        for row in range(4)
        for column in range(4)
    ]


def _make_frames(
    model: Any,
    data: MotionData | None,
    output: Path,
    *,
    fps: float,
    size: tuple[int, int],
    camera_position: tuple[float, float, float] | None,
    camera_focal_point: tuple[float, float, float] | None,
    camera_view_up: tuple[float, float, float] | None,
    show_ground: bool,
    show_muscles: bool,
    show_markers: bool,
    show_axes: bool,
    show_reachability: bool,
    show_vision_cone: bool,
    path_map: Any,
) -> dict[str, Any]:
    import vtk

    assets = output / "assets"
    geometry_dir = assets / "geometry"
    muscle_dir = assets / "muscles"
    tables_dir = assets / "tables"
    geometry_dir.mkdir(parents=True)
    muscle_dir.mkdir(parents=True)

    visualizer = VTKVisualizer(model, offscreen=True, size=size)
    try:
        visualizer.set_ground_visible(show_ground)
        visualizer.set_muscles_visible(show_muscles)
        visualizer.set_markers_visible(show_markers)
        visualizer.set_axes_visible(show_axes)
        visualizer.set_reachability_visible(show_reachability)
        visualizer.set_vision_cone_visible(show_vision_cone)
        if any(
            vector is not None
            for vector in (camera_position, camera_focal_point, camera_view_up)
        ):
            visualizer.set_camera(
                position=camera_position,
                focal_point=camera_focal_point,
                view_up=camera_view_up,
            )

        body_actors = []
        for index, (body, actor) in enumerate(visualizer._actor_bodies):
            mapper = actor.GetMapper()
            mapper.Update()
            geometry_path = Path("geometry") / f"body_{index:05d}.vtp"
            _write_polydata(vtk, mapper.GetInput(), assets / geometry_path)
            body_name = body.getName().rsplit("/", 1)[-1]
            category = (
                "reachability"
                if body_name.startswith("reach_")
                else "vision_cone"
                if body_name.startswith("vision_cone")
                else "body"
            )
            prop = actor.GetProperty()
            body_actors.append(
                {
                    "geometry": geometry_path.as_posix(),
                    "name": visualizer._actor_components[actor][1],
                    "category": category,
                    "color": list(prop.GetColor()),
                    "opacity": prop.GetOpacity(),
                }
            )

        marker_actors = []
        for index, (marker, actor) in enumerate(visualizer._marker_items):
            mapper = actor.GetMapper()
            mapper.Update()
            geometry_path = Path("geometry") / f"marker_{index:05d}.vtp"
            _write_polydata(vtk, mapper.GetInput(), assets / geometry_path)
            marker_actors.append(
                {
                    "geometry": geometry_path.as_posix(),
                    "name": marker.getName(),
                    "color": list(actor.GetProperty().GetColor()),
                    "opacity": actor.GetProperty().GetOpacity(),
                }
            )

        muscle_actors = [
            {
                "name": muscle.getName(),
                "color": list(actor.GetProperty().GetColor()),
            }
            for muscle, _, actor in visualizer._muscle_items
        ]
        if data is None:
            frame_times = [float(model.state.getTime())]
            start_time = end_time = frame_times[0]
        else:
            start_time = data.start_time
            end_time = data.end_time
            frame_count = max(1, round((end_time - start_time) * fps))
            frame_times = [
                start_time + (end_time - start_time) * index / frame_count
                for index in range(frame_count + 1)
            ]

        frames = []
        for frame_index, time in enumerate(frame_times):
            if data is not None:
                _export._set_pose(model, data, time)
            else:
                model.model.realizePosition(model.state)
            visualizer.show(model.state)

            frame_muscles = []
            for muscle_index, (_, mapper, _) in enumerate(visualizer._muscle_items):
                mapper.Update()
                polydata = mapper.GetInput()
                if polydata is None or polydata.GetNumberOfPoints() == 0:
                    frame_muscles.append(None)
                    continue
                path = Path("muscles") / f"m{muscle_index:05d}_{frame_index:06d}.vtp"
                _write_polydata(vtk, polydata, assets / path)
                frame_muscles.append(path.as_posix())

            frames.append(
                {
                    "time": time,
                    "body_matrices": [
                        _actor_matrix(actor)
                        for _, actor in visualizer._actor_bodies
                    ],
                    "muscles": frame_muscles,
                    "markers": [
                        list(actor.GetPosition())
                        for _, actor in visualizer._marker_items
                    ],
                }
            )

        camera = visualizer._renderer.GetActiveCamera()
        camera_state = {
            "position": list(camera.GetPosition()),
            "focal_point": list(camera.GetFocalPoint()),
            "view_up": list(camera.GetViewUp()),
        }
        tables = _export.export_tables_to_xlsx(
            model, data, tables_dir, path_map=path_map, overwrite=True
        )
        return {
            "title": model.getName() or "opensim-frameworks",
            "fps": fps,
            "start_time": start_time,
            "end_time": end_time,
            "loop": False,
            "has_motion": data is not None,
            "camera": camera_state,
            "size": list(size),
            "body_actors": body_actors,
            "muscle_actors": muscle_actors,
            "marker_actors": marker_actors,
            "frames": frames,
            "tables": [path.name for path in tables],
            "show_ground": show_ground,
            "show_muscles": show_muscles,
            "show_markers": show_markers,
            "show_axes": show_axes,
            "show_reachability": show_reachability,
            "show_vision_cone": show_vision_cone,
        }
    finally:
        visualizer.close()


def save_player_executable(
    model: Any,
    motion: MotionData | None,
    path: str | Path,
    *,
    fps: float = 30.0,
    clean: bool = False,
    size: tuple[int, int] = _DEFAULT_PLAYER_SIZE,
    loop: bool = False,
    camera_position: tuple[float, float, float] | None = None,
    camera_focal_point: tuple[float, float, float] | None = None,
    camera_view_up: tuple[float, float, float] | None = None,
    show_ground: bool = True,
    show_muscles: bool = True,
    show_markers: bool = True,
    show_axes: bool = True,
    show_reachability: bool = True,
    show_vision_cone: bool = True,
    overwrite: bool = True,
    path_map: Any = None,
) -> Path:
    """Build a self-contained Windows executable for an interactive player.

    OpenSim is used only while generating the data-only VTK scene. The
    resulting executable contains the VTK geometry, per-frame transforms,
    and precomputed XLSX exports, but does not package or import OpenSim.
    With ``clean=True``, the model-independent PyInstaller runtime is rebuilt
    instead of reusing the compatible cached executable.
    Existing destination files are replaced by default; pass
    ``overwrite=False`` to reject an existing executable.
    """
    if not sys.platform.startswith("win"):
        raise RuntimeError("standalone .exe export is supported only on Windows")
    if (
        not isinstance(fps, (int, float))
        or isinstance(fps, bool)
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("fps must be a positive finite number")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")
    if not isinstance(clean, bool):
        raise TypeError("clean must be a boolean")
    if not isinstance(loop, bool):
        raise TypeError("loop must be a boolean")
    visibility = {
        "show_ground": show_ground,
        "show_muscles": show_muscles,
        "show_markers": show_markers,
        "show_axes": show_axes,
        "show_reachability": show_reachability,
        "show_vision_cone": show_vision_cone,
    }
    if any(not isinstance(value, bool) for value in visibility.values()):
        raise TypeError("visibility options must be booleans")
    if (
        len(size) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in size)
    ):
        raise ValueError("size must contain two positive integers")
    destination = Path(path)
    if destination.suffix.lower() != ".exe":
        raise ValueError("path must have a .exe extension")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    native_model = getattr(model, "model", model)
    opensim = getattr(model, "opensim", None)
    if opensim is None:
        raise TypeError("model must be an OpenSim model adapter")
    try:
        initial_state = opensim.State(model.state)
    except RuntimeError:
        initial_state = opensim.State(native_model.getWorkingState())
    export_model = _ViewerModel(native_model, initial_state)
    export_model.path_map = path_map
    if motion is not None and not isinstance(motion, MotionData):
        raise TypeError("motion must be MotionData or None")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        runtime = stack.enter_context(_player_runtime(clean=clean))
        work_dir = Path(
            stack.enter_context(tempfile.TemporaryDirectory(prefix="opensim-player-"))
        )
        manifest = _make_frames(
            export_model,
            motion,
            work_dir,
            fps=float(fps),
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
            path_map=path_map,
        )
        manifest["loop"] = loop
        manifest_path = work_dir / "assets" / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, separators=(",", ":"), ensure_ascii=True),
            encoding="utf-8",
        )
        archive_path = work_dir / "player-data.zip"
        assets = work_dir / "assets"
        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for source in assets.rglob("*"):
                if source.is_file():
                    archive.write(source, source.relative_to(assets).as_posix())

        with tempfile.TemporaryDirectory(
            prefix="opensim-player-exe-", dir=destination.parent
        ) as staging:
            staged_executable = Path(staging) / destination.name
            with runtime.open("rb") as source, staged_executable.open("wb") as target:
                shutil.copyfileobj(source, target)
                archive_size = archive_path.stat().st_size
                with archive_path.open("rb") as data:
                    shutil.copyfileobj(data, target)
                target.write(struct.pack("!8sQ", _OVERLAY_MAGIC, archive_size))
            if overwrite:
                os.replace(staged_executable, destination)
            else:
                os.rename(staged_executable, destination)
    return destination
