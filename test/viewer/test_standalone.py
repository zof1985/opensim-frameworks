from __future__ import annotations

import json
import struct
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

opensim = pytest.importorskip("opensim")
pytest.importorskip("openpyxl")

from opensim_frameworks.viewer.animation import _ViewerModel
from opensim_frameworks.viewer.player import MotionData
from opensim_frameworks.viewer import standalone as standalone_module
from opensim_frameworks.viewer.standalone import _make_frames
from opensim_frameworks.viewer import standalone_player as standalone_player_module
from opensim_frameworks.viewer.standalone_player import Player, _prepare_assets


def make_model():
    model = opensim.Model()
    model.setName("portable")
    body = opensim.Body("link", 1.0, opensim.Vec3(0), opensim.Inertia(1))
    body.attachGeometry(opensim.Brick(opensim.Vec3(0.1, 0.2, 0.3)))
    model.addBody(body)
    joint = opensim.PinJoint(
        "hinge",
        model.getGround(),
        opensim.Vec3(0),
        opensim.Vec3(0),
        body,
        opensim.Vec3(0),
        opensim.Vec3(0),
    )
    joint.updCoordinate().setName("angle")
    model.addJoint(joint)
    model.addMarker(opensim.Marker("tip", body, opensim.Vec3(1, 0, 0)))
    model.initSystem()
    return model


def make_motion():
    table = opensim.TimeSeriesTable()
    table.setColumnLabels(["angle"])
    table.addTableMetaDataString("inDegrees", "yes")
    table.appendRow(0.0, opensim.RowVector([0.0]))
    table.appendRow(1.0, opensim.RowVector([90.0]))
    return table


def test_precomputes_geometry_and_per_frame_motion_without_runtime_opensim(tmp_path):
    model = make_model()
    adapter = _ViewerModel(model, opensim.State(model.getWorkingState()))
    data = MotionData(adapter, make_motion())

    manifest = _make_frames(
        adapter,
        data,
        tmp_path,
        fps=2.0,
        size=(160, 120),
        camera_position=None,
        camera_focal_point=None,
        camera_view_up=None,
        show_ground=True,
        show_muscles=True,
        show_markers=True,
        show_axes=True,
        show_reachability=True,
        show_vision_cone=True,
        path_map=None,
    )

    assert len(manifest["frames"]) == 3
    assert manifest["has_motion"] is True
    assert manifest["body_actors"]
    assert manifest["marker_actors"]
    first, last = manifest["frames"][0], manifest["frames"][-1]
    assert first["body_matrices"] != last["body_matrices"]
    assert first["markers"] != last["markers"]
    geometry = tmp_path / "assets" / manifest["body_actors"][0]["geometry"]
    assert geometry.is_file()
    assert b'compressor="vtkZLibDataCompressor"' in geometry.read_bytes()
    assert (tmp_path / "assets" / "tables" / "portable.xlsx").is_file()


def test_executable_passes_clean_and_appends_scene_archive(tmp_path, monkeypatch):
    model = make_model()
    adapter = _ViewerModel(model, opensim.State(model.getWorkingState()))
    runtime = tmp_path / "runtime.exe"
    runtime.write_bytes(b"generic-player-runtime")
    clean_values = []

    def fake_make_frames(model, data, output, **options):
        assets = output / "assets"
        assets.mkdir()
        (assets / "manifest.json").write_text(
            json.dumps({"has_motion": False}), encoding="utf-8"
        )
        (assets / "compressed.bin").write_bytes(b"player-geometry-" * 1000)
        return {"has_motion": False}

    monkeypatch.setattr(sys, "platform", "win32")

    @contextmanager
    def fake_runtime(*, clean=False):
        clean_values.append(clean)
        yield runtime

    monkeypatch.setattr(standalone_module, "_player_runtime", fake_runtime)
    monkeypatch.setattr(standalone_module, "_make_frames", fake_make_frames)
    destination = tmp_path / "player.exe"
    destination.write_bytes(b"old-executable")

    result = standalone_module.save_player_executable(
        adapter, None, destination, clean=True
    )

    assert result == destination
    assert clean_values == [True]
    with zipfile.ZipFile(destination) as archive:
        assert json.loads(archive.read("manifest.json")) == {
            "has_motion": False,
            "loop": False,
        }
        asset = archive.getinfo("compressed.bin")
        assert asset.compress_type == zipfile.ZIP_DEFLATED
        assert archive.read(asset) == b"player-geometry-" * 1000
        assert asset.compress_size < asset.file_size
    trailer_size = struct.calcsize("!8sQ")
    with destination.open("rb") as executable:
        executable.seek(-trailer_size, 2)
        magic, archive_size = struct.unpack("!8sQ", executable.read(trailer_size))
    assert magic == standalone_module._OVERLAY_MAGIC
    assert archive_size > 0
    assert destination.read_bytes().startswith(b"generic-player-runtime")


def test_build_runtime_bundles_only_required_dynamic_runtime_files(
    tmp_path, monkeypatch
):
    import PyInstaller.__main__

    arguments = []

    def fake_run(options):
        arguments.extend(options)
        dist_path = Path(options[options.index("--distpath") + 1])
        (dist_path / "player-runtime.exe").write_bytes(b"runtime")

    monkeypatch.setattr(PyInstaller.__main__, "run", fake_run)

    runtime = standalone_module._build_runtime(tmp_path)

    assert runtime.read_bytes() == b"runtime"
    assert "--collect-all" not in arguments
    splash_path = Path(arguments[arguments.index("--splash") + 1])
    assert splash_path.is_file()
    from PIL import Image

    with Image.open(splash_path) as splash:
        assert splash.size == (520, 180)
    assert arguments[arguments.index("--collect-binaries") + 1] == "imageio_ffmpeg"
    excluded_modules = {
        arguments[index + 1]
        for index, option in enumerate(arguments[:-1])
        if option == "--exclude-module"
    }
    assert excluded_modules == {"opensim", "opensim_frameworks", "pandas", "scipy"}


def test_standalone_player_closes_the_pyinstaller_splash_when_frozen(
    monkeypatch,
):
    closed = []
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setitem(
        sys.modules, "pyi_splash", SimpleNamespace(close=lambda: closed.append(True))
    )

    standalone_player_module._close_startup_splash()

    assert closed == [True]


def test_executable_does_not_replace_existing_file_when_overwrite_is_false(
    tmp_path, monkeypatch
):
    destination = tmp_path / "player.exe"
    destination.write_bytes(b"existing-executable")
    monkeypatch.setattr(sys, "platform", "win32")

    with pytest.raises(FileExistsError):
        standalone_module.save_player_executable(
            None, None, destination, overwrite=False
        )

    assert destination.read_bytes() == b"existing-executable"


def test_executable_keeps_precomputed_animation_in_scene_archive(tmp_path, monkeypatch):
    model = make_model()
    adapter = _ViewerModel(model, opensim.State(model.getWorkingState()))
    motion = MotionData(adapter, make_motion())
    runtime = tmp_path / "runtime.exe"
    runtime.write_bytes(b"generic-player-runtime")

    @contextmanager
    def fake_runtime(*, clean=False):
        yield runtime

    def fake_make_frames(model, data, output, **options):
        assert data is motion
        assets = output / "assets"
        assets.mkdir()
        return {
            "has_motion": True,
            "frames": [{"time": 0.0}, {"time": 1.0}],
        }

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(standalone_module, "_player_runtime", fake_runtime)
    monkeypatch.setattr(standalone_module, "_make_frames", fake_make_frames)
    destination = tmp_path / "animated-player.exe"

    standalone_module.save_player_executable(adapter, motion, destination)

    with zipfile.ZipFile(destination) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["has_motion"] is True
    assert [frame["time"] for frame in manifest["frames"]] == [0.0, 1.0]


def test_standalone_player_selects_nearest_frame_with_binary_search():
    player = Player.__new__(Player)
    player.times = [0.0, 1.0, 2.0]

    assert player._frame_at_time(-1.0) == 0
    assert player._frame_at_time(0.5) == 0
    assert player._frame_at_time(0.8) == 1
    assert player._frame_at_time(3.0) == 2


def test_standalone_player_frame_step_pauses_and_clamps():
    player = Player.__new__(Player)
    player.has_motion = True
    player.frames = [{"time": 0.0}, {"time": 1.0}, {"time": 2.0}]
    player.frame_index = 1
    player.current_time = 1.0
    player.start_time = 0.0
    player.end_time = 2.0
    player.speed = 2.0
    player.playing = True
    button_updates = []
    player.play_button = SimpleNamespace(
        config=lambda **options: button_updates.append(options)
    )
    player.slider_var = SimpleNamespace(set=lambda value: setattr(player, "slider", value))
    player.time_var = SimpleNamespace(set=lambda value: setattr(player, "label", value))
    applied_frames = []

    def apply_frame(index):
        applied_frames.append(index)
        player.frame_index = index
        player.current_time = float(player.frames[index]["time"])

    player._apply_frame = apply_frame

    player._step_frame(1)

    assert applied_frames == [2]
    assert player.playing is False
    assert button_updates == [{"text": "▶"}]
    assert player.slider == 1000.0
    assert player.label.startswith("  2.00 / 2.00 s")

    player._step_frame(1)
    assert applied_frames == [2, 2]


def test_standalone_player_playback_hotkeys_dispatch_to_controls():
    player = Player.__new__(Player)
    actions = []
    player._play_pause = lambda: actions.append("toggle")
    player._step_frame = lambda direction: actions.append(direction)

    assert player._handle_playback_key("space") is True
    assert player._handle_playback_key("Left") is True
    assert player._handle_playback_key("Right") is True
    assert player._handle_playback_key("Escape") is False
    assert actions == ["toggle", -1, 1]


def test_standalone_player_schedules_only_remaining_frame_interval(monkeypatch):
    player = Player.__new__(Player)
    player._last_tick = 10.0
    player.has_motion = False
    player.fps = 30.0
    delays = []
    player.root = SimpleNamespace(
        after=lambda delay, _callback: delays.append(delay)
    )
    player.interactor = SimpleNamespace(ProcessEvents=lambda: None)
    player._show_hover = lambda: None
    monotonic_values = iter((10.01, 10.035))
    monkeypatch.setattr(
        standalone_player_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    player._tick()

    assert delays == [8]


def test_standalone_player_exports_animation_with_ffmpeg_only(tmp_path, monkeypatch):
    player = Player.__new__(Player)
    frames_sent = []
    writer_options = {}
    writer = SimpleNamespace(
        send=frames_sent.append,
        close=lambda: frames_sent.append("closed"),
    )
    monkeypatch.setattr(
        standalone_player_module.filedialog,
        "asksaveasfilename",
        lambda **_options: str(tmp_path / "animation.mp4"),
    )

    def fake_write_frames(path, size, **options):
        writer_options.update(path=path, size=size, **options)
        return writer

    monkeypatch.setattr(
        standalone_player_module.imageio_ffmpeg, "write_frames", fake_write_frames
    )
    player.current_time = 0.5
    player.playing = True
    player.fps = 24
    player.frames = [{}, {}]
    player.window = SimpleNamespace(GetSize=lambda: (160, 120))
    player.root = SimpleNamespace(config=lambda **_options: None, update=lambda: None)
    player._apply_frame = lambda _index: None
    player._capture = lambda: standalone_player_module.np.zeros(
        (2, 2, 3), dtype="uint8"
    )
    player._frame_at_time = lambda _time: 0

    player._save_animation()

    assert writer_options == {
        "path": str(tmp_path / "animation.mp4"),
        "size": (160, 120),
        "fps": 24,
        "codec": "libx264",
        "quality": 8,
        "macro_block_size": 1,
    }
    assert frames_sent[0] is None
    assert frames_sent[1:] == [bytes(12), bytes(12), "closed"]
    assert player.playing is True


def test_player_runtime_builds_once_and_reuses_the_cached_executable(
    tmp_path, monkeypatch
):
    cache = tmp_path / "cache"
    builds = []

    def fake_build_runtime(work_dir):
        executable = work_dir / "player-runtime.exe"
        executable.write_bytes(b"runtime")
        builds.append(executable)
        return executable

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(standalone_module, "_runtime_cache_directory", lambda: cache)
    monkeypatch.setattr(standalone_module, "_build_runtime", fake_build_runtime)

    with standalone_module._player_runtime() as first_runtime:
        assert first_runtime.read_bytes() == b"runtime"
    with standalone_module._player_runtime() as second_runtime:
        assert second_runtime.read_bytes() == b"runtime"

    assert first_runtime == second_runtime == cache / "player-runtime.exe"
    assert len(builds) == 1


def test_player_runtime_clean_rebuilds_existing_cached_executable(
    tmp_path, monkeypatch
):
    cache = tmp_path / "cache"
    cache.mkdir()
    runtime = cache / "player-runtime.exe"
    runtime.write_bytes(b"old-runtime")
    builds = []

    def fake_build_runtime(work_dir):
        executable = work_dir / "player-runtime.exe"
        executable.write_bytes(b"rebuilt-runtime")
        builds.append(executable)
        return executable

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(standalone_module, "_runtime_cache_directory", lambda: cache)
    monkeypatch.setattr(standalone_module, "_build_runtime", fake_build_runtime)

    with standalone_module._player_runtime(clean=True) as rebuilt_runtime:
        assert rebuilt_runtime == runtime
        assert rebuilt_runtime.read_bytes() == b"rebuilt-runtime"

    assert len(builds) == 1


def test_player_runtime_falls_back_to_temporary_directory_and_cleans_it(
    tmp_path, monkeypatch
):
    cache = tmp_path / "cache"
    original_temporary_directory = standalone_module.tempfile.TemporaryDirectory

    def reject_cache_directory(*, prefix, dir=None):
        if dir is not None:
            raise PermissionError("cache is not writable")
        return original_temporary_directory(prefix=prefix, dir=dir)

    def fake_build_runtime(work_dir):
        executable = work_dir / "player-runtime.exe"
        executable.write_bytes(b"runtime")
        return executable

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(standalone_module, "_runtime_cache_directory", lambda: cache)
    monkeypatch.setattr(
        standalone_module.tempfile, "TemporaryDirectory", reject_cache_directory
    )
    monkeypatch.setattr(standalone_module, "_build_runtime", fake_build_runtime)

    with pytest.warns(RuntimeWarning, match="temporary directory"):
        with standalone_module._player_runtime() as runtime:
            temporary_directory = runtime.parent
            assert runtime.read_bytes() == b"runtime"
            assert temporary_directory.exists()

    assert not temporary_directory.exists()


def test_standalone_runtime_extracts_the_appended_scene_archive(tmp_path, monkeypatch):
    executable_path = tmp_path / "player.exe"
    archive_path = tmp_path / "assets.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", '{"title":"portable"}')
    with executable_path.open("wb") as executable:
        executable.write(b"runtime")
        executable.write(archive_path.read_bytes())
        executable.write(
            struct.pack(
                "!8sQ",
                standalone_module._OVERLAY_MAGIC,
                archive_path.stat().st_size,
            )
        )
    extracted = tmp_path / "bundle"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(extracted), raising=False)
    monkeypatch.setattr(sys, "executable", str(executable_path))

    _prepare_assets()

    assert json.loads((extracted / "assets" / "manifest.json").read_text()) == {
        "title": "portable"
    }
