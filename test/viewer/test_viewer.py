from __future__ import annotations

import json
import zipfile

import pytest

opensim = pytest.importorskip("opensim")

from opensim_frameworks.framework import Framework
from opensim_frameworks.models.model import Model
from opensim_frameworks.viewer import FrameworkViewer
from opensim_frameworks.viewer import viewer as viewer_module
from opensim_frameworks.viewer.animation import _ViewerModel
from opensim_frameworks.viewer.visualizer import VTKVisualizer


def make_model() -> Model:
    model = Model()
    body = opensim.Body("body", 1.0, opensim.Vec3(0), opensim.Inertia(1))
    model.addBody(body)
    joint = opensim.PinJoint(
        "root",
        model.getGround(),
        opensim.Vec3(0),
        opensim.Vec3(0),
        body,
        opensim.Vec3(0),
        opensim.Vec3(0),
    )
    joint.upd_coordinates(0).setName("rotation")
    model.addJoint(joint)
    model.initSystem()
    return model


def make_trajectory(model):
    trajectory = opensim.StatesTrajectory()
    coordinate = model.getCoordinateSet().get("rotation")
    for time, value in ((0.0, 0.0), (1.0, 1.0), (2.0, 0.0)):
        state = opensim.State(model.state)
        coordinate.setValue(state, value, False)
        state.setTime(time)
        model.realizePosition(state)
        trajectory.append(state)
    return trajectory


def test_framework_viewer_normalizes_inputs_and_exposes_setters(monkeypatch):
    model = make_model()
    trajectory = make_trajectory(model)
    viewer = FrameworkViewer(
        model,
        trajectories=trajectory,
        sampling_frequency_hz=2.0,
    )
    assert isinstance(viewer.framework, Framework)
    assert viewer.framework["model"] is not model
    assert viewer.framework.states_trajectory.getSize() == 3
    assert viewer.trajectories is trajectory
    assert viewer.sampling_frequency_hz == 2.0

    viewer.set_sampling_frequency_hz(4.0)
    assert viewer.sampling_frequency_hz == 4.0
    with pytest.raises(ValueError, match="finite and positive"):
        viewer.set_sampling_frequency_hz(0)
    with pytest.raises(ValueError, match="only be supplied"):
        FrameworkViewer(Framework(model=model), trajectories=trajectory)

    framework = Framework(model=make_model())
    viewer.set_framework(framework)
    assert viewer.framework is framework
    assert viewer.trajectories is None


def test_show_sampling_frequency_overrides_constructor_value(monkeypatch):
    model = make_model()
    viewer = FrameworkViewer(
        model,
        trajectories=make_trajectory(model),
        sampling_frequency_hz=2.0,
    )
    captured = {}
    expected = object()

    def fake_start_player(adapter, motion, *, loop, fps):
        captured.update(adapter=adapter, motion=motion, loop=loop, fps=fps)
        return expected

    monkeypatch.setattr(viewer_module, "start_player", fake_start_player)
    assert viewer.show(sampling_frequency_hz=4.0, loop=True) is expected
    assert isinstance(captured["adapter"], _ViewerModel)
    assert captured["motion"].getNumRows() == 9
    assert captured["loop"] is True
    assert captured["fps"] == 4.0


def test_snapshot_interpolates_recorded_and_sampled_times_and_rejects_others():
    model = make_model()
    viewer = FrameworkViewer(
        model,
        trajectories=make_trajectory(model),
        sampling_frequency_hz=2.0,
    )
    state = viewer._snapshot_state(frame=None, time_s=1.0)
    assert state.getTime() == pytest.approx(1.0)
    assert viewer.framework.composite.getCoordinateSet().get("rotation").getValue(
        state
    ) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="match a recorded state or a sampled time"):
        viewer._snapshot_state(frame=None, time_s=0.3)
    with pytest.raises(ValueError, match="not both"):
        viewer._snapshot_state(frame=0, time_s=0.0)


def test_static_framework_disables_animation_and_limits_snapshot_time():
    viewer = FrameworkViewer(make_model())
    assert viewer._motion_table() is None
    with pytest.raises(ValueError, match="at least two recorded states"):
        viewer.save_animation("motion.mp4")
    with pytest.raises(ValueError, match="unavailable"):
        viewer._snapshot_state(frame=None, time_s=1.0)


def test_save_animation_uses_interpolated_framework_trajectory(monkeypatch, tmp_path):
    model = make_model()
    viewer = FrameworkViewer(
        model,
        trajectories=make_trajectory(model),
        sampling_frequency_hz=4.0,
    )
    captured = {}

    def fake_render(model, table, path, **options):
        captured.update(model=model, table=table, path=path, options=options)
        return path

    monkeypatch.setattr(viewer_module, "_render_animation", fake_render)
    destination = tmp_path / "motion.mp4"
    assert viewer.save_animation(destination) == destination
    assert captured["table"].getNumRows() == 9
    assert captured["path"] == destination
    assert isinstance(captured["model"], opensim.Model)


def test_save_framework_writes_osf_manifest_models_and_honors_overwrite(
    tmp_path,
):
    viewer = FrameworkViewer(Framework(model=make_model()))
    destination = tmp_path / "framework.osf"
    assert viewer.save_framework(destination) == destination

    with zipfile.ZipFile(destination) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["format"] == "opensim-framework"
        assert manifest["members"][0]["name"] == "model"
        assert manifest["members"][0]["config"] == {}
        assert manifest["composite_model"] == "composite/model.osim"
        assert "models/model/model.osim" in archive.namelist()
        assert "composite/model.osim" in archive.namelist()

    with pytest.raises(FileExistsError):
        viewer.save_framework(destination, overwrite=False)


def test_save_framework_delegates_to_framework_to_osf(monkeypatch, tmp_path):
    framework = Framework(model=make_model())
    viewer = FrameworkViewer(framework)
    destination = tmp_path / "delegated.osf"
    captured = {}

    def fake_to_osf(self, file, *, overwrite=True):
        captured.update(framework=self, file=file, overwrite=overwrite)
        return file

    monkeypatch.setattr(Framework, "to_osf", fake_to_osf)

    assert viewer.save_framework(destination, overwrite=False) == destination
    assert captured == {
        "framework": framework,
        "file": destination,
        "overwrite": False,
    }


def test_from_osf_restores_models_and_synchronized_trajectories(tmp_path):
    model = make_model()
    framework = Framework(model=model)
    coordinate = model.getCoordinateSet().get("rotation")
    for time, value in ((0.0, 0.0), (1.0, 1.0), (2.0, 0.0)):
        state = opensim.State(model.state)
        coordinate.setValue(state, value, False)
        model.state = state
        framework.record_state(time)

    destination = tmp_path / "animated.osf"
    assert framework.to_osf(destination) == destination
    with pytest.raises(FileExistsError):
        framework.to_osf(destination, overwrite=False)
    loaded = Framework.from_osf(destination)

    assert loaded.state_times == (0.0, 1.0, 2.0)
    assert loaded.states_trajectories["model"].getSize() == 3
    assert loaded.states_trajectory.getSize() == 3
    restored_coordinate = loaded["model"].getCoordinateSet().get("rotation")
    restored_values = [
        restored_coordinate.getValue(loaded.states_trajectories["model"].get(index))
        for index in range(3)
    ]
    assert restored_values == pytest.approx([0.0, 1.0, 0.0])


def test_from_osf_rejects_archive_path_traversal(tmp_path):
    destination = tmp_path / "unsafe.osf"
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr("../outside.txt", "not extracted")

    with pytest.raises(ValueError, match="invalid OSF archive path"):
        Framework.from_osf(destination)
    assert not (tmp_path.parent / "outside.txt").exists()


def test_overlay_controls_report_unavailable_layers_for_plain_models():
    visualizer = VTKVisualizer(_ViewerModel(make_model()), offscreen=True)
    try:
        assert not visualizer.has_vision_cone_and_axis
        assert not visualizer.has_reachability_regions
        visualizer.set_vision_cone_visible(False)
        visualizer.set_reachability_visible(False)
    finally:
        visualizer.close()


def test_visualizer_skips_geometry_marked_invisible():
    model = make_model()
    body = model.getBodySet().get("body")
    geometry = opensim.Brick(opensim.Vec3(0.1, 0.2, 0.3))
    geometry.upd_Appearance().set_visible(False)
    body.attachGeometry(geometry)
    model.finalizeConnections()
    model.initSystem()

    visualizer = VTKVisualizer(_ViewerModel(model), offscreen=True)
    try:
        assert not visualizer._actor_bodies
    finally:
        visualizer.close()
