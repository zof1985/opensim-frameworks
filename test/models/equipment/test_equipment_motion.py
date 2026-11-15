import importlib
from unittest.mock import MagicMock

import numpy as np
import pytest

from opensim_frameworks import Equipment
from opensim_frameworks.models.equipment.equipment import opensim
from opensim_frameworks.viewer.player import MotionData, _apply_time

viewer_module = importlib.import_module("opensim_frameworks.viewer")


def patch_viewer(monkeypatch, viewer):
    captured = {}

    def create_viewer(model, *, sampling_frequency_hz):
        captured["model"] = model
        captured["sampling_frequency_hz"] = sampling_frequency_hz
        return viewer

    monkeypatch.setattr(viewer_module, "FrameworkViewer", create_viewer)
    return captured


class MovingEquipment(Equipment):
    def __init__(self, *, coupled=False, prescribed=False):
        opensim.Model.__init__(self)
        for name, joint_type, limits in (
            ("rotation", opensim.PinJoint, (-1.0, 1.0)),
            ("translation", opensim.SliderJoint, (0.0, 0.3)),
        ):
            body = opensim.Body(name, 1.0, opensim.Vec3(0), opensim.Inertia(1))
            self.addBody(body)
            joint = joint_type(
                name + "_joint",
                self.getGround(),
                opensim.Vec3(0),
                opensim.Vec3(0),
                body,
                opensim.Vec3(0),
                opensim.Vec3(0),
            )
            coordinate = joint.upd_coordinates(0)
            coordinate.setName(name)
            coordinate.setRangeMin(limits[0])
            coordinate.setRangeMax(limits[1])
            if prescribed and name == "rotation":
                coordinate.setPrescribedFunction(opensim.LinearFunction(0.1, 0.0))
                coordinate.setDefaultIsPrescribed(True)
            self.addJoint(joint)
        if coupled:
            coupler = opensim.CoordinateCouplerConstraint()
            independent = opensim.ArrayStr()
            independent.append("rotation")
            coupler.setIndependentCoordinateNames(independent)
            coupler.setDependentCoordinateName("translation")
            coupler.setFunction(opensim.LinearFunction(0.1, 0.15))
            self.addConstraint(coupler)
        self.state = self.initSystem()
        self.realizePosition(self.state)

    def _build(self):
        pass

    def fit_user(self, *args, **kwargs):
        pass


def test_show_motion_animates_all_coordinates_without_mutating_original(monkeypatch):
    model = MovingEquipment()
    coordinate = model.getCoordinateSet().get("rotation")
    coordinate.set_locked(True)
    coordinate.setLocked(model.state, True)
    viewer = MagicMock()
    captured = patch_viewer(monkeypatch, viewer)

    model.show_motion(duration=1.0, fps=3.0)

    kwargs = viewer.show.call_args.kwargs
    animated = captured["model"]
    table = kwargs["motion"]
    assert animated is not model
    assert kwargs["loop"] is True
    assert captured["sampling_frequency_hz"] == 3.0
    assert list(table.getColumnLabels()) == ["rotation", "translation"]
    assert table.getTableMetaDataAsString("inDegrees") == "no"
    np.testing.assert_allclose(
        table.getDependentColumn("rotation").to_numpy(), [-1, 0, 1, 0, -1]
    )
    np.testing.assert_allclose(
        table.getDependentColumn("translation").to_numpy(), [0, 0.15, 0.3, 0.15, 0]
    )
    _apply_time(animated, MotionData(animated, table), 0.5)
    assert animated.getCoordinateSet().get("rotation").getValue(
        animated.state
    ) == pytest.approx(1.0)
    assert animated.getCoordinateSet().get("translation").getValue(
        animated.state
    ) == pytest.approx(0.3)
    assert coordinate.get_locked() and coordinate.getLocked(model.state)
    viewer.show.return_value.wait_closed.assert_called_once()


def test_show_motion_respects_coordinate_coupling(monkeypatch):
    model = MovingEquipment(coupled=True)
    viewer = MagicMock()
    captured = patch_viewer(monkeypatch, viewer)

    model.show_motion(duration=1.0, fps=4.0)

    table = viewer.show.call_args.kwargs["motion"]
    rotation = table.getDependentColumn("rotation").to_numpy()
    translation = table.getDependentColumn("translation").to_numpy()
    np.testing.assert_allclose(translation, 0.1 * rotation + 0.15, atol=1e-8)
    animated = captured["model"]
    _apply_time(animated, MotionData(animated, table), 0.5)
    assert animated.getCoordinateSet().get("translation").getValue(
        animated.state
    ) == pytest.approx(0.25)


def test_show_motion_sweeps_prescribed_coordinate_on_copy(monkeypatch):
    model = MovingEquipment(prescribed=True)
    viewer = MagicMock()
    captured = patch_viewer(monkeypatch, viewer)

    model.show_motion(duration=1.0, fps=4.0)

    animated = captured["model"]
    original = model.getCoordinateSet().get("rotation")
    coordinate = animated.getCoordinateSet().get("rotation")
    assert original.isPrescribed(model.state)
    assert not coordinate.isPrescribed(animated.state)
    table = viewer.show.call_args.kwargs["motion"]
    np.testing.assert_allclose(
        table.getDependentColumn("rotation").to_numpy(), [-1, 0, 1, 0, -1]
    )


@pytest.mark.parametrize("duration,fps", [(0, 30), (1, 0), (float("nan"), 30)])
def test_show_motion_rejects_invalid_timing(duration, fps):
    with pytest.raises(ValueError):
        MovingEquipment().show_motion(duration=duration, fps=fps)
