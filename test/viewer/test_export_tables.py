"""Tests for the xlsx table export (viewer Export button and ``save_tables``)."""

from __future__ import annotations

import math

import pytest

opensim = pytest.importorskip("opensim")
openpyxl = pytest.importorskip("openpyxl")

from opensim_frameworks.framework import Framework
from opensim_frameworks.viewer import VTKViewer
from opensim_frameworks.viewer.export import export_tables_to_xlsx, xlsx_export_paths
from opensim_frameworks.viewer.player import MotionData


def make_arm(name: str) -> opensim.Model:
    """Return a one-body pendulum (pin ``hinge``) with one marker ``tip``."""
    model = opensim.Model()
    model.setName(name)
    body = opensim.Body("link", 1.0, opensim.Vec3(0), opensim.Inertia(1))
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


def make_motion(model: opensim.Model, degrees: list[float]) -> opensim.TimeSeriesTable:
    table = opensim.TimeSeriesTable()
    table.setColumnLabels(opensim.StdVectorString(["angle"]))
    for index, value in enumerate(degrees):
        table.appendRow(float(index), opensim.RowVector([value]))
    table.addTableMetaDataString("inDegrees", "yes")
    return table


def read_sheet(path, sheet):
    workbook = openpyxl.load_workbook(path, read_only=True)
    return list(workbook[sheet].iter_rows(values_only=True))


def test_single_model_rows_columns_and_values(tmp_path):
    model = make_arm("arm")
    files = VTKViewer().save_tables(model, tmp_path / "out", make_motion(model, [0, 90, 180]))
    assert [file.name for file in files] == ["arm.xlsx"]
    markers = read_sheet(files[0], "markers")
    joints = read_sheet(files[0], "joints")
    assert markers[0] == ("time [s]", "tip.X [m]", "tip.Y [m]", "tip.Z [m]")
    assert joints[0] == ("time [s]", "angle [deg]")
    assert len(markers) == len(joints) == 4
    # Row 1 is 90 degrees about Z: the marker at (1, 0, 0) moves to (0, 1, 0).
    assert joints[2] == (1.0, pytest.approx(90.0))
    assert markers[2][1:] == (
        pytest.approx(0.0, abs=1e-9),
        pytest.approx(1.0),
        pytest.approx(0.0, abs=1e-9),
    )


def test_no_motion_gives_one_row_at_time_zero(tmp_path):
    files = VTKViewer().save_tables(make_arm("arm"), tmp_path)
    assert [row[0] for row in read_sheet(files[0], "joints")[1:]] == [0.0]


def test_framework_writes_one_file_per_member_with_original_names(tmp_path):
    framework = Framework()
    framework["first"] = make_arm("first")
    framework["second"] = make_arm("second")
    files = framework.save_tables(tmp_path)
    assert sorted(file.name for file in files) == ["first.xlsx", "second.xlsx"]
    for file in files:
        # The colliding names are prefixed in the composite, not in the files.
        assert read_sheet(file, "markers")[0][1] == "tip.X [m]"
        assert read_sheet(file, "joints")[0][1] == "angle [deg]"


def test_overwrite_policy(tmp_path):
    model = make_arm("arm")
    viewer = VTKViewer()
    viewer.save_tables(model, tmp_path)
    with pytest.raises(FileExistsError):
        viewer.save_tables(model, tmp_path)
    assert viewer.save_tables(model, tmp_path, overwrite=True)


def test_export_does_not_change_the_given_model_pose(tmp_path):
    model = make_arm("arm")
    state = model.initSystem()
    VTKViewer().save_tables(model, tmp_path, make_motion(model, [0, 90]))
    assert math.isclose(model.getCoordinateSet().get("angle").getValue(state), 0.0)


def test_xlsx_export_paths_rejects_duplicate_file_names(tmp_path):
    framework = Framework()
    framework["a b"] = make_arm("one")
    framework["a/b"] = make_arm("two")
    composite, path_map = framework.merge_to_model(return_mapping=True)
    with pytest.raises(ValueError):
        xlsx_export_paths(composite, tmp_path, path_map=path_map)


def test_export_function_accepts_motion_data_directly(tmp_path):
    model = make_arm("arm")
    data = MotionData(model, make_motion(model, [0, 45]))
    files = export_tables_to_xlsx(model, data, tmp_path)
    assert len(read_sheet(files[0], "joints")) == 3
