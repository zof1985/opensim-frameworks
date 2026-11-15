from __future__ import annotations

import copy

import pytest

opensim = pytest.importorskip("opensim")

from opensim_frameworks import Framework, FrameworkModel, row_to_state, state_to_row


def make_model(name: str = "model"):
    model = FrameworkModel()
    model.setName(name)
    return model


def make_single_body_model() -> opensim.Model:
    model = FrameworkModel()
    body = opensim.Body("body", 1.0, opensim.Vec3(0), opensim.Inertia(1))
    model.addBody(body)
    model.addJoint(
        opensim.WeldJoint(
            "root",
            model.getGround(),
            opensim.Vec3(0),
            opensim.Vec3(0),
            body,
            opensim.Vec3(0),
            opensim.Vec3(0),
        )
    )
    model.state = model.initSystem()
    return model


def make_pin_model() -> opensim.Model:
    model = FrameworkModel()
    body = opensim.Body("body", 1.0, opensim.Vec3(0), opensim.Inertia(1))
    model.addBody(body)
    model.addJoint(
        opensim.PinJoint(
            "root",
            model.getGround(),
            opensim.Vec3(0),
            opensim.Vec3(0),
            body,
            opensim.Vec3(0),
            opensim.Vec3(0),
        )
    )
    model.state = model.initSystem()
    return model


def test_framework_is_a_dict_of_native_models():
    model = make_model()

    framework = Framework({"user": model})

    assert isinstance(framework, dict)
    assert isinstance(framework["user"], opensim.Model)
    assert framework.user is model


def test_framework_accepts_assignment_add_and_attribute_access():
    first = make_model("first")
    second = make_model("second")
    framework = Framework()

    framework["first"] = first
    assert framework.add(second, "second") is second
    assert framework.first is first
    assert framework.second is second
    assert "first" in dir(framework)


def test_framework_rejects_non_models_and_invalid_names():
    framework = Framework()

    with pytest.raises(TypeError, match="FrameworkModel"):
        framework["body"] = opensim.Body("body", 1.0, opensim.Vec3(0), opensim.Inertia(1))
    with pytest.raises(TypeError, match="str"):
        framework[1] = make_model()
    with pytest.raises(ValueError, match="identifier"):
        framework["not a name"] = make_model()
    with pytest.raises(ValueError, match="reserved"):
        framework["items"] = make_model()


def test_framework_does_not_store_one_model_under_multiple_names():
    model = make_model()
    framework = Framework({"user": model})

    with pytest.raises(ValueError, match="multiple names"):
        framework["other"] = model
    assert list(framework) == ["user"]


def test_update_is_atomic_and_validates_all_entries():
    framework = Framework({"existing": make_model("existing")})
    replacement = make_model("replacement")
    invalid = opensim.Body("body", 1.0, opensim.Vec3(0), opensim.Inertia(1))

    with pytest.raises(TypeError, match="FrameworkModel"):
        framework.update({"new": replacement, "invalid": invalid})

    assert list(framework) == ["existing"]
    assert framework["existing"].getName() == "existing"


def test_framework_mutation_methods_and_attribute_assignment():
    first = make_model("first")
    second = make_model("second")
    framework = Framework()

    framework.update({"first": first})
    framework |= {"second": second}
    framework.setdefault("first", make_model("unused"))
    assert framework.first is first
    assert framework.pop("second") is second

    framework.third = make_model("third")
    assert framework["third"].getName() == "third"
    del framework.third
    assert "third" not in framework

    framework.clear()
    assert framework == {}


def test_framework_copies_are_independent_containers():
    model = make_model("source")
    framework = Framework({"source": model})

    shallow = copy.copy(framework)
    deep = copy.deepcopy(framework)

    assert shallow["source"] is model
    assert deep["source"] is not model
    assert isinstance(deep["source"], opensim.Model)
    assert deep["source"].getName() == "source"


def test_merge_to_model_clones_members_and_resolves_component_name_collisions():
    first = make_single_body_model()
    second = make_single_body_model()
    framework = Framework({"first": first, "second": second})

    merged = framework.merge_to_model()
    state = merged.initSystem()

    assert type(merged) is opensim.Model
    assert merged.getBodySet().getSize() == 2
    assert merged.getJointSet().getSize() == 2
    assert merged.getBodySet().contains("body")
    assert merged.getBodySet().contains("second_body")
    assert merged.getTotalMass(state) == pytest.approx(2.0)
    assert first.getBodySet().getSize() == second.getBodySet().getSize() == 1


def test_merge_to_model_carries_member_meshes(tmp_path):
    from opensim_frameworks.bodies.shapes import MESHES_DIR

    member = make_single_body_model()
    member.geometry_directories = (tmp_path,)
    merged = Framework({"first": member, "other": make_single_body_model()}).merge_to_model()

    assert tmp_path in merged.geometry_directories
    assert MESHES_DIR in merged.geometry_directories


def test_composite_is_a_fresh_snapshot_with_coordinate_path_mappings():
    member = make_pin_model()
    framework = Framework({"arm": member})

    first = framework.composite
    framework.initSystem()
    second = framework.composite

    assert first is not second
    assert first.getBodySet().getSize() == 1
    assert member.getBodySet().getSize() == 1
    source_path = member.getCoordinateSet().get(0).getAbsolutePathString()
    composite_path = framework.path_map[("arm", source_path)]
    assert framework.reverse_path_map[composite_path] == ("arm", source_path)
    assert first.getComponent(composite_path) is not None


def test_update_state_copies_composite_coordinates_and_speeds_to_members():
    first = make_pin_model()
    second = make_pin_model()
    framework = Framework({"first": first, "second": second})
    framework.initSystem()
    state = framework.state
    composite = framework.composite

    source_path = second.getCoordinateSet().get(0).getAbsolutePathString()
    composite_path = framework.path_map[("second", source_path)]
    coordinate = opensim.Coordinate.safeDownCast(composite.getComponent(composite_path))
    assert coordinate is not None
    coordinate.setValue(state, 0.5, False)
    coordinate.setSpeedValue(state, 1.25)

    framework.update_state()

    member_coordinate = second.getCoordinateSet().get(0)
    member_state = second.getWorkingState()
    assert member_coordinate.getValue(member_state) == pytest.approx(0.5)
    assert member_coordinate.getSpeedValue(member_state) == pytest.approx(1.25)
    assert first.getCoordinateSet().get(0).getValue(first.getWorkingState()) == pytest.approx(
        0.0
    )


def test_composite_preserves_member_gravity_and_rejects_conflicts():
    member = make_single_body_model()
    member.setGravity(opensim.Vec3(0, -3.0, 0))

    composite = Framework({"member": member}).composite

    assert tuple(composite.getGravity().get(i) for i in range(3)) == pytest.approx(
        (0.0, -3.0, 0.0)
    )
    other = make_single_body_model()
    with pytest.raises(ValueError, match="same gravity"):
        Framework({"member": member, "other": other}).composite


def test_set_gravity_updates_every_member_and_invalidates_the_snapshot():
    first = make_single_body_model()
    second = make_single_body_model()
    framework = Framework({"first": first, "second": second})
    snapshot = framework.composite

    framework.set_gravity(opensim.Vec3(0, -4.0, 0))

    assert framework.gravity == pytest.approx((0.0, -4.0, 0.0))
    assert first.getGravity().get(1) == pytest.approx(-4.0)
    assert second.getGravity().get(1) == pytest.approx(-4.0)
    assert framework.composite is not snapshot
    with pytest.raises(ValueError, match="three finite values"):
        framework.set_gravity((0.0, float("nan"), 0.0))


def test_forward_integrates_and_synchronizes_the_member_state():
    member = make_pin_model()
    coordinate = member.getCoordinateSet().get(0)
    coordinate.setSpeedValue(member.state, 1.0)
    framework = Framework({"arm": member})

    state = framework.forward(0.01)

    composite_coordinate = framework.composite.getCoordinateSet().get(0)
    assert state.model is framework.composite
    assert state.getTime() == pytest.approx(0.01)
    assert coordinate.getValue(member.state) == pytest.approx(
        composite_coordinate.getValue(state)
    )
    assert coordinate.getSpeedValue(member.state) == pytest.approx(
        composite_coordinate.getSpeedValue(state)
    )
    settled_state = framework.settle_under_gravity(0.01)
    assert settled_state.model is framework.composite
    assert settled_state.getTime() == pytest.approx(0.02)


def test_forward_rejects_free_joint_before_native_manager_integration():
    model = FrameworkModel()
    body = opensim.Body("body", 1.0, opensim.Vec3(0), opensim.Inertia(1))
    model.addBody(body)
    model.addJoint(opensim.FreeJoint("free", model.getGround(), body))
    model.state = model.initSystem()
    framework = Framework({"free_body": model})

    with pytest.raises(RuntimeError, match="opensim.FreeJoint"):
        framework.forward(0.01)


def test_framework_state_keeps_its_originating_composite_model():
    framework = Framework({"first": make_pin_model()})
    state = framework.state
    owner_id = id(state.model)

    framework["second"] = make_pin_model()

    assert isinstance(state, opensim.State)
    assert id(state.model) == owner_id
    state.model.realizePosition(state)


def test_framework_state_row_round_trip_preserves_values_speeds_and_time():
    framework = Framework({"arm": make_pin_model()})
    state = framework.state
    coordinate = framework.composite.getCoordinateSet().get(0)
    coordinate.setValue(state, 0.75, False)
    coordinate.setSpeedValue(state, 1.25)
    state.setTime(0.5)

    time, labels, row = state_to_row(framework)

    coordinate.setValue(state, -0.25, False)
    coordinate.setSpeedValue(state, -0.5)
    state.setTime(0.0)
    restored = row_to_state(framework, row, labels, time)

    assert restored is framework.state
    assert restored.model is framework.composite
    assert restored.getTime() == pytest.approx(0.5)
    assert coordinate.getValue(restored) == pytest.approx(0.75)
    assert coordinate.getSpeedValue(restored) == pytest.approx(1.25)


def test_row_to_state_reorders_columns_by_state_variable_name():
    framework = Framework({"arm": make_pin_model()})
    _, labels, row = state_to_row(framework)
    reversed_row = opensim.RowVector(
        [row[index] for index in reversed(range(row.size()))]
    )

    row_to_state(framework, reversed_row, tuple(reversed(labels)), 0.25)

    assert framework.state.getTime() == pytest.approx(0.25)


def test_row_to_state_rejects_missing_state_variable_columns():
    framework = Framework({"arm": make_pin_model()})
    time, labels, row = state_to_row(framework)
    incomplete_row = opensim.RowVector(
        [row[index] for index in range(row.size() - 1)]
    )

    with pytest.raises(ValueError, match="column_labels"):
        row_to_state(framework, incomplete_row, labels[:-1], time)


def test_get_motion_table_resamples_with_cubic_spline():
    model = make_pin_model()
    framework = Framework({"arm": model})
    coordinate = model.getCoordinateSet().get(0)
    for time in range(4):
        coordinate.setValue(model.state, 0.1 * time**3, False)
        model.realizePosition(model.state)
        framework.record_state(float(time))

    table = framework.get_motion_table(sampling_frequency_hz=2.0)

    times = table.getIndependentColumn()
    assert [times[index] for index in range(table.getNumRows())] == pytest.approx(
        [index / 2 for index in range(7)]
    )
    labels = list(table.getColumnLabels())
    value_column = next(
        index for index, label in enumerate(labels) if label.endswith("/value")
    )
    values = table.getDependentColumnAtIndex(value_column)
    assert [values[index] for index in range(table.getNumRows())] == pytest.approx(
        [0.1 * time**3 for time in [index / 2 for index in range(7)]]
    )

    undersampled = framework.get_motion_table(sampling_frequency_hz=1.5)
    sample_times = undersampled.getIndependentColumn()
    assert [sample_times[index] for index in range(undersampled.getNumRows())] == pytest.approx(
        [index / 1.5 for index in range(5)]
    )


@pytest.mark.parametrize(
    "sampling_frequency_hz",
    [0.0, -1.0, float("nan"), float("inf"), True],
)
def test_get_motion_table_rejects_invalid_sampling_frequency(sampling_frequency_hz):
    with pytest.raises(ValueError, match="sampling_frequency_hz"):
        Framework().get_motion_table(sampling_frequency_hz)


def test_get_motion_table_rejects_single_state_trajectory():
    model = make_pin_model()
    framework = Framework({"arm": model})
    framework.record_state(0.0)

    with pytest.raises(ValueError, match="at least two recorded states"):
        framework.get_motion_table(30.0)
