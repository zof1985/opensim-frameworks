"""Closed-form kinematic chains of OpenSim joints."""

from __future__ import annotations

import numpy as np
import pytest

import opensim_frameworks as osf

opensim = pytest.importorskip("opensim")


@pytest.fixture(scope="module")
def user() -> osf.User:
    return osf.User("M", height_cm=175)


def _randomize(user: osf.User, names, rng) -> None:
    coordinates = user.getCoordinateSet()
    for name in names:
        coordinate = coordinates.get(name)
        coordinate.setLocked(user.state, False)
        coordinate.setValue(
            user.state,
            rng.uniform(coordinate.getRangeMin(), coordinate.getRangeMax()),
            False,
        )


@pytest.mark.parametrize("end", ["hand_r", "scapula_l", "humerus_l"])
def test_chain_reproduces_the_model_at_random_poses(user, end):
    chain = osf.KinematicChain(user, "torso", end)
    rng = np.random.default_rng(0)

    for _ in range(10):
        _randomize(user, chain.coordinate_names, rng)
        position_error, rotation_error = chain.max_error(user.state)
        assert position_error < 1e-9
        assert rotation_error < 1e-9


def test_chain_lists_bodies_and_coordinates_from_root_to_end(user):
    chain = osf.KinematicChain(user, "torso", "hand_r")

    assert chain.body_names[0] == "torso"
    assert chain.body_names[-1] == "hand_r"
    assert "elbow_flex_r" in chain.coordinate_names
    assert "sc_protraction_r" in chain.coordinate_names
    assert "elbow_flex_l" not in chain.coordinate_names


def test_batch_poses_match_row_by_row_evaluation(user):
    chain = osf.KinematicChain(user, "torso", "hand_r")
    values = np.random.default_rng(1).uniform(-1.0, 1.0, (5, len(chain.coordinate_names)))

    rotations, positions = chain.poses(values)

    assert rotations.shape == (5, len(chain.body_names), 3, 3)
    assert positions.shape == (5, len(chain.body_names), 3)
    for index in range(5):
        single_rotations, single_positions = chain.poses(values[index : index + 1])
        np.testing.assert_allclose(rotations[index], single_rotations[0])
        np.testing.assert_allclose(positions[index], single_positions[0])


def test_chain_rejects_invalid_arguments(user):
    with pytest.raises(ValueError):
        osf.KinematicChain(user, "hand_r", "torso")  # not a descendant
    with pytest.raises(ValueError):
        osf.KinematicChain(user, "torso", "no_such_body")
    chain = osf.KinematicChain(user, "torso", "hand_r")
    with pytest.raises(ValueError):
        chain.poses(np.zeros((2, 3)))
