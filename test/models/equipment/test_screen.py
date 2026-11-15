import math

import numpy as np
import pytest

from opensim_frameworks.bodies import Box, make_body
from opensim_frameworks.models.equipment import Screen
from opensim_frameworks.utils.opensim import import_opensim, vec3_to_numpy

opensim = import_opensim()


def test_screen_is_an_opensim_body_with_generated_mesh():
    screen = Screen(0.06, 0.11, 0.006, 0.1)

    assert isinstance(screen, opensim.Body)
    assert isinstance(screen.mesh, opensim.Mesh)
    assert screen.getName() == "screen"
    assert [
        screen.mesh.get_scale_factors().get(index) for index in range(3)
    ] == pytest.approx((0.006, 0.11, 0.06))
    assert screen.getMass() == pytest.approx(0.1)
    assert screen.width == pytest.approx(0.06)
    assert screen.height == pytest.approx(0.11)
    assert screen.depth == pytest.approx(0.006)
    assert screen.corners.shape == (8, 3)
    assert screen.samples(2).shape == (4, 3)


def test_from_existing_body_fits_mesh_and_copies_body_properties():
    source, _ = make_body(
        "display",
        [
            Box(
                size=(0.006, 0.11, 0.06),
                orientation=(0.0, 0.0, math.radians(-50.0)),
            )
        ],
        density=1000.0,
    )

    screen = Screen.from_existing_body(source)

    assert isinstance(screen, Screen)
    assert isinstance(screen, opensim.Body)
    assert screen.getName() == source.getName()
    assert screen.width_mm == pytest.approx(60.0)
    assert screen.height_mm == pytest.approx(110.0)
    assert screen.depth_mm == pytest.approx(6.0)
    assert screen.inclination_deg == pytest.approx(40.0)
    assert screen.getMass() == pytest.approx(source.getMass())
    assert vec3_to_numpy(screen.getMassCenter()) == pytest.approx(
        vec3_to_numpy(source.getMassCenter())
    )
    assert vec3_to_numpy(screen.getInertia().getMoments()) == pytest.approx(
        vec3_to_numpy(source.getInertia().getMoments())
    )
    assert vec3_to_numpy(screen.getInertia().getProducts()) == pytest.approx(
        vec3_to_numpy(source.getInertia().getProducts())
    )
