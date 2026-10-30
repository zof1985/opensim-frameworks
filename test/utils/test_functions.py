from __future__ import annotations

import numpy as np

from opensim_frameworks.utils.functions import global_to_local, local_to_global
from opensim_frameworks.utils.functions import import_opensim

opensim = import_opensim()


def _posed_body():
    model = opensim.Model()
    body = opensim.Body(
        "test_body",
        1.0,
        opensim.Vec3(0, 0, 0),
        opensim.Inertia(1, 1, 1, 0, 0, 0),
    )
    model.addBody(body)
    model.addJoint(
        opensim.WeldJoint(
            "test_weld",
            model.getGround(),
            opensim.Vec3(1, 2, 3),
            opensim.Vec3(0, 0, np.pi / 2),
            body,
            opensim.Vec3(0, 0, 0),
            opensim.Vec3(0, 0, 0),
        )
    )
    state = model.initSystem()
    model.realizePosition(state)
    return model, body, state


def test_local_to_global_applies_frame_translation_and_rotation():
    _model, body, state = _posed_body()

    global_point = local_to_global(body, state, (1, 0, 0))

    np.testing.assert_allclose(
        [global_point.get(index) for index in range(3)],
        [1, 3, 3],
        atol=1e-12,
    )


def test_global_to_local_inverts_frame_translation_and_rotation():
    _model, body, state = _posed_body()

    local_point = global_to_local(body, state, opensim.Vec3(1, 3, 3))

    np.testing.assert_allclose(
        [local_point.get(index) for index in range(3)],
        [1, 0, 0],
        atol=1e-12,
    )
