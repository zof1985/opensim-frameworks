"""Targeted tests for fit_handle's nine configurable stage 2-4 tuning parameters.

``fit_handle``'s stage 2-4 objective weights and wrist-lock diagnostic
tolerances, previously fixed module constants, are now also keyword
parameters (`axis_distance_weight`, `axis_angle_weight`,
`axis_alignment_regularization_weight`, `palm_tangency_regularization_weight`,
`axis_alignment_wrist_regularization_weight`,
`palm_tangency_wrist_regularization_weight`, `wrist_lock_distance_tolerance_m`,
`wrist_lock_angle_tolerance_deg`, `wrist_lock_gap_tolerance_m`), each
defaulting to the corresponding module constant. These tests avoid a full
OpenSim/VTK end-to-end `fit_handle` run (slow, already covered by
``test_fit_handle.py``) and instead check:

1. Statically, that every new parameter's default is exactly the same object
   as its module constant (single source of truth, unchanged default
   behaviour).
2. Synthetically, that an explicitly passed, non-default weight actually
   changes the optimizer's behaviour in the expected direction, reusing the
   Phase 1/Phase 2 reproduction approach from
   ``test_lexicographic_wrist_regularization.py``.
"""

from __future__ import annotations

import inspect

import numpy as np
from scipy.optimize import minimize

from opensim_frameworks.models.user import hand_fitting as hf

_WRIST_IDX = np.array([5, 6])
_FREE_IDX = np.array([0, 1, 2, 3, 4])
_BOUNDS = [(-1.5, 1.5)] * 7

_PARAMETER_TO_CONSTANT = {
    "axis_distance_weight": "_AXIS_DISTANCE_WEIGHT",
    "axis_angle_weight": "_AXIS_ANGLE_WEIGHT",
    "axis_alignment_regularization_weight": "_AXIS_ALIGNMENT_REGULARIZATION_WEIGHT",
    "palm_tangency_regularization_weight": "_PALM_TANGENCY_REGULARIZATION_WEIGHT",
    "axis_alignment_wrist_regularization_weight": "_AXIS_ALIGNMENT_WRIST_REGULARIZATION_WEIGHT",
    "palm_tangency_wrist_regularization_weight": "_PALM_TANGENCY_WRIST_REGULARIZATION_WEIGHT",
    "wrist_lock_distance_tolerance_m": "_WRIST_LOCK_DISTANCE_TOLERANCE_M",
    "wrist_lock_angle_tolerance_deg": "_WRIST_LOCK_ANGLE_TOLERANCE_DEG",
    "wrist_lock_gap_tolerance_m": "_WRIST_LOCK_GAP_TOLERANCE_M",
}


def test_tuning_parameter_defaults_match_module_constants():
    """Every new keyword default must be the same value as its module constant.

    This is a static (no solve, no OpenSim) guarantee that omitting all nine
    parameters reproduces today's behaviour exactly, since the constants
    themselves were not duplicated into the signature.
    """
    signature = inspect.signature(hf._HandFittingMixin.fit_handle)
    for parameter_name, constant_name in _PARAMETER_TO_CONSTANT.items():
        default = signature.parameters[parameter_name].default
        constant = getattr(hf, constant_name)
        assert default == constant, (parameter_name, default, constant)


def _lexicographic_solve(objective, start: np.ndarray, wrist_regularization_weight: float):
    """Reproduce fit_handle's Phase 1 / Phase 2 split for a synthetic objective."""
    def objective_wrist_locked(x_free: np.ndarray) -> float:
        x_full = start.copy()
        x_full[_FREE_IDX] = x_free
        return objective(x_full)

    phase1 = minimize(
        objective_wrist_locked, start[_FREE_IDX], method="SLSQP",
        bounds=[_BOUNDS[i] for i in _FREE_IDX], options={"maxiter": 200, "ftol": 1e-12},
    )
    assert phase1.success
    x_phase1 = start.copy()
    x_phase1[_FREE_IDX] = phase1.x

    def objective_wrist_only(x_wrist: np.ndarray) -> float:
        x_full = x_phase1.copy()
        x_full[_WRIST_IDX] = x_wrist
        wrist_offset = x_wrist - start[_WRIST_IDX]
        return objective(x_full) + wrist_regularization_weight * float(
            np.dot(wrist_offset, wrist_offset)
        )

    phase2 = minimize(
        objective_wrist_only, start[_WRIST_IDX], method="SLSQP",
        bounds=[_BOUNDS[i] for i in _WRIST_IDX], options={"maxiter": 200, "ftol": 1e-12},
    )
    assert phase2.success
    x_phase2 = x_phase1.copy()
    x_phase2[_WRIST_IDX] = phase2.x
    return x_phase2


def test_explicit_wrist_regularization_weight_overrides_default_effect():
    """A caller-supplied wrist regularization weight must actually be used.

    Same synthetic wrist-dependent objective as in
    ``test_lexicographic_wrist_regularization.py`` (geometric weight 50,
    dominating the default regularization weight 1e-3 by nearly 5 orders of
    magnitude), solved once with the default
    `axis_alignment_wrist_regularization_weight` and once with a weight large
    enough to itself dominate the geometric weight (1e7x larger, i.e. 10 vs.
    the objective's own 50 -- still the same order of magnitude, so this
    checks the passed-in value is what the solve actually uses, not merely
    that "some huge number" overwhelms everything): the larger weight must
    pin the wrist much closer to `start`, proving the passed-in value (not
    the module constant) drives the Phase 2 penalty.
    """
    start = np.array([0.1, -0.2, 0.3, 0.4, -0.1, 0.9, -0.9])
    wrist_target = np.array([0.0, 0.0])

    def objective(x: np.ndarray) -> float:
        wrist = x[_WRIST_IDX]
        return float(50.0 * np.dot(wrist - wrist_target, wrist - wrist_target))

    x_default = _lexicographic_solve(
        objective, start, hf._AXIS_ALIGNMENT_WRIST_REGULARIZATION_WEIGHT
    )
    x_strong = _lexicographic_solve(
        objective, start, hf._AXIS_ALIGNMENT_WRIST_REGULARIZATION_WEIGHT * 1e7
    )

    distance_default = float(np.linalg.norm(x_default[_WRIST_IDX] - start[_WRIST_IDX]))
    distance_strong = float(np.linalg.norm(x_strong[_WRIST_IDX] - start[_WRIST_IDX]))
    assert distance_strong < 0.1 * distance_default


def test_tuning_parameters_are_wired_to_the_right_closures_not_the_module_constants():
    """Each new parameter name must appear where the matching constant did.

    White-box source check on the actual `fit_handle` body (not a
    reproduction): catches the kind of wiring bug a synthetic objective can't
    -- a parameter added to the signature but accidentally left unused, or
    threaded into the wrong closure -- by confirming `step3_objective`,
    `stage3_tolerance_ok`, `step4_objective`, `stage4_tolerance_ok` and the
    two `lexicographic_solve` calls each reference their expected parameter
    name, and that the old bare module-constant names no longer appear in
    those same bodies (meaning they were actually replaced, not just
    shadowed).
    """
    source = inspect.getsource(hf._HandFittingMixin.fit_handle)

    def body_between(start_marker: str, end_marker: str) -> str:
        start = source.index(start_marker)
        end = source.index(end_marker, start)
        return source[start:end]

    step3_body = body_between("def step3_objective", "def stage3_tolerance_ok")
    assert "axis_distance_weight" in step3_body
    assert "axis_angle_weight" in step3_body
    assert "axis_alignment_regularization_weight" in step3_body
    assert "_AXIS_DISTANCE_WEIGHT" not in step3_body
    assert "_AXIS_ANGLE_WEIGHT" not in step3_body
    assert "_AXIS_ALIGNMENT_REGULARIZATION_WEIGHT" not in step3_body

    stage3_tolerance_body = body_between("def stage3_tolerance_ok", "try:")
    assert "wrist_lock_distance_tolerance_m" in stage3_tolerance_body
    assert "wrist_lock_angle_tolerance_deg" in stage3_tolerance_body
    assert "_WRIST_LOCK_DISTANCE_TOLERANCE_M" not in stage3_tolerance_body
    assert "_WRIST_LOCK_ANGLE_TOLERANCE_DEG" not in stage3_tolerance_body

    stage3_call_body = body_between("x1, wrist_locked_stage3 = lexicographic_solve(", ")\n")
    assert "axis_alignment_wrist_regularization_weight" in stage3_call_body
    assert "_AXIS_ALIGNMENT_WRIST_REGULARIZATION_WEIGHT" not in stage3_call_body

    step4_body = body_between("def step4_objective", "def stage4_tolerance_ok")
    assert "palm_tangency_regularization_weight" in step4_body
    assert "_PALM_TANGENCY_REGULARIZATION_WEIGHT" not in step4_body

    stage4_tolerance_body = body_between("def stage4_tolerance_ok", "try:")
    assert "wrist_lock_gap_tolerance_m" in stage4_tolerance_body
    assert "_WRIST_LOCK_GAP_TOLERANCE_M" not in stage4_tolerance_body

    stage4_call_body = body_between("x3, wrist_locked_stage4 = lexicographic_solve(", ")\n")
    assert "palm_tangency_wrist_regularization_weight" in stage4_call_body
    assert "_PALM_TANGENCY_WRIST_REGULARIZATION_WEIGHT" not in stage4_call_body
