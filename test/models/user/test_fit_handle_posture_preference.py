"""Targeted tests for fit_handle's optional soft posture-preference term.

``fit_handle`` gained four new keyword parameters (`posture_regularization_weight`,
`arm_add_target_rad`, `elbow_flex_target_rad`, `pro_sup_target_rad`) adding an
optional soft quadratic nudge of `arm_add`/`elbow_flex`/`pro_sup` towards
caller-supplied targets, inside both `step3_objective` and `step4_objective`.
As with the wrist-regularization tests, `lexicographic_solve` is a private
closure and is not independently callable, so these tests reproduce its
Phase 1/Phase 2 split plus the posture term against a minimal synthetic
7-coordinate objective, checking the two properties the task requires:

(a) with weight=0.0/targets=None (the defaults) the result is identical to
    not having the term at all -- a strict no-op;
(b) with weight>0.0 and targets set, the targeted coordinates move closer to
    their targets relative to the no-preference case, without changing
    `wrist_flex`/`wrist_dev` at all -- the structural property documented in
    `hand_fitting.py` (the posture term depends only on coordinates in
    `free_idx`, never on `wrist_idx`, so it is a Phase-2-constant and cannot
    affect the wrist's own two-variable re-solve).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

_WRIST_IDX = np.array([5, 6])
_FREE_IDX = np.array([0, 1, 2, 3, 4])
_ARM_ADD_IDX, _ELBOW_FLEX_IDX, _PRO_SUP_IDX = 1, 3, 4
_BOUNDS = [(-1.5, 1.5)] * 7


def _posture_regularization_term(x: np.ndarray, targets: dict[int, float]) -> float:
    if not targets:
        return 0.0
    return float(sum((x[index] - target) ** 2 for index, target in targets.items()))


def _lexicographic_solve(
    geometric_objective,
    start: np.ndarray,
    posture_weight: float,
    posture_targets: dict[int, float],
):
    """Reproduce fit_handle's Phase 1/Phase 2 split plus the posture term."""
    def objective(x: np.ndarray) -> float:
        return geometric_objective(x) + posture_weight * _posture_regularization_term(
            x, posture_targets
        )

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
        # Same wrist regularization weight used throughout this module's
        # earlier tests; its value is irrelevant here (the posture term's
        # non-interference with the wrist is what's under test).
        return objective(x_full) + 1e-3 * float(np.dot(wrist_offset, wrist_offset))

    phase2 = minimize(
        objective_wrist_only, start[_WRIST_IDX], method="SLSQP",
        bounds=[_BOUNDS[i] for i in _WRIST_IDX], options={"maxiter": 200, "ftol": 1e-12},
    )
    assert phase2.success
    x_phase2 = x_phase1.copy()
    x_phase2[_WRIST_IDX] = phase2.x
    return x_phase2


def _geometric_objective(x: np.ndarray) -> float:
    """A stage-3-shaped synthetic objective with a flat (degenerate) arm null space.

    Only the wrist coordinates are penalized directly; `arm_add`/`elbow_flex`/
    `pro_sup` (and `arm_flex`/`arm_rot`) are otherwise free, so Phase 1 has no
    preference among them absent the posture term -- mirroring the reported
    M-150 case, where the geometric objective alone left the arm free to
    settle into a strongly abducted pose.
    """
    wrist = x[_WRIST_IDX]
    wrist_target = np.array([0.2, -0.1])
    return float(50.0 * np.dot(wrist - wrist_target, wrist - wrist_target))


def test_posture_preference_is_a_strict_no_op_at_default_weight_and_targets():
    """weight=0.0 (regardless of targets) must reproduce the no-term result exactly."""
    start = np.array([0.3, -0.9, 0.1, 1.0, 0.05, 0.2, -0.1])

    targets = {_ARM_ADD_IDX: 1.5, _ELBOW_FLEX_IDX: 0.0, _PRO_SUP_IDX: 1.5707963267948966}
    baseline = _lexicographic_solve(_geometric_objective, start, posture_weight=0.0, posture_targets={})
    with_targets_but_zero_weight = _lexicographic_solve(
        _geometric_objective, start, posture_weight=0.0, posture_targets=targets
    )
    np.testing.assert_allclose(baseline, with_targets_but_zero_weight, atol=1e-10)


def test_posture_preference_moves_target_coordinates_without_touching_wrist():
    """weight>0.0 with targets set must pull the targeted coordinates closer.

    `arm_add`/`elbow_flex`/`pro_sup` are otherwise unconstrained by
    `_geometric_objective` (it only penalizes the wrist), so Phase 1 would
    leave them exactly at `start` absent the posture term; with the term
    enabled they must move substantially towards their targets, while
    `wrist_flex`/`wrist_dev` (indices 5, 6) must stay exactly the same as the
    no-preference case -- the structural non-interference property.
    """
    start = np.array([0.3, -0.9, 0.1, 1.0, 0.05, 0.2, -0.1])
    targets = {_ARM_ADD_IDX: 1.5, _ELBOW_FLEX_IDX: 0.0, _PRO_SUP_IDX: 1.5707963267948966}

    baseline = _lexicographic_solve(_geometric_objective, start, posture_weight=0.0, posture_targets={})
    with_preference = _lexicographic_solve(
        _geometric_objective, start, posture_weight=0.5, posture_targets=targets
    )

    # The wrist is untouched by the posture term: identical in both solves.
    np.testing.assert_allclose(with_preference[_WRIST_IDX], baseline[_WRIST_IDX], atol=1e-9)

    # Each targeted coordinate moved strictly closer to its target.
    for index, target in targets.items():
        baseline_distance = abs(baseline[index] - target)
        preference_distance = abs(with_preference[index] - target)
        assert preference_distance < baseline_distance, (index, baseline_distance, preference_distance)
