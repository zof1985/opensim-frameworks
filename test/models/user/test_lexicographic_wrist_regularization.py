"""Targeted tests for fit_handle's Phase-2 explicit wrist regularization.

``lexicographic_solve`` is a private closure defined inside
``User.fit_handle`` (see ``hand_fitting.py``) and is not independently
callable, so these tests reproduce its two-phase structure -- Phase 1 solves
the five non-wrist coordinates with the wrist locked at `start`, Phase 2
always re-solves `wrist_flex`/`wrist_dev` alone with an explicit quadratic
penalty towards `start`'s own wrist values -- against a minimal synthetic
7-coordinate objective, using the real module-level regularization weight.
This avoids requiring OpenSim/VTK or a real `fit_handle` end-to-end run,
while still exercising the actual tuned constant
(``_AXIS_ALIGNMENT_WRIST_REGULARIZATION_WEIGHT``) introduced alongside it.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from opensim_frameworks.models.user import hand_fitting as hf

_WRIST_IDX = np.array([5, 6])
_FREE_IDX = np.array([0, 1, 2, 3, 4])
_BOUNDS = [(-1.5, 1.5)] * 7


def _lexicographic_solve(objective, start: np.ndarray, wrist_regularization_weight: float):
    """Reproduce fit_handle's Phase 1 / Phase 2 split for a synthetic objective."""
    # Phase 1: wrist locked at `start`, only the five free coordinates vary.
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

    # Phase 2: wrist-only re-solve, always run, with the explicit penalty
    # towards `start`'s own wrist values added on top of `objective`.
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


def test_phase2_keeps_wrist_close_to_start_when_gain_is_negligible():
    """A wrist-independent objective should leave Phase 2's wrist near `start`.

    The objective only depends on the five free coordinates, so moving the
    wrist away from `start` brings zero geometric gain; the explicit
    regularization should pull Phase 2's wrist solution back to (numerically
    very close to) `start`'s own wrist value.
    """
    start = np.array([0.1, -0.2, 0.3, 0.4, -0.1, 0.25, -0.15])

    def objective(x: np.ndarray) -> float:
        free = x[_FREE_IDX]
        target = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        return float(50.0 * np.dot(free - target, free - target))

    x_final = _lexicographic_solve(
        objective, start, hf._AXIS_ALIGNMENT_WRIST_REGULARIZATION_WEIGHT
    )
    np.testing.assert_allclose(x_final[_WRIST_IDX], start[_WRIST_IDX], atol=1e-6)


def test_phase2_still_corrects_wrist_when_geometric_gain_justifies_it():
    """A wrist-dependent objective with a large gain should still move it.

    The same weak regularization weight must not block Phase 2 from
    substantially closing a wrist-dependent geometric term (mirroring the
    axis-alignment distance/angle terms in `fit_handle`'s stage 3, whose
    weights -- 50 and 5 -- dominate the weak regularization weight
    ``_AXIS_ALIGNMENT_WRIST_REGULARIZATION_WEIGHT`` by several orders of
    magnitude), otherwise the wrist-posture lexicographic policy (correct the
    wrist only when actually needed, but still correct it) would be broken.
    """
    start = np.array([0.1, -0.2, 0.3, 0.4, -0.1, 0.9, -0.9])
    wrist_target = np.array([0.0, 0.0])

    def objective(x: np.ndarray) -> float:
        wrist = x[_WRIST_IDX]
        return float(50.0 * np.dot(wrist - wrist_target, wrist - wrist_target))

    x_final = _lexicographic_solve(
        objective, start, hf._AXIS_ALIGNMENT_WRIST_REGULARIZATION_WEIGHT
    )
    # The regularization is weak relative to the 50x geometric weight, so the
    # wrist should move most of the way from `start` towards the target,
    # not stay pinned at `start`.
    initial_distance = float(np.linalg.norm(start[_WRIST_IDX] - wrist_target))
    final_distance = float(np.linalg.norm(x_final[_WRIST_IDX] - wrist_target))
    assert final_distance < 0.1 * initial_distance
