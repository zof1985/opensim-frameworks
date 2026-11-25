"""Generate and validate the height-independent bilinear shoulder maps."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from ... import User
from .scapular_map import _COEFFICIENT_NAMES, _OUTPUT_NAMES, _ASSETS

_REFERENCE_COHORTS: dict[str, tuple[tuple[str, str, float], ...]] = {
    "M": (
        ("height_150_cm", "height_cm", 150.0),
        ("p10", "percentile", 10.0),
        ("p50", "percentile", 50.0),
        ("p90", "percentile", 90.0),
        ("height_205_cm", "height_cm", 205.0),
    ),
    "F": (
        ("height_140_cm", "height_cm", 140.0),
        ("p10", "percentile", 10.0),
        ("p50", "percentile", 50.0),
        ("p90", "percentile", 90.0),
        ("height_195_cm", "height_cm", 195.0),
    ),
}
_TRAINING_ANGLES = (-10.0, 0.0, 10.0, 20.0, 30.0)
_VALIDATION_ANGLES = (-5.0, 5.0, 15.0, 25.0)
_AC_NAMES = _OUTPUT_NAMES[2:]
_GROUND_TRUTH_PATH = _ASSETS / "scapulothoracic_map_ground_truth.csv"
_VALIDATION_PATH = _ASSETS / "scapulothoracic_map_validation.csv"


def _set_pose(users: list[User], protraction: float, elevation: float, ac: np.ndarray) -> None:
    for user in users:
        coordinates = user.getCoordinateSet()
        coordinates.get("sc_protraction_r").setValue(user.state, protraction, False)
        coordinates.get("sc_elevation_r").setValue(user.state, elevation, False)
        for name, value in zip(_AC_NAMES, ac):
            coordinates.get(f"{name}_r").setValue(user.state, float(value), False)
        user.realizePosition(user.state)


def _ac_is_in_range(users: list[User], ac: np.ndarray) -> bool:
    for user in users:
        coordinates = user.getCoordinateSet()
        if any(
            not coordinates.get(f"{name}_r").getRangeMin() - 1e-10
            <= value
            <= coordinates.get(f"{name}_r").getRangeMax() + 1e-10
            for name, value in zip(_AC_NAMES, ac)
        ):
            return False
    return True


def _fit_reference_pose(
    users: list[User],
    protraction_deg: float,
    elevation_deg: float,
    previous: np.ndarray | None,
) -> tuple[np.ndarray, float, list[float], list[float]]:
    shoulder = np.radians([protraction_deg, elevation_deg])
    contacts = [user._scapula_mesh_contact("r") for user in users]
    first_coordinates = users[0].getCoordinateSet()
    lower = np.array(
        [first_coordinates.get(f"{name}_r").getRangeMin() for name in _AC_NAMES]
    )
    upper = np.array(
        [first_coordinates.get(f"{name}_r").getRangeMax() for name in _AC_NAMES]
    )

    def residual(ac: np.ndarray) -> np.ndarray:
        _set_pose(users, shoulder[0], shoulder[1], ac)
        return np.concatenate(
            [np.sqrt(contact.distances_to_rib_center()) for contact in contacts]
        )

    starts = [
        np.clip(np.zeros(3), lower, upper),
        np.clip(np.radians([10.0, 0.0, 15.0]), lower, upper),
    ]
    if previous is not None:
        starts.insert(0, np.clip(previous, lower, upper))

    sampled: list[tuple[float, np.ndarray]] = []
    sobol = qmc.Sobol(d=3, scramble=False).random_base2(m=12)

    def sample(points: np.ndarray) -> None:
        for point in points:
            candidate = lower + point * (upper - lower)
            objective = residual(candidate)
            if not any(contact.intersects_ribs() for contact in contacts):
                sampled.append((float(objective @ objective), candidate.copy()))

    sample(sobol[:128])
    if not sampled:
        sample(sobol[128:])
    starts.extend(
        candidate
        for _, candidate in sorted(sampled, key=lambda item: item[0])[:3]
        if not any(np.allclose(candidate, existing) for existing in starts)
    )

    candidates: list[tuple[float, np.ndarray]] = []
    for start in starts:
        objective = residual(start)
        if not any(contact.intersects_ribs() for contact in contacts):
            candidates.append((float(objective @ objective), start.copy()))
        fit = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            max_nfev=100,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        objective = residual(fit.x)
        if not any(contact.intersects_ribs() for contact in contacts):
            candidates.append((float(objective @ objective), fit.x.copy()))

    if not candidates:
        raise RuntimeError(
            f"No common collision-free pose for gender={users[0].gender}, "
            f"protraction={protraction_deg:g}°, elevation={elevation_deg:g}°"
        )
    objective, ac = min(candidates, key=lambda candidate: candidate[0])
    _set_pose(users, shoulder[0], shoulder[1], ac)
    gaps = [contact.distance() for contact in contacts]
    per_user_objectives = [
        float(np.sum(contact.distances_to_rib_center())) for contact in contacts
    ]
    return ac, objective, gaps, per_user_objectives


def _features(protraction_deg: float, elevation_deg: float) -> np.ndarray:
    return np.array(
        [1.0, protraction_deg, elevation_deg, protraction_deg * elevation_deg]
    )


def _make_users(gender: str) -> list[User]:
    users = []
    for _, method, value in _REFERENCE_COHORTS[gender]:
        if method == "height_cm":
            users.append(User(gender, height_cm=value))
        else:
            users.append(User(gender, percentile=value))
    return users


def _row(
    gender: str,
    split: str,
    protraction: float,
    elevation: float,
    ac: np.ndarray,
    objective: float,
    gaps: list[float],
    per_user_objectives: list[float],
    users: list[User],
) -> dict[str, str | float]:
    row: dict[str, str | float] = {
        "gender": gender,
        "split": split,
        "reference_cohorts": "|".join(
            name for name, _, _ in _REFERENCE_COHORTS[gender]
        ),
        "protraction_deg": protraction,
        "elevation_deg": elevation,
        "sc_protraction_deg": protraction,
        "sc_elevation_deg": elevation,
        "ac_protraction_deg": float(np.degrees(ac[0])),
        "ac_upward_rotation_deg": float(np.degrees(ac[1])),
        "ac_axial_rotation_deg": float(np.degrees(ac[2])),
        "objective_sum_distance_m": objective,
    }
    for index, (cohort, user, gap, user_objective) in enumerate(
        zip(_REFERENCE_COHORTS[gender], users, gaps, per_user_objectives), 1
    ):
        name, _, _ = cohort
        prefix = f"cohort_{index}"
        row[f"{prefix}_name"] = name
        row[f"{prefix}_height_cm"] = float(
            user._reference.values["stature_m"] * 100.0
        )
        row[f"{prefix}_percentile"] = float(user._reference.percentile)
        row[f"{prefix}_minimum_gap_m"] = gap
        row[f"{prefix}_objective_sum_distance_m"] = user_objective
        row[f"{prefix}_intersects"] = 0
    return row


def _write_rows(path: Path, rows: list[dict[str, str | float]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty dataset to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fit_coefficients(
    training: list[dict[str, str | float]],
) -> dict[str, np.ndarray]:
    fitted: dict[str, np.ndarray] = {}
    for gender in ("M", "F"):
        rows = [row for row in training if row["gender"] == gender]
        design = np.array(
            [
                _features(
                    float(row["protraction_deg"]), float(row["elevation_deg"])
                )
                for row in rows
            ]
        )
        targets = np.array(
            [
                [float(row[f"{name}_deg"]) for name in _OUTPUT_NAMES]
                for row in rows
            ]
        )
        coefficients, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
        if rank != 4:
            raise RuntimeError(f"bilinear design matrix is rank deficient for {gender}")
        fitted[gender] = coefficients
    return fitted


def _write_coefficients(coefficients: dict[str, np.ndarray]) -> None:
    path = _ASSETS / "scapulothoracic_map_coefficients.csv"
    fields = [
        "gender",
        "coefficient",
        "protraction_min_deg",
        "protraction_max_deg",
        "elevation_min_deg",
        "elevation_max_deg",
        *_OUTPUT_NAMES,
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for gender in ("M", "F"):
            for index, name in enumerate(_COEFFICIENT_NAMES):
                row: dict[str, str | float] = {
                    "gender": gender,
                    "coefficient": name,
                    "protraction_min_deg": min(_TRAINING_ANGLES),
                    "protraction_max_deg": max(_TRAINING_ANGLES),
                    "elevation_min_deg": min(_TRAINING_ANGLES),
                    "elevation_max_deg": max(_TRAINING_ANGLES),
                }
                row.update(
                    {
                        coordinate: float(value)
                        for coordinate, value in zip(
                            _OUTPUT_NAMES, coefficients[gender][index]
                        )
                    }
                )
                writer.writerow(row)


def generate() -> None:
    training: list[dict[str, str | float]] = []
    validation: list[dict[str, str | float]] = []
    fitted_coefficients: dict[str, np.ndarray] = {}

    for gender in ("M", "F"):
        users = _make_users(gender)
        previous: np.ndarray | None = None
        for split, values in (
            ("training", _TRAINING_ANGLES),
            ("validation", _VALIDATION_ANGLES),
        ):
            target_rows = training if split == "training" else validation
            previous = None
            for protraction in values:
                for elevation in values:
                    ac, objective, gaps, per_user_objectives = _fit_reference_pose(
                        users, protraction, elevation, previous
                    )
                    previous = ac
                    target_rows.append(
                        _row(
                            gender,
                            split,
                            protraction,
                            elevation,
                            ac,
                            objective,
                            gaps,
                            per_user_objectives,
                            users,
                        )
                    )

    fitted_coefficients = _fit_coefficients(training)
    _write_coefficients(fitted_coefficients)

    # Fit and evaluate each gender independently. Validation poses are kept
    # separate from the training table and never contribute to coefficients.
    validation_summary: dict[str, dict[str, float | int]] = {}
    for gender in ("M", "F"):
        users = _make_users(gender)
        gender_validation = [row for row in validation if row["gender"] == gender]
        errors: list[np.ndarray] = []
        regrets: list[float] = []
        collisions = 0
        user_pose_collisions = 0
        out_of_range_predictions = 0
        for row in gender_validation:
            p = float(row["protraction_deg"])
            e = float(row["elevation_deg"])
            prediction_deg = _features(p, e) @ fitted_coefficients[gender]
            prediction_deg[:2] = [p, e]
            reference_ac = np.radians(
                [
                    float(row["ac_protraction_deg"]),
                    float(row["ac_upward_rotation_deg"]),
                    float(row["ac_axial_rotation_deg"]),
                ]
            )
            predicted_ac = np.radians(prediction_deg[2:])
            error = prediction_deg[2:] - np.degrees(reference_ac)
            errors.append(error)

            contacts = [user._scapula_mesh_contact("r") for user in users]
            in_range = _ac_is_in_range(users, predicted_ac)
            if in_range:
                _set_pose(users, np.radians(p), np.radians(e), predicted_ac)
                intersects = [contact.intersects_ribs() for contact in contacts]
                collisions += int(any(intersects))
                user_pose_collisions += sum(intersects)
                prediction_objective = sum(
                    float(np.sum(contact.distances_to_rib_center()))
                    for contact in contacts
                )
                reference_objective = sum(
                    float(row[f"cohort_{index}_objective_sum_distance_m"])
                    for index in range(1, len(users) + 1)
                )
                regret = 100.0 * (
                    prediction_objective / reference_objective - 1.0
                )
                regrets.append(regret)
            else:
                out_of_range_predictions += 1
                intersects = [False] * len(users)
                regret = float("nan")

            row.update(
                {
                    f"predicted_{name}_deg": float(value)
                    for name, value in zip(_AC_NAMES, prediction_deg[2:])
                }
            )
            row.update(
                {
                    f"error_{name}_deg": float(value)
                    for name, value in zip(_AC_NAMES, error)
                }
            )
            row["predicted_intersections"] = int(any(intersects))
            row.update(
                {
                    f"predicted_intersections_cohort_{index}": int(value)
                    for index, value in enumerate(intersects, 1)
                }
            )
            row["objective_regret_pct"] = regret
            row["prediction_out_of_range"] = int(not in_range)

        error_array = np.asarray(errors)
        validation_summary[gender] = {
            "validation_points": len(gender_validation),
            "ac_protraction_mae_deg": float(np.mean(np.abs(error_array[:, 0]))),
            "ac_upward_rotation_mae_deg": float(np.mean(np.abs(error_array[:, 1]))),
            "ac_axial_rotation_mae_deg": float(np.mean(np.abs(error_array[:, 2]))),
            "ac_protraction_max_abs_error_deg": float(np.max(np.abs(error_array[:, 0]))),
            "ac_upward_rotation_max_abs_error_deg": float(np.max(np.abs(error_array[:, 1]))),
            "ac_axial_rotation_max_abs_error_deg": float(np.max(np.abs(error_array[:, 2]))),
            "colliding_predictions": collisions,
            "colliding_user_poses": user_pose_collisions,
            "out_of_range_predictions": out_of_range_predictions,
            "mean_objective_regret_pct": float(np.mean(regrets)) if regrets else float("nan"),
            "max_objective_regret_pct": float(np.max(regrets)) if regrets else float("nan"),
        }
        print(f"{gender}: {validation_summary[gender]}")

    _write_rows(_GROUND_TRUTH_PATH, training)
    _write_rows(_VALIDATION_PATH, validation)


if __name__ == "__main__":
    generate()
