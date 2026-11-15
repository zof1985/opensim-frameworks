"""Normalized velocity profiles and corresponding position/time mappings."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from scipy.special import betainc, betaincinv, betaln, erf

__all__ = [
    "beta_velocity_profile",
    "beta_position_profile",
    "beta_time_for_position",
    "gaussian_velocity_profile",
    "gaussian_position_profile",
    "gaussian_time_for_position",
]

_FloatArray = npt.NDArray[np.float64]


def _fraction_array(
    values: npt.ArrayLike, name: str
) -> tuple[_FloatArray, bool]:
    try:
        fractions = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain finite values in [0, 1]") from error
    if not np.all(np.isfinite(fractions)) or np.any(
        (fractions < 0.0) | (fractions > 1.0)
    ):
        raise ValueError(f"{name} must contain finite values in [0, 1]")
    return fractions, fractions.ndim == 0


def _restore_shape(values: _FloatArray, scalar: bool) -> float | _FloatArray:
    return float(values) if scalar else values


def _beta_shapes(shape_a: float, shape_b: float) -> tuple[float, float]:
    try:
        shape_a = float(shape_a)
        shape_b = float(shape_b)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("beta shape parameters must be finite and greater than 1") from error
    if (
        not math.isfinite(shape_a)
        or not math.isfinite(shape_b)
        or shape_a <= 1.0
        or shape_b <= 1.0
    ):
        raise ValueError("beta shape parameters must be finite and greater than 1")
    return shape_a, shape_b


def _sigma_value(sigma: float) -> float:
    try:
        sigma = float(sigma)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("sigma must be finite and positive") from error
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    return sigma


def beta_velocity_profile(
    time_fraction: npt.ArrayLike,
    shape_a: float = 3.0,
    shape_b: float = 5.0,
) -> float | _FloatArray:
    """Evaluate a unit-area Beta velocity profile over normalized phase time.

    The default ``Beta(3, 5)`` profile has a speed peak at one third of the
    phase and zero speed at both endpoints.

    Parameters
    ----------
    time_fraction : array-like
        Normalized phase time in ``[0, 1]``.
    shape_a, shape_b : float, default=3.0, 5.0
        Beta distribution shape parameters. Both must exceed 1.

    Returns
    -------
    float or numpy.ndarray
        Normalized velocity values whose integral over the phase is 1.

    Raises
    ------
    ValueError
        If the input fractions or shape parameters are invalid.
    """
    times, scalar = _fraction_array(time_fraction, "time_fraction")
    shape_a, shape_b = _beta_shapes(shape_a, shape_b)
    velocity = (
        np.power(times, shape_a - 1.0)
        * np.power(1.0 - times, shape_b - 1.0)
        * math.exp(-float(betaln(shape_a, shape_b)))
    )
    return _restore_shape(velocity, scalar)


def beta_position_profile(
    time_fraction: npt.ArrayLike,
    shape_a: float = 3.0,
    shape_b: float = 5.0,
) -> float | _FloatArray:
    """Map normalized Beta-profile time to normalized traveled distance.

    Parameters
    ----------
    time_fraction : array-like
        Normalized phase time in ``[0, 1]``.
    shape_a, shape_b : float, default=3.0, 5.0
        Beta distribution shape parameters. Both must exceed 1.

    Returns
    -------
    float or numpy.ndarray
        Normalized position fraction in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the input fractions or shape parameters are invalid.
    """
    times, scalar = _fraction_array(time_fraction, "time_fraction")
    shape_a, shape_b = _beta_shapes(shape_a, shape_b)
    position = betainc(shape_a, shape_b, times)
    return _restore_shape(np.asarray(position, dtype=float), scalar)


def beta_time_for_position(
    position_fraction: npt.ArrayLike,
    shape_a: float = 3.0,
    shape_b: float = 5.0,
) -> float | _FloatArray:
    """Invert the Beta position profile to obtain normalized phase time.

    Parameters
    ----------
    position_fraction : array-like
        Normalized traveled distance in ``[0, 1]``.
    shape_a, shape_b : float, default=3.0, 5.0
        Beta distribution shape parameters. Both must exceed 1.

    Returns
    -------
    float or numpy.ndarray
        Normalized phase time in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the input fractions or shape parameters are invalid.
    """
    positions, scalar = _fraction_array(position_fraction, "position_fraction")
    shape_a, shape_b = _beta_shapes(shape_a, shape_b)
    times = betaincinv(shape_a, shape_b, positions)
    return _restore_shape(np.asarray(times, dtype=float), scalar)


def _gaussian_area(time_fraction: _FloatArray, sigma: float) -> _FloatArray:
    endpoint = math.exp(-0.5 * (0.5 / sigma) ** 2)
    scaled = (time_fraction - 0.5) / (math.sqrt(2.0) * sigma)
    start = -0.5 / (math.sqrt(2.0) * sigma)
    return (
        sigma
        * math.sqrt(math.pi / 2.0)
        * (erf(scaled) - erf(start))
        - endpoint * time_fraction
    )


def _gaussian_total_area(sigma: float) -> float:
    total = float(_gaussian_area(np.asarray(1.0), sigma))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("sigma does not define a numerically usable profile")
    return total


def gaussian_velocity_profile(
    time_fraction: npt.ArrayLike, sigma: float = 0.2
) -> float | _FloatArray:
    """Evaluate a unit-area Gaussian velocity profile with zero endpoint speed.

    The Gaussian is centered at the phase midpoint. Its endpoint value is
    subtracted before normalization so the speed is exactly zero at both ends.

    Parameters
    ----------
    time_fraction : array-like
        Normalized phase time in ``[0, 1]``.
    sigma : float, default=0.2
        Gaussian standard deviation in normalized phase time.

    Returns
    -------
    float or numpy.ndarray
        Normalized velocity values whose integral over the phase is 1.

    Raises
    ------
    ValueError
        If the input fractions or ``sigma`` are invalid.
    """
    times, scalar = _fraction_array(time_fraction, "time_fraction")
    sigma = _sigma_value(sigma)
    endpoint = math.exp(-0.5 * (0.5 / sigma) ** 2)
    raw_velocity = np.exp(-0.5 * ((times - 0.5) / sigma) ** 2) - endpoint
    velocity = np.maximum(raw_velocity, 0.0) / _gaussian_total_area(sigma)
    return _restore_shape(velocity, scalar)


def gaussian_position_profile(
    time_fraction: npt.ArrayLike, sigma: float = 0.2
) -> float | _FloatArray:
    """Map normalized Gaussian-profile time to normalized traveled distance.

    Parameters
    ----------
    time_fraction : array-like
        Normalized phase time in ``[0, 1]``.
    sigma : float, default=0.2
        Gaussian standard deviation in normalized phase time.

    Returns
    -------
    float or numpy.ndarray
        Normalized position fraction in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the input fractions or ``sigma`` are invalid.
    """
    times, scalar = _fraction_array(time_fraction, "time_fraction")
    sigma = _sigma_value(sigma)
    position = _gaussian_area(times, sigma) / _gaussian_total_area(sigma)
    return _restore_shape(np.asarray(position, dtype=float), scalar)


def gaussian_time_for_position(
    position_fraction: npt.ArrayLike, sigma: float = 0.2
) -> float | _FloatArray:
    """Invert the Gaussian position profile to obtain normalized phase time.

    Parameters
    ----------
    position_fraction : array-like
        Normalized traveled distance in ``[0, 1]``.
    sigma : float, default=0.2
        Gaussian standard deviation in normalized phase time.

    Returns
    -------
    float or numpy.ndarray
        Normalized phase time in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the input fractions or ``sigma`` are invalid.
    """
    positions, scalar = _fraction_array(position_fraction, "position_fraction")
    sigma = _sigma_value(sigma)
    total_area = _gaussian_total_area(sigma)

    def invert(position: float) -> float:
        if position == 0.0 or position == 1.0:
            return position
        target_area = position * total_area
        return float(
            brentq(
                lambda time: float(_gaussian_area(np.asarray(time), sigma))
                - target_area,
                0.0,
                1.0,
            )
        )

    times = np.fromiter(
        (invert(float(position)) for position in positions.flat),
        dtype=float,
        count=positions.size,
    ).reshape(positions.shape)
    return _restore_shape(times, scalar)
