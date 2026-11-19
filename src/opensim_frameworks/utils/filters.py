"""Reusable digital signal filters."""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfilt, sosfiltfilt

__all__ = ["filter_array"]


def filter_array(
    array_object: npt.ArrayLike,
    cutoff: float | tuple[float, float],
    order: int,
    filter_type: Literal["lowpass", "highpass", "bandpass", "bandstop"] | str,
    phase_corrected: bool = True,
    sampling_frequency: float | None = None,
) -> npt.NDArray[np.float64]:
    """Filter an array along its first axis.

    If ``sampling_frequency`` is provided, ``cutoff`` is specified in Hz.
    Otherwise, ``cutoff`` is normalized to the Nyquist frequency and must lie
    between 0 and 1. Multidimensional arrays are filtered independently along
    each column and trailing dimension.

    Parameters
    ----------
    array_object : array-like
        Samples to filter. The first axis is treated as time.
    cutoff : float or tuple of float
        Normalized cutoff frequency or frequency pair for the selected filter.
    order : int
        Butterworth filter order; must be positive.
    filter_type : {"lowpass", "highpass", "bandpass", "bandstop"} or str
        Filter type passed to :func:`scipy.signal.butter`.
    phase_corrected : bool, default=True
        Use forward-backward filtering to avoid phase shift when true;
        otherwise apply a single forward pass.
    sampling_frequency : float or None, default=None
        Sampling frequency in Hz. When provided, ``cutoff`` is also interpreted
        in Hz.

    Returns
    -------
    numpy.ndarray
        Filtered floating-point array with the same shape as the input.

    Raises
    ------
    TypeError
        If ``order``, ``phase_corrected``, or ``sampling_frequency`` has an
        invalid type.
    ValueError
        If ``order`` is not positive, or the input has fewer than two samples
        along its first axis, or ``sampling_frequency`` is not positive and
        finite. Invalid cutoff frequencies and filter types are also rejected
        by :func:`scipy.signal.butter`.
    """
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError("order must be an integer")
    if order < 1:
        raise ValueError("order must be positive")
    if not isinstance(phase_corrected, bool):
        raise TypeError("phase_corrected must be a bool")
    if sampling_frequency is not None:
        if isinstance(sampling_frequency, bool):
            raise TypeError("sampling_frequency must be a number or None")
        try:
            sampling_frequency = float(sampling_frequency)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError("sampling_frequency must be a number or None") from error
        if not np.isfinite(sampling_frequency) or sampling_frequency <= 0.0:
            raise ValueError("sampling_frequency must be positive and finite")

    samples = np.asarray(array_object, dtype=float)
    if samples.ndim == 0 or samples.shape[0] < 2:
        raise ValueError("array_object must contain at least two samples")

    if sampling_frequency is None:
        coefficients = butter(order, cutoff, btype=filter_type, output="sos")
    else:
        coefficients = butter(
            order,
            cutoff,
            btype=filter_type,
            fs=sampling_frequency,
            output="sos",
        )
    if not phase_corrected:
        return sosfilt(coefficients, samples, axis=0)

    padlen = min(3 * (2 * len(coefficients) + 1), samples.shape[0] - 1)
    return sosfiltfilt(coefficients, samples, axis=0, padlen=padlen)
