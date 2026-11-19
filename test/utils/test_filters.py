import numpy as np
import pytest

from opensim_frameworks.utils import filter_array


def test_filter_array_suppresses_high_frequency_and_preserves_shape():
    sample_rate_hz = 100.0
    times = np.arange(1000) / sample_rate_hz
    low_frequency = np.sin(2.0 * np.pi * 2.0 * times)
    high_frequency = np.sin(2.0 * np.pi * 30.0 * times)
    samples = np.column_stack((low_frequency, high_frequency))

    filtered = filter_array(
        samples,
        cutoff=0.2,
        order=4,
        filter_type="lowpass",
    )
    filtered_hz = filter_array(
        samples,
        cutoff=10.0,
        order=4,
        filter_type="lowpass",
        sampling_frequency=sample_rate_hz,
    )

    assert filtered.shape == samples.shape
    assert filtered_hz == pytest.approx(filtered)
    assert np.corrcoef(filtered[100:-100, 0], low_frequency[100:-100])[0, 1] > 0.999
    assert np.sqrt(np.mean(filtered[100:-100, 1] ** 2)) < 0.02


def test_filter_array_supports_single_pass_without_phase_correction():
    samples = np.zeros(201)
    samples[100] = 1.0

    phase_corrected = filter_array(
        samples,
        cutoff=0.2,
        order=2,
        filter_type="lowpass",
    )
    single_pass = filter_array(
        samples,
        cutoff=0.2,
        order=2,
        filter_type="lowpass",
        phase_corrected=False,
    )

    assert phase_corrected.shape == samples.shape
    assert single_pass.shape == samples.shape
    assert phase_corrected[90:100] == pytest.approx(phase_corrected[101:111][::-1])
    assert np.all(single_pass[:100] == 0.0)
    assert np.any(single_pass[101:] != 0.0)


@pytest.mark.parametrize("order", [0, -1])
def test_filter_array_rejects_nonpositive_order(order):
    with pytest.raises(ValueError, match="order must be positive"):
        filter_array(np.arange(10), cutoff=0.2, order=order, filter_type="lowpass")


@pytest.mark.parametrize("sampling_frequency", [0.0, -1.0, np.inf, np.nan])
def test_filter_array_rejects_invalid_sampling_frequency(sampling_frequency):
    with pytest.raises(ValueError, match="sampling_frequency must be positive and finite"):
        filter_array(
            np.arange(10),
            cutoff=2.0,
            order=2,
            filter_type="lowpass",
            sampling_frequency=sampling_frequency,
        )
