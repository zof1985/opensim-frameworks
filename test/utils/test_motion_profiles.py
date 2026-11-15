import numpy as np
import pytest

from opensim_frameworks.utils.motion_profiles import (
    beta_position_profile,
    beta_time_for_position,
    beta_velocity_profile,
    gaussian_position_profile,
    gaussian_time_for_position,
    gaussian_velocity_profile,
)


def test_beta_profile_round_trip_and_peak():
    times = np.linspace(0.0, 1.0, 101)
    positions = np.linspace(0.0, 1.0, 101)

    assert beta_position_profile([0.0, 1.0]) == pytest.approx([0.0, 1.0])
    assert beta_time_for_position(beta_position_profile(times)) == pytest.approx(times)
    assert beta_velocity_profile([0.0, 1.0]) == pytest.approx([0.0, 0.0])
    peak_time = times[np.argmax(beta_velocity_profile(times))]
    assert peak_time == pytest.approx(1 / 3, abs=0.01)


def test_gaussian_profile_round_trip_and_zero_endpoint_speed():
    fractions = np.linspace(0.0, 1.0, 101)
    times = gaussian_time_for_position(fractions)

    assert gaussian_position_profile([0.0, 1.0]) == pytest.approx([0.0, 1.0])
    assert gaussian_position_profile(times) == pytest.approx(fractions)
    assert gaussian_velocity_profile([0.0, 1.0]) == pytest.approx([0.0, 0.0])
    assert gaussian_velocity_profile(0.5) == pytest.approx(
        max(gaussian_velocity_profile(fractions))
    )


@pytest.mark.parametrize(
    "profile",
    [
        beta_position_profile,
        beta_time_for_position,
        beta_velocity_profile,
        gaussian_position_profile,
        gaussian_time_for_position,
        gaussian_velocity_profile,
    ],
)
def test_profiles_reject_fractions_outside_unit_interval(profile):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        profile(1.1)
