"""Base chest-press equipment model with arm and wrist fitting."""

from __future__ import annotations

import abc
import math
from typing import Any

import numpy as np
import opensim as osim
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

from opensim_frameworks import Equipment, SHOULDER_GIRDLE_COORDINATES
from opensim_frameworks.framework.framework import Framework
from opensim_frameworks.models.user.user import User
from opensim_frameworks.utils.geometry import body_transform_array
from opensim_frameworks.utils.kinematics import marker_position_array
from opensim_frameworks.utils.motion_profiles import (
    beta_time_for_position,
    beta_velocity_profile,
    gaussian_time_for_position,
    gaussian_velocity_profile,
)
from .handlefit import (
    handle_fit_failure_message,
    fit_handle,
    lock_hands_to_handles,
)
from .userpositioning import position_user

__all__ = ["ChestPress"]

_TRAJECTORY_FILTER_ORDER = 4
_TRAJECTORY_FILTER_CUTOFF_HZ = 1.0
_TRAJECTORY_FILTER_SAMPLE_RATE_HZ = 30.0
_UPPER_LIMB_COORDINATE_PREFIXES = (
    "arm_",
    "elbow_",
    "pro_sup_",
    "wrist_",
    "index_",
    "middle_",
    "ring_",
    "little_",
    "thumb_",
)


def _phase_times_and_speeds(
    state_angles: list[float],
    state_phases: list[str],
    lower_angle: float,
    concentric_duration_s: float,
    eccentric_duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    angles = np.asarray(state_angles, dtype=float)
    phases = np.asarray(state_phases)
    concentric_indices = np.flatnonzero(phases == "concentric")
    eccentric_indices = np.flatnonzero(phases == "eccentric")
    peak_angle = float(angles[concentric_indices[-1]])
    travel = peak_angle - lower_angle
    if travel <= 0.0:
        raise ValueError("the fitted press stroke must span a positive angle")

    times = np.empty(angles.size, dtype=float)
    angular_speeds = np.zeros(angles.size, dtype=float)
    for index in concentric_indices:
        progress = float(np.clip((angles[index] - lower_angle) / travel, 0.0, 1.0))
        time_fraction = float(beta_time_for_position(progress))
        times[index] = concentric_duration_s * time_fraction
        angular_speeds[index] = (
            travel / concentric_duration_s * float(beta_velocity_profile(time_fraction))
        )

    for index in eccentric_indices:
        progress = float(np.clip((peak_angle - angles[index]) / travel, 0.0, 1.0))
        time_fraction = float(gaussian_time_for_position(progress))
        times[index] = concentric_duration_s + eccentric_duration_s * time_fraction
        angular_speeds[index] = (
            -travel
            / eccentric_duration_s
            * float(gaussian_velocity_profile(time_fraction))
        )
    return times, angular_speeds


def _set_trajectory_coordinate_speeds(
    framework: Framework,
    states: list[tuple[osim.State, osim.State]],
    state_angles: list[float],
    state_phases: list[str],
    angular_speeds: np.ndarray,
) -> None:
    angles = np.asarray(state_angles, dtype=float)
    phases = np.asarray(state_phases)
    concentric_indices = np.flatnonzero(phases == "concentric")
    eccentric_indices = np.flatnonzero(phases == "eccentric")
    eccentric_segment_indices = np.concatenate(
        (concentric_indices[-1:], eccentric_indices)
    )

    for member_index, model in enumerate((framework.user, framework.equipment)):
        coordinates = model.getCoordinateSet()
        for coordinate_index in range(coordinates.getSize()):
            coordinate = coordinates.get(coordinate_index)
            positions = np.asarray(
                [coordinate.getValue(state[member_index]) for state in states],
                dtype=float,
            )
            speeds = np.zeros(len(states), dtype=float)
            if concentric_indices.size >= 2:
                spline = CubicSpline(
                    angles[concentric_indices], positions[concentric_indices]
                )
                speeds[concentric_indices] = (
                    spline(angles[concentric_indices], 1)
                    * angular_speeds[concentric_indices]
                )
            if eccentric_segment_indices.size >= 2 and eccentric_indices.size:
                ordered_indices = eccentric_segment_indices[
                    np.argsort(angles[eccentric_segment_indices])
                ]
                spline = CubicSpline(
                    angles[ordered_indices], positions[ordered_indices]
                )
                speeds[eccentric_indices] = (
                    spline(angles[eccentric_indices], 1)
                    * angular_speeds[eccentric_indices]
                )
            for state_index, state in enumerate(states):
                coordinate.setSpeedValue(
                    state[member_index], float(speeds[state_index])
                )


def _filter_upper_limb_trajectories(
    user: User,
    states: list[tuple[osim.State, osim.State]],
    state_times: np.ndarray,
) -> None:
    times = np.asarray(state_times, dtype=float)
    if (
        times.ndim != 1
        or times.size != len(states)
        or times.size < 2
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("state_times must be finite, increasing, and match states")

    upper_limb_coordinates = _upper_limb_coordinates(user)
    if not upper_limb_coordinates:
        return

    sample_count = max(
        int(math.ceil((times[-1] - times[0]) * _TRAJECTORY_FILTER_SAMPLE_RATE_HZ)) + 1,
        2,
    )
    uniform_times = np.linspace(times[0], times[-1], sample_count)
    sample_rate_hz = (sample_count - 1) / (times[-1] - times[0])
    sos = butter(
        _TRAJECTORY_FILTER_ORDER,
        _TRAJECTORY_FILTER_CUTOFF_HZ,
        btype="lowpass",
        fs=sample_rate_hz,
        output="sos",
    )
    padlen = min(3 * (2 * len(sos) + 1), sample_count - 1)

    positions = np.asarray(
        [
            [coordinate.getValue(user_state) for coordinate in upper_limb_coordinates]
            for user_state, _ in states
        ],
        dtype=float,
    )
    uniform_positions = np.column_stack(
        [
            np.interp(uniform_times, times, positions[:, index])
            for index in range(positions.shape[1])
        ]
    )
    filtered_uniform_positions = sosfiltfilt(
        sos, uniform_positions, axis=0, padlen=padlen
    )
    filtered_positions = np.column_stack(
        [
            np.interp(times, uniform_times, filtered_uniform_positions[:, index])
            for index in range(filtered_uniform_positions.shape[1])
        ]
    )

    for index, coordinate in enumerate(upper_limb_coordinates):
        filtered_positions[:, index] = np.clip(
            filtered_positions[:, index],
            coordinate.getRangeMin(),
            coordinate.getRangeMax(),
        )
    for state_index, (user_state, _) in enumerate(states):
        for coordinate_index, coordinate in enumerate(upper_limb_coordinates):
            coordinate.setValue(
                user_state,
                float(filtered_positions[state_index, coordinate_index]),
                False,
            )
        user.realizePosition(user_state)


def _upper_limb_coordinates(user: User) -> list[Any]:
    coordinates = user.getCoordinateSet()
    result = []
    for index in range(coordinates.getSize()):
        coordinate = coordinates.get(index)
        name = coordinate.getName()
        if (
            name.startswith(_UPPER_LIMB_COORDINATE_PREFIXES)
            or name.rsplit("_", 1)[0] in SHOULDER_GIRDLE_COORDINATES
        ):
            result.append(coordinate)
    return result


def _restore_filtered_palm_contact(
    framework: Framework,
    states: list[tuple[osim.State, osim.State]],
    state_angles: list[float],
    state_phases: list[str],
    unfiltered_coordinates: list[dict[str, float]],
    fit_log: list[dict[str, Any]],
) -> None:
    """Refit both palms to their handle anchors after trajectory filtering.

    Parameters
    ----------
    framework : opensim_frameworks.Framework
        Framework containing the user, equipment, and fitted palm anchors.
    states : list of (opensim.State, opensim.State)
        User and equipment states to correct, in trajectory order.
    state_angles : list of float
        Matching lever angles in radians.
    state_phases : list of str
        Matching movement phases.
    unfiltered_coordinates : list of dict
        Original upper-limb coordinate values, used only if filtered contact
        cannot be restored.
    fit_log : list of dict
        Simulation log to extend with post-filter contact diagnostics.

    Raises
    ------
    RuntimeError
        If contact cannot be restored for any recorded state.
    """
    contact_report: dict[str, Any] = {
        "phase": "post_filter_contact",
        "samples": [],
    }
    fit_log.append(contact_report)
    for (user_state, equipment_state), angle, phase, original_coordinates in zip(
        states, state_angles, state_phases, unfiltered_coordinates, strict=True
    ):
        framework.user.state = user_state
        framework.equipment.state = equipment_state
        sample_log = [{"messages": []}]
        contact_maintained = lock_hands_to_handles(
            framework,
            mirror_left_from_right=True,
            fit_log=sample_log,
        )
        sample = {
            "phase": phase,
            "lever_angle_deg": math.degrees(angle),
            "messages": sample_log[0]["messages"],
        }
        if not isinstance(contact_maintained, bool):
            error = RuntimeError(
                "post-filter palm-anchor fitting did not report contact status"
            )
            setattr(error, "handle_fit_log", fit_log)
            raise error
        if not contact_maintained:
            coordinates = framework.user.getCoordinateSet()
            for name, value in original_coordinates.items():
                coordinates.get(name).setValue(user_state, value, False)
            framework.user.realizePosition(user_state)
            fallback_log = [{"messages": []}]
            contact_maintained = lock_hands_to_handles(
                framework,
                mirror_left_from_right=True,
                fit_log=fallback_log,
            )
            sample["messages"].append(
                {
                    "kind": "post_filter_contact_fallback",
                    "level": "WARNING",
                    "message": (
                        "restored original upper-limb coordinates after filtered "
                        "palm-anchor fitting failed"
                    ),
                }
            )
            sample["messages"].extend(fallback_log[0]["messages"])
            sample["fallback"] = "unfiltered_upper_limb_pose"
            if not isinstance(contact_maintained, bool):
                error = RuntimeError(
                    "palm-anchor fitting did not report contact status after "
                    "restoring original coordinates"
                )
                setattr(error, "handle_fit_log", fit_log)
                raise error
        contact_report["samples"].append(sample)
        if not contact_maintained:
            error = RuntimeError(
                "Palm contact could not be restored after filtering at "
                f"{math.degrees(angle):.2f} degrees during the {phase} phase"
            )
            setattr(error, "handle_fit_log", fit_log)
            raise error


def _palm_position_in_handle(user: User, equipment: Equipment, side: str) -> np.ndarray:
    suffix = "r" if side == "right" else "l"
    handle = equipment.getBodySet().get(f"{side}_handle")
    handle_marker = equipment.getMarkerSet().get(f"{side}_handle")
    palm_marker = user.getMarkerSet().get("RPALM" if suffix == "r" else "LPALM")
    transform = body_transform_array(handle, equipment.state)
    palm = marker_position_array(palm_marker, user.state)
    center = marker_position_array(handle_marker, equipment.state)
    return transform[:3, :3].T @ (palm - center)


class ChestPress(Equipment, abc.ABC):
    """Base class for chest-press equipment with user fitting and simulation.

    Subclasses must implement the ``_build`` method to construct the OpenSim
    model geometry and structure. After construction (via ``build``), the
    instance verifies that required bodies exist for fitting and simulation
    to function correctly.
    """

    @property
    @abc.abstractmethod
    def _backrest_angle_deg(self) -> float:
        """Backrest inclination in degrees relative to the Ground x axis.

        Returns
        -------
        float
            Backrest angle in degrees.
        """
        ...

    def __init__(self) -> None:
        """Initialize the chest-press equipment.

        Calls parent initialization which triggers ``_build`` and then verifies
        that required bodies exist for fitting and simulation.

        Raises
        ------
        RuntimeError
            If required bodies (``left_handle``, ``right_handle``,
            ``back_pad``, ``seat_pad``) are not found in the model.
        """
        super().__init__()
        self._verify_required_bodies()

    def _verify_required_bodies(self) -> None:
        """Verify that all bodies required by fit_user and simulate_kinematics exist.

        Raises
        ------
        RuntimeError
            If any required body is missing.
        """
        required = {
            "left_handle",
            "right_handle",
            "back_pad",
            "seat_pad",
        }
        body_set = self.getBodySet()
        available = {body_set.get(i).getName() for i in range(body_set.getSize())}
        missing = required - available
        if missing:
            raise RuntimeError(
                f"Missing required bodies for fitting: {', '.join(sorted(missing))}"
            )
        for side in ("left", "right"):
            if not self.handle_meshes(side):
                raise RuntimeError(f"No meshes attached to {side}_handle")

    def handle_meshes(self, side: str) -> list[osim.Mesh]:
        """Return meshes attached to the handle and its child frames.

        Parameters
        ----------
        side : {"left", "right"}
            Handle side.

        Returns
        -------
        list of opensim.Mesh
            Meshes attached to the handle body and its physical frames.
        """
        body = self.getBodySet().get(f"{side}_handle")
        frames = [body]
        frames.extend(
            frame
            for component in body.getComponentsList()
            if (frame := osim.PhysicalFrame.safeDownCast(component)) is not None
        )
        return [
            mesh
            for frame in frames
            for index in range(frame.getPropertyByName("attached_geometry").size())
            if (mesh := osim.Mesh.safeDownCast(frame.get_attached_geometry(index)))
            is not None
        ]

    def fit_user(
        self,
        user: User,
        *,
        wrist_flex_weight: float = 10.0,
        wrist_dev_weight: float = 0.2,
        knuckle_weight: float = 10.0,
        knuckle_end_weight: float = 10.0,
        protraction_weight: float = 1.0,
        scapular_elevation_weight: float = 1.0,
        arm_weight: float = 1.0,
        pro_sup_weight: float = 1.0,
        plane_distance_weight: float = 1.0,
        shoulder_wrist_axis_distance_weight: float = 10.0,
        palm_contact_weight: float = 10.0,
    ) -> tuple[Framework, dict[str, Any]]:
        """Fit a user on the machine.

        The user is first positioned by :func:`position_user`: the shoulder
        girdle and trunk are aligned, the pelvis is seated, the hips are
        abducted, the feet are returned to ground contact, and both shoulders
        and forearms are rotated by 90 degrees. The right arm is fitted against
        the handle with :func:`fit_handle` and the hand is closed around it. The right arm is
        mirrored to the left; the right and left palm markers are anchored in
        their respective moving handle frames on a best-effort basis.

        Parameters
        ----------
        user : opensim_frameworks.User
            Scaled user whose feet are initially in contact with ground.
        wrist_flex_weight : float, default=10.0
            Weight favoring at least 30 degrees of wrist extension.
        wrist_dev_weight : float, default=0.2
            Weight for neutral radial/ulnar deviation.
        knuckle_weight : float, default=10.0
            Weight for directed alignment of the index-to-little MCP vector
            with the handle axis.
        knuckle_end_weight : float, default=10.0
            Weight for keeping both MCP projections within their endpoint
            limits.
        protraction_weight : float, default=1.0
            Weight penalizing scapular protraction relative to its coordinate
            range minimum.
        scapular_elevation_weight : float, default=1.0
            Weight penalizing positive scapular elevation.
        arm_weight : float, default=1.0
            Weight penalizing external upper-arm rotation.
        pro_sup_weight : float, default=1.0
            Weight favoring forearm supination near 90 degrees.
        plane_distance_weight : float, default=1.0
            Weight for keeping the handle axis in the plane defined by the
            shoulder, elbow and wrist joint centre.
        shoulder_wrist_axis_distance_weight : float, default=10.0
            Weight for minimizing the distance between the shoulder-wrist line
            and the handle axis.
        palm_contact_weight : float, default=10.0
            Weight for matching the palm-marker and middle MCP-to-axis distances
            to the cylinder radius.
        Returns
        -------
        tuple of (opensim_frameworks.Framework, dict)
            Framework containing ``user`` and ``equipment``, and a report with
            the keys ``back``, ``seat``, ``hips``, ``feet`` and ``grip``.
            ``grip["handle_fit_log"]`` contains messages from the initial fit
            and the mirrored contact fit. Anchor residuals are included in
            each fit-log entry and do not by themselves stop the simulation.
            ``grip["fingers"]["right"]`` holds the right grasp report; the
            left finger posture is mirrored from the right.

        Raises
        ------
        TypeError
            If ``user`` is not an ``opensim_frameworks.User`` instance.
        RuntimeError
            If a lever's range does not include zero degrees or the initial
            arm fit does not converge.

        Notes
        -----
        If the initial fit fails, the raised exception exposes its messages as
        ``handle_fit_log``.
        """
        if not isinstance(user, User):
            raise TypeError("user must be an opensim_frameworks.User")
        framework = Framework(user=user, equipment=self)
        user = framework.user
        equipment = framework.equipment

        lever_coordinates = {
            side: equipment.getCoordinateSet().get(f"{side}_lever-angle")
            for side in ("right", "left")
        }
        for side, coordinate in lever_coordinates.items():
            if not coordinate.getRangeMin() <= 0.0 <= coordinate.getRangeMax():
                raise RuntimeError(
                    f"The {side} lever range does not include zero degrees"
                )
            coordinate.setValue(equipment.state, 0.0, False)
        equipment.realizePosition(equipment.state)

        report = position_user(
            user,
            equipment,
            backrest_angle_deg=self._backrest_angle_deg,
        )
        elbow_flexion, fitted_palm_position, messages = fit_handle(
            user=user,
            equipment=equipment,
            handle_name="right_handle",
            side="right",
            palm_contact_weight=palm_contact_weight,
            knuckle_weight=knuckle_weight,
            knuckle_end_weight=knuckle_end_weight,
            plane_distance_weight=plane_distance_weight,
            shoulder_wrist_axis_distance_weight=shoulder_wrist_axis_distance_weight,
            pro_sup_weight=pro_sup_weight,
            wrist_flex_weight=wrist_flex_weight,
            wrist_dev_weight=wrist_dev_weight,
            protraction_weight=protraction_weight,
            scapular_elevation_weight=scapular_elevation_weight,
            arm_weight=arm_weight,
        )
        fit_log = [
            {
                "phase": "initial",
                "lever_angle_deg": 0.0,
                "messages": messages,
            }
        ]
        if elbow_flexion is None or fitted_palm_position is None:
            error = RuntimeError("The initial right arm fit did not converge")
            setattr(error, "handle_fit_log", fit_log)
            raise error

        right_grasp = user.grasp_handle(
            "right",
            equipment.getBodySet().get("right_handle"),
            equipment.handle_meshes("right"),
            handle_state=equipment.state,
        )
        lock_hands_to_handles(
            framework=framework,
            mirror_left_from_right=True,
            fit_log=fit_log,
        )
        report["grip"] = {
            "elbow_flexion_deg": elbow_flexion,
            "palm_handle_position_m": fitted_palm_position,
            "left_palm_handle_position_m": _palm_position_in_handle(
                user, equipment, "left"
            ),
            "fingers": {
                "right": right_grasp,
                "left": {"mirrored_from_right": True},
            },
            "handle_fit_log": fit_log,
        }
        return framework, report

    def simulate_kinematics(
        self,
        user: User,
        step_deg: float = 1.0,
        concentric_duration_s: float = 1.5,
        eccentric_duration_s: float = 2.5,
        *,
        wrist_flex_weight: float = 10.0,
        wrist_dev_weight: float = 0.2,
        knuckle_weight: float = 10.0,
        knuckle_end_weight: float = 10.0,
        protraction_weight: float = 1.0,
        scapular_elevation_weight: float = 1.0,
        arm_weight: float = 1.0,
        pro_sup_weight: float = 1.0,
        plane_distance_weight: float = 1.0,
        shoulder_wrist_axis_distance_weight: float = 10.0,
        palm_contact_weight: float = 10.0,
    ) -> tuple[Framework, list[dict[str, Any]]]:
        """Fit and record one outward-and-return press movement.

        Starting from the fitted zero-angle pose, the right lever advances by
        ``step_deg`` while both palm anchors remain within 1 mm of their
        positions on the moving handles. The first step that exceeds this
        tolerance ends the concentric phase; that detached state is not
        recorded, and the eccentric phase returns from the last attached
        position. At each angle the right arm is refitted, the fingers grasp
        the handle, and the palm anchors are fit before the arm is mirrored
        left. The same fitting and grasping sequence is repeated on return to
        zero.

        Parameters
        ----------
        user : opensim_frameworks.User
            Scaled user to fit and simulate.
        step_deg : float, default=1.0
            Lever-angle increment, in degrees.
        concentric_duration_s : float, default=1.5
            Duration of the concentric press phase, in seconds.
        eccentric_duration_s : float, default=2.5
            Duration of the eccentric return phase, in seconds.
        wrist_flex_weight : float, default=10.0
            Weight favoring at least 30 degrees of wrist extension.
        wrist_dev_weight : float, default=0.2
            Weight for neutral radial/ulnar deviation.
        knuckle_weight : float, default=10.0
            Weight for directed index-to-little MCP alignment with the handle.
        knuckle_end_weight : float, default=10.0
            Weight for keeping both MCP projections within their endpoint limits.
        protraction_weight : float, default=1.0
            Weight penalizing scapular protraction relative to its coordinate
            range minimum.
        scapular_elevation_weight : float, default=1.0
            Weight penalizing positive scapular elevation.
        arm_weight : float, default=1.0
            Weight penalizing external upper-arm rotation.
        pro_sup_weight : float, default=1.0
            Weight favoring forearm supination near 90 degrees.
        plane_distance_weight : float, default=1.0
            Weight for keeping the handle axis in the plane defined by the
            shoulder, elbow and wrist joint centre.
        shoulder_wrist_axis_distance_weight : float, default=10.0
            Weight for minimizing the distance between the shoulder-wrist line
            and the handle axis.
        palm_contact_weight : float, default=10.0
            Weight for matching the palm-marker and middle MCP-to-axis distances
            to the cylinder radius.
        Returns
        -------
        tuple of (opensim_frameworks.Framework, list of dict)
            Framework containing synchronized trajectories and the handle-fit
            log, including the initial fit and each outward and return step.
            Each entry contains ``phase``, ``lever_angle_deg`` and ``messages``;
            movement-step messages include the right-arm fit and its mirror.

        Raises
        ------
        ValueError
            If a numeric parameter is invalid, a simulated right-lever angle
            is outside the left lever's range.
        RuntimeError
            If no positive attached stroke is possible, contact is lost during
            the eccentric phase, or a handle fit fails.

        Notes
        -----
        Concentric state times follow the inverse CDF of a Beta(3, 5) speed
        profile; eccentric state times follow the inverse CDF of a normalized
        Gaussian speed profile with zero endpoint speed. Contact fitting uses
        the previous attached pose as its initial guess. Coordinate speeds are
        derived from the fitted coordinate positions and lever-angle profile.
        Before trajectory recording, shoulder-girdle, arm, wrist, and finger
        coordinates are resampled at 30 Hz and filtered with a fourth-order,
        1 Hz zero-phase Butterworth low-pass filter. Palm anchors are then
        refitted at every recorded state so filtering cannot separate the
        hands from the moving handles. The trajectories retain their original
        sample times. Accelerations are not separately recorded. Sampling for
        playback is handled separately by the framework viewer. Simulation
        failures raise an exception with the accumulated ``handle_fit_log``.
        Palm-anchor residuals are recorded in millimetres with the optimizer's
        convergence status; residuals do not cause failure during the raw
        stroke fit, but failure to restore contact after filtering aborts the
        simulation.
        """
        if not math.isfinite(step_deg) or step_deg <= 0.0:
            raise ValueError("step_deg must be finite and positive")
        if not math.isfinite(concentric_duration_s) or concentric_duration_s <= 0.0:
            raise ValueError("concentric_duration_s must be finite and positive")
        if not math.isfinite(eccentric_duration_s) or eccentric_duration_s <= 0.0:
            raise ValueError("eccentric_duration_s must be finite and positive")
        framework, report = self.fit_user(
            user=user,
            wrist_dev_weight=wrist_dev_weight,
            knuckle_weight=knuckle_weight,
            knuckle_end_weight=knuckle_end_weight,
            protraction_weight=protraction_weight,
            scapular_elevation_weight=scapular_elevation_weight,
            arm_weight=arm_weight,
            pro_sup_weight=pro_sup_weight,
            plane_distance_weight=plane_distance_weight,
            shoulder_wrist_axis_distance_weight=shoulder_wrist_axis_distance_weight,
            palm_contact_weight=palm_contact_weight,
            wrist_flex_weight=wrist_flex_weight,
        )
        fit_log = list(report["grip"]["handle_fit_log"])
        fit_options = {
            "palm_contact_weight": palm_contact_weight,
            "knuckle_weight": knuckle_weight,
            "knuckle_end_weight": knuckle_end_weight,
            "plane_distance_weight": plane_distance_weight,
            "shoulder_wrist_axis_distance_weight": shoulder_wrist_axis_distance_weight,
            "pro_sup_weight": pro_sup_weight,
            "wrist_flex_weight": wrist_flex_weight,
            "wrist_dev_weight": wrist_dev_weight,
            "protraction_weight": protraction_weight,
            "scapular_elevation_weight": scapular_elevation_weight,
            "arm_weight": arm_weight,
        }

        def fit_at_current_angle():
            result = fit_handle(
                user=framework.user,
                equipment=framework.equipment,
                handle_name="right_handle",
                side="right",
                fit_shoulder_girdle=True,
                **fit_options,
            )
            if result[0] is not None:
                return result

            retry = fit_handle(
                user=framework.user,
                equipment=framework.equipment,
                handle_name="right_handle",
                side="right",
                fit_shoulder_girdle=False,
                **fit_options,
            )
            return (
                retry[0],
                retry[1],
                [
                    *result[2],
                    {"message": "retried with the previous safe shoulder-girdle pose"},
                    *retry[2],
                ],
            )

        coordinates = framework.equipment.getCoordinateSet()
        right_lever_coord = coordinates.get("right_lever-angle")
        left_lever_coord = coordinates.get("left_lever-angle")
        lower = 0.0
        upper = right_lever_coord.getRangeMax()
        if not right_lever_coord.getRangeMin() <= lower <= upper:
            raise RuntimeError("The right lever range does not include zero degrees")
        if (
            not left_lever_coord.getRangeMin()
            <= lower
            <= left_lever_coord.getRangeMax()
        ):
            raise RuntimeError("The left lever range does not include zero degrees")
        step_rad = math.radians(step_deg)
        states: list[tuple[osim.State, osim.State]] = [
            (
                osim.State(framework.user.state),
                osim.State(framework.equipment.state),
            )
        ]
        state_angles: list[float] = [lower]
        state_phases: list[str] = ["concentric"]

        def fit_grasp_and_save(angle: float, phase: str) -> bool:
            if (
                not left_lever_coord.getRangeMin()
                <= angle
                <= left_lever_coord.getRangeMax()
            ):
                raise ValueError("simulated angle is outside the left lever range")
            framework.user.state = osim.State(states[-1][0])
            framework.equipment.state = osim.State(states[-1][1])
            right_lever_coord.setValue(framework.equipment.state, angle, False)
            left_lever_coord.setValue(framework.equipment.state, angle, False)
            framework.equipment.realizePosition(framework.equipment.state)
            elbow_flexion, _, messages = fit_at_current_angle()
            fit_log.append(
                {
                    "phase": phase,
                    "lever_angle_deg": math.degrees(angle),
                    "messages": messages,
                }
            )
            if elbow_flexion is None:
                error = RuntimeError(
                    handle_fit_failure_message(
                        f"{phase} stroke", math.degrees(angle), messages
                    )
                )
                setattr(error, "handle_fit_log", fit_log)
                raise error
            framework.user.grasp_handle(
                "right",
                framework.equipment.getBodySet().get("right_handle"),
                framework.equipment.handle_meshes("right"),
                handle_state=framework.equipment.state,
            )
            contact_maintained = lock_hands_to_handles(
                framework, mirror_left_from_right=True, fit_log=fit_log
            )
            if not isinstance(contact_maintained, bool):
                raise RuntimeError("palm-anchor fitting did not report contact status")
            if not contact_maintained:
                return False
            states.append(
                (
                    osim.State(framework.user.state),
                    osim.State(framework.equipment.state),
                )
            )
            state_angles.append(angle)
            state_phases.append(phase)
            return True

        angles = np.linspace(lower, upper, int((upper - lower) // step_rad) + 1)
        pbar = tqdm(angles, desc="Fitting concentric phase")
        for angle in pbar:
            out = fit_grasp_and_save(angle, "concentric")
            if not out:
                angles = angles[angles < angle]
                pbar.close()
                break
            pbar.update(1)
        for angle in tqdm(angles[::-1], desc="Fitting eccentric phase"):
            out = fit_grasp_and_save(angle, "eccentric")

        state_times, angular_speeds = _phase_times_and_speeds(
            state_angles,
            state_phases,
            lower,
            concentric_duration_s,
            eccentric_duration_s,
        )
        upper_limb_coordinates = _upper_limb_coordinates(framework.user)
        unfiltered_coordinates = [
            {
                coordinate.getName(): float(coordinate.getValue(user_state))
                for coordinate in upper_limb_coordinates
            }
            for user_state, _ in states
        ]
        _filter_upper_limb_trajectories(
            framework.user,
            states,
            state_times,
        )
        _restore_filtered_palm_contact(
            framework,
            states,
            state_angles,
            state_phases,
            unfiltered_coordinates,
            fit_log,
        )
        _set_trajectory_coordinate_speeds(
            framework,
            states,
            state_angles,
            state_phases,
            angular_speeds,
        )

        # Record each member snapshot so the framework builds synchronized
        # trajectories.
        for time, (user_state, equipment_state) in zip(
            state_times, states, strict=True
        ):
            framework.user.state = user_state
            framework.equipment.state = equipment_state
            framework.record_state(float(time))

        return framework, fit_log
