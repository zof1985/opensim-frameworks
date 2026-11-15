"""Base chest-press equipment model with arm and wrist fitting."""

from __future__ import annotations

import abc
import math
import warnings
from typing import Any

import numpy as np
import opensim as osim
from scipy.interpolate import CubicSpline

from opensim_frameworks import Equipment
from opensim_frameworks.framework.framework import Framework
from opensim_frameworks.models.user.hand import FINGERS
from opensim_frameworks.models.user.user import User
from opensim_frameworks.utils.motion_profiles import (
    beta_time_for_position,
    beta_velocity_profile,
    gaussian_time_for_position,
    gaussian_velocity_profile,
)
from .handlefit import (
    handle_fit_failure_message,
    fit_handle,
    mirror_arm_with_log,
)
from .userpositioning import position_user

__all__ = ["ChestPress"]


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
            travel
            / concentric_duration_s
            * float(beta_velocity_profile(time_fraction))
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
                [
                    coordinate.getValue(state[member_index])
                    for state in states
                ],
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
        step_deg: float = 1.0,
        wrist_dev_weight: float = 1000.0,
        knuckle_weight: float = 1e5,
        knuckle_end_weight: float = 1e8,
        protraction_weight: float = 5.0,
        scapular_elevation_weight: float = 1000.0,
        arm_weight: float = 100000.0,
        pro_sup_weight: float = 100000.0,
        plane_distance_weight: float = 1e6,
        wrist_flexion_weight: float = 10000.0,
        palm_contact_weight: float = 1e7,
    ) -> tuple[Framework, dict[str, Any]]:
        """Fit a user on the machine.

        The user is first positioned by :func:`position_user`: the shoulder
        girdle and trunk are aligned, the pelvis is seated, the hips are
        abducted, the feet are returned to ground contact, and both shoulders
        and forearms are rotated by 90 degrees. The fitting then proceeds:

        1. Advance the right lever by ``step_deg`` and fit the right arm and
           wrist at each angle to find the reachable sweep.
        2. Return the right lever to zero and refit the arm and wrist,
           starting from the initial zero-lever fit.
        3. Close the right fingers on the handle, then mirror all upper-limb
           coordinates to the left side.

        The weights shared with :func:`fit_handle` are passed to each
        handle fit during the lever sweep and to the subsequent arm fit.

        The initial foot pose is recorded before rotating the body so the feet
        can be returned to their original ground contact in phase 4.

        Parameters
        ----------
        user : opensim_frameworks.User
            Scaled user whose feet are initially in contact with ground.
        step_deg : float, default=1.0
            Independent lever-angle increment used for the arm-and-wrist sweep
            to 1 degree of elbow flexion.
        wrist_dev_weight : float, default=1000.0
            Weight for neutral radial/ulnar deviation.
        knuckle_weight : float, default=1e5
            Weight for alignment of the knuckle line and handle axis.
        knuckle_end_weight : float, default=1e8
            Weight penalizing the index MCP center's deviation from its target
            distance to the handle center.
        protraction_weight : float, default=5.0
            Weight penalizing scapular protraction relative to its coordinate
            range minimum.
        scapular_elevation_weight : float, default=1000.0
            Weight penalizing positive scapular elevation.
        arm_weight : float, default=100000.0
            Weight penalizing external upper-arm rotation.
        pro_sup_weight : float, default=100000.0
            Weight penalizing forearm rotation outside 75-115 degrees.
        plane_distance_weight : float, default=1e6
            Weight for the elbow's distance from the shoulder-wrist plane
            parallel to the handle axis.
        wrist_flexion_weight : float, default=10000.0
            Weight strongly penalizing positive wrist flexion.
        palm_contact_weight : float, default=1e7
            Weight for the wrist-center-to-axis distance target used by
            :func:`fit_handle`.
        Returns
        -------
        tuple of (opensim_frameworks.Framework, dict)
            Framework containing ``user`` and ``equipment``, and a report with
            the keys ``back``, ``seat``, ``hips``, ``feet`` and ``grip``.
            ``grip["handle_fit_log"]`` contains one entry per handle fit,
            with its phase, lever angle in degrees and messages. The final
            entry also includes messages from mirroring the left arm.
            ``grip["fingers"]["left"]`` is None: the mirrored left grasp is
            not independently fitted or checked against the handle.

        Raises
        ------
        ValueError
            If ``step_deg`` is invalid or no handle fit succeeds in the sweep.
        RuntimeError
            If a lever's range does not include zero degrees or the final
            arm fit does not converge.

        Notes
        -----
        If all sweep fits fail or the final fit fails, the raised exception
        exposes the accumulated messages as ``handle_fit_log``.
        """
        if not isinstance(user, User):
            raise TypeError("user must be an opensim_frameworks.User")
        if not math.isfinite(step_deg) or step_deg <= 0.0:
            raise ValueError("step_deg must be finite and positive")
        framework = Framework(user=user, equipment=self)
        user = framework.user
        equipment = framework.equipment

        report = position_user(
            user,
            equipment,
            backrest_angle_deg=self._backrest_angle_deg,
        )

        lever_coordinates = {
            side: equipment.getCoordinateSet().get(f"{side}_lever-angle")
            for side in ("right", "left")
        }
        zero_angle = 0.0
        for side, coordinate in lever_coordinates.items():
            if not coordinate.getRangeMin() <= zero_angle <= coordinate.getRangeMax():
                raise RuntimeError(
                    f"The {side} lever range does not include zero degrees"
                )
            coordinate.setValue(equipment.state, zero_angle, False)
        equipment.realizePosition(equipment.state)

        fit_log = []
        initial_pose = None
        angle = zero_angle
        upper = lever_coordinates["right"].getRangeMax()
        while True:
            lever_coordinates["right"].setValue(equipment.state, angle, False)
            equipment.realizePosition(equipment.state)
            elbow_flexion, _, messages = fit_handle(
                user=user,
                equipment=equipment,
                handle_name="right_handle",
                side="right",
                palm_contact_weight=palm_contact_weight,
                knuckle_weight=knuckle_weight,
                knuckle_end_weight=knuckle_end_weight,
                plane_distance_weight=plane_distance_weight,
                pro_sup_weight=pro_sup_weight,
                wrist_dev_weight=wrist_dev_weight,
                protraction_weight=protraction_weight,
                scapular_elevation_weight=scapular_elevation_weight,
                wrist_flexion_weight=wrist_flexion_weight,
                arm_weight=arm_weight,
            )
            fit_log.append(
                {
                    "phase": "sweep",
                    "lever_angle_deg": math.degrees(angle),
                    "messages": messages,
                }
            )
            if elbow_flexion is None:
                break
            if initial_pose is None:
                coordinates = user.getCoordinateSet()
                initial_pose = [
                    (
                        coordinates.get(index),
                        coordinates.get(index).getValue(user.state),
                    )
                    for index in range(coordinates.getSize())
                ]
            if elbow_flexion < 1.0:
                break
            next_angle = min(angle + math.radians(step_deg), upper)
            if next_angle <= angle:
                warnings.warn(
                    f"The right elbow reached only {angle} degrees of flexion"
                )
                break
            angle = next_angle

        if initial_pose is None:
            error = ValueError("No valid handle fits found")
            setattr(error, "handle_fit_log", fit_log)
            raise error

        lever_coordinates["right"].setValue(equipment.state, zero_angle, False)
        equipment.realizePosition(equipment.state)
        for coordinate, value in initial_pose:
            if coordinate.getValue(user.state) != value:
                coordinate.setValue(user.state, value, False)
        user.realizePosition(user.state)
        elbow_flexion, fitted_palm_position, messages = fit_handle(
            user=user,
            equipment=equipment,
            handle_name="right_handle",
            side="right",
            palm_contact_weight=palm_contact_weight,
            knuckle_weight=knuckle_weight,
            knuckle_end_weight=knuckle_end_weight,
            plane_distance_weight=plane_distance_weight,
            pro_sup_weight=pro_sup_weight,
            wrist_dev_weight=wrist_dev_weight,
            protraction_weight=protraction_weight,
            scapular_elevation_weight=scapular_elevation_weight,
            wrist_flexion_weight=wrist_flexion_weight,
            arm_weight=arm_weight,
        )
        fit_log.append(
            {
                "phase": "final",
                "lever_angle_deg": math.degrees(zero_angle),
                "messages": messages,
            }
        )

        if elbow_flexion is None or fitted_palm_position is None:
            error = RuntimeError("The final right arm fit did not converge")
            setattr(error, "handle_fit_log", fit_log)
            raise error

        grasp = {
            "right": user.grasp_handle(
                "right",
                equipment.getBodySet().get("right_handle"),
                equipment.handle_meshes("right"),
            ),
            "left": None,
        }
        mirror_arm_with_log(user, fit_log)
        for finger in FINGERS:
            for joint in ("mcp", "pip", "dip"):
                user.set_finger_coordinate_locked("left", finger, joint, "flexion", True)
            user.set_finger_coordinate_locked("left", finger, "mcp", "abduction", True)

        report["grip"] = {
            "sweep_angle_deg": math.degrees(angle),
            "elbow_flexion_deg": elbow_flexion,
            "palm_handle_position_m": fitted_palm_position,
            "fingers": grasp,
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
        wrist_dev_weight: float = 1000.0,
        knuckle_weight: float = 1e5,
        knuckle_end_weight: float = 1e8,
        protraction_weight: float = 5.0,
        scapular_elevation_weight: float = 1000.0,
        arm_weight: float = 100000.0,
        pro_sup_weight: float = 100000.0,
        plane_distance_weight: float = 1e6,
        wrist_flexion_weight: float = 10000.0,
        palm_contact_weight: float = 1e7,
    ) -> tuple[Framework, list[dict[str, Any]]]:
        """Fit and record one outward-and-return press movement.

        The user is fitted first. The right arm is refitted at successive
        lever angles on the outward and return strokes, with the left arm
        mirrored from the right. Every successful composite state is retained.

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
        wrist_dev_weight : float, default=1000.0
            Weight for neutral radial/ulnar deviation.
        knuckle_weight : float, default=1e5
            Weight for alignment of the knuckle line and handle axis.
        knuckle_end_weight : float, default=1e8
            Weight penalizing the index MCP center's deviation from its target
            distance to the handle center.
        protraction_weight : float, default=5.0
            Weight penalizing scapular protraction relative to its coordinate
            range minimum.
        scapular_elevation_weight : float, default=1000.0
            Weight penalizing positive scapular elevation.
        arm_weight : float, default=100000.0
            Weight penalizing external upper-arm rotation.
        pro_sup_weight : float, default=100000.0
            Weight penalizing forearm rotation outside 75-115 degrees.
        plane_distance_weight : float, default=1e6
            Weight for the elbow's distance from the shoulder-wrist plane
            parallel to the handle axis.
        wrist_flexion_weight : float, default=10000.0
            Weight penalizing positive wrist flexion.
        palm_contact_weight : float, default=1e7
            Weight for the wrist-center-to-axis distance target used by
            :func:`fit_handle`.
        Returns
        -------
        tuple of (opensim_frameworks.Framework, list of dict)
            Framework containing synchronized trajectories and the handle-fit
            log, including the initial fit and each outward and return step.
            Each entry contains ``phase``, ``lever_angle_deg`` and ``messages``;
            successful steps include messages from mirroring the left arm.

        Raises
        ------
        ValueError
            If a numeric parameter is invalid, a simulated right-lever angle
            is outside the left lever's range, or fewer than two valid states
            are produced.

        Notes
        -----
        Concentric state times follow the inverse CDF of a Beta(3, 5) speed
        profile; eccentric state times follow the inverse CDF of a normalized
        Gaussian speed profile with zero endpoint speed. Coordinate speeds are
        derived from the fitted coordinate positions and lever-angle profile.
        Accelerations are not separately recorded. Sampling for playback is
        handled separately by the framework viewer. If fewer than two states
        are produced, the exception exposes the accumulated handle-fit
        messages as ``handle_fit_log``.
        """
        if not math.isfinite(step_deg) or step_deg <= 0.0:
            raise ValueError("step_deg must be finite and positive")
        if not math.isfinite(concentric_duration_s) or concentric_duration_s <= 0.0:
            raise ValueError("concentric_duration_s must be finite and positive")
        if not math.isfinite(eccentric_duration_s) or eccentric_duration_s <= 0.0:
            raise ValueError("eccentric_duration_s must be finite and positive")
        framework, report = self.fit_user(
            user=user,
            step_deg=step_deg,
            wrist_dev_weight=wrist_dev_weight,
            knuckle_weight=knuckle_weight,
            knuckle_end_weight=knuckle_end_weight,
            protraction_weight=protraction_weight,
            scapular_elevation_weight=scapular_elevation_weight,
            arm_weight=arm_weight,
            pro_sup_weight=pro_sup_weight,
            plane_distance_weight=plane_distance_weight,
            wrist_flexion_weight=wrist_flexion_weight,
            palm_contact_weight=palm_contact_weight,
        )

        right_lever_coord = framework.equipment.getCoordinateSet()
        right_lever_coord = right_lever_coord.get("right_lever-angle")
        lower = right_lever_coord.getRangeMin()
        upper = right_lever_coord.getRangeMax()
        left_lever_coord = framework.equipment.getCoordinateSet().get(
            "left_lever-angle"
        )
        fit_log = list(report["grip"]["handle_fit_log"])
        step_rad = math.radians(step_deg)
        angle = lower
        states: list[tuple[osim.State, osim.State]] = []
        state_angles: list[float] = []
        state_phases: list[str] = []
        while True:

            if (
                not left_lever_coord.getRangeMin()
                <= angle
                <= left_lever_coord.getRangeMax()
            ):
                raise ValueError("simulated angle is outside the left lever range")
            right_lever_coord.setValue(framework.equipment.state, angle, False)
            left_lever_coord.setValue(framework.equipment.state, angle, False)
            framework.equipment.realizePosition(framework.equipment.state)

            elbow_flexion, _, messages = fit_handle(
                user=framework.user,
                equipment=framework.equipment,
                handle_name="right_handle",
                side="right",
                palm_contact_weight=palm_contact_weight,
                knuckle_weight=knuckle_weight,
                knuckle_end_weight=knuckle_end_weight,
                plane_distance_weight=plane_distance_weight,
                pro_sup_weight=pro_sup_weight,
                wrist_dev_weight=wrist_dev_weight,
                protraction_weight=protraction_weight,
                scapular_elevation_weight=scapular_elevation_weight,
                wrist_flexion_weight=wrist_flexion_weight,
                arm_weight=arm_weight,
            )
            fit_log.append(
                {
                    "phase": "outward",
                    "lever_angle_deg": math.degrees(angle),
                    "messages": messages,
                }
            )

            if elbow_flexion is None:
                warnings.warn(
                    handle_fit_failure_message(
                        "concentric press (outward)",
                        math.degrees(angle),
                        messages,
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            mirror_arm_with_log(framework.user, fit_log)
            states.append(
                (
                    osim.State(framework.user.state),
                    osim.State(framework.equipment.state),
                )
            )
            state_angles.append(angle)
            state_phases.append("concentric")

            if elbow_flexion < 1.0:
                break
            if angle >= upper:
                warnings.warn(
                    f"The right lever reached {math.degrees(angle):.2f} degrees "
                    f"with the elbow still flexed by {elbow_flexion:.2f} degrees",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break
            angle = min(angle + step_rad, upper)

        if states:
            # Resume the return stroke from the last successful outward fit.
            framework.user.state = osim.State(states[-1][0])
            framework.equipment.state = osim.State(states[-1][1])
            angle = right_lever_coord.getValue(framework.equipment.state)
            while angle > lower:
                angle = max(lower, angle - step_rad)
                right_lever_coord.setValue(framework.equipment.state, angle, False)
                left_lever_coord.setValue(framework.equipment.state, angle, False)
                framework.equipment.realizePosition(framework.equipment.state)

                elbow_flexion, _, messages = fit_handle(
                    user=framework.user,
                    equipment=framework.equipment,
                    handle_name="right_handle",
                    side="right",
                    palm_contact_weight=palm_contact_weight,
                    knuckle_weight=knuckle_weight,
                    knuckle_end_weight=knuckle_end_weight,
                    plane_distance_weight=plane_distance_weight,
                    pro_sup_weight=pro_sup_weight,
                    wrist_dev_weight=wrist_dev_weight,
                    protraction_weight=protraction_weight,
                    scapular_elevation_weight=scapular_elevation_weight,
                    wrist_flexion_weight=wrist_flexion_weight,
                    arm_weight=arm_weight,
                )
                fit_log.append(
                    {
                        "phase": "return",
                        "lever_angle_deg": math.degrees(angle),
                        "messages": messages,
                    }
                )

                if elbow_flexion is None:
                    warnings.warn(
                        handle_fit_failure_message(
                            "eccentric return",
                            math.degrees(angle),
                            messages,
                        ),
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    break

                mirror_arm_with_log(framework.user, fit_log)
                states.append(
                    (
                        osim.State(framework.user.state),
                        osim.State(framework.equipment.state),
                    )
                )
                state_angles.append(angle)
                state_phases.append("eccentric")

        if len(states) < 2:
            error = ValueError("fewer than two valid movement states were produced")
            setattr(error, "handle_fit_log", fit_log)
            raise error

        state_times, angular_speeds = _phase_times_and_speeds(
            state_angles,
            state_phases,
            lower,
            concentric_duration_s,
            eccentric_duration_s,
        )
        _set_trajectory_coordinate_speeds(
            framework,
            states,
            state_angles,
            state_phases,
            angular_speeds,
        )

        # Record each member snapshot so the framework builds synchronized trajectories.
        for time, (user_state, equipment_state) in zip(
            state_times, states, strict=True
        ):
            framework.user.state = user_state
            framework.equipment.state = equipment_state
            framework.record_state(float(time))

        return framework, fit_log
