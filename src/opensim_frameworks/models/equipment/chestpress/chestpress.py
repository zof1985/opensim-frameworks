"""Base chest-press equipment model with arm and wrist fitting."""

from __future__ import annotations

import abc
import math
import sys
import warnings
from typing import Any

import numpy as np
import opensim as osim

from opensim_frameworks import Equipment
from opensim_frameworks.framework.framework import Framework
from opensim_frameworks.models.user.hand import FINGERS
from opensim_frameworks.models.user.user import User
from .handlefit import _capture_messages, fit_handle, mirror_arm
from .userpositioning import position_user

__all__ = ["ChestPress"]


def _mirror_arm_with_log(user: User, fit_log: list[dict[str, Any]]) -> None:
    with _capture_messages() as messages:
        completed = False
        try:
            mirror_arm(user=user, to_mirror="right")
            completed = True
        finally:
            fit_log[-1]["messages"].extend(messages)
            error = sys.exc_info()[1] if not completed else None
            if error is not None:
                BaseException.__setattr__(error, "handle_fit_log", fit_log)


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
            Weight penalizing knuckle projections outside the handle ends.
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
            Weight for the elbow's distance from the shoulder-palm plane
            parallel to the handle axis.
        wrist_flexion_weight : float, default=10000.0
            Weight strongly penalizing positive wrist flexion.
        palm_contact_weight : float, default=1e7
            Weight for palm contact.
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
        _mirror_arm_with_log(user, fit_log)
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
        duration_s: float = 5.0,
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
        duration_s : float, default=5.0
            Total duration assigned to the state sequence, in seconds.
        wrist_dev_weight : float, default=1000.0
            Weight for neutral radial/ulnar deviation.
        knuckle_weight : float, default=1e5
            Weight for alignment of the knuckle line and handle axis.
        knuckle_end_weight : float, default=1e8
            Weight penalizing knuckle projections outside the handle ends.
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
            Weight for the elbow's distance from the shoulder-palm plane
            parallel to the handle axis.
        wrist_flexion_weight : float, default=10000.0
            Weight penalizing positive wrist flexion.
        palm_contact_weight : float, default=1e7
            Weight for palm contact.
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
        The successful steps are uniformly distributed over ``duration_s``.
        Sampling for playback is handled separately by the framework viewer.
        If fewer than two states are produced, the exception exposes the
        accumulated handle-fit messages as ``handle_fit_log``.
        """
        if not math.isfinite(step_deg) or step_deg <= 0.0:
            raise ValueError("step_deg must be finite and positive")
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
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
                    f"Outward handle fit failed at {math.degrees(angle):.2f} degrees",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            _mirror_arm_with_log(framework.user, fit_log)
            states.append(
                (
                    osim.State(framework.user.state),
                    osim.State(framework.equipment.state),
                )
            )

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
                        f"Return handle fit failed at {math.degrees(angle):.2f} degrees",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    break

                _mirror_arm_with_log(framework.user, fit_log)
                states.append(
                    (
                        osim.State(framework.user.state),
                        osim.State(framework.equipment.state),
                    )
                )

        if len(states) < 2:
            error = ValueError("fewer than two valid movement states were produced")
            setattr(error, "handle_fit_log", fit_log)
            raise error

        # Record each member snapshot so the framework builds synchronized trajectories.
        for time, (user_state, equipment_state) in zip(
            np.linspace(0.0, duration_s, len(states)), states, strict=True
        ):
            framework.user.state = user_state
            framework.equipment.state = equipment_state
            framework.record_state(float(time))

        return framework, fit_log
