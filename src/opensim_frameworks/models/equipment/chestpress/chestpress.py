"""Base chest-press equipment model with arm and wrist fitting."""

from __future__ import annotations

import abc
import math
import warnings
from typing import Any

import numpy as np
import opensim as osim

from opensim_frameworks import Equipment
from opensim_frameworks.framework.framework import Framework
from opensim_frameworks.models.user.user import User
from .handlefit import fit_handle, mirror_arm
from .userpositioning import position_user

__all__ = ["ChestPress"]


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
        """Verify that all bodies required by fit_user and simulate_on_user exist.

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

    def fit_user(
        self,
        user: User,
        *,
        step_deg: float = 1.0,
        wrist_extension_deg: float = 45.0,
        palm_handle_distance_m: float = 0.005,
        wrist_dev_weight: float = 1000.0,
        knuckle_weight: float = 10000.0,
        protraction_weight: float = 5.0,
        scapular_elevation_weight: float = 1000.0,
        arm_weight: float = 0.05,
        pro_sup_weight: float = 100.0,
        plane_distance_weight: float = 10000.0,
        palm_facing_weight: float = 1000.0,
        wrist_extension_weight: float = 1000.0,
        wrist_flexion_weight: float = 10000.0,
        palm_contact_weight: float = 1e6,
        handle_end_weight: float = 1e6,
    ) -> tuple[Framework, dict[str, Any]]:
        """Fit a user on the machine.

        The user is first positioned by :func:`position_user`: the shoulder
        girdle and trunk are aligned, the pelvis is seated, the hips are
        abducted, the feet are returned to ground contact, and both shoulders
        and forearms are rotated by 90 degrees. The fitting then proceeds:

        1. Advance the right lever by ``step_deg`` and fit the right arm and
           wrist at each angle, recording each successful palm position.
        2. Return the right lever to zero and refit the arm and wrist to the
           saved palm position.
        3. Mirror the fitted right arm and hand coordinates to the left arm.

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
        wrist_extension_deg : float, default=45.0
            Wrist extension target for the arm and wrist fits.
        palm_handle_distance_m : float, default=0.005
            Target distance in metres between the palm marker and the
            enclosing cylindrical approximation of the handle surface.
        wrist_dev_weight : float, default=1000.0
            Weight for neutral radial/ulnar deviation.
        knuckle_weight : float, default=10000.0
            Weight for the index MCP abduction coordinate.
        protraction_weight : float, default=5.0
            Weight penalizing scapular protraction relative to its coordinate
            range minimum.
        scapular_elevation_weight : float, default=1000.0
            Weight penalizing positive scapular elevation.
        arm_weight : float, default=0.05
            Weight for the press-posture tie-break.
        pro_sup_weight : float, default=100.0
            Weight for the neutral forearm rotation of 90 degrees.
        plane_distance_weight : float, default=10000.0
            Weight for the palm's local z displacement from the handle marker.
        palm_facing_weight : float, default=1000.0
            Weight for the angle of the palm position in the handle body's
            local yz plane; it does not measure palm orientation.
        wrist_extension_weight : float, default=1000.0
            Weight for the phase-3 wrist-extension target.
        wrist_flexion_weight : float, default=10000.0
            Weight strongly penalizing positive wrist flexion.
        palm_contact_weight : float, default=1e6
            Weight for palm contact.
        handle_end_weight : float, default=1e6
            Weight for keeping the hand within the ends of the handle.

        Returns
        -------
        tuple of (opensim_frameworks.Framework, dict)
            Framework containing ``user`` and ``equipment``, and a report with
            the keys ``back``, ``seat``, ``hips``, ``feet`` and ``grip``.

        Raises
        ------
        ValueError
            If ``step_deg`` is invalid or no valid palm positions are found.
        RuntimeError
            If a lever's range does not include zero degrees or the final
            arm fit does not converge.
        """
        if not isinstance(user, User):
            raise TypeError("user must be an opensim_frameworks.User")
        if not math.isfinite(step_deg) or step_deg <= 0.0:
            raise ValueError("step_deg must be finite and positive")

        F = Framework(user=user, equipment=self)
        user = F.user
        equipment = F.equipment

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

        palm_positions = []
        angle = zero_angle
        upper = lever_coordinates["right"].getRangeMax()
        while True:
            lever_coordinates["right"].setValue(equipment.state, angle, False)
            equipment.realizePosition(equipment.state)
            elbow_flexion, palm_position = fit_handle(
                user=user,
                equipment=equipment,
                handle_name="right_handle",
                side="right",
                palm_position_handle=None,
                palm_handle_distance_m=palm_handle_distance_m,
                wrist_extension_deg=wrist_extension_deg,
                palm_contact_weight=palm_contact_weight,
                palm_facing_weight=palm_facing_weight,
                knuckle_weight=knuckle_weight,
                plane_distance_weight=plane_distance_weight,
                handle_end_weight=handle_end_weight,
                pro_sup_weight=pro_sup_weight,
                wrist_dev_weight=wrist_dev_weight,
                wrist_extension_weight=wrist_extension_weight,
                protraction_weight=protraction_weight,
                scapular_elevation_weight=scapular_elevation_weight,
                wrist_flexion_weight=wrist_flexion_weight,
                arm_weight=arm_weight,
            )
            if elbow_flexion is None:
                break
            palm_positions.append(palm_position)
            if elbow_flexion < 1.0:
                break
            next_angle = min(angle + math.radians(step_deg), upper)
            if next_angle <= angle:
                warnings.warn(
                    f"The right elbow reached only {angle} degrees of flexion"
                )
                break
            angle = next_angle

        if len(palm_positions) == 0:
            raise ValueError("No valid palm positions found")
        mean_palm_position = np.mean(palm_positions, axis=0)

        lever_coordinates["right"].setValue(equipment.state, zero_angle, False)
        equipment.realizePosition(equipment.state)
        elbow_flexion, fitted_palm_position = fit_handle(
            user=user,
            equipment=equipment,
            handle_name="right_handle",
            side="right",
            palm_position_handle=mean_palm_position,
            palm_handle_distance_m=palm_handle_distance_m,
            wrist_extension_deg=wrist_extension_deg,
            palm_contact_weight=palm_contact_weight,
            palm_facing_weight=palm_facing_weight,
            knuckle_weight=knuckle_weight,
            plane_distance_weight=plane_distance_weight,
            handle_end_weight=handle_end_weight,
            pro_sup_weight=pro_sup_weight,
            wrist_dev_weight=wrist_dev_weight,
            wrist_extension_weight=wrist_extension_weight,
            protraction_weight=protraction_weight,
            scapular_elevation_weight=scapular_elevation_weight,
            wrist_flexion_weight=wrist_flexion_weight,
            arm_weight=arm_weight,
        )

        if elbow_flexion is None or fitted_palm_position is None:
            raise RuntimeError("The final right arm fit did not converge")

        mirror_arm(user=user, to_mirror="right")

        report["grip"] = {
            "sweep_angle_deg": math.degrees(angle),
            "elbow_flexion_deg": elbow_flexion,
            "palm_handle_position_m": fitted_palm_position,
        }

        return F, report

    def simulate_on_user(
        self,
        user: User,
        step_deg: float = 1.0,
        duration_s: float = 5.0,
        fps: int = 30,
        *,
        wrist_extension_deg: float = 45.0,
        palm_handle_distance_m: float = 0.005,
        wrist_dev_weight: float = 1000.0,
        knuckle_weight: float = 10000.0,
        protraction_weight: float = 5.0,
        scapular_elevation_weight: float = 1000.0,
        arm_weight: float = 0.05,
        pro_sup_weight: float = 100.0,
        plane_distance_weight: float = 10000.0,
        palm_facing_weight: float = 1000.0,
        wrist_extension_weight: float = 1000.0,
        wrist_flexion_weight: float = 10000.0,
        palm_contact_weight: float = 1e6,
        handle_end_weight: float = 1e6,
    ) -> Framework:
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
        fps : int, default=30
            Positive playback frame rate in frames per second. It does not
            change the number of solver states retained in the trajectory.
        wrist_extension_deg : float, default=45.0
            Target wrist extension for each handle fit.
        palm_handle_distance_m : float, default=0.005
            Target distance between the palm marker and the enclosing
            cylindrical approximation of the handle surface.
        wrist_dev_weight : float, default=1000.0
            Weight for neutral radial/ulnar deviation.
        knuckle_weight : float, default=10000.0
            Weight for the index MCP abduction coordinate.
        protraction_weight : float, default=5.0
            Weight penalizing scapular protraction relative to its coordinate
            range minimum.
        scapular_elevation_weight : float, default=1000.0
            Weight penalizing positive scapular elevation.
        arm_weight : float, default=0.05
            Weight for the arm-posture tie-break.
        pro_sup_weight : float, default=100.0
            Weight for neutral forearm rotation.
        plane_distance_weight : float, default=10000.0
            Weight for the palm's local z displacement from the handle marker.
        palm_facing_weight : float, default=1000.0
            Weight for the angle of the palm position in the handle body's
            local yz plane; it does not measure palm orientation.
        wrist_extension_weight : float, default=1000.0
            Weight for the wrist-extension target.
        wrist_flexion_weight : float, default=10000.0
            Weight penalizing positive wrist flexion.
        palm_contact_weight : float, default=1e6
            Weight for palm contact.
        handle_end_weight : float, default=1e6
            Weight for keeping the hand within the handle ends.

        Returns
        -------
        framework : opensim_frameworks.Framework
            Framework containing the fitted user and equipment with recorded
            movement states.

        Raises
        ------
        ValueError
            If a numeric parameter is invalid or fewer than two valid states
            are produced.

        Notes
        -----
        The state count is determined by the successful lever-step fits.
        ``fps`` is a playback setting for visualization, not a resampling rate.
        """
        if not math.isfinite(step_deg) or step_deg <= 0.0:
            raise ValueError("step_deg must be finite and positive")
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
            raise ValueError("fps must be a positive integer")

        framework, report = self.fit_user(
            user=user,
            step_deg=step_deg,
            wrist_extension_deg=wrist_extension_deg,
            palm_handle_distance_m=palm_handle_distance_m,
            wrist_dev_weight=wrist_dev_weight,
            knuckle_weight=knuckle_weight,
            protraction_weight=protraction_weight,
            scapular_elevation_weight=scapular_elevation_weight,
            arm_weight=arm_weight,
            pro_sup_weight=pro_sup_weight,
            plane_distance_weight=plane_distance_weight,
            palm_facing_weight=palm_facing_weight,
            wrist_extension_weight=wrist_extension_weight,
            wrist_flexion_weight=wrist_flexion_weight,
            palm_contact_weight=palm_contact_weight,
            handle_end_weight=handle_end_weight,
        )

        right_lever_coord = framework.equipment.getCoordinateSet()
        right_lever_coord = right_lever_coord.get("right_lever-angle")
        lower = right_lever_coord.getRangeMin()
        upper = right_lever_coord.getRangeMax()

        palm_position_m = report["grip"]["palm_handle_position_m"]
        step_rad = math.radians(step_deg)
        angle = lower
        max_angle = lower
        states = []
        while True:

            right_lever_coord.setValue(framework.equipment.state, angle, False)
            framework.equipment.realizePosition(framework.equipment.state)

            elbow_flexion, _ = fit_handle(
                user=framework.user,
                equipment=framework.equipment,
                handle_name="right_handle",
                side="right",
                palm_position_handle=palm_position_m,
                palm_handle_distance_m=palm_handle_distance_m,
                wrist_extension_deg=wrist_extension_deg,
                palm_contact_weight=palm_contact_weight,
                palm_facing_weight=palm_facing_weight,
                knuckle_weight=knuckle_weight,
                plane_distance_weight=plane_distance_weight,
                handle_end_weight=handle_end_weight,
                pro_sup_weight=pro_sup_weight,
                wrist_dev_weight=wrist_dev_weight,
                wrist_extension_weight=wrist_extension_weight,
                protraction_weight=protraction_weight,
                scapular_elevation_weight=scapular_elevation_weight,
                wrist_flexion_weight=wrist_flexion_weight,
                arm_weight=arm_weight,
            )

            mirror_arm(framework.user, "right")

            if elbow_flexion is None:
                break
            if elbow_flexion < 1.0:
                break

            states.append(osim.State(framework.state))

            next_angle = angle + step_rad
            if next_angle <= upper:
                angle = next_angle
                max_angle = max(max_angle, angle)
            else:
                warnings.warn(
                    f"The right elbow reached only {max_angle} degrees of flexion"
                )
                break

        while True:
            next_angle = angle - step_rad
            if next_angle >= lower:
                angle = next_angle
                max_angle = max(max_angle, angle)
                right_lever_coord.setValue(framework.equipment.state, angle, False)
                framework.equipment.realizePosition(framework.equipment.state)

                elbow_flexion, _ = fit_handle(
                    user=framework.user,
                    equipment=framework.equipment,
                    handle_name="right_handle",
                    side="right",
                    palm_position_handle=palm_position_m,
                    palm_handle_distance_m=palm_handle_distance_m,
                    wrist_extension_deg=wrist_extension_deg,
                    palm_contact_weight=palm_contact_weight,
                    palm_facing_weight=palm_facing_weight,
                    knuckle_weight=knuckle_weight,
                    plane_distance_weight=plane_distance_weight,
                    handle_end_weight=handle_end_weight,
                    pro_sup_weight=pro_sup_weight,
                    wrist_dev_weight=wrist_dev_weight,
                    wrist_extension_weight=wrist_extension_weight,
                    protraction_weight=protraction_weight,
                    scapular_elevation_weight=scapular_elevation_weight,
                    wrist_flexion_weight=wrist_flexion_weight,
                    arm_weight=arm_weight,
                )

                if elbow_flexion is None or elbow_flexion < 1.0:
                    break

                mirror_arm(framework.user, "right")
                states.append(osim.State(framework.state))
            else:
                break

        return framework
