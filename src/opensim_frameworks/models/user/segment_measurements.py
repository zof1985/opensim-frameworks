"""Body-segment measurement properties, geometric and ANSUR-derived.

Each property is either a fixed ANSUR measurement (via
``self._side_measurement_value``/``self._reference.values``, independent of
posture -- see :mod:`.scaling`) or a posture-dependent distance between two
joint centres (via ``self._joint_center``, see :mod:`.landmarks`). This is
the read-only counterpart to the explicit setters in
:mod:`.measurement_generated`.
"""

from __future__ import annotations

import math

__all__ = []


class _SegmentMeasurementMixin:
    """Every body-segment length/circumference/depth/width property."""

    @property
    def left_foot_length(self) -> float:
        """Return left foot length, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``footlength`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres;
        does not depend on the model's current posture or on any joint
        centre.

        Returns
        -------
        float
            Foot length, in metres.
        """
        return self._side_measurement_value("left_foot_length", "footlength")

    @property
    def right_foot_length(self) -> float:
        """Return right foot length, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``footlength`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres;
        does not depend on the model's current posture or on any joint
        centre. ANSUR does not record left/right foot length separately, so
        this is identical to :attr:`left_foot_length`.

        Returns
        -------
        float
            Foot length, in metres.
        """
        return self._side_measurement_value("right_foot_length", "footlength")

    @property
    def left_foot_height(self) -> float:
        """Return the left foot's height above the ground.

        Approximated as the ``y`` coordinate of :attr:`left_ankle` (the
        left ankle joint centre), at the model's current posture -- this is
        an approximation of the sole's height, not a direct measurement of
        the foot's own geometry.

        Returns
        -------
        float
            Height above the ground, in metres.
        """
        return self._joint_center("ankle_l")[1]

    @property
    def right_foot_height(self) -> float:
        """Return the right foot's height above the ground.

        Approximated as the ``y`` coordinate of :attr:`right_ankle` (the
        right ankle joint centre), at the model's current posture -- this
        is an approximation of the sole's height, not a direct measurement
        of the foot's own geometry.

        Returns
        -------
        float
            Height above the ground, in metres.
        """
        return self._joint_center("ankle_r")[1]

    @property
    def left_thigh_length(self) -> float:
        """Return the left thigh length, at the model's current posture.

        Euclidean distance between :attr:`left_hip` and :attr:`left_knee`
        in the ground frame; since both depend on posture, this changes as
        the hip/knee angles change (it is not a fixed ANSUR measurement).

        Returns
        -------
        float
            Hip-to-knee distance, in metres.
        """
        return math.dist(
            self._joint_center("hip_l"), self._joint_center("walker_knee_l")
        )

    @property
    def right_thigh_length(self) -> float:
        """Return the right thigh length, at the model's current posture.

        Euclidean distance between :attr:`right_hip` and :attr:`right_knee`
        in the ground frame; since both depend on posture, this changes as
        the hip/knee angles change (it is not a fixed ANSUR measurement).

        Returns
        -------
        float
            Hip-to-knee distance, in metres.
        """
        return math.dist(
            self._joint_center("hip_r"), self._joint_center("walker_knee_r")
        )

    @property
    def left_shank_length(self) -> float:
        """Return the left shank length, at the model's current posture.

        Euclidean distance between :attr:`left_knee` and :attr:`left_ankle`
        in the ground frame; since both depend on posture, this changes as
        the knee/ankle angles change (it is not a fixed ANSUR measurement).

        Returns
        -------
        float
            Knee-to-ankle distance, in metres.
        """
        return math.dist(
            self._joint_center("walker_knee_l"), self._joint_center("ankle_l")
        )

    @property
    def right_shank_length(self) -> float:
        """Return the right shank length, at the model's current posture.

        Euclidean distance between :attr:`right_knee` and
        :attr:`right_ankle` in the ground frame; since both depend on
        posture, this changes as the knee/ankle angles change (it is not a
        fixed ANSUR measurement).

        Returns
        -------
        float
            Knee-to-ankle distance, in metres.
        """
        return math.dist(
            self._joint_center("walker_knee_r"), self._joint_center("ankle_r")
        )

    @property
    def torso_height(self) -> float:
        """Return the trunk height, at the model's current posture.

        Euclidean distance between a "hip centre" and a "shoulder centre",
        each the midpoint between the corresponding left/right joint
        centres (:attr:`left_hip`/:attr:`right_hip` and
        :attr:`left_shoulder`/:attr:`right_shoulder`). Since all four
        inputs depend on posture, this changes with the current hip/spine
        posture (it is not a fixed ANSUR measurement).

        Returns
        -------
        float
            Hip-centre-to-shoulder-centre distance, in metres.
        """
        left_hip, right_hip = self._joint_center("hip_l"), self._joint_center("hip_r")
        left_shoulder, right_shoulder = (
            self._joint_center("acromial_l"),
            self._joint_center("acromial_r"),
        )
        hip_center = tuple((a + b) / 2 for a, b in zip(left_hip, right_hip))
        shoulder_center = tuple(
            (a + b) / 2 for a, b in zip(left_shoulder, right_shoulder)
        )
        return math.dist(hip_center, shoulder_center)

    @property
    def shoulder_width(self) -> float:
        """Return the left-to-right shoulder joint centre distance.

        Euclidean distance between :attr:`left_shoulder` and
        :attr:`right_shoulder`, at the model's current posture -- a
        geometric measurement, not read from ANSUR. See
        :attr:`biacromial_breadth` for the ANSUR-measured equivalent; the
        two are independent and generally do not coincide numerically (see
        the README's "Joint centres and derived measurements" section).

        Returns
        -------
        float
            Shoulder-to-shoulder distance, in metres.
        """
        return math.dist(
            self._joint_center("acromial_l"), self._joint_center("acromial_r")
        )

    @property
    def biacromial_breadth(self) -> float:
        """Return the biacromial breadth, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``biacromialbreadth``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres. See :attr:`shoulder_width` for the
        posture-dependent geometric equivalent derived from the model's
        shoulder joint centres.

        Returns
        -------
        float
            Biacromial breadth, in metres.
        """
        return self._reference.values["biacromialbreadth"] / 1000.0

    @property
    def left_arm_length(self) -> float:
        """Return the left upper-arm length, at the model's current posture.

        Euclidean distance between :attr:`left_shoulder` and
        :attr:`left_elbow` in the ground frame; since both depend on
        posture, this changes as the shoulder/elbow angles change (it is
        not a fixed ANSUR measurement).

        Returns
        -------
        float
            Shoulder-to-elbow distance, in metres.
        """
        return math.dist(
            self._joint_center("acromial_l"), self._joint_center("elbow_l")
        )

    @property
    def right_arm_length(self) -> float:
        """Return the right upper-arm length, at the model's current posture.

        Euclidean distance between :attr:`right_shoulder` and
        :attr:`right_elbow` in the ground frame; since both depend on
        posture, this changes as the shoulder/elbow angles change (it is
        not a fixed ANSUR measurement).

        Returns
        -------
        float
            Shoulder-to-elbow distance, in metres.
        """
        return math.dist(
            self._joint_center("acromial_r"), self._joint_center("elbow_r")
        )

    @property
    def left_forearm_length(self) -> float:
        """Return the left forearm length, at the model's current posture.

        Euclidean distance between :attr:`left_elbow` and :attr:`left_wrist`
        in the ground frame; since both depend on posture, this changes as
        the elbow/wrist/forearm angles change (it is not a fixed ANSUR
        measurement).

        Returns
        -------
        float
            Elbow-to-wrist distance, in metres.
        """
        return math.dist(
            self._joint_center("elbow_l"), self._joint_center("radius_hand_l")
        )

    @property
    def right_forearm_length(self) -> float:
        """Return the right forearm length, at the model's current posture.

        Euclidean distance between :attr:`right_elbow` and
        :attr:`right_wrist` in the ground frame; since both depend on
        posture, this changes as the elbow/wrist/forearm angles change (it
        is not a fixed ANSUR measurement).

        Returns
        -------
        float
            Elbow-to-wrist distance, in metres.
        """
        return math.dist(
            self._joint_center("elbow_r"), self._joint_center("radius_hand_r")
        )

    @property
    def left_palm_length(self) -> float:
        """Return left palm length, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``palmlength`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.
        This is the palm only, not including the fingers (ANSUR II and the
        underlying OpenSim model do not provide individual finger lengths,
        see the README).

        Returns
        -------
        float
            Palm length, in metres.
        """
        return self._side_measurement_value("left_palm_length", "palmlength")

    @property
    def right_palm_length(self) -> float:
        """Return right palm length, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``palmlength`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.
        This is the palm only, not including the fingers. ANSUR does not
        record left/right palm length separately, so this is identical to
        :attr:`left_palm_length`.

        Returns
        -------
        float
            Palm length, in metres.
        """
        return self._side_measurement_value("right_palm_length", "palmlength")

    @property
    def left_arm_circumference(self) -> float:
        """Return left flexed-biceps circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``bicepscircumferenceflexed``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Flexed biceps circumference, in metres.
        """
        return self._side_measurement_value(
            "left_arm_circumference", "bicepscircumferenceflexed"
        )

    @property
    def right_arm_circumference(self) -> float:
        """Return right flexed-biceps circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``bicepscircumferenceflexed``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres. ANSUR does not record left/right
        circumference separately, so this is identical to
        :attr:`left_arm_circumference`.

        Returns
        -------
        float
            Flexed biceps circumference, in metres.
        """
        return self._side_measurement_value(
            "right_arm_circumference", "bicepscircumferenceflexed"
        )

    @property
    def left_forearm_circumference(self) -> float:
        """Return left flexed-forearm circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR
        ``forearmcircumferenceflexed`` measurement (see
        :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Flexed forearm circumference, in metres.
        """
        return self._side_measurement_value(
            "left_forearm_circumference", "forearmcircumferenceflexed"
        )

    @property
    def right_forearm_circumference(self) -> float:
        """Return right flexed-forearm circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR
        ``forearmcircumferenceflexed`` measurement (see
        :attr:`anthropometry`), converted from millimetres to metres.
        ANSUR does not record left/right circumference separately, so this
        is identical to :attr:`left_forearm_circumference`.

        Returns
        -------
        float
            Flexed forearm circumference, in metres.
        """
        return self._side_measurement_value(
            "right_forearm_circumference", "forearmcircumferenceflexed"
        )

    @property
    def neck_circumference(self) -> float:
        """Return neck circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``neckcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Neck circumference, in metres.
        """
        return self._reference.values["neckcircumference"] / 1000.0

    @property
    def chest_circumference(self) -> float:
        """Return chest circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``chestcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Chest circumference, in metres.
        """
        return self._reference.values["chestcircumference"] / 1000.0

    @property
    def chest_depth(self) -> float:
        """Return chest (sagittal) depth, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``chestdepth`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Chest depth (front-to-back), in metres.
        """
        return self._reference.values["chestdepth"] / 1000.0

    @property
    def chest_width(self) -> float:
        """Return chest width (breadth), from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``chestbreadth`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Chest width (side-to-side), in metres.
        """
        return self._reference.values["chestbreadth"] / 1000.0

    @property
    def waist_circumference(self) -> float:
        """Return waist circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``waistcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Waist circumference, in metres.
        """
        return self._reference.values["waistcircumference"] / 1000.0

    @property
    def waist_depth(self) -> float:
        """Return waist (sagittal) depth, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``waistdepth`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Waist depth (front-to-back), in metres.
        """
        return self._reference.values["waistdepth"] / 1000.0

    @property
    def waist_width(self) -> float:
        """Return waist width (breadth), from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``waistbreadth`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Waist width (side-to-side), in metres.
        """
        return self._reference.values["waistbreadth"] / 1000.0

    @property
    def hip_circumference(self) -> float:
        """Return hip (buttock) circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``buttockcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Hip/buttock circumference, in metres.
        """
        return self._reference.values["buttockcircumference"] / 1000.0

    @property
    def hip_depth(self) -> float:
        """Return hip (buttock, sagittal) depth, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``buttockdepth`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Hip/buttock depth (front-to-back), in metres.
        """
        return self._reference.values["buttockdepth"] / 1000.0

    @property
    def hip_width(self) -> float:
        """Return hip width (breadth), from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``hipbreadth`` measurement
        (see :attr:`anthropometry`), converted from millimetres to metres.

        Returns
        -------
        float
            Hip width (side-to-side), in metres.
        """
        return self._reference.values["hipbreadth"] / 1000.0

    @property
    def left_thigh_circumference(self) -> float:
        """Return left thigh circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``thighcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Thigh circumference, in metres.
        """
        return self._side_measurement_value(
            "left_thigh_circumference", "thighcircumference"
        )

    @property
    def right_thigh_circumference(self) -> float:
        """Return right thigh circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``thighcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres. ANSUR does not record left/right
        circumference separately, so this is identical to
        :attr:`left_thigh_circumference`.

        Returns
        -------
        float
            Thigh circumference, in metres.
        """
        return self._side_measurement_value(
            "right_thigh_circumference", "thighcircumference"
        )

    @property
    def left_calf_circumference(self) -> float:
        """Return left calf circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``calfcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres.

        Returns
        -------
        float
            Calf circumference, in metres.
        """
        return self._side_measurement_value(
            "left_calf_circumference", "calfcircumference"
        )

    @property
    def right_calf_circumference(self) -> float:
        """Return right calf circumference, from ANSUR, independent of posture.

        Read directly from the resolved ANSUR ``calfcircumference``
        measurement (see :attr:`anthropometry`), converted from
        millimetres to metres. ANSUR does not record left/right
        circumference separately, so this is identical to
        :attr:`left_calf_circumference`.

        Returns
        -------
        float
            Calf circumference, in metres.
        """
        return self._side_measurement_value(
            "right_calf_circumference", "calfcircumference"
        )

    @property
    def left_thigh_depth(self) -> float:
        """Return left thigh depth, assuming a circular cross-section.

        ANSUR has no thigh breadth measurement to fit an ellipse against,
        so this derives a diameter from the thigh circumference instead
        (``circumference / pi``) -- an approximation, not a direct ANSUR
        measurement. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated thigh depth (front-to-back), in metres.
        """
        return self.left_thigh_circumference / math.pi

    @property
    def right_thigh_depth(self) -> float:
        """Return right thigh depth, assuming a circular cross-section.

        ANSUR has no thigh breadth measurement to fit an ellipse against,
        so this derives a diameter from the thigh circumference instead
        (``circumference / pi``) -- an approximation, not a direct ANSUR
        measurement. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated thigh depth (front-to-back), in metres.
        """
        return self.right_thigh_circumference / math.pi

    @property
    def left_thigh_width(self) -> float:
        """Return left thigh width, assuming a circular cross-section.

        Equal to :attr:`left_thigh_depth`: a circle has a single diameter,
        so under the circular-section assumption, width and depth
        coincide. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated thigh width (side-to-side), in metres.
        """
        return self.left_thigh_depth

    @property
    def right_thigh_width(self) -> float:
        """Return right thigh width, assuming a circular cross-section.

        Equal to :attr:`right_thigh_depth`: a circle has a single diameter,
        so under the circular-section assumption, width and depth
        coincide. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated thigh width (side-to-side), in metres.
        """
        return self.right_thigh_depth

    @property
    def left_calf_depth(self) -> float:
        """Return left calf depth, assuming a circular cross-section.

        ANSUR has no calf breadth measurement to fit an ellipse against, so
        this derives a diameter from the calf circumference instead
        (``circumference / pi``) -- an approximation, not a direct ANSUR
        measurement. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated calf depth (front-to-back), in metres.
        """
        return self.left_calf_circumference / math.pi

    @property
    def right_calf_depth(self) -> float:
        """Return right calf depth, assuming a circular cross-section.

        ANSUR has no calf breadth measurement to fit an ellipse against, so
        this derives a diameter from the calf circumference instead
        (``circumference / pi``) -- an approximation, not a direct ANSUR
        measurement. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated calf depth (front-to-back), in metres.
        """
        return self.right_calf_circumference / math.pi

    @property
    def left_calf_width(self) -> float:
        """Return left calf width, assuming a circular cross-section.

        Equal to :attr:`left_calf_depth`: a circle has a single diameter,
        so under the circular-section assumption, width and depth
        coincide. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated calf width (side-to-side), in metres.
        """
        return self.left_calf_depth

    @property
    def right_calf_width(self) -> float:
        """Return right calf width, assuming a circular cross-section.

        Equal to :attr:`right_calf_depth`: a circle has a single diameter,
        so under the circular-section assumption, width and depth
        coincide. Independent of the model's current posture.

        Returns
        -------
        float
            Estimated calf width (side-to-side), in metres.
        """
        return self.right_calf_depth
