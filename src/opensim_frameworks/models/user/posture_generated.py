from __future__ import annotations

__all__ = []


class _PostureMixin:
    """Every posture setter/getter pair.

    Every angle setter accepts optional ``range_min`` and ``range_max``
    arguments in degrees. Supplied limits replace the current coordinate
    bounds; omitted limits remain unchanged. When either limit is supplied,
    the requested angle must lie within the resulting range.
    """

    def set_left_hip_flexionextension(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the left hip flexion/extension angle, in degrees.

        Writes the OpenSim coordinate ``hip_flexion_l``. Positive values
        flex the hip (thigh swings forward/upward relative to the pelvis);
        negative values extend it (thigh swings backward). This direction
        is inferred from the coordinate's own range in the bundled model
        (-30 deg of extension vs. 120 deg of flexion), which mirrors the
        real anatomical asymmetry between hip flexion and extension.

        Like every posture setter, this only writes the raw coordinate
        value (see "Raw posture setters" in the project README):
        call :meth:`update_state` before reading anything derived from the
        new posture (joint centres, ``com``, muscle lengths/forces...).

        Parameters
        ----------
        degrees : float
            Target hip flexion/extension angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-30, 120]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built, regardless of what the
            base .osim file specifies -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('hip_flexion_l', degrees, range_min, range_max)

    @property
    def left_hip_flexionextension(self) -> float:
        """Return the current left hip flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``hip_flexion_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_hip_flexionextension` or any other
        writer of this coordinate). Positive values mean the hip is flexed
        (thigh forward/upward relative to the pelvis); negative values
        mean it is extended (thigh backward).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current left hip flexion/extension angle, in degrees.
        """
        return self._coordinate_degrees('hip_flexion_l')

    def set_right_hip_flexionextension(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the right hip flexion/extension angle, in degrees.

        Writes the OpenSim coordinate ``hip_flexion_r``. Positive values
        flex the hip (thigh swings forward/upward relative to the pelvis);
        negative values extend it (thigh swings backward). This direction
        is inferred from the coordinate's own range in the bundled model
        (-30 deg of extension vs. 120 deg of flexion), which mirrors the
        real anatomical asymmetry between hip flexion and extension.

        Like every posture setter, this only writes the raw coordinate
        value (see "Raw posture setters" in the project README):
        call :meth:`update_state` before reading anything derived from the
        new posture (joint centres, ``com``, muscle lengths/forces...).

        Parameters
        ----------
        degrees : float
            Target hip flexion/extension angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-30, 120]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built, regardless of what the
            base .osim file specifies -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('hip_flexion_r', degrees, range_min, range_max)

    @property
    def right_hip_flexionextension(self) -> float:
        """Return the current right hip flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``hip_flexion_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_hip_flexionextension` or any other
        writer of this coordinate). Positive values mean the hip is flexed
        (thigh forward/upward relative to the pelvis); negative values
        mean it is extended (thigh backward).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current right hip flexion/extension angle, in degrees.
        """
        return self._coordinate_degrees('hip_flexion_r')

    def set_left_hip_adduction(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the left hip adduction/abduction angle, in degrees.

        Writes the OpenSim coordinate ``hip_adduction_l``. Positive values
        adduct the hip (thigh swings toward the midline); negative values
        abduct it (thigh swings away from the midline) -- consistent both
        with the coordinate's own name and with its asymmetric range in
        the bundled model (-50 deg of abduction vs. 30 deg of adduction,
        matching the real ROM asymmetry between the two).

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target hip adduction/abduction angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-50, 30]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('hip_adduction_l', degrees, range_min, range_max)

    @property
    def left_hip_adduction(self) -> float:
        """Return the current left hip adduction/abduction angle, in degrees.

        Reads the OpenSim coordinate ``hip_adduction_l`` (raw state read,
        no :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_hip_adduction` or any other writer of
        this coordinate). Positive values mean the hip is adducted (thigh
        toward the midline); negative values mean it is abducted (thigh
        away from the midline).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current left hip adduction/abduction angle, in degrees.
        """
        return self._coordinate_degrees('hip_adduction_l')

    def set_right_hip_adduction(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the right hip adduction/abduction angle, in degrees.

        Writes the OpenSim coordinate ``hip_adduction_r``. Positive values
        adduct the hip (thigh swings toward the midline); negative values
        abduct it (thigh swings away from the midline) -- consistent both
        with the coordinate's own name and with its asymmetric range in
        the bundled model (-50 deg of abduction vs. 30 deg of adduction,
        matching the real ROM asymmetry between the two).

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target hip adduction/abduction angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-50, 30]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('hip_adduction_r', degrees, range_min, range_max)

    @property
    def right_hip_adduction(self) -> float:
        """Return the current right hip adduction/abduction angle, in degrees.

        Reads the OpenSim coordinate ``hip_adduction_r`` (raw state read,
        no :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_hip_adduction` or any other writer of
        this coordinate). Positive values mean the hip is adducted (thigh
        toward the midline); negative values mean it is abducted (thigh
        away from the midline).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current right hip adduction/abduction angle, in degrees.
        """
        return self._coordinate_degrees('hip_adduction_r')

    def set_left_hip_rotation(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the left hip (axial) rotation angle, in degrees.

        Writes the OpenSim coordinate ``hip_rotation_l``, rotating the
        thigh about its own long axis. Positive values internally rotate
        the thigh (anterior surface turns toward the body's midline);
        negative values externally rotate it (away from the midline).
        This direction is verified from the bundled model's femur-frame
        rotation.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target hip rotation angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-40, 40]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('hip_rotation_l', degrees, range_min, range_max)

    @property
    def left_hip_rotation(self) -> float:
        """Return the current left hip (axial) rotation angle, in degrees.

        Reads the OpenSim coordinate ``hip_rotation_l`` (raw state read,
        no :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_hip_rotation` or any other writer of
        this coordinate). Positive values internally rotate the thigh
        (anterior surface toward the body's midline); negative values
        externally rotate it (away from the midline).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current left hip rotation angle, in degrees.
        """
        return self._coordinate_degrees('hip_rotation_l')

    def set_right_hip_rotation(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the right hip (axial) rotation angle, in degrees.

        Writes the OpenSim coordinate ``hip_rotation_r``, rotating the
        thigh about its own long axis. Positive values internally rotate
        the thigh (anterior surface turns toward the body's midline);
        negative values externally rotate it (away from the midline).
        This direction is verified from the bundled model's femur-frame
        rotation.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target hip rotation angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-40, 40]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('hip_rotation_r', degrees, range_min, range_max)

    @property
    def right_hip_rotation(self) -> float:
        """Return the current right hip (axial) rotation angle, in degrees.

        Reads the OpenSim coordinate ``hip_rotation_r`` (raw state read,
        no :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_hip_rotation` or any other writer of
        this coordinate). Positive values internally rotate the thigh
        (anterior surface toward the body's midline); negative values
        externally rotate it (away from the midline).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current right hip rotation angle, in degrees.
        """
        return self._coordinate_degrees('hip_rotation_r')

    def set_left_knee_flexionextension(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the left knee flexion/extension angle, in degrees.

        Writes the OpenSim coordinate ``knee_angle_l``. Positive values
        flex the knee (shank swings backward relative to the thigh); the
        coordinate's range starts exactly at 0 degrees (full extension)
        and only goes positive (up to 140 degrees): positive values flex
        the knee (shank swings backward), while negative values would
        extend it past straight if a custom range allowed them.

        The patella (:attr:`left_patella`) is kinematically coupled to
        this coordinate through a ``CoordinateCouplerConstraint`` in the
        base model. :meth:`update_state` does **not** resolve that
        coupling (a documented limitation, see the README): call
        :meth:`reinitialize` instead if the patella position needs to
        reflect the new knee angle for something other than ``export()``.

        Parameters
        ----------
        degrees : float
            Target knee flexion/extension angle, in degrees.
            The bundled model's initial coordinate range is
            ``[0, 140]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('knee_angle_l', degrees, range_min, range_max)

    @property
    def left_knee_flexionextension(self) -> float:
        """Return the current left knee flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``knee_angle_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_knee_flexionextension` or any other
        writer of this coordinate). Positive values mean the knee is
        flexed (shank swung backward relative to the thigh); negative
        values would extend it past straight if a custom range allowed
        them. The coordinate cannot go below 0 degrees (full extension) in this
        model.
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.
        Note that the coupled patella position
        (:attr:`left_patella`) does not automatically reflect a knee angle
        written mid-edit until :meth:`reinitialize` is called (see
        :meth:`set_left_knee_flexionextension`).

        Returns
        -------
        float
            Current left knee flexion/extension angle, in degrees.
        """
        return self._coordinate_degrees('knee_angle_l')

    def set_right_knee_flexionextension(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the right knee flexion/extension angle, in degrees.

        Writes the OpenSim coordinate ``knee_angle_r``. Positive values
        flex the knee (shank swings backward relative to the thigh); the
        coordinate's range starts exactly at 0 degrees (full extension)
        and only goes positive (up to 140 degrees): positive values flex
        the knee (shank swings backward), while negative values would
        extend it past straight if a custom range allowed them.

        The patella (:attr:`right_patella`) is kinematically coupled to
        this coordinate through a ``CoordinateCouplerConstraint`` in the
        base model. :meth:`update_state` does **not** resolve that
        coupling (a documented limitation, see the README): call
        :meth:`reinitialize` instead if the patella position needs to
        reflect the new knee angle for something other than ``export()``.

        Parameters
        ----------
        degrees : float
            Target knee flexion/extension angle, in degrees.
            The bundled model's initial coordinate range is
            ``[0, 140]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('knee_angle_r', degrees, range_min, range_max)

    @property
    def right_knee_flexionextension(self) -> float:
        """Return the current right knee flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``knee_angle_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_knee_flexionextension` or any other
        writer of this coordinate). Positive values mean the knee is
        flexed (shank swung backward relative to the thigh); negative
        values would extend it past straight if a custom range allowed
        them. The coordinate cannot go below 0 degrees (full extension) in this
        model.
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.
        Note that the coupled patella position
        (:attr:`right_patella`) does not automatically reflect a knee
        angle written mid-edit until :meth:`reinitialize` is called (see
        :meth:`set_right_knee_flexionextension`).

        Returns
        -------
        float
            Current right knee flexion/extension angle, in degrees.
        """
        return self._coordinate_degrees('knee_angle_r')

    def set_left_ankle_flexiondorsiflexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the left ankle flexion/dorsiflexion angle, in degrees.

        Writes the OpenSim coordinate ``ankle_angle_l``, which combines
        plantarflexion and dorsiflexion into one degree of freedom.
        Positive values dorsiflex the ankle (toes move toward the shin);
        negative values plantarflex it (toes point away from the shin),
        as verified from the calcaneus frame in the bundled model.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target ankle flexion/dorsiflexion angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-50, 50]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('ankle_angle_l', degrees, range_min, range_max)

    @property
    def left_ankle_flexiondorsiflexion(self) -> float:
        """Return the current left ankle flexion/dorsiflexion angle, in degrees.

        Reads the OpenSim coordinate ``ankle_angle_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_ankle_flexiondorsiflexion` or any other
        writer of this coordinate). Positive values dorsiflex the ankle
        (toes toward the shin); negative values plantarflex it (toes away
        from the shin).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current left ankle flexion/dorsiflexion angle, in degrees.
        """
        return self._coordinate_degrees('ankle_angle_l')

    def set_right_ankle_flexiondorsiflexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the right ankle flexion/dorsiflexion angle, in degrees.

        Writes the OpenSim coordinate ``ankle_angle_r``, which combines
        plantarflexion and dorsiflexion into one degree of freedom.
        Positive values dorsiflex the ankle (toes move toward the shin);
        negative values plantarflex it (toes point away from the shin),
        as verified from the calcaneus frame in the bundled model.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target ankle flexion/dorsiflexion angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-50, 50]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('ankle_angle_r', degrees, range_min, range_max)

    @property
    def right_ankle_flexiondorsiflexion(self) -> float:
        """Return the current right ankle flexion/dorsiflexion angle, in degrees.

        Reads the OpenSim coordinate ``ankle_angle_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_ankle_flexiondorsiflexion` or any
        other writer of this coordinate). Positive values dorsiflex the
        ankle (toes toward the shin); negative values plantarflex it (toes
        away from the shin).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current right ankle flexion/dorsiflexion angle, in degrees.
        """
        return self._coordinate_degrees('ankle_angle_r')

    def set_left_subtalar_inversion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the left subtalar inversion/eversion angle, in degrees.

        Writes the OpenSim coordinate ``subtalar_angle_l``. Positive
        values invert the foot (sole turns to face the body's midline);
        negative values evert it (sole turns outward) -- per the
        coordinate's own name.

        In the bundled .osim file this coordinate ships locked
        (``<locked>true</locked>``); it accepts values here regardless,
        because every ``User`` instance force-unlocks every coordinate at
        construction time (see ``User._unlock_coordinates`` and
        the README's "Changing posture" section), overriding whatever
        the base file specifies.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target subtalar inversion/eversion angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-35, 35]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked again after construction (e.g. via
            ``coordinate("subtalar_angle_l").set_locked(True)``).
        """
        self._set_coordinate_degrees('subtalar_angle_l', degrees, range_min, range_max)

    @property
    def left_subtalar_inversion(self) -> float:
        """Return the current left subtalar inversion/eversion angle, in degrees.

        Reads the OpenSim coordinate ``subtalar_angle_l`` (raw state read,
        no :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_subtalar_inversion` or any other writer
        of this coordinate). Positive values mean the foot is inverted
        (sole turned toward the midline); negative values mean it is
        everted (sole turned outward).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.
        This coordinate ships locked in the base
        .osim file but is force-unlocked for every ``User`` instance (see
        :meth:`set_left_subtalar_inversion`).

        Returns
        -------
        float
            Current left subtalar inversion/eversion angle, in degrees.
        """
        return self._coordinate_degrees('subtalar_angle_l')

    def set_right_subtalar_inversion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the right subtalar inversion/eversion angle, in degrees.

        Writes the OpenSim coordinate ``subtalar_angle_r``. Positive
        values invert the foot (sole turns to face the body's midline);
        negative values evert it (sole turns outward) -- per the
        coordinate's own name.

        In the bundled .osim file this coordinate ships locked
        (``<locked>true</locked>``); it accepts values here regardless,
        because every ``User`` instance force-unlocks every coordinate at
        construction time (see ``User._unlock_coordinates`` and
        the README's "Changing posture" section), overriding whatever
        the base file specifies.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target subtalar inversion/eversion angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-35, 35]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked again after construction (e.g. via
            ``coordinate("subtalar_angle_r").set_locked(True)``).
        """
        self._set_coordinate_degrees('subtalar_angle_r', degrees, range_min, range_max)

    @property
    def right_subtalar_inversion(self) -> float:
        """Return the current right subtalar inversion/eversion angle, in degrees.

        Reads the OpenSim coordinate ``subtalar_angle_r`` (raw state read,
        no :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_subtalar_inversion` or any other
        writer of this coordinate). Positive values mean the foot is
        inverted (sole turned toward the midline); negative values mean it
        is everted (sole turned outward).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.
        This coordinate ships locked in
        the base .osim file but is force-unlocked for every ``User``
        instance (see :meth:`set_right_subtalar_inversion`).

        Returns
        -------
        float
            Current right subtalar inversion/eversion angle, in degrees.
        """
        return self._coordinate_degrees('subtalar_angle_r')

    def set_left_mtp_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the left metatarsophalangeal (toe) flexion angle, in degrees.

        Writes the OpenSim coordinate ``mtp_angle_l``. Positive values
        flex the toes (curl downward/plantarward); negative values extend
        them (dorsiflexion, toes point up, as in late stance/toe-off) --
        per the coordinate's own name and its asymmetric range in the
        bundled model (45 deg of extension vs. 30 deg of flexion), which
        matches the real MTP joint's larger dorsiflexion ROM.

        In the bundled .osim file this coordinate ships locked
        (``<locked>true</locked>``); it accepts values here regardless,
        because every ``User`` instance force-unlocks every coordinate at
        construction time (see ``User._unlock_coordinates``),
        overriding whatever the base file specifies.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target MTP flexion/extension angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-45, 30]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked again after construction.
        """
        self._set_coordinate_degrees('mtp_angle_l', degrees, range_min, range_max)

    @property
    def left_mtp_flexion(self) -> float:
        """Return the current left MTP (toe) flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``mtp_angle_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_mtp_flexion` or any other writer of
        this coordinate). Positive values mean the toes are flexed (curled
        plantarward/downward); negative values mean they are extended
        (dorsiflexed, pointing up).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.
        This coordinate ships locked in the base
        .osim file but is force-unlocked for every ``User`` instance (see
        :meth:`set_left_mtp_flexion`).

        Returns
        -------
        float
            Current left MTP flexion/extension angle, in degrees.
        """
        return self._coordinate_degrees('mtp_angle_l')

    def set_right_mtp_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the right metatarsophalangeal (toe) flexion angle, in degrees.

        Writes the OpenSim coordinate ``mtp_angle_r``. Positive values
        flex the toes (curl downward/plantarward); negative values extend
        them (dorsiflexion, toes point up, as in late stance/toe-off) --
        per the coordinate's own name and its asymmetric range in the
        bundled model (45 deg of extension vs. 30 deg of flexion), which
        matches the real MTP joint's larger dorsiflexion ROM.

        In the bundled .osim file this coordinate ships locked
        (``<locked>true</locked>``); it accepts values here regardless,
        because every ``User`` instance force-unlocks every coordinate at
        construction time (see ``User._unlock_coordinates``),
        overriding whatever the base file specifies.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target MTP flexion/extension angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-45, 30]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked again after construction (see
            ``coordinate("mtp_angle_r").set_locked(True)``).
        """
        self._set_coordinate_degrees('mtp_angle_r', degrees, range_min, range_max)

    @property
    def right_mtp_flexion(self) -> float:
        """Return the current right MTP (toe) flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``mtp_angle_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_mtp_flexion` or any other writer of
        this coordinate). Positive values mean the toes are flexed (curled
        plantarward/downward); negative values mean they are extended
        (dorsiflexed, pointing up).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.
        This coordinate ships locked in the base
        .osim file but is force-unlocked for every ``User`` instance (see
        :meth:`set_right_mtp_flexion`).

        Returns
        -------
        float
            Current right MTP flexion/extension angle, in degrees.
        """
        return self._coordinate_degrees('mtp_angle_r')

    def set_left_shoulder_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the left shoulder flexion/extension angle, in degrees.

        Writes the OpenSim coordinate ``arm_flex_l``. Positive values flex
        the shoulder (arm swings forward and up, toward overhead);
        negative values extend it (arm swings backward) -- per the
        coordinate's own name.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture (joint centres, ``com``, muscle
        lengths/forces...).

        Parameters
        ----------
        degrees : float
            Target shoulder flexion/extension angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-90, 90]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('arm_flex_l', degrees, range_min, range_max)

    @property
    def left_shoulder_flexion(self) -> float:
        """Return the current left shoulder flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``arm_flex_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_shoulder_flexion` or any other writer
        of this coordinate). Positive values mean the shoulder is flexed
        (arm forward/up, toward overhead); negative values mean it is
        extended (arm backward).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current left shoulder flexion/extension angle, in degrees.
        """
        return self._coordinate_degrees('arm_flex_l')

    def set_right_shoulder_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the right shoulder flexion/extension angle, in degrees.

        Writes the OpenSim coordinate ``arm_flex_r``. Positive values flex
        the shoulder (arm swings forward and up, toward overhead);
        negative values extend it (arm swings backward) -- per the
        coordinate's own name.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target shoulder flexion/extension angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-90, 90]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('arm_flex_r', degrees, range_min, range_max)

    @property
    def right_shoulder_flexion(self) -> float:
        """Return the current right shoulder flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``arm_flex_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_shoulder_flexion` or any other writer
        of this coordinate). Positive values mean the shoulder is flexed
        (arm forward/up, toward overhead); negative values mean it is
        extended (arm backward).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current right shoulder flexion/extension angle, in degrees.
        """
        return self._coordinate_degrees('arm_flex_r')

    def set_left_shoulder_adduction(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the left shoulder adduction/abduction angle, in degrees.

        Writes the OpenSim coordinate ``arm_add_l``. Positive values
        adduct the shoulder (arm swings toward/across the body); negative
        values abduct it (arm swings away from the body, overhead) -- per
        the coordinate's own name and its asymmetric range in the bundled
        model (120 deg of abduction vs. 90 deg of adduction), matching the
        real ROM asymmetry between the two (abduction reaches much further
        than adduction past the midline).

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target shoulder adduction/abduction angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-120, 90]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('arm_add_l', degrees, range_min, range_max)

    @property
    def left_shoulder_adduction(self) -> float:
        """Return the current left shoulder adduction/abduction angle, in degrees.

        Reads the OpenSim coordinate ``arm_add_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_shoulder_adduction` or any other writer
        of this coordinate). Positive values mean the shoulder is adducted
        (arm toward/across the body); negative values mean it is abducted
        (arm away from the body, overhead).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current left shoulder adduction/abduction angle, in degrees.
        """
        return self._coordinate_degrees('arm_add_l')

    def set_right_shoulder_adduction(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the right shoulder adduction/abduction angle, in degrees.

        Writes the OpenSim coordinate ``arm_add_r``. Positive values
        adduct the shoulder (arm swings toward/across the body); negative
        values abduct it (arm swings away from the body, overhead) -- per
        the coordinate's own name and its asymmetric range in the bundled
        model (120 deg of abduction vs. 90 deg of adduction), matching the
        real ROM asymmetry between the two (abduction reaches much further
        than adduction past the midline).

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target shoulder adduction/abduction angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-120, 90]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('arm_add_r', degrees, range_min, range_max)

    @property
    def right_shoulder_adduction(self) -> float:
        """Return the current right shoulder adduction/abduction angle, in degrees.

        Reads the OpenSim coordinate ``arm_add_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_shoulder_adduction` or any other
        writer of this coordinate). Positive values mean the shoulder is
        adducted (arm toward/across the body); negative values mean it is
        abducted (arm away from the body, overhead).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current right shoulder adduction/abduction angle, in degrees.
        """
        return self._coordinate_degrees('arm_add_r')

    def set_left_shoulder_rotation(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the left shoulder (axial) rotation angle, in degrees.

        Writes the OpenSim coordinate ``arm_rot_l``, rotating the upper
        arm about its own long axis. Positive values internally rotate
        the arm (anterior surface turns toward the body's midline);
        negative values externally rotate it (away from the midline).
        This direction is verified from the bundled model's humerus-frame
        rotation.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target shoulder rotation angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-90, 90]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('arm_rot_l', degrees, range_min, range_max)

    @property
    def left_shoulder_rotation(self) -> float:
        """Return the current left shoulder (axial) rotation angle, in degrees.

        Reads the OpenSim coordinate ``arm_rot_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_shoulder_rotation` or any other writer
        of this coordinate). Positive values internally rotate the arm
        (anterior surface toward the body's midline); negative values
        externally rotate it (away from the midline).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current left shoulder rotation angle, in degrees.
        """
        return self._coordinate_degrees('arm_rot_l')

    def set_right_shoulder_rotation(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the right shoulder (axial) rotation angle, in degrees.

        Writes the OpenSim coordinate ``arm_rot_r``, rotating the upper
        arm about its own long axis. Positive values internally rotate
        the arm (anterior surface turns toward the body's midline);
        negative values externally rotate it (away from the midline).
        This direction is verified from the bundled model's humerus-frame
        rotation.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target shoulder rotation angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-90, 90]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('arm_rot_r', degrees, range_min, range_max)

    @property
    def right_shoulder_rotation(self) -> float:
        """Return the current right shoulder (axial) rotation angle, in degrees.

        Reads the OpenSim coordinate ``arm_rot_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_shoulder_rotation` or any other writer
        of this coordinate). Positive values internally rotate the arm
        (anterior surface toward the body's midline); negative values
        externally rotate it (away from the midline).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current right shoulder rotation angle, in degrees.
        """
        return self._coordinate_degrees('arm_rot_r')

    def set_left_elbow_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the left elbow flexion angle, in degrees.

        Writes the OpenSim coordinate ``elbow_flex_l``. Positive values
        flex the elbow (forearm swings toward the upper arm); the
        coordinate's range starts exactly at 0 degrees (full extension)
        and only goes positive (up to 150 degrees): positive values flex
        the elbow (forearm swings toward the upper arm), while negative
        values would extend it past straight if a custom range allowed
        them.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target elbow flexion angle, in degrees.
            The bundled model's initial coordinate range is
            ``[0, 150]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('elbow_flex_l', degrees, range_min, range_max)

    @property
    def left_elbow_flexion(self) -> float:
        """Return the current left elbow flexion angle, in degrees.

        Reads the OpenSim coordinate ``elbow_flex_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_elbow_flexion` or any other writer of
        this coordinate). Positive values mean the elbow is flexed
        (forearm toward the upper arm); negative values would extend it
        past straight if a custom range allowed them. The coordinate
        has a default minimum of 0 degrees (full extension).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current left elbow flexion angle, in degrees.
        """
        return self._coordinate_degrees('elbow_flex_l')

    def set_right_elbow_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the right elbow flexion angle, in degrees.

        Writes the OpenSim coordinate ``elbow_flex_r``. Positive values
        flex the elbow (forearm swings toward the upper arm); the
        coordinate's range starts exactly at 0 degrees (full extension)
        and only goes positive (up to 150 degrees): positive values flex
        the elbow (forearm swings toward the upper arm), while negative
        values would extend it past straight if a custom range allowed
        them.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target elbow flexion angle, in degrees.
            The bundled model's initial coordinate range is
            ``[0, 150]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('elbow_flex_r', degrees, range_min, range_max)

    @property
    def right_elbow_flexion(self) -> float:
        """Return the current right elbow flexion angle, in degrees.

        Reads the OpenSim coordinate ``elbow_flex_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_elbow_flexion` or any other writer of
        this coordinate). Positive values mean the elbow is flexed
        (forearm toward the upper arm); negative values would extend it
        past straight if a custom range allowed them. The coordinate
        has a default minimum of 0 degrees (full extension).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current right elbow flexion angle, in degrees.
        """
        return self._coordinate_degrees('elbow_flex_r')

    def set_left_wrist_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the left wrist flexion/extension angle, in degrees.

        Writes the OpenSim coordinate ``wrist_flex_l``. Positive values
        flex the wrist (palmar flexion, palm moves toward the forearm);
        negative values extend it (dorsiflexion, back of the hand moves
        toward the forearm) -- per the coordinate's own name.

        In the bundled .osim file this coordinate ships locked
        (``<locked>true</locked>``); it accepts values here regardless,
        because every ``User`` instance force-unlocks every coordinate at
        construction time (see ``User._unlock_coordinates``),
        overriding whatever the base file specifies.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target wrist flexion/extension angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-70, 70]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked again after construction (e.g. via
            ``coordinate("wrist_flex_l").set_locked(True)``).
        """
        self._set_coordinate_degrees('wrist_flex_l', degrees, range_min, range_max)

    @property
    def left_wrist_flexion(self) -> float:
        """Return the current left wrist flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``wrist_flex_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_wrist_flexion` or any other writer of
        this coordinate). Positive values mean the wrist is flexed
        (palmar flexion); negative values mean it is extended
        (dorsiflexion).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.
        This coordinate ships locked in the base .osim file
        but is force-unlocked for every ``User`` instance (see
        :meth:`set_left_wrist_flexion`).

        Returns
        -------
        float
            Current left wrist flexion/extension angle, in degrees.
        """
        return self._coordinate_degrees('wrist_flex_l')

    def set_right_wrist_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the right wrist flexion/extension angle, in degrees.

        Writes the OpenSim coordinate ``wrist_flex_r``. Positive values
        flex the wrist (palmar flexion, palm moves toward the forearm);
        negative values extend it (dorsiflexion, back of the hand moves
        toward the forearm) -- per the coordinate's own name.

        In the bundled .osim file this coordinate ships locked
        (``<locked>true</locked>``); it accepts values here regardless,
        because every ``User`` instance force-unlocks every coordinate at
        construction time (see ``User._unlock_coordinates``),
        overriding whatever the base file specifies.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target wrist flexion/extension angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-70, 70]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked again after construction (e.g. via
            ``coordinate("wrist_flex_r").set_locked(True)``).
        """
        self._set_coordinate_degrees('wrist_flex_r', degrees, range_min, range_max)

    @property
    def right_wrist_flexion(self) -> float:
        """Return the current right wrist flexion/extension angle, in degrees.

        Reads the OpenSim coordinate ``wrist_flex_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_wrist_flexion` or any other writer of
        this coordinate). Positive values mean the wrist is flexed
        (palmar flexion); negative values mean it is extended
        (dorsiflexion).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.
        This coordinate ships locked in the base .osim file
        but is force-unlocked for every ``User`` instance (see
        :meth:`set_right_wrist_flexion`).

        Returns
        -------
        float
            Current right wrist flexion/extension angle, in degrees.
        """
        return self._coordinate_degrees('wrist_flex_r')

    def set_left_wrist_deviation(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the left wrist radial/ulnar deviation angle, in degrees.

        Writes the OpenSim coordinate ``wrist_dev_l``, which deviates the
        hand sideways relative to the forearm. Positive values produce
        ulnar deviation (toward the little finger); negative values
        produce radial deviation (toward the thumb), as verified from the
        hand-frame motion in the bundled model.

        In the bundled .osim file this coordinate ships locked
        (``<locked>true</locked>``); it accepts values here regardless,
        because every ``User`` instance force-unlocks every coordinate at
        construction time (see ``User._unlock_coordinates``),
        overriding whatever the base file specifies.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target wrist deviation angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-25, 35]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked again after construction (e.g. via
            ``coordinate("wrist_dev_l").set_locked(True)``).
        """
        self._set_coordinate_degrees('wrist_dev_l', degrees, range_min, range_max)

    @property
    def left_wrist_deviation(self) -> float:
        """Return the current left wrist radial/ulnar deviation angle, in degrees.

        Reads the OpenSim coordinate ``wrist_dev_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_wrist_deviation` or any other writer of
        this coordinate). Positive values produce ulnar deviation (toward
        the little finger); negative values produce radial deviation
        (toward the thumb).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.
        This coordinate ships locked in
        the base .osim file but is force-unlocked for every ``User``
        instance.

        Returns
        -------
        float
            Current left wrist deviation angle, in degrees.
        """
        return self._coordinate_degrees('wrist_dev_l')

    def set_right_wrist_deviation(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the right wrist radial/ulnar deviation angle, in degrees.

        Writes the OpenSim coordinate ``wrist_dev_r``, which deviates the
        hand sideways relative to the forearm. Positive values produce
        ulnar deviation (toward the little finger); negative values
        produce radial deviation (toward the thumb), as verified from the
        hand-frame motion in the bundled model.

        In the bundled .osim file this coordinate ships locked
        (``<locked>true</locked>``); it accepts values here regardless,
        because every ``User`` instance force-unlocks every coordinate at
        construction time (see ``User._unlock_coordinates``),
        overriding whatever the base file specifies.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target wrist deviation angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-25, 35]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked again after construction (e.g. via
            ``coordinate("wrist_dev_r").set_locked(True)``).
        """
        self._set_coordinate_degrees('wrist_dev_r', degrees, range_min, range_max)

    @property
    def right_wrist_deviation(self) -> float:
        """Return the current right wrist radial/ulnar deviation angle, in degrees.

        Reads the OpenSim coordinate ``wrist_dev_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_wrist_deviation` or any other writer
        of this coordinate). Positive values produce ulnar deviation
        (toward the little finger); negative values produce radial
        deviation (toward the thumb).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.
        This coordinate ships locked in
        the base .osim file but is force-unlocked for every ``User``
        instance.

        Returns
        -------
        float
            Current right wrist deviation angle, in degrees.
        """
        return self._coordinate_degrees('wrist_dev_r')

    def set_left_forearm_pronation(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the left forearm pronation/supination angle, in degrees.

        Writes the OpenSim coordinate ``pro_sup_l``. Positive values
        pronate the forearm (palm turns to face backward/downward);
        negative values supinate it (palm turns toward the reference
        posture). The coordinate's range starts exactly at 0 degrees and only goes
        positive (up to approximately 120 degrees), consistent with 0
        being the fully supinated reference posture and positive values
        moving toward pronation -- per the coordinate's own name.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target forearm pronation/supination angle, in degrees.
            The bundled model's initial coordinate range is
            approximately ``[0, 119.75]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('pro_sup_l', degrees, range_min, range_max)

    @property
    def left_forearm_pronation(self) -> float:
        """Return the current left forearm pronation/supination angle, in degrees.

        Reads the OpenSim coordinate ``pro_sup_l`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_left_forearm_pronation` or any other writer
        of this coordinate). Positive values mean the forearm is pronated
        (palm facing backward/downward); negative values move it toward
        supination. The coordinate cannot go below 0
        degrees (the fully supinated reference posture) in this model.
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current left forearm pronation/supination angle, in degrees.
        """
        return self._coordinate_degrees('pro_sup_l')

    def set_right_forearm_pronation(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the right forearm pronation/supination angle, in degrees.

        Writes the OpenSim coordinate ``pro_sup_r``. Positive values
        pronate the forearm (palm turns to face backward/downward);
        negative values supinate it (palm turns toward the reference
        posture). The coordinate's range starts exactly at 0 degrees and only goes
        positive (up to approximately 120 degrees), consistent with 0
        being the fully supinated reference posture and positive values
        moving toward pronation -- per the coordinate's own name.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target forearm pronation/supination angle, in degrees.
            The bundled model's initial coordinate range is
            approximately ``[0, 119.75]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('pro_sup_r', degrees, range_min, range_max)

    @property
    def right_forearm_pronation(self) -> float:
        """Return the current right forearm pronation/supination angle, in degrees.

        Reads the OpenSim coordinate ``pro_sup_r`` (raw state read, no
        :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_right_forearm_pronation` or any other writer
        of this coordinate). Positive values mean the forearm is pronated
        (palm facing backward/downward); negative values move it toward
        supination. The coordinate cannot go below 0
        degrees (the fully supinated reference posture) in this model.
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current right forearm pronation/supination angle, in degrees.
        """
        return self._coordinate_degrees('pro_sup_r')

    def set_lumbar_extension(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the lumbar extension/flexion angle, in degrees.

        Writes the OpenSim coordinate ``lumbar_extension``. This is the
        only lumbar coordinate without a left/right pair (the trunk has a
        single lumbar joint in this model). Positive values extend the
        trunk (leans backward); negative values flex it (leans forward)
        -- per the coordinate's own name.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture (joint centres, ``com``, muscle
        lengths/forces...).

        Parameters
        ----------
        degrees : float
            Target lumbar extension/flexion angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-90, 90]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('lumbar_extension', degrees, range_min, range_max)

    @property
    def lumbar_extension(self) -> float:
        """Return the current lumbar extension/flexion angle, in degrees.

        Reads the OpenSim coordinate ``lumbar_extension`` (raw state read,
        no :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_lumbar_extension` or any other writer of
        this coordinate). There is no left/right pair: the trunk has a
        single lumbar joint in this model. Positive values mean the trunk
        is extended (leaning backward); negative values mean it is flexed
        (leaning forward).
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current lumbar extension/flexion angle, in degrees.
        """
        return self._coordinate_degrees('lumbar_extension')

    def set_lumbar_bending(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the lumbar lateral bending angle, in degrees.

        Writes the OpenSim coordinate ``lumbar_bending`` (no left/right
        pair: the trunk has a single lumbar joint in this model). Positive
        values bend the trunk toward the user's right; negative values
        bend it toward the left, as verified from the torso-frame motion
        in the bundled model.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target lumbar lateral bending angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-90, 90]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('lumbar_bending', degrees, range_min, range_max)

    @property
    def lumbar_bending(self) -> float:
        """Return the current lumbar lateral bending angle, in degrees.

        Reads the OpenSim coordinate ``lumbar_bending`` (raw state read,
        no :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_lumbar_bending` or any other writer of this
        coordinate). There is no left/right pair: the trunk has a single
        lumbar joint in this model. Positive values bend the trunk toward
        the user's right; negative values bend it toward the left.
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current lumbar lateral bending angle, in degrees.
        """
        return self._coordinate_degrees('lumbar_bending')

    def set_lumbar_rotation(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ):
        """Set the lumbar (axial) rotation angle, in degrees.

        Writes the OpenSim coordinate ``lumbar_rotation`` (no left/right
        pair: the trunk has a single lumbar joint in this model), rotating
        the trunk about its own long axis. Positive values turn the trunk
        toward the user's right; negative values turn it left, as verified
        from the torso-frame motion in the bundled model.

        Like every posture setter, this only writes the raw coordinate
        value: call :meth:`update_state` before reading anything derived
        from the new posture.

        Parameters
        ----------
        degrees : float
            Target lumbar rotation angle, in degrees.
            The bundled model's initial coordinate range is
            ``[-90, 90]`` degrees.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. When either limit is supplied, the requested
            angle must lie within the resulting range.

        Raises
        ------
        ValueError
            If ``degrees`` is not finite, or if this coordinate has been
            explicitly re-locked after construction (every coordinate is
            force-unlocked when the model is built -- see
            ``User._unlock_coordinates``).
        """
        self._set_coordinate_degrees('lumbar_rotation', degrees, range_min, range_max)

    @property
    def lumbar_rotation(self) -> float:
        """Return the current lumbar (axial) rotation angle, in degrees.

        Reads the OpenSim coordinate ``lumbar_rotation`` (raw state read,
        no :meth:`update_state` needed: it reflects the exact value last
        written by :meth:`set_lumbar_rotation` or any other writer of this
        coordinate). There is no left/right pair: the trunk has a single
        lumbar joint in this model. Positive values turn the trunk toward
        the user's right; negative values turn it left.
        The setter may override the range with ``range_min`` and
        ``range_max``; this property returns the value currently held by
        OpenSim.

        Returns
        -------
        float
            Current lumbar rotation angle, in degrees.
        """
        return self._coordinate_degrees('lumbar_rotation')
