"""Posture setters/getters for the custom neck/skull joints.

These coordinates (``neck_flexion``, ``neck_base_flexion``, ...) belong to
the ``torso_neck``/``neck_skull`` joints added in
:meth:`User._add_anatomical_bodies`, not to the bundled Rajagopal-Lai-Uhlrich
model's own native coordinates -- unlike :mod:`.posture_generated`, which
covers exactly that native, pre-existing coordinate set (and is exercised by
an exhaustive test enumerating it). Kept as a separate mixin so that
boundary stays explicit rather than conflated.
"""

from __future__ import annotations

__all__ = []


class _NeckPostureMixin:
    """Neck and cervical-spine posture setter/getter pairs."""

    def set_neck_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set head flexion/extension relative to the neck base.

        Parameters
        ----------
        degrees : float
            Target angle in degrees. Positive values extend the head
            backward; negative values flex it forward (chin toward chest).

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. The requested angle must lie within the resulting
            range.
        """
        self._set_coordinate_degrees("neck_flexion", degrees, range_min, range_max)

    @property
    def neck_flexion(self) -> float:
        """Return the current head flexion/extension angle in degrees.

        Positive values extend the head backward; negative values flex it
        forward (chin toward chest).

        Returns
        -------
        float
            ``neck_flexion`` coordinate in degrees.
        """
        return self._coordinate_degrees("neck_flexion")

    def set_neck_lateral_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set lateral flexion of the head relative to the neck base.

        Parameters
        ----------
        degrees : float
            Target lateral-flexion angle in degrees. Positive values bend
            the head toward the user's right; negative values bend it left.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. The requested angle must lie within the resulting
            range.
        """
        self._set_coordinate_degrees(
            "neck_lateral_flexion", degrees, range_min, range_max
        )

    @property
    def neck_lateral_flexion(self) -> float:
        """Return the current head lateral-flexion angle in degrees.

        Positive values bend the head toward the user's right; negative
        values bend it left.

        Returns
        -------
        float
            ``neck_lateral_flexion`` coordinate in degrees.
        """
        return self._coordinate_degrees("neck_lateral_flexion")

    def set_neck_rotation(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set axial rotation of the head relative to the neck base.

        Parameters
        ----------
        degrees : float
            Target axial-rotation angle in degrees. Positive values turn
            the head toward the user's right; negative values turn it left.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. The requested angle must lie within the resulting
            range.
        """
        self._set_coordinate_degrees("neck_rotation", degrees, range_min, range_max)

    @property
    def neck_rotation(self) -> float:
        """Return the current head axial-rotation angle in degrees.

        Positive values turn the head toward the user's right; negative
        values turn it left.

        Returns
        -------
        float
            ``neck_rotation`` coordinate in degrees.
        """
        return self._coordinate_degrees("neck_rotation")

    def set_neck_base_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set flexion/extension of the cervical spine relative to the torso.

        Parameters
        ----------
        degrees : float
            Target angle in degrees. Positive values extend the neck
            backward; negative values flex it forward.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. The requested angle must lie within the resulting
            range.
        """
        self._set_coordinate_degrees("neck_base_flexion", degrees, range_min, range_max)

    @property
    def neck_base_flexion(self) -> float:
        """Return cervical flexion/extension relative to the torso.

        Positive values extend the neck backward; negative values flex it
        forward.

        Returns
        -------
        float
            ``neck_base_flexion`` coordinate in degrees.
        """
        return self._coordinate_degrees("neck_base_flexion")

    def set_neck_base_lateral_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set cervical lateral flexion relative to the torso.

        Parameters
        ----------
        degrees : float
            Target lateral-flexion angle in degrees. Positive values bend
            the neck toward the user's right; negative values bend it left.

        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current corresponding bounds; omitted limits remain
            unchanged. The requested angle must lie within the resulting
            range.
        """
        self._set_coordinate_degrees(
            "neck_base_lateral_flexion", degrees, range_min, range_max
        )

    @property
    def neck_base_lateral_flexion(self) -> float:
        """Return cervical lateral flexion relative to the torso.

        Positive values bend the neck toward the user's right; negative
        values bend it left.

        Returns
        -------
        float
            ``neck_base_lateral_flexion`` coordinate in degrees.
        """
        return self._coordinate_degrees("neck_base_lateral_flexion")
