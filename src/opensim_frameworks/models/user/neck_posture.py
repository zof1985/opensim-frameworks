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

    def set_neck_flexion(self, degrees: float) -> None:
        """Set head flexion/extension relative to the neck base.

        Parameters
        ----------
        degrees : float
            Target angle in degrees. Positive values follow the OpenSim
            coordinate's flexion direction.
        """
        self._set_coordinate_degrees("neck_flexion", degrees)

    @property
    def neck_flexion(self) -> float:
        """Return the current head flexion/extension angle.

        Returns
        -------
        float
            ``neck_flexion`` coordinate in degrees.
        """
        return self._coordinate_degrees("neck_flexion")

    def set_neck_lateral_flexion(self, degrees: float) -> None:
        """Set lateral flexion of the head relative to the neck base.

        Parameters
        ----------
        degrees : float
            Target lateral-flexion angle in degrees.
        """
        self._set_coordinate_degrees("neck_lateral_flexion", degrees)

    @property
    def neck_lateral_flexion(self) -> float:
        """Return the current head lateral-flexion angle.

        Returns
        -------
        float
            ``neck_lateral_flexion`` coordinate in degrees.
        """
        return self._coordinate_degrees("neck_lateral_flexion")

    def set_neck_rotation(self, degrees: float) -> None:
        """Set axial rotation of the head relative to the neck base.

        Parameters
        ----------
        degrees : float
            Target axial-rotation angle in degrees.
        """
        self._set_coordinate_degrees("neck_rotation", degrees)

    @property
    def neck_rotation(self) -> float:
        """Return the current head axial-rotation angle.

        Returns
        -------
        float
            ``neck_rotation`` coordinate in degrees.
        """
        return self._coordinate_degrees("neck_rotation")

    def set_neck_base_flexion(self, degrees: float) -> None:
        """Set flexion/extension of the cervical spine relative to the torso.

        Parameters
        ----------
        degrees : float
            Target angle in degrees.
        """
        self._set_coordinate_degrees("neck_base_flexion", degrees)

    @property
    def neck_base_flexion(self) -> float:
        """Return cervical flexion/extension relative to the torso.

        Returns
        -------
        float
            ``neck_base_flexion`` coordinate in degrees.
        """
        return self._coordinate_degrees("neck_base_flexion")

    def set_neck_base_lateral_flexion(self, degrees: float) -> None:
        """Set cervical lateral flexion relative to the torso.

        Parameters
        ----------
        degrees : float
            Target lateral-flexion angle in degrees.
        """
        self._set_coordinate_degrees("neck_base_lateral_flexion", degrees)

    @property
    def neck_base_lateral_flexion(self) -> float:
        """Return cervical lateral flexion relative to the torso.

        Returns
        -------
        float
            ``neck_base_lateral_flexion`` coordinate in degrees.
        """
        return self._coordinate_degrees("neck_base_lateral_flexion")
