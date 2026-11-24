"""Named setters and getters for every articulated finger coordinate."""

from __future__ import annotations

__all__ = []


class _FingerPostureMixin:
    """Explicit left/right posture aliases for the finger joints.

    Coordinates are relative to the corrected reference pose and start
    locked. Setters delegate to :meth:`set_finger_angle`, including its
    degree-based range handling and validation.
    """

    def set_left_index_mcp_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left index metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "index", "mcp", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def left_index_mcp_flexion(self) -> float:
        """Return left index metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "index", "mcp", "flexion")

    def set_left_index_mcp_abduction(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left index metacarpophalangeal (MCP) abduction in degrees.

        Positive values abduct the digit (spread it away from the hand's
        midline); negative values adduct it toward the midline.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "index", "mcp", "abduction", degrees,
            range_min, range_max,
        )

    @property
    def left_index_mcp_abduction(self) -> float:
        """Return left index metacarpophalangeal (MCP) abduction in degrees.

        Positive values abduct the digit (spread it away from the hand's
        midline); negative values adduct it toward the midline.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "index", "mcp", "abduction")

    def set_left_index_pip_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left index proximal interphalangeal (PIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "index", "pip", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def left_index_pip_flexion(self) -> float:
        """Return left index proximal interphalangeal (PIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "index", "pip", "flexion")

    def set_left_index_dip_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left index distal interphalangeal (DIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "index", "dip", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def left_index_dip_flexion(self) -> float:
        """Return left index distal interphalangeal (DIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "index", "dip", "flexion")

    def set_left_middle_mcp_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left middle metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "middle", "mcp", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def left_middle_mcp_flexion(self) -> float:
        """Return left middle metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "middle", "mcp", "flexion")

    def set_left_middle_mcp_abduction(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left middle metacarpophalangeal (MCP) abduction in degrees.

        Positive values abduct the digit (spread it away from the hand's
        midline); negative values adduct it toward the midline.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "middle", "mcp", "abduction", degrees,
            range_min, range_max,
        )

    @property
    def left_middle_mcp_abduction(self) -> float:
        """Return left middle metacarpophalangeal (MCP) abduction in degrees.

        Positive values abduct the digit (spread it away from the hand's
        midline); negative values adduct it toward the midline.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "middle", "mcp", "abduction")

    def set_left_middle_pip_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left middle proximal interphalangeal (PIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "middle", "pip", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def left_middle_pip_flexion(self) -> float:
        """Return left middle proximal interphalangeal (PIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "middle", "pip", "flexion")

    def set_left_middle_dip_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left middle distal interphalangeal (DIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "middle", "dip", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def left_middle_dip_flexion(self) -> float:
        """Return left middle distal interphalangeal (DIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "middle", "dip", "flexion")

    def set_left_ring_mcp_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left ring metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "ring", "mcp", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def left_ring_mcp_flexion(self) -> float:
        """Return left ring metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "ring", "mcp", "flexion")

    def set_left_ring_mcp_abduction(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left ring metacarpophalangeal (MCP) abduction in degrees.

        Positive values abduct the digit (spread it away from the hand's
        midline); negative values adduct it toward the midline.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "ring", "mcp", "abduction", degrees,
            range_min, range_max,
        )

    @property
    def left_ring_mcp_abduction(self) -> float:
        """Return left ring metacarpophalangeal (MCP) abduction in degrees.

        Positive values abduct the digit (spread it away from the hand's
        midline); negative values adduct it toward the midline.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "ring", "mcp", "abduction")

    def set_left_ring_pip_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left ring proximal interphalangeal (PIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "ring", "pip", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def left_ring_pip_flexion(self) -> float:
        """Return left ring proximal interphalangeal (PIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "ring", "pip", "flexion")

    def set_left_ring_dip_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left ring distal interphalangeal (DIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "ring", "dip", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def left_ring_dip_flexion(self) -> float:
        """Return left ring distal interphalangeal (DIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "ring", "dip", "flexion")

    def set_left_little_mcp_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left little metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "little", "mcp", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def left_little_mcp_flexion(self) -> float:
        """Return left little metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "little", "mcp", "flexion")

    def set_left_little_mcp_abduction(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left little metacarpophalangeal (MCP) abduction in degrees.

        Positive values abduct the digit (spread it away from the hand's
        midline); negative values adduct it toward the midline.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "little", "mcp", "abduction", degrees,
            range_min, range_max,
        )

    @property
    def left_little_mcp_abduction(self) -> float:
        """Return left little metacarpophalangeal (MCP) abduction in degrees.

        Positive values abduct the digit (spread it away from the hand's
        midline); negative values adduct it toward the midline.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "little", "mcp", "abduction")

    def set_left_little_pip_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left little proximal interphalangeal (PIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "little", "pip", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def left_little_pip_flexion(self) -> float:
        """Return left little proximal interphalangeal (PIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "little", "pip", "flexion")

    def set_left_little_dip_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left little distal interphalangeal (DIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "little", "dip", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def left_little_dip_flexion(self) -> float:
        """Return left little distal interphalangeal (DIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "little", "dip", "flexion")

    def set_left_thumb_cmc_opposition(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left thumb carpometacarpal (CMC) opposition in degrees.

        Positive values oppose the thumb across the palm toward the
        fingers; negative values move it away from opposition.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "thumb", "cmc", "opposition", degrees,
            range_min, range_max,
        )

    @property
    def left_thumb_cmc_opposition(self) -> float:
        """Return left thumb carpometacarpal (CMC) opposition in degrees.

        Positive values oppose the thumb across the palm toward the
        fingers; negative values move it away from opposition.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "thumb", "cmc", "opposition")

    def set_left_thumb_cmc_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left thumb carpometacarpal (CMC) flexion in degrees.

        Positive values flex/curl the thumb; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "thumb", "cmc", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def left_thumb_cmc_flexion(self) -> float:
        """Return left thumb carpometacarpal (CMC) flexion in degrees.

        Positive values flex/curl the thumb; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "thumb", "cmc", "flexion")

    def set_left_thumb_mcp_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left thumb metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the thumb; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "thumb", "mcp", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def left_thumb_mcp_flexion(self) -> float:
        """Return left thumb metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the thumb; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "thumb", "mcp", "flexion")

    def set_left_thumb_ip_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set left thumb interphalangeal (IP) flexion in degrees.

        Positive values flex/curl the thumb; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "left", "thumb", "ip", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def left_thumb_ip_flexion(self) -> float:
        """Return left thumb interphalangeal (IP) flexion in degrees.

        Positive values flex/curl the thumb; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("left", "thumb", "ip", "flexion")

    def set_right_index_mcp_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right index metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "index", "mcp", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def right_index_mcp_flexion(self) -> float:
        """Return right index metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "index", "mcp", "flexion")

    def set_right_index_mcp_abduction(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right index metacarpophalangeal (MCP) abduction in degrees.

        Positive values abduct the digit (spread it away from the hand's
        midline); negative values adduct it toward the midline.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "index", "mcp", "abduction", degrees,
            range_min, range_max,
        )

    @property
    def right_index_mcp_abduction(self) -> float:
        """Return right index metacarpophalangeal (MCP) abduction in degrees.

        Positive values abduct the digit (spread it away from the hand's
        midline); negative values adduct it toward the midline.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "index", "mcp", "abduction")

    def set_right_index_pip_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right index proximal interphalangeal (PIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "index", "pip", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def right_index_pip_flexion(self) -> float:
        """Return right index proximal interphalangeal (PIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "index", "pip", "flexion")

    def set_right_index_dip_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right index distal interphalangeal (DIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "index", "dip", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def right_index_dip_flexion(self) -> float:
        """Return right index distal interphalangeal (DIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "index", "dip", "flexion")

    def set_right_middle_mcp_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right middle metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "middle", "mcp", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def right_middle_mcp_flexion(self) -> float:
        """Return right middle metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "middle", "mcp", "flexion")

    def set_right_middle_mcp_abduction(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right middle metacarpophalangeal (MCP) abduction in degrees.

        Positive values abduct the digit (spread it away from the hand's
        midline); negative values adduct it toward the midline.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "middle", "mcp", "abduction", degrees,
            range_min, range_max,
        )

    @property
    def right_middle_mcp_abduction(self) -> float:
        """Return right middle metacarpophalangeal (MCP) abduction in degrees.

        Positive values abduct the digit (spread it away from the hand's
        midline); negative values adduct it toward the midline.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "middle", "mcp", "abduction")

    def set_right_middle_pip_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right middle proximal interphalangeal (PIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "middle", "pip", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def right_middle_pip_flexion(self) -> float:
        """Return right middle proximal interphalangeal (PIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "middle", "pip", "flexion")

    def set_right_middle_dip_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right middle distal interphalangeal (DIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "middle", "dip", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def right_middle_dip_flexion(self) -> float:
        """Return right middle distal interphalangeal (DIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "middle", "dip", "flexion")

    def set_right_ring_mcp_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right ring metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "ring", "mcp", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def right_ring_mcp_flexion(self) -> float:
        """Return right ring metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "ring", "mcp", "flexion")

    def set_right_ring_mcp_abduction(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right ring metacarpophalangeal (MCP) abduction in degrees.

        Positive values abduct the digit (spread it away from the hand's
        midline); negative values adduct it toward the midline.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "ring", "mcp", "abduction", degrees,
            range_min, range_max,
        )

    @property
    def right_ring_mcp_abduction(self) -> float:
        """Return right ring metacarpophalangeal (MCP) abduction in degrees.

        Positive values abduct the digit (spread it away from the hand's
        midline); negative values adduct it toward the midline.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "ring", "mcp", "abduction")

    def set_right_ring_pip_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right ring proximal interphalangeal (PIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "ring", "pip", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def right_ring_pip_flexion(self) -> float:
        """Return right ring proximal interphalangeal (PIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "ring", "pip", "flexion")

    def set_right_ring_dip_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right ring distal interphalangeal (DIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "ring", "dip", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def right_ring_dip_flexion(self) -> float:
        """Return right ring distal interphalangeal (DIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "ring", "dip", "flexion")

    def set_right_little_mcp_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right little metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "little", "mcp", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def right_little_mcp_flexion(self) -> float:
        """Return right little metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "little", "mcp", "flexion")

    def set_right_little_mcp_abduction(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right little metacarpophalangeal (MCP) abduction in degrees.

        Positive values abduct the digit (spread it away from the hand's
        midline); negative values adduct it toward the midline.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "little", "mcp", "abduction", degrees,
            range_min, range_max,
        )

    @property
    def right_little_mcp_abduction(self) -> float:
        """Return right little metacarpophalangeal (MCP) abduction in degrees.

        Positive values abduct the digit (spread it away from the hand's
        midline); negative values adduct it toward the midline.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "little", "mcp", "abduction")

    def set_right_little_pip_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right little proximal interphalangeal (PIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "little", "pip", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def right_little_pip_flexion(self) -> float:
        """Return right little proximal interphalangeal (PIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "little", "pip", "flexion")

    def set_right_little_dip_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right little distal interphalangeal (DIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "little", "dip", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def right_little_dip_flexion(self) -> float:
        """Return right little distal interphalangeal (DIP) flexion in degrees.

        Positive values flex/curl the digit; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "little", "dip", "flexion")

    def set_right_thumb_cmc_opposition(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right thumb carpometacarpal (CMC) opposition in degrees.

        Positive values oppose the thumb across the palm toward the
        fingers; negative values move it away from opposition.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "thumb", "cmc", "opposition", degrees,
            range_min, range_max,
        )

    @property
    def right_thumb_cmc_opposition(self) -> float:
        """Return right thumb carpometacarpal (CMC) opposition in degrees.

        Positive values oppose the thumb across the palm toward the
        fingers; negative values move it away from opposition.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "thumb", "cmc", "opposition")

    def set_right_thumb_cmc_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right thumb carpometacarpal (CMC) flexion in degrees.

        Positive values flex/curl the thumb; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "thumb", "cmc", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def right_thumb_cmc_flexion(self) -> float:
        """Return right thumb carpometacarpal (CMC) flexion in degrees.

        Positive values flex/curl the thumb; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "thumb", "cmc", "flexion")

    def set_right_thumb_mcp_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right thumb metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the thumb; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "thumb", "mcp", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def right_thumb_mcp_flexion(self) -> float:
        """Return right thumb metacarpophalangeal (MCP) flexion in degrees.

        Positive values flex/curl the thumb; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "thumb", "mcp", "flexion")

    def set_right_thumb_ip_flexion(
        self,
        degrees: float,
        range_min: float | None = None,
        range_max: float | None = None,
    ) -> None:
        """Set right thumb interphalangeal (IP) flexion in degrees.

        Positive values flex/curl the thumb; negative values extend it.
        Unlock the coordinate with :meth:`set_finger_coordinate_locked`
        before calling this setter.

        Parameters
        ----------
        degrees : float
            Target angle relative to the corrected reference pose.
        range_min, range_max : float or None, optional
            Exact coordinate limits in degrees. Supplied limits replace
            the current bounds; omitted limits remain unchanged.

        Raises
        ------
        ValueError
            If the coordinate is locked or the angle or limits are invalid.
        """
        self.set_finger_angle(
            "right", "thumb", "ip", "flexion", degrees,
            range_min, range_max,
        )

    @property
    def right_thumb_ip_flexion(self) -> float:
        """Return right thumb interphalangeal (IP) flexion in degrees.

        Positive values flex/curl the thumb; negative values extend it.
        The value is relative to the corrected reference pose.

        Returns
        -------
        float
            Current coordinate angle in degrees.
        """
        return self.finger_angle("right", "thumb", "ip", "flexion")
