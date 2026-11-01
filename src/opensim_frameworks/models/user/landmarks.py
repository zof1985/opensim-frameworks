"""Aggregated joint-centre and marker landmarks, in the ground frame.

Complements :mod:`.joint_centers_generated` (per-joint named properties such
as ``pelvis``/``left_hip``, each a thin wrapper around ``_joint_center``):
this module defines ``_joint_center`` itself, the aggregated
``joint_centers`` dict, marker-based landmarks (eye/palm/foot markers,
individual heel/toe/metatarsal properties), the whole-body centre of mass
and its ground projection, and :meth:`User.set_position`.
"""

from __future__ import annotations

import math

from .joint_center_table import JOINT_CENTER_NAMES as _JOINT_CENTER_NAMES

__all__ = []

_PELVIS_TRANSLATION_COORDINATES = ("pelvis_tx", "pelvis_ty", "pelvis_tz")

# Friendly name -> OpenSim marker name. The foot has no 1st-metatarsal
# marker in this model (only the 5th): no landmark is exposed for it rather
# than guessing one.
_FOOT_MARKER_NAMES = {
    "left_heel": "LCAL",
    "right_heel": "RCAL",
    "left_toe": "LTOE",
    "right_toe": "RTOE",
    "left_mt5": "LMT5",
    "right_mt5": "RMT5",
}


class _LandmarkMixin:
    """Aggregated joint-centre and marker landmark properties."""

    def _joint_center(self, joint_name: str) -> tuple[float, float, float]:
        self.realizePosition(self.state)
        frame = self.getJointSet().get(joint_name).getChildFrame()
        position = frame.getPositionInGround(self.state)
        return (position.get(0), position.get(1), position.get(2))

    @property
    def joint_centers(self) -> dict[str, tuple[float, float, float]]:
        """Return every named joint centre at once, in the ground frame.

        Equivalent to calling every individual joint-centre property below
        (``pelvis``, ``left_hip``, ``right_knee``, ...) and collecting the
        results into one dictionary, keyed by the same friendly names.
        Reflects the model's current posture: each position is computed via
        ``getPositionInGround`` with an automatic ``realizePosition``, so it
        is correct immediately after a posture setter, with no need to call
        :meth:`update_state` first.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Mapping from friendly joint name (``"pelvis"``, ``"left_hip"``,
            ``"right_ankle"``, ...) to that joint's centre as an ``(x, y,
            z)`` position in the ground frame, in metres.
        """
        return {
            name: self._joint_center(joint)
            for name, joint in _JOINT_CENTER_NAMES.items()
        }

    def _marker_location(self, marker_name: str) -> tuple[float, float, float]:
        self.realizePosition(self.state)
        position = self.getMarkerSet().get(marker_name).getLocationInGround(self.state)
        return (position.get(0), position.get(1), position.get(2))

    @property
    def foot_markers(self) -> dict[str, tuple[float, float, float]]:
        """Return every foot landmark marker at once, in the ground frame.

        Equivalent to calling each of ``left_heel``/``right_heel``,
        ``left_toe``/``right_toe``, and ``left_mt5``/``right_mt5`` and
        collecting the results into one dictionary. Reflects the model's
        current posture (``realizePosition`` is called automatically; no
        need to call :meth:`update_state` first). There is no 1st-metatarsal
        marker in this model (only the 5th, see :attr:`left_mt5`), so none
        is exposed here either.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Mapping from friendly marker name (``"left_heel"``,
            ``"right_toe"``, ``"left_mt5"``, ...) to that marker's location
            as an ``(x, y, z)`` position in the ground frame, in metres.
        """
        return {
            name: self._marker_location(marker)
            for name, marker in _FOOT_MARKER_NAMES.items()
        }

    @property
    def eye_markers(self) -> dict[str, tuple[float, float, float]]:
        """Return the right and left orbital marker positions.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Mapping from ``"right"`` and ``"left"`` to the corresponding eye
            marker's ``(x, y, z)`` position in metres in the ground frame.
        """
        return {
            side: self._marker_location(marker)
            for side, marker in (("right", "REYE"), ("left", "LEYE"))
        }

    @property
    def palm_markers(self) -> dict[str, tuple[float, float, float]]:
        """Return the articulated right and left palm-centre marker positions.

        Each marker is attached to its corresponding hand body at the centre
        of the resolved palm, halfway from the wrist along the hand's local
        negative y axis. The hand bodies are connected to the radii by
        universal wrist joints with flexion/extension and radial/ulnar
        deviation coordinates. Marker positions therefore follow the current
        wrist posture and are expressed in the ground frame.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            Mapping from ``"right"`` and ``"left"`` to the corresponding
            palm-centre ``(x, y, z)`` positions in metres in the ground frame.
        """
        return {
            side: self._marker_location(marker)
            for side, marker in (("right", "RPALM"), ("left", "LPALM"))
        }

    @property
    def left_heel(self) -> tuple[float, float, float]:
        """Return the left heel marker location, in the ground frame.

        Reads the OpenSim marker ``LCAL``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("LCAL")

    @property
    def right_heel(self) -> tuple[float, float, float]:
        """Return the right heel marker location, in the ground frame.

        Reads the OpenSim marker ``RCAL``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("RCAL")

    @property
    def left_toe(self) -> tuple[float, float, float]:
        """Return the left toe marker location, in the ground frame.

        Reads the OpenSim marker ``LTOE``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("LTOE")

    @property
    def right_toe(self) -> tuple[float, float, float]:
        """Return the right toe marker location, in the ground frame.

        Reads the OpenSim marker ``RTOE``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("RTOE")

    @property
    def left_mt5(self) -> tuple[float, float, float]:
        """Return the left 5th-metatarsal marker location, in the ground frame.

        Reads the OpenSim marker ``LMT5``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first). There is no 1st-metatarsal marker in
        this model, so this is the only metatarsal landmark available.

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("LMT5")

    @property
    def right_mt5(self) -> tuple[float, float, float]:
        """Return the right 5th-metatarsal marker location, in the ground frame.

        Reads the OpenSim marker ``RMT5``, reflecting the model's current
        posture (``realizePosition`` is called automatically; no need for
        :meth:`update_state` first). There is no 1st-metatarsal marker in
        this model, so this is the only metatarsal landmark available.

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        return self._marker_location("RMT5")

    @property
    def com(self) -> tuple[float, float, float]:
        """Return the whole-body centre of mass, in the ground frame.

        Computed natively by OpenSim (``Model.calcMassCenterPosition``),
        reflecting the model's current posture (``realizePosition`` is
        called automatically; no need to call :meth:`update_state` first).

        Returns
        -------
        tuple[float, float, float]
            ``(x, y, z)`` position in metres, in the ground frame.
        """
        self.realizePosition(self.state)
        position = self.calcMassCenterPosition(self.state)
        return (position.get(0), position.get(1), position.get(2))

    @property
    def cop(self) -> tuple[float, float, float]:
        """Return the ground projection of the centre of mass.

        This is a purely kinematic projection of :attr:`com` straight down
        onto the ground plane (``y = 0``): the ``x``/``z`` coordinates are
        copied unchanged from :attr:`com` and ``y`` is forced to ``0.0``.
        Despite the name, this is **not** a dynamically computed centre of
        pressure derived from contact/ground-reaction forces.

        Returns
        -------
        tuple[float, float, float]
            ``(x, 0.0, z)`` position in metres, in the ground frame, where
            ``x``/``z`` match :attr:`com`.
        """
        x, _, z = self.com
        return (x, 0.0, z)

    def set_position(
        self, reference: tuple[float, float, float], x: float, y: float, z: float
    ) -> None:
        """Translate the whole model so that ``reference`` ends up at ``(x, y, z)``.

        ``reference`` is any ground-frame point belonging to this model --
        e.g. :attr:`com`, :attr:`cop`, a :attr:`joint_centers` entry, or one
        of the named joint-centre properties (``left_ankle``, ``pelvis``,
        ...). The whole model is translated rigidly through its root
        (``pelvis_tx``/``pelvis_ty``/``pelvis_tz``): relative posture and
        joint angles are unaffected. Calls :meth:`update_state` internally,
        so derived quantities are immediately up to date.

        Parameters
        ----------
        reference : tuple[float, float, float]
            Current ground-frame position, in metres, of the point to move
            (read it right before calling, since it depends on the current
            posture).
        x, y, z : float
            Target ground-frame coordinates for that point, in metres.

        Raises
        ------
        ValueError
            If any coordinate is not finite.
        """
        if not all(math.isfinite(value) for value in (x, y, z, *reference)):
            raise ValueError("position must be finite")
        targets = (x, y, z)
        for name, target, current in zip(
            _PELVIS_TRANSLATION_COORDINATES, targets, reference
        ):
            coordinate = self.getCoordinateSet().get(name)
            coordinate.setValue(
                self.state,
                coordinate.getValue(self.state) + (target - current),
                False,
            )
        self.update_state()
