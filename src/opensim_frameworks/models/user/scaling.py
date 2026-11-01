"""Anthropometric scaling: mass calibration and per-measurement rescaling.

Backs the explicit setters in :mod:`.measurement_generated`
(``set_height``, ``set_left_foot_length``, ...) and the read-only
``gender``/``height_cm``/``percentile``/``mass_kg``/``anthropometry``
properties, all of which recompute body scale factors against the ANSUR
reference resolved for this user (see :mod:`.data` and :mod:`.mapping`) and
reapply them through :meth:`User.scale_bodies`.
"""

from __future__ import annotations

import math
from dataclasses import replace

from .mapping import segment_scale_factors

__all__ = []


def _validate_mass_kg(mass_kg: float) -> None:
    """Raise ``ValueError`` unless ``mass_kg`` is a finite, strictly positive number.

    Shared by the constructor's ``mass_kg`` parameter and
    :meth:`User.set_mass_kg`, so both reject the same values with the same
    message.

    Parameters
    ----------
    mass_kg : float
        Requested body mass in kilograms.

    Raises
    ------
    ValueError
        If the mass is not finite and strictly positive.
    """
    if not math.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("mass_kg must be a positive finite value in kilograms")


def _axis_scale_factors(
    ratio: float, axes: tuple[str, ...]
) -> tuple[float, float, float]:
    return (
        ratio if "x" in axes else 1.0,
        ratio if "y" in axes else 1.0,
        ratio if "z" in axes else 1.0,
    )


class _AnthropometricScalingMixin:
    """Mass calibration and per-measurement anthropometric rescaling."""

    def _rescale_total_mass(self, mass_kg: float) -> None:
        """Uniformly rescale every body's mass and inertia so the model's total mass matches ``mass_kg`` exactly.

        Applies one scalar correction factor (``mass_kg`` divided by the
        model's current ``getTotalMass(self.state)``, i.e. whatever the geometric,
        height-driven scaling in :meth:`scale_bodies` produced
        volumetrically) to every body via ``opensim.Body.scaleMass`` --
        confirmed directly against a hand-built body that this native
        OpenSim method scales both ``mass`` *and* the full inertia tensor
        (moments and products alike) by the same factor, leaving
        ``mass_center`` untouched. A single uniform factor across every
        body keeps each body's relative share of the total mass (and each
        one's own radius of gyration/shape) exactly as the geometric
        scaling left it; only the overall scale of "how heavy" changes.

        Mass feeds into the multibody system's mass matrix, built once in
        ``initSystem()`` -- the same reason
        :meth:`~opensim_frameworks.components.Body.set_mass` calls
        :meth:`User.reinitialize` after
        changing a single body's mass -- so this does the same once every
        body has been rescaled, preserving whatever posture is current when
        it runs (the model's still-default posture, the first time this
        runs inside the constructor; otherwise whatever :meth:`set_mass_kg`
        found current when it was called).

        Parameters
        ----------
        mass_kg : float
            Target total body mass, in kilograms. Expected to already be
            validated (finite, strictly positive) by the caller -- see
            the constructor's own ``mass_kg`` validation.
        """
        scale_factor = mass_kg / self.getTotalMass(self.state)
        body_set = self.getBodySet()
        for index in range(body_set.getSize()):
            body_set.get(index).scaleMass(scale_factor)
        self.reinitialize()

    @staticmethod
    def _validate_measurement(name: str, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite value")
        return float(value)

    def _apply_reference_scale(self, reference) -> None:
        current_factors = self._expand_bilateral_bodies(
            segment_scale_factors(self._reference, self._baseline)
        )
        target_factors = self._expand_bilateral_bodies(
            segment_scale_factors(reference, self._baseline)
        )
        incremental_factors: dict[str, tuple[float, float, float]] = {}
        for body_name, target in target_factors.items():
            current = current_factors[body_name]
            ratios = (
                target[0] / current[0],
                target[1] / current[1],
                target[2] / current[2],
            )
            if any(
                not math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=1e-12)
                for ratio in ratios
            ):
                incremental_factors[body_name] = ratios
        if incremental_factors:
            self.scale_bodies(incremental_factors)
            self._rescale_total_mass(self._mass_kg)

    def _set_reference_measurement(
        self,
        property_name: str,
        source_measure: str,
        value: float,
        *,
        source_multiplier: float,
    ) -> None:
        """Update one symmetric ANSUR input and apply its scale delta.

        Parameters
        ----------
        property_name : str
            Name of the ``User`` measurement property to update.
        source_measure : str
            ANSUR reference column backing the property.
        value : float
            New measurement in the property's units.
        source_multiplier : float
            Conversion factor from the property value to the ANSUR column
            units.
        """
        measurement = self._validate_measurement(property_name, value)
        updated_values = self._reference.values.copy()
        updated_values[source_measure] = measurement * source_multiplier
        updated_reference = replace(self._reference, values=updated_values)
        self._apply_reference_scale(updated_reference)
        self._reference = updated_reference

    def _set_side_measurement(
        self,
        property_name: str,
        value: float,
        *,
        body_axes: tuple[tuple[str, tuple[str, ...]], ...],
        store_override: bool,
    ) -> None:
        """Apply a side-specific measurement without changing its opposite.

        Parameters
        ----------
        property_name : str
            Name of the side-specific ``User`` measurement property.
        value : float
            New measurement in the property's units.
        body_axes : tuple
            Body names and axes affected by the measurement scale.
        store_override : bool
            Whether to retain the new side-specific value as an override.

        Raises
        ------
        RuntimeError
            If a body required for scaling is absent from the model.
        ValueError
            If the measurement is invalid.
        """
        measurement = self._validate_measurement(property_name, value)
        body_set = self.getBodySet()
        for body_name, _ in body_axes:
            if not body_set.contains(body_name):
                raise RuntimeError(
                    f"Cannot apply {property_name!r}: body {body_name!r} is absent"
                )

        current = float(getattr(self, property_name))
        if store_override:
            ratio = measurement / current
            if not math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=1e-12):
                scale_factors = {
                    body_name: _axis_scale_factors(ratio, axes)
                    for body_name, axes in body_axes
                }
                self.scale_bodies(scale_factors)
                self._rescale_total_mass(self._mass_kg)
            self._side_measurements[property_name] = measurement
            return

        for _ in range(8):
            current = float(getattr(self, property_name))
            ratio = measurement / current
            if math.isclose(ratio, 1.0, rel_tol=1e-6, abs_tol=1e-9):
                return
            scale_factors = {
                body_name: _axis_scale_factors(ratio, axes)
                for body_name, axes in body_axes
            }
            self.scale_bodies(scale_factors)
            self._rescale_total_mass(self._mass_kg)

        current = float(getattr(self, property_name))
        if not math.isclose(current, measurement, rel_tol=1e-6, abs_tol=1e-9):
            raise RuntimeError(
                f"Scaling {property_name!r} did not reach the requested value "
                f"{measurement:g} m; the resulting value is {current:g} m"
            )

    def _side_measurement_value(self, property_name: str, source_measure: str) -> float:
        override = self._side_measurements.get(property_name)
        if override is not None:
            return override
        return self._reference.values[source_measure] / 1000.0

    @property
    def gender(self):
        """Return the normalized sex code used to resolve this user's anthropometry.

        Returns
        -------
        str
            ``"M"`` or ``"F"``, matching the ``gender`` passed to the
            constructor (normalization only affects case/whitespace, not
            the two accepted values themselves).
        """
        return self._reference.gender

    @property
    def height_cm(self):
        """Return this user's resolved stature, in centimetres.

        If the constructor was called with ``height_cm=...``, this is exactly
        that value (even if it was outside the ANSUR range and triggered
        extrapolation). Otherwise it is the ANSUR stature at the resolved
        ``percentile`` for this user's ``gender``.

        Returns
        -------
        float
            Stature, in centimetres.
        """
        return self._reference.height_cm

    @property
    def percentile(self):
        """Return the ANSUR percentile associated with this user.

        When the constructor was called with ``percentile=...`` (the
        default path), this is exactly that value, applied independently to
        every ANSUR measurement. When it was called with ``height_cm=...``
        instead, every measurement is resolved directly from the requested
        height (not from this percentile), and this property instead
        reports the *empirical* percentile of that height within the ANSUR
        population of this user's ``gender`` -- informational only, see the
        constructor's ``height_cm`` parameter.

        Returns
        -------
        float
            Percentile in the inclusive range ``[0.1, 99.9]``.
        """
        return self._reference.percentile

    @property
    def mass_kg(self):
        """Return this user's resolved and calibrated total body mass, in kilograms.

        If the constructor was called with ``mass_kg=...`` (or
        :meth:`set_mass_kg` was called afterward), this is exactly that
        value. Otherwise, it is the ANSUR ``weight_kg`` resolved for this
        user's height/percentile (see :attr:`anthropometry`). Either way,
        this is not just a label: ``self.getTotalMass(self.state)``
        is calibrated to match it exactly (see :meth:`_rescale_total_mass`),
        unlike the purely volumetric, height-driven mass that the
        geometric per-body scaling alone would otherwise leave in place.

        Returns
        -------
        float
            Total body mass, in kilograms.
        """
        return self._mass_kg

    def set_mass_kg(self, mass_kg: float) -> None:
        """Recalibrate this user's total body mass to ``mass_kg``, in kilograms.

        This re-applies the same independent
        mass correction the constructor itself uses (see
        :meth:`_rescale_total_mass`) directly to the already geometrically
        scaled model, rather than rebuilding it from scratch. Safe to call
        more than once, and safe to call regardless of whether the
        constructor was given an explicit ``mass_kg`` or not -- it always
        rescales from whatever the model's current total mass happens to be
        right now to the new target.

        Parameters
        ----------
        mass_kg : float
            New target total body mass, in kilograms. Must be finite and
            strictly positive.

        Raises
        ------
        ValueError
            If ``mass_kg`` is not finite or not strictly positive.
        """
        _validate_mass_kg(mass_kg)
        self._rescale_total_mass(mass_kg)
        self._mass_kg = mass_kg

    @property
    def anthropometry(self):
        """Return every anthropometric measurement resolved for this user.

        Returns
        -------
        opensim_frameworks.models.user.data.AnthropometricReference
            Internal reference object exposing ``gender``, ``percentile``,
            ``values`` (a ``dict`` mapping every numeric ANSUR II column
            resolved for this user to its value in the dataset's original
            units -- predominantly millimetres, with ``values["stature_m"]``
            in metres; see "ANSUR-resolved values" in the README), and the
            convenience properties ``height_m``/``height_cm`` (stature only,
            derived from ``values["stature_m"]``). This type is an internal
            implementation detail (not part of the package's re-exported
            public surface), but the object itself and its attributes
            remain freely readable once a :class:`User` has been built.
        """
        return self._reference
