"""Equipment models and screen geometry."""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np

from ...utils.opensim import import_opensim
from ..model import FrameworkModel
from .screen import Screen

opensim = import_opensim()

__all__ = ["Equipment"]


class Equipment(FrameworkModel, ABC):
    """Base class for user-facing OpenSim equipment.

    Concrete equipment provides its own user-fitting procedure. If it has a
    screen, :attr:`screen` returns its OpenSim body;
    otherwise it returns ``None``. The default initializer loads a serialized
    ``.osim`` model from the subclass's ``_model_path`` asset directory,
    calling ``_build`` first when that file is missing. Subclasses that build
    parameter-specific models in memory can provide their own initializer.
    """

    _model_path: ClassVar[Path | None] = None
    _model_filename: ClassVar[str] = "model.osim"
    _rebuild_model_on_init: ClassVar[bool] = False

    def __init__(self) -> None:
        self.opensim = opensim
        model_directory = self.model_path
        mesh_directory = self.mesh_path
        model_file = self.model_file
        if model_directory is None:
            raise NotImplementedError(
                f"{type(self).__name__} must define its _model_path"
            )
        if not isinstance(model_directory, Path):
            raise TypeError("model_path must be a pathlib.Path")
        if not isinstance(mesh_directory, Path):
            raise TypeError("mesh_path must be a pathlib.Path")
        if not isinstance(model_file, Path):
            raise TypeError("model_file must be a pathlib.Path")

        self._model_path = model_directory
        self._mesh_path = mesh_directory
        self._model_file = model_file
        if not self.model_file.is_file() or self._rebuild_model_on_init:
            model_directory.mkdir(parents=True, exist_ok=True)
            mesh_directory.mkdir(parents=True, exist_ok=True)
            self._build()
        model_file = self.model_file
        if model_file is None or not model_file.is_file():
            raise FileNotFoundError(
                f"{type(self).__name__}._build did not create {self.model_file}"
            )
        opensim.Model.__init__(self, str(model_file))
        self.finalizeConnections()
        self.state = self.initSystem()
        self.realizePosition(self.state)

    def set_config(self, config: dict[str, Any]) -> None:
        """Restore equipment-specific Python attributes after .osf loading.

        Parameters
        ----------
        config : dict[str, Any]
            Equipment configuration returned by :meth:`get_config`.
        """
        super().set_config(config)
        model_file = self._osf_model_file
        model_directory = model_file.parent
        self.opensim = opensim
        self._model_path = model_directory
        self._mesh_path = model_directory / "Geometry"
        self._model_file = model_file
        self.geometry_directories = (self._mesh_path,)

    @property
    def model_path(self) -> Path | None:
        """Path to the assets directory."""
        return getattr(self, "_model_path", None)

    @property
    def mesh_path(self) -> Path | None:
        """Path to the mesh directory."""
        mesh_path = getattr(self, "_mesh_path", None)
        if mesh_path is not None:
            return mesh_path
        model_path = self.model_path
        return None if model_path is None else model_path / "meshes"

    @property
    def meshes(self) -> list[Path]:
        """List of all mesh files in the mesh directory."""
        mesh_path = self.mesh_path
        if mesh_path is None:
            return []
        return sorted(mesh_path.glob("*.vtp"))

    @property
    def model_file(self) -> Path | None:
        """Path to the model file."""
        model_file = getattr(self, "_model_file", None)
        if model_file is not None:
            return model_file
        model_path = self.model_path
        return None if model_path is None else model_path / self._model_filename

    @property
    def screen(self) -> Screen | None:
        """Return this equipment's screen body, if present.

        Returns
        -------
        Screen or None
            Screen body and mesh, or ``None`` when this equipment has no screen.
        """
        return None

    def has_screen(self) -> bool:
        """Report whether this equipment provides a screen.

        Returns
        -------
        bool
            ``True`` when :attr:`screen` contains a screen body.
        """
        return self.screen is not None

    def screen_panel_corners(
        self,
        position: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        """Return the screen's four panel corners in Ground.

        Parameters
        ----------
        position : tuple of float or None, optional
            Candidate equipment-specific screen pose. ``None`` uses the
            currently attached screen.

        Returns
        -------
        numpy.ndarray, shape (4, 3)
            Screen panel corners in metres.

        Raises
        ------
        NotImplementedError
            If a candidate pose is supplied but the equipment does not
            implement candidate screen placement.
        ValueError
            If the equipment has no screen.
        """
        screen = self.screen
        if screen is None:
            raise ValueError(f"{type(self).__name__} does not provide a screen")
        if position is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement candidate screen poses"
            )
        return screen.surface_corners

    def screen_panel_samples(
        self,
        grid_size: int,
        position: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        """Sample the screen panel in Ground coordinates.

        Parameters
        ----------
        grid_size : int
            Number of evenly spaced cell centres along each panel dimension.
        position : tuple of float or None, optional
            Candidate equipment-specific screen pose. ``None`` uses the
            currently attached screen.

        Returns
        -------
        numpy.ndarray, shape (grid_size**2, 3)
            Screen panel sample points in metres.

        Raises
        ------
        NotImplementedError
            If a candidate pose is supplied but the equipment does not
            implement candidate screen placement.
        ValueError
            If the equipment has no screen or ``grid_size`` is invalid.
        """
        screen = self.screen
        if screen is None:
            raise ValueError(f"{type(self).__name__} does not provide a screen")
        if position is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement candidate screen poses"
            )
        return screen.samples(grid_size)

    @property
    def screen_offset_minima(self) -> tuple[float | None, float | None]:
        """Return equipment-specific lower bounds for screen offsets.

        Returns
        -------
        tuple of float or None
            Minimum horizontal and vertical offsets in millimetres. ``None``
            means that the equipment does not impose a separate lower bound.
        """
        return None, None

    @property
    def screen_position(self) -> tuple[float, float, float]:
        """Return the equipment-specific current screen pose.

        Returns
        -------
        tuple of float
            Current ``(horizontal_offset_mm, vertical_offset_mm,
            inclination_deg)``.

        Raises
        ------
        NotImplementedError
            If the equipment does not expose a configurable screen pose.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a configurable screen pose"
        )

    def set_screen_position(
        self,
        horizontal_offset_mm: float,
        vertical_offset_mm: float,
        inclination_deg: float,
    ) -> None:
        """Apply a candidate pose to the equipment screen.

        Screen-bearing equipment must override this method so optimizers can
        apply the selected screen pose to its native model.

        Parameters
        ----------
        horizontal_offset_mm, vertical_offset_mm : float
            Screen mounting offsets from its equipment-specific reference, in
            millimetres.
        inclination_deg : float
            Directed panel inclination in degrees.

        Raises
        ------
        NotImplementedError
            If the equipment has a screen but does not support repositioning.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement screen repositioning"
        )

    def expand_screen_offset_limits(
        self,
        horizontal_min_mm: float | None = None,
        vertical_min_mm: float | None = None,
    ) -> None:
        """Expand equipment-specific lower bounds for screen offsets.

        Screen-bearing equipment with configurable offset limits should
        override this method. The default is suitable for equipment whose
        screen offsets do not have separate lower limits.

        Parameters
        ----------
        horizontal_min_mm : float or None, optional
            Requested minimum horizontal screen offset in millimetres.
        vertical_min_mm : float or None, optional
            Requested minimum vertical screen offset in millimetres.

        Raises
        ------
        NotImplementedError
            If the equipment has a screen with separate offset limits but
            does not support expanding them.
        """
        horizontal_limit, vertical_limit = self.screen_offset_minima
        if self.screen is not None and (
            (
                horizontal_min_mm is not None
                and horizontal_limit is not None
                and horizontal_min_mm < horizontal_limit
            )
            or (
                vertical_min_mm is not None
                and vertical_limit is not None
                and vertical_min_mm < vertical_limit
            )
        ):
            raise NotImplementedError(
                f"{type(self).__name__} does not support expanding screen offset limits"
            )

    def screen_position_constraints(
        self, position: tuple[float, float, float]
    ) -> np.ndarray:
        """Return feasibility margins for a candidate screen pose.

        Positive values indicate satisfied constraints. Equipment without
        special placement constraints inherits an always-feasible margin.

        Parameters
        ----------
        position : tuple of float
            Candidate ``(horizontal_offset_mm, vertical_offset_mm,
            inclination_deg)``.

        Returns
        -------
        numpy.ndarray
            Non-negative feasibility margins, or negative values for violated
            constraints.
        """
        return np.ones(1)

    def clone(self) -> Self:
        """Create an independent native and Python-level copy of this equipment.

        Concrete subclasses with non-copyable Python state may override this
        method. The copied OpenSim model is reinitialized so its working state
        belongs to the clone.

        Returns
        -------
        Equipment
            Independent copy of this equipment instance.
        """
        result = type(self).__new__(type(self))
        opensim.Model.__init__(result, self)
        for name, value in vars(self).items():
            if name == "opensim":
                setattr(result, name, value)
            elif name == "_visualizer":
                setattr(result, name, None)
            elif name == "state":
                continue
            else:
                setattr(result, name, copy.deepcopy(value))
        result.state = result.initSystem()
        result.realizePosition(result.state)
        return result

    def show_motion(self, duration: float = 5.0, fps: float = 30.0) -> None:
        """Display a repeating excursion of the equipment's movable joints.

        Each independent coordinate moves from its minimum to its maximum
        and back, in radians or metres. Dependent coordinates follow the
        model's kinematic constraints. Coordinate locks and prescribed
        motions are disabled on a copy for the sweep, leaving the equipment's
        state and properties unchanged.

        Parameters
        ----------
        duration : float, default=5.0
            Duration of one complete cycle in seconds.
        fps : float, default=30.0
            Number of motion samples per second.

        Raises
        ------
        ValueError
            If the timing is invalid, an independent coordinate has no
            finite range or the model has no independently movable coordinates.
        """
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration must be positive and finite")
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps must be positive and finite")

        from ...viewer import FrameworkViewer

        model = opensim.Model(self)
        model.geometry_directories = (
            *getattr(self, "geometry_directories", ()),
            *((self.mesh_path,) if self.mesh_path is not None else ()),
        )
        model.state = model.initSystem()
        model.realizePosition(model.state)
        coordinates = model.getCoordinateSet()
        for index in range(coordinates.getSize()):
            coordinate = coordinates.get(index)
            coordinate.set_prescribed(False)
            coordinate.setIsPrescribed(model.state, False)
            coordinate.set_locked(False)
            coordinate.setLocked(model.state, False)
        model.realizePosition(model.state)

        independent = []
        dependent = []
        for index in range(coordinates.getSize()):
            coordinate = coordinates.get(index)
            if coordinate.isDependent(model.state):
                dependent.append(coordinate)
                continue
            low, high = coordinate.getRangeMin(), coordinate.getRangeMax()
            if not math.isfinite(low) or not math.isfinite(high):
                raise ValueError(
                    f"Coordinate {coordinate.getName()!r} must have a finite range"
                )
            if low < high:
                independent.append(coordinate)
        if not independent:
            raise ValueError("equipment has no independently movable coordinates")

        intervals = max(2, 2 * math.ceil(duration * fps / 2))
        times = np.linspace(0.0, duration, intervals + 1)
        motion = opensim.TimeSeriesTable()
        animated = [*independent, *dependent]
        motion.setColumnLabels([coordinate.getName() for coordinate in animated])
        motion.addTableMetaDataString("inDegrees", "no")
        for time in times:
            progress = 1.0 - abs(2.0 * float(time) / duration - 1.0)
            for coordinate in independent:
                low, high = coordinate.getRangeMin(), coordinate.getRangeMax()
                coordinate.setValue(model.state, low + progress * (high - low), False)
            if dependent:
                model.assemble(model.state)
            model.realizePosition(model.state)
            motion.appendRow(
                float(time),
                opensim.RowVector(
                    [float(coordinate.getValue(model.state)) for coordinate in animated]
                ),
            )

        window = FrameworkViewer(model, sampling_frequency_hz=fps).show(
            motion=motion, loop=True
        )
        window.wait_closed()

    @abstractmethod
    def _build(self) -> None:
        """Construct this equipment's OpenSim model.

        Subclasses using the default initializer must also save the model and
        any display meshes under ``_model_path``.

        Raises
        ------
        NotImplementedError
            If the concrete equipment does not implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def fit_user(self, *args: Any, **kwargs: Any) -> Any:
        """Fit a user to this equipment using a subclass-specific procedure.

        Raises
        ------
        NotImplementedError
            If the concrete equipment does not implement a fitting procedure.
        """
        raise NotImplementedError

    def simulate_on_user(self, user: Any, **kwargs: Any) -> Any:
        """Simulate a user performing this equipment's exercise gesture.

        Equipment that models a repeatable exercise gesture must override
        this method so callers can generate an animated, biomechanically
        fitted repetition starting from :meth:`fit_user`'s pose. This is a
        concrete method, not an :func:`abc.abstractmethod`, so that existing
        subclasses without a simulated gesture (e.g. ``Bike``) keep working
        unchanged, matching :meth:`set_screen_position`'s pattern.

        Parameters
        ----------
        user : Any
            User to simulate, in whatever type the concrete equipment's
            ``fit_user`` expects.
        **kwargs : Any
            Equipment-specific simulation parameters.

        Raises
        ------
        NotImplementedError
            If the equipment does not implement gesture simulation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement user motion simulation"
        )
