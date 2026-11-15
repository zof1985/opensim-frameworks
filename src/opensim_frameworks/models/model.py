"""Base class and registration protocol for persisted framework models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Self

from ..utils.opensim import import_opensim, register_geometry_search_path

opensim = import_opensim()

__all__ = ["FrameworkModel", "register_framework_model"]

_MODEL_CLASSES: dict[str, type[FrameworkModel]] = {}


def _class_identifier(model_class: type) -> str:
    return f"{model_class.__module__}:{model_class.__qualname__}"


def register_framework_model(model_class: type[FrameworkModel]) -> type[FrameworkModel]:
    """Register a FrameworkModel subclass for safe .osf deserialization.

    Parameters
    ----------
    model_class : type[FrameworkModel]
        Model class whose module has already been imported by the application.

    Returns
    -------
    type[FrameworkModel]
        The registered class, allowing this function to be used as a decorator.

    Raises
    ------
    TypeError
        If ``model_class`` does not derive from ``FrameworkModel``.
    """
    if not isinstance(model_class, type) or not issubclass(
        model_class, FrameworkModel
    ):
        raise TypeError("model_class must derive from FrameworkModel")
    _MODEL_CLASSES[_class_identifier(model_class)] = model_class
    return model_class


class FrameworkModel(opensim.Model):
    """OpenSim model base required for membership in a :class:`Framework`.

    Subclasses may override :meth:`get_config` and :meth:`set_config` to
    serialize Python-side configuration in addition to the native OpenSim
    model stored in a ``.osf`` archive.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register subclasses when their application module is imported."""
        super().__init_subclass__(**kwargs)
        register_framework_model(cls)

    def __init__(self) -> None:
        """Initialize an empty native OpenSim model."""
        opensim.Model.__init__(self)
        object.__setattr__(self, "_osf_state", None)
        object.__setattr__(self, "_geometry_dirs", [])

    def initSystem(self) -> opensim.State:
        """Initialize the native system and make its state current.

        Returns
        -------
        opensim.State
            Initial state owned by this model.
        """
        state = opensim.Model.initSystem(self)
        object.__setattr__(self, "_osf_state", state)
        return state

    @property
    def state(self) -> opensim.State:
        """Return this model's current native state, initializing it lazily.

        Returns
        -------
        opensim.State
            Current state owned by this model.
        """
        state = getattr(self, "_osf_state", None)
        if state is None:
            state = self.initSystem()
            object.__setattr__(self, "_osf_state", state)
        return state

    @state.setter
    def state(self, state: opensim.State) -> None:
        """Set the current native state for this model.

        Parameters
        ----------
        state : opensim.State
            State created by this model's initialized system.

        Raises
        ------
        TypeError
            If ``state`` is not an ``opensim.State``.
        """
        if not isinstance(state, opensim.State):
            raise TypeError("state must be an opensim.State")
        object.__setattr__(self, "_osf_state", state)

    @property
    def geometry_directories(self) -> tuple[Path, ...]:
        """Return the registered geometry search directories.

        Returns
        -------
        tuple[pathlib.Path, ...]
            Resolved geometry directories in registration order.
        """
        return tuple(getattr(self, "_geometry_dirs", ()))

    @geometry_directories.setter
    def geometry_directories(self, directories: Iterable[str | Path]) -> None:
        """Replace the geometry search directories for this model.

        Parameters
        ----------
        directories : iterable of str or pathlib.Path
            Directories searched for geometry and mesh files.

        Raises
        ------
        TypeError
            If ``directories`` is not iterable.
        """
        if isinstance(directories, (str, bytes)):
            raise TypeError("directories must be an iterable of paths")
        object.__setattr__(self, "_geometry_dirs", [])
        for directory in directories:
            self.add_geometry_directory(directory)

    def add_geometry_directory(self, directory: str | Path) -> None:
        """Register a directory for resolving this model's geometry assets.

        Parameters
        ----------
        directory : str or pathlib.Path
            Geometry search directory.
        """
        path = Path(directory).expanduser().resolve()
        directories = list(getattr(self, "_geometry_dirs", ()))
        if path not in directories:
            register_geometry_search_path(opensim, path)
            directories.append(path)
            object.__setattr__(self, "_geometry_dirs", directories)

    def get_config(self) -> dict[str, Any]:
        """Return JSON-compatible Python configuration for this model.

        Returns
        -------
        dict[str, Any]
            Configuration restored by :meth:`set_config`. The native model
            structure and state are stored separately by ``Framework``.
        """
        return {}

    def set_config(self, config: Mapping[str, Any]) -> None:
        """Restore Python-side configuration after loading the native model.

        Parameters
        ----------
        config : collections.abc.Mapping
            Configuration previously returned by :meth:`get_config`.

        Raises
        ------
        TypeError
            If ``config`` is not a mapping.
        ValueError
            If this class does not accept any configuration but ``config`` is
            not empty.
        """
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        if config:
            raise ValueError(
                f"{type(self).__name__}.set_config does not accept configuration"
            )

    @classmethod
    def from_osf_model(
        cls,
        model_file: str | Path,
        config: Mapping[str, Any],
    ) -> Self:
        """Create a model instance from an extracted .osf model file.

        Parameters
        ----------
        model_file : str or pathlib.Path
            Path to the model's embedded ``.osim`` file.
        config : collections.abc.Mapping
            Python-side configuration returned by :meth:`get_config`.

        Returns
        -------
        FrameworkModel
            Model initialized from the native file and restored configuration.
        """
        instance = cls.__new__(cls)
        opensim.Model.__init__(instance, str(model_file))
        resolved_model_file = Path(model_file).resolve()
        geometry_directory = resolved_model_file.parent / "Geometry"
        object.__setattr__(instance, "_osf_state", None)
        object.__setattr__(instance, "_osf_model_file", resolved_model_file)
        object.__setattr__(instance, "_osf_assets_directory", resolved_model_file.parent)
        object.__setattr__(
            instance,
            "_geometry_dirs",
            [geometry_directory] if geometry_directory.is_dir() else [],
        )
        if geometry_directory.is_dir():
            register_geometry_search_path(opensim, geometry_directory)
        instance.set_config(config)
        return instance

    @classmethod
    def from_model(
        cls,
        source: opensim.Model,
        trajectories: opensim.StatesTrajectory | None = None,
    ) -> Self:
        """Create a FrameworkModel copy of a native OpenSim model.

        Parameters
        ----------
        source : opensim.Model
            Model to copy. FrameworkModel subclasses retain their Python
            configuration.
        trajectories : opensim.StatesTrajectory or None, optional
            Optional sequence of source-model states to attach to the copy.

        Returns
        -------
        FrameworkModel
            Independent model instance with its initial state copied.

        Raises
        ------
        TypeError
            If ``source`` is not an OpenSim model or ``trajectories`` is not
            an OpenSim StatesTrajectory.
        """
        if not isinstance(source, opensim.Model):
            raise TypeError("source must be an opensim.Model")
        if trajectories is not None and not isinstance(
            trajectories, opensim.StatesTrajectory
        ):
            raise TypeError("trajectories must be an opensim.StatesTrajectory or None")

        source_is_framework_model = isinstance(source, FrameworkModel)
        model_class = type(source) if source_is_framework_model else cls
        instance = model_class.__new__(model_class)
        opensim.Model.__init__(instance, source)
        object.__setattr__(instance, "_osf_state", None)
        object.__setattr__(instance, "_osf_trajectory", None)
        model_file = getattr(source, "model_file", None)
        object.__setattr__(
            instance,
            "_osf_model_file",
            Path(model_file).resolve() if model_file is not None else None,
        )
        asset_directories = getattr(source, "geometry_directories", ())
        object.__setattr__(
            instance,
            "_osf_assets_directory",
            Path(asset_directories[0]).resolve() if asset_directories else None,
        )
        object.__setattr__(
            instance,
            "_geometry_dirs",
            list(asset_directories),
        )
        if source_is_framework_model:
            instance.set_config(source.get_config())

        cloned_state = opensim.Model.initSystem(instance)
        if source_is_framework_model:
            source_state = source.state
        else:
            try:
                source_state = source.getWorkingState()
            except RuntimeError:
                source_state = None
            if source_state is None:
                source_state = source.initSystem()
        source_names = source.getStateVariableNames()
        clone_names = instance.getStateVariableNames()
        clone_indices = {
            clone_names.get(index): index for index in range(clone_names.getSize())
        }
        source_values = source.getStateVariableValues(source_state)
        for index in range(source_names.getSize()):
            name = source_names.get(index)
            clone_index = clone_indices.get(name)
            if clone_index is not None:
                instance.setStateVariableValue(
                    cloned_state, name, source_values.get(index)
                )
        cloned_state.setTime(source_state.getTime())
        object.__setattr__(instance, "_osf_state", cloned_state)

        if trajectories is not None:
            converted_trajectory = opensim.StatesTrajectory()
            for trajectory_index in range(trajectories.getSize()):
                source_trajectory_state = trajectories.get(trajectory_index)
                trajectory_values = source.getStateVariableValues(
                    source_trajectory_state
                )
                target_state = opensim.State(cloned_state)
                for source_index in range(source_names.getSize()):
                    name = source_names.get(source_index)
                    if name in clone_indices:
                        instance.setStateVariableValue(
                            target_state,
                            name,
                            trajectory_values.get(source_index),
                        )
                target_state.setTime(source_trajectory_state.getTime())
                instance.realizePosition(target_state)
                converted_trajectory.append(target_state)
            object.__setattr__(instance, "_osf_trajectory", converted_trajectory)
        return instance

register_framework_model(FrameworkModel)


def _registered_model_class(identifier: str) -> type[FrameworkModel] | None:
    """Return a previously imported model class by its stable identifier."""
    return _MODEL_CLASSES.get(identifier)
