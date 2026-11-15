"""OpenSim screen bodies and their display geometry."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ...bodies import Box, MESHES_DIR, mass_properties, matrix_to_euler_xyz
from ...utils.constants import MM_TO_M
from ...utils.geometry import enclosing_parallelepiped, get_mesh_vertices_in_ground
from ...utils.opensim import import_opensim

opensim = import_opensim()

__all__ = ["Screen"]

_DEFAULT_COLOR = (0.25, 0.65, 0.8)


class Screen(opensim.Body):
    """OpenSim body for a screen with a generated parallelepiped mesh.

    Width, height, and depth use metres; mass uses kilograms. The body frame is
    centred on the generated mesh. Geometry is attached to the body, which can
    then be added and welded into an equipment model.

    Parameters
    ----------
    width : float
        Screen width in metres.
    height : float
        Screen height in metres.
    depth : float
        Screen depth in metres.
    mass : float
        Body mass in kilograms.
    name : str, default="screen"
        OpenSim body name.
    color : sequence of float, default=(0.25, 0.65, 0.8)
        RGB display color, with each component in the range [0, 1].

    Raises
    ------
    ValueError
        If dimensions, mass, or color are invalid.
    """

    def __init__(
        self,
        width: float,
        height: float,
        depth: float,
        mass: float,
        *,
        name: str = "screen",
        color: Sequence[float] = _DEFAULT_COLOR,
    ) -> None:
        dimensions = tuple(
            self._positive(field, value)
            for field, value in (
                ("width", width),
                ("height", height),
                ("depth", depth),
                ("mass", mass),
            )
        )
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        display_color = np.asarray(color, dtype=float)
        if (
            display_color.shape != (3,)
            or not np.all(np.isfinite(display_color))
            or np.any((display_color < 0.0) | (display_color > 1.0))
        ):
            raise ValueError("color must contain three values in [0, 1]")

        width_value, height_value, depth_value, mass_value = dimensions
        shape = Box(size=(depth_value, height_value, width_value))
        density = mass_value / shape.volume
        computed_mass, center_of_mass, inertia_matrix = mass_properties(
            [shape], density
        )
        inertia = opensim.Inertia(
            float(inertia_matrix[0, 0]),
            float(inertia_matrix[1, 1]),
            float(inertia_matrix[2, 2]),
            float(inertia_matrix[0, 1]),
            float(inertia_matrix[0, 2]),
            float(inertia_matrix[1, 2]),
        )
        opensim.Body.__init__(
            self,
            name,
            computed_mass,
            opensim.Vec3(*map(float, center_of_mass)),
            inertia,
        )

        mesh = shape._geometry()
        mesh.setColor(opensim.Vec3(*map(float, display_color)))
        frame = opensim.PhysicalOffsetFrame()
        frame.setName(f"{name}_mesh_frame")
        frame.setParentFrame(self)
        frame.set_translation(opensim.Vec3(0.0, 0.0, 0.0))
        frame.set_orientation(opensim.Vec3(0.0, 0.0, 0.0))
        previous_log_level = opensim.Logger.getLevelString()
        opensim.Logger.setLevelString("Off")
        try:
            frame.attachGeometry(mesh)
            self.addComponent(frame)
        finally:
            opensim.Logger.setLevelString(previous_log_level)

        object.__setattr__(self, "_color", tuple(map(float, display_color)))
        object.__setattr__(
            self, "_dimensions", (width_value, height_value, depth_value)
        )
        object.__setattr__(self, "_mesh", mesh)
        object.__setattr__(self, "_mesh_frame", frame)

    @classmethod
    def from_existing_body(
        cls,
        body: opensim.Body,
        *,
        geometry_directories: Sequence[str | Path] = (),
        state: opensim.State | None = None,
    ) -> Screen:
        """Create a screen by fitting its mesh to an existing body's geometry.

        All meshes attached to ``body`` are combined in Ground and enclosed by
        an oriented parallelepiped. The inferred depth, height, width, and
        ground orientation define a new screen body with its own generated
        cuboid mesh; the source body's name and mass properties are retained.

        Parameters
        ----------
        body : opensim.Body
            Existing body whose attached mesh geometry defines the screen.
        geometry_directories : sequence of str or pathlib.Path, optional
            Additional directories searched for relative mesh filenames.
        state : opensim.State or None, optional
            State used to determine the source body's Ground pose. When
            omitted, the model's working state or a newly initialized state is
            used if the body belongs to a model.

        Returns
        -------
        Screen
            New screen body fitted to the source mesh.

        Raises
        ------
        TypeError
            If ``body`` is not an ``opensim.Body``.
        ValueError
            If the body has no valid three-dimensional mesh.
        FileNotFoundError
            If an attached mesh file cannot be resolved.
        """
        if not isinstance(body, opensim.Body):
            raise TypeError("body must be an opensim.Body")
        vertices = get_mesh_vertices_in_ground(
            body,
            state=state,
            directories=(MESHES_DIR, *geometry_directories),
            owner_label="Screen",
        )
        dimensions, orientation, center, _ = enclosing_parallelepiped(vertices)
        screen = cls(
            width=float(dimensions[2]),
            height=float(dimensions[1]),
            depth=float(dimensions[0]),
            mass=float(body.getMass()),
            name=body.getName() or "screen",
        )
        screen._mesh_frame.set_translation(opensim.Vec3(*map(float, center)))
        screen._mesh_frame.set_orientation(
            opensim.Vec3(*matrix_to_euler_xyz(orientation))
        )
        screen.setMass(body.getMass())
        screen.setMassCenter(body.getMassCenter())
        screen.setInertia(body.getInertia())
        return screen

    @staticmethod
    def _positive(name: str, value: float) -> float:
        """Validate and normalize a positive finite scalar.

        Parameters
        ----------
        name : str
            Parameter name used in the validation error.
        value : float
            Value to validate.

        Returns
        -------
        float
            Normalized scalar value.

        Raises
        ------
        ValueError
            If ``value`` is not positive and finite.
        """
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite value")
        return value

    @property
    def mesh(self) -> opensim.Mesh:
        """Return the generated display mesh attached to this body.

        Returns
        -------
        opensim.Mesh
            Screen panel mesh.
        """
        return self._mesh

    @property
    def name(self) -> str:
        """Return the OpenSim body name."""
        return self.getName()

    @name.setter
    def name(self, value: str) -> None:
        self.set_name(value)

    def set_name(self, value: str) -> None:
        """Set the body name and keep its attached mesh frame name aligned."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("name must be a non-empty string")
        self.setName(value)
        frame = getattr(self, "_mesh_frame", None)
        if frame is not None:
            frame.setName(f"{value}_mesh_frame")

    @property
    def mass(self) -> float:
        """Return the body mass in kilograms."""
        return float(self.getMass())

    @mass.setter
    def mass(self, value: float) -> None:
        self.set_mass(value)

    def set_mass(self, value: float) -> None:
        """Set body mass and update the cuboid inertia tensor."""
        mass = self._positive("mass", value)
        width, height, depth = self._dimensions
        self.setMass(mass)
        self.setInertia(
            opensim.Inertia(
                mass * (height**2 + width**2) / 12.0,
                mass * (depth**2 + width**2) / 12.0,
                mass * (depth**2 + height**2) / 12.0,
                0.0,
                0.0,
                0.0,
            )
        )

    @property
    def color(self) -> tuple[float, float, float]:
        """Return the mesh display color as RGB components."""
        return self._color

    @color.setter
    def color(self, value: Sequence[float]) -> None:
        self.set_color(value)

    def set_color(self, value: Sequence[float]) -> None:
        """Set the mesh display color using RGB values in [0, 1]."""
        color = np.asarray(value, dtype=float)
        if (
            color.shape != (3,)
            or not np.all(np.isfinite(color))
            or np.any((color < 0.0) | (color > 1.0))
        ):
            raise ValueError("color must contain three values in [0, 1]")
        normalized = tuple(float(component) for component in color)
        self._mesh.setColor(opensim.Vec3(*normalized))
        self._color = normalized

    @property
    def corners(self) -> np.ndarray:
        """Return the eight corners of the mesh's circumscribing box in Ground.

        Returns
        -------
        numpy.ndarray, shape (8, 3)
            Ground-frame corners ordered by depth, height, and width signs.
        """
        return self._fitted_parallelepiped[3]

    @property
    def surface_corners(self) -> np.ndarray:
        """Return the four corners of the screen's centre panel in Ground.

        Returns
        -------
        numpy.ndarray, shape (4, 3)
            Ground-frame panel corners, excluding the front and back faces'
            depth offsets.
        """
        dimensions, orientation, center, _ = self._fitted_parallelepiped
        return np.asarray(
            [
                center
                + height * dimensions[1] / 2 * orientation[:, 1]
                + width * dimensions[2] / 2 * orientation[:, 2]
                for height in (-1.0, 1.0)
                for width in (-1.0, 1.0)
            ]
        )

    @property
    def width(self) -> float:
        """Return the fitted screen width in metres.

        Returns
        -------
        float
            Screen width derived from the circumscribing parallelepiped.
        """
        return float(self._fitted_parallelepiped[0][2])

    @width.setter
    def width(self, value: float) -> None:
        self.set_width(value)

    def set_width(self, value: float) -> None:
        """Set screen width in metres and rebuild its mass properties."""
        self._resize(width=self._positive("width", value))

    @property
    def height(self) -> float:
        """Return the fitted screen height in metres.

        Returns
        -------
        float
            Screen height derived from the circumscribing parallelepiped.
        """
        return float(self._fitted_parallelepiped[0][1])

    @height.setter
    def height(self, value: float) -> None:
        self.set_height(value)

    def set_height(self, value: float) -> None:
        """Set screen height in metres and rebuild its mass properties."""
        self._resize(height=self._positive("height", value))

    @property
    def depth(self) -> float:
        """Return the fitted screen depth in metres.

        Returns
        -------
        float
            Screen depth derived from the circumscribing parallelepiped.
        """
        return float(self._fitted_parallelepiped[0][0])

    @depth.setter
    def depth(self, value: float) -> None:
        self.set_depth(value)

    def set_depth(self, value: float) -> None:
        """Set screen depth in metres and rebuild its mass properties."""
        self._resize(depth=self._positive("depth", value))

    @property
    def orientation(self) -> np.ndarray:
        """Return the fitted screen orientation matrix relative to Ground.

        Returns
        -------
        numpy.ndarray, shape (3, 3)
            Rotation matrix with columns along depth, height, and width.
        """
        return self._fitted_parallelepiped[1]

    @property
    def inclination_deg(self) -> float:
        """Return the screen's upward-axis inclination relative to Ground.

        Returns
        -------
        float
            Directed angle from Ground ``+x`` to the fitted panel-up axis,
            in degrees.
        """
        up = self.orientation[:, 1]
        return math.degrees(math.atan2(float(up[1]), float(up[0])))

    @property
    def width_mm(self) -> float:
        """Return the fitted screen width in millimetres."""
        return self.width / MM_TO_M

    @property
    def height_mm(self) -> float:
        """Return the fitted screen height in millimetres."""
        return self.height / MM_TO_M

    @property
    def depth_mm(self) -> float:
        """Return the fitted screen depth in millimetres."""
        return self.depth / MM_TO_M

    def samples(self, grid_size: int) -> np.ndarray:
        """Sample a uniform grid over the screen's centre panel in Ground.

        Parameters
        ----------
        grid_size : int
            Number of evenly spaced cell centres along each panel dimension.

        Returns
        -------
        numpy.ndarray, shape (grid_size**2, 3)
            Sample points in metres.

        Raises
        ------
        ValueError
            If ``grid_size`` is not a positive integer.
        """
        if (
            not isinstance(grid_size, int)
            or isinstance(grid_size, bool)
            or grid_size < 1
        ):
            raise ValueError("grid_size must be a positive integer")
        dimensions, orientation, center, _ = self._fitted_parallelepiped
        values = (np.arange(grid_size) + 0.5) / grid_size - 0.5
        return np.asarray(
            [
                center
                + height * dimensions[1] * orientation[:, 1]
                + width * dimensions[2] * orientation[:, 2]
                for height in values
                for width in values
            ]
        )

    @property
    def _fitted_parallelepiped(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit the attached screen mesh in Ground coordinates."""
        vertices = get_mesh_vertices_in_ground(
            self,
            state=getattr(self, "_ground_state", None),
            directories=(MESHES_DIR,),
            owner_label="Screen",
        )
        return enclosing_parallelepiped(vertices)

    def _resize(
        self,
        *,
        width: float | None = None,
        height: float | None = None,
        depth: float | None = None,
    ) -> None:
        current_width, current_height, current_depth = self._dimensions
        dimensions = (
            current_width if width is None else width,
            current_height if height is None else height,
            current_depth if depth is None else depth,
        )
        mass = self.mass
        self._mesh.set_scale_factors(
            opensim.Vec3(dimensions[2], dimensions[1], dimensions[0])
        )
        self._dimensions = dimensions
        self.set_mass(mass)

    def _set_ground_state(self, state: opensim.State) -> None:
        """Set the state used to evaluate this body's Ground-frame geometry."""
        object.__setattr__(self, "_ground_state", state)
