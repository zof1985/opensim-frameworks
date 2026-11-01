"""The ANSUR-based anthropometric :class:`User`, backed by an OpenSim model.

Construction, scaling orchestration, geometry/visualization IO and the
anatomical-body setup (torso/neck/skull split, hand articulation) live here.
Each other concern is a private mixin composed into :class:`User`:
explicit measurement setters (:mod:`.measurement_generated`), posture
setter/getter pairs for the bundled model's native coordinates
(:mod:`.posture_generated`) and for the custom neck/skull joints
(:mod:`.neck_posture`), per-joint centre properties
(:mod:`.joint_centers_generated`) and aggregated landmark/marker properties
(:mod:`.landmarks`), anthropometric scaling internals (:mod:`.scaling`),
derived body-segment measurements (:mod:`.segment_measurements`), and
hand-to-grip fitting (:mod:`.hand_fitting`).
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ...bodies.shapes import MESHES_DIR
from ...utils.opensim import (
    get_referenced_meshes,
    import_opensim,
    read_vtp_vertices,
    register_geometry_search_path,
)
from .data import DEFAULT_DATASET, resolve_reference
from .hand import MOVING_HAND_SEGMENTS, add_articulated_hands
from .hand_fitting import _HandFittingMixin
from .joint_center_table import JOINT_CENTER_NAMES as _JOINT_CENTER_NAMES
from .joint_centers_generated import _JointCenterMixin
from .landmarks import _FOOT_MARKER_NAMES, _LandmarkMixin
from .mapping import segment_scale_factors
from .measurement_generated import _MeasurementMixin
from .neck_posture import _NeckPostureMixin
from .posture_generated import _PostureMixin
from .scaling import _AnthropometricScalingMixin, _validate_mass_kg
from .segment_measurements import _SegmentMeasurementMixin

__all__ = ["User"]

opensim = import_opensim()

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_MODEL_PATH = _ASSETS_DIR / "rajagopalaiulrich2023.osim"
DEFAULT_MESHES_DIR = _ASSETS_DIR / "meshes"


def _remove_segment_mass(body: Any, mass: float, center: np.ndarray) -> None:
    """Split a rigid segment's mass while preserving its first moment.

    Parameters
    ----------
    body : opensim.Body
        Body whose mass and centre of mass are updated in place.
    mass : float
        Mass in kilograms to remove from the body.
    center : numpy.ndarray, shape (3,)
        Centre of mass of the removed mass in metres, in the body frame.

    Raises
    ------
    ValueError
        If removing ``mass`` would leave the original body with no positive
        mass.
    """
    original_mass = body.getMass()
    remaining_mass = original_mass - mass
    if remaining_mass <= 0:
        raise ValueError(f"cannot split {mass:g} kg from body {body.getName()!r}")
    original_center = np.array([body.getMassCenter().get(i) for i in range(3)])
    remaining_center = (
        original_mass * original_center - mass * center
    ) / remaining_mass
    body.scaleMass(remaining_mass / original_mass)
    body.setMassCenter(opensim.Vec3(*remaining_center))


class User(
    opensim.Model,
    _MeasurementMixin,
    _PostureMixin,
    _JointCenterMixin,
    _NeckPostureMixin,
    _AnthropometricScalingMixin,
    _LandmarkMixin,
    _SegmentMeasurementMixin,
    _HandFittingMixin,
):
    """An ANSUR-based anthropometric user backed by an OpenSim model.

    Built from the Rajagopal-Lai-Uhlrich full-body OpenSim model
    (``rajagopalaiulrich2023.osim``, bundled under ``assets/``), scaled
    per-body to the requested anthropometry using ANSUR II reference data
    (see the README's "Anthropometric scaling" section), then calibrated
    so the model's total mass matches a real anthropometric mass reference
    (``mass_kg``, if given, otherwise the resolved ANSUR ``weight_kg`` --
    see :attr:`mass_kg`) rather than whatever the geometric scaling alone
    produces volumetrically; unlike every other anthropometric parameter
    here, total mass can also be recalibrated after construction, via
    :meth:`set_mass_kg`. Body posture angles are exposed through
    ``set_<joint>_<motion>``/``<joint>_<motion>`` setter/getter pairs.
    Finger coordinates are accessed with :meth:`finger_coordinate`,
    :meth:`set_finger_angle`, and :meth:`finger_angle`. They are initially
    locked at the corrected reference pose, which is not an anatomical zero.
    Finger joint geometry and inertial properties are approximations, and
    the model does not include finger muscles or skin-contact geometry.
    Anthropometric dimensions have explicit measurement setters and
    properties. Bilateral setters (for example ``set_left_arm_length`` and
    ``set_right_arm_length``) scale only their corresponding side; ANSUR's
    shared measurements provide equal initial values until one side is
    overridden. Each hand body is connected to its radius by a universal
    wrist joint with independent flexion/extension and radial/ulnar-deviation
    coordinates. ``RPALM`` and ``LPALM`` are hand-attached markers at the
    anthropometric palm centres, repositioned after body scaling.
    After scaling, the model is translated vertically so the lowest vertices
    of both foot meshes touch the ground plane (``y = 0``). This is a
    geometry-based approximation and does not model soft-tissue contact.

    Parameters
    ----------
    gender : str | Literal['M', 'F']
        Sex code used to select the ANSUR II reference population. Must be
        exactly ``"M"`` or ``"F"`` (case-sensitive, no other values
        accepted).
    height_cm : float or None, optional
        Requested stature, in centimetres. If provided, every anthropometric
        measurement (stature included) is resolved directly from this
        height via a per-measurement PCHIP regression against the ANSUR
        subjects of the requested ``gender``, and takes precedence over
        ``percentile`` (which is still computed afterwards, purely for
        information -- see :attr:`percentile`). A height outside the ANSUR
        range observed for that gender is not rejected: it is extrapolated
        (PCHIP with linear tails) and raises a :class:`UserWarning`, with
        extrapolation reliability dropping the further the requested height
        is from the observed range.
    percentile : float, optional
        Common ANSUR percentile applied to every numeric measurement when
        ``height_cm`` is not provided. Defaults to ``50.0``. Must lie in the
        inclusive interval ``[0.1, 99.9]``; this applies ``numpy.percentile``
        independently per measurement (no single reference subject is
        selected, and no interpolation is performed across different
        measurements).
    mass_kg : float or None, optional
        Requested total body mass, in kilograms. If provided, it is used
        as an *independent* scaling input, exactly like ``height_cm``: once
        every body has been scaled geometrically (see the README's
        "Anthropometric scaling" section), every body's mass *and* inertia
        tensor are uniformly rescaled (a single scalar factor, via
        ``opensim.Body.scaleMass``) so the whole model's total mass
        (``self.getTotalMass(self.state)``) matches ``mass_kg`` exactly --
        independent of whatever the height-driven geometric scaling alone
        would have produced volumetrically. If not provided (the default),
        the same rescaling is still applied, but targeting the ANSUR
        ``weight_kg`` resolved for this user's height/percentile (see
        :attr:`anthropometry`) -- i.e. mass is always resolved "like the
        other measurements" (from height when given, via
        :func:`~opensim_frameworks.models.user.data.resolve_reference`),
        never left as a pure side effect of geometric scaling. See
        :attr:`mass_kg` to read back the resolved value either way.
    dataset : str or pathlib.Path, optional
        Path to the ANSUR II reference CSV to resolve measurements from.
        Defaults to the dataset bundled with this package.
    Raises
    ------
    ValueError
        If ``gender`` is not ``"M"``/``"F"``, if ``height_cm`` is given but is
        not a positive finite number, if ``percentile`` is not finite or
        falls outside ``[0.1, 99.9]``, if ``mass_kg`` is not a positive finite
        number, or if both feet cannot be grounded by vertical translation.
    """

    def __init__(
        self,
        gender: str | Literal["M", "F"],
        height_cm: float | None = None,
        percentile: float = 50.0,
        mass_kg: float | None = None,
        *,
        dataset: str | Path = DEFAULT_DATASET,
    ):
        """Resolve anthropometry, scale the base model, and calibrate its mass.

        The resolved reference determines segment scaling relative to the
        50th-percentile baseline for the same gender. The model's total mass
        is then calibrated to ``mass_kg`` or, when omitted, to the resolved
        ANSUR ``weight_kg``. The scaled foot meshes are placed on ground by
        translating the pelvis vertically without changing joint angles.

        Parameters
        ----------
        gender : {"M", "F"}
            ANSUR population used to resolve anthropometric measurements.
        height_cm : float or None, optional
            Requested stature in centimetres. When omitted, ``percentile``
            selects the reference.
        percentile : float, optional
            ANSUR percentile used when resolving the reference. Defaults to
            50.
        mass_kg : float or None, optional
            Target total model mass in kilograms. Defaults to the resolved
            ANSUR body mass.
        dataset : str or pathlib.Path, optional
            Path to an alternative ANSUR II reference CSV.

        Raises
        ------
        ValueError
            If the supplied gender, stature, percentile, or mass is invalid,
            or if both feet cannot be grounded by vertical translation.
        UserWarning
            If the requested stature lies outside the ANSUR range.
        """
        if mass_kg is not None:
            _validate_mass_kg(mass_kg)
        self._reference = resolve_reference(gender, height_cm, percentile, dataset)
        self.opensim = opensim
        self._geometry_dirs: list[Path] = []
        self._side_measurements: dict[str, float] = {}
        self._visualizer = None
        # Meshes are resolved while the native model file is loaded.
        register_geometry_search_path(opensim, DEFAULT_MESHES_DIR)
        opensim.Model.__init__(self, str(DEFAULT_MODEL_PATH))
        self._unlock_coordinates()
        self._baseline = resolve_reference(
            self.gender, percentile=50.0, dataset=dataset
        )
        self._add_anatomical_bodies()
        # Split the hand meshes into articulated bodies before initialization.
        add_articulated_hands(self, opensim, DEFAULT_MESHES_DIR)
        self.finalizeConnections()
        self.state = self.initSystem()
        self.add_geometry_directory(DEFAULT_MESHES_DIR)
        factors = segment_scale_factors(self._reference, self._baseline)
        self.scale_bodies(self._expand_bilateral_bodies(factors))
        # Geometric scaling alone only moves mass as a volumetric side
        # effect (OpenSim's own preserveMassDist=True, driven purely by the
        # height-derived per-body scale factors above) -- not calibrated
        # against any real mass reference. This always (re)calibrates the
        # model's total mass against either the explicit mass_kg or, if not
        # given, the ANSUR weight_kg resolved for this user, exactly as
        # every other anthropometric measurement is already resolved "from
        # height when given, else from percentile".
        target_mass_kg = (
            mass_kg if mass_kg is not None else self._reference.values["weight_kg"]
        )
        self._rescale_total_mass(target_mass_kg)
        self._mass_kg = target_mass_kg
        self._place_feet_on_ground()

    def _place_feet_on_ground(self) -> None:
        """Translate the scaled model so both foot meshes touch ground."""
        self.realizePosition(self.state)
        foot_heights = []
        for side in ("r", "l"):
            mesh_heights = []
            for body_name in (f"calcn_{side}", f"toes_{side}"):
                body = self.getBodySet().get(body_name)
                for index in range(body.getPropertyByName("attached_geometry").size()):
                    mesh = opensim.Mesh.safeDownCast(body.get_attached_geometry(index))
                    if mesh is None:
                        continue
                    mesh_path = DEFAULT_MESHES_DIR / mesh.get_mesh_file()
                    vertices = read_vtp_vertices(mesh_path)
                    scale = np.array(
                        [mesh.get_scale_factors().get(axis) for axis in range(3)]
                    )
                    transform = mesh.getFrame().getTransformInGround(self.state)
                    vertical = np.array(
                        [transform.R().get(1, axis) for axis in range(3)]
                    )
                    heights = (vertices * scale) @ vertical + transform.p().get(1)
                    mesh_heights.append(float(np.min(heights)))
            if not mesh_heights:
                raise ValueError(f"No foot mesh is attached on side {side!r}")
            foot_heights.append(min(mesh_heights))

        if not all(math.isfinite(height) for height in foot_heights):
            raise ValueError("Foot mesh heights must be finite")
        if not math.isclose(foot_heights[0], foot_heights[1], abs_tol=1e-8):
            raise ValueError(
                "The scaled feet cannot both touch ground by vertical translation: "
                f"right height {foot_heights[0]:.9g} m, "
                f"left height {foot_heights[1]:.9g} m"
            )

        coordinate = self.getCoordinateSet().get("pelvis_ty")
        target = coordinate.getValue(self.state) - min(foot_heights)
        if not coordinate.getRangeMin() <= target <= coordinate.getRangeMax():
            raise ValueError(
                f"Grounding the feet requires pelvis_ty={target:.9g} m outside "
                f"[{coordinate.getRangeMin():.9g}, {coordinate.getRangeMax():.9g}] m"
            )
        coordinate.setDefaultValue(target)
        coordinate.setValue(self.state, target, False)
        self.realizePosition(self.state)

    def _add_anatomical_bodies(self) -> None:
        """Create articulated head segments and anatomical eye/palm markers.

        The skull and neck are separated from the torso, their mesh geometry
        is reassigned, and cervical joints are added. Eye markers are attached
        to the skull. Palm-centre markers are attached to the existing
        articulated hand bodies; their final local offsets are updated after
        anthropometric scaling.
        """
        torso = self.getBodySet().get("torso")
        head_meshes = []
        spine_mesh = None
        for index in range(torso.getPropertyByName("attached_geometry").size()):
            mesh = opensim.Mesh.safeDownCast(torso.get_attached_geometry(index))
            if mesh is None:
                continue
            if mesh.get_mesh_file() in {"hat_skull.vtp", "hat_jaw.vtp"}:
                mesh.upd_Appearance().set_visible(False)
                head_meshes.append(mesh)
            elif mesh.get_mesh_file() == "hat_spine.vtp":
                mesh.set_mesh_file("hat_spine_torso.vtp")
                spine_mesh = mesh

        neck_center = np.array([0.0, 0.445, 0.0])
        neck_mass = 1000.0 * math.pi * 0.045**2 * 0.09
        neck_radial_inertia = neck_mass * (3 * 0.045**2 + 0.09**2) / 12
        neck_axial_inertia = 0.5 * neck_mass * 0.045**2
        if spine_mesh is None:
            raise RuntimeError("The bundled model does not contain hat_spine.vtp")
        neck = opensim.Body(
            "neck",
            neck_mass,
            opensim.Vec3(*neck_center),
            opensim.Inertia(
                neck_radial_inertia,
                neck_axial_inertia,
                neck_radial_inertia,
            ),
        )
        neck_mesh = opensim.Mesh("hat_neck.vtp")
        neck_mesh.setName("neck_spine")
        neck_mesh.set_scale_factors(spine_mesh.get_scale_factors())
        neck_mesh.setColor(spine_mesh.getColor())

        skull_mass = 4.5
        skull_center = np.array([0.0, 0.57, 0.0])
        skull_inertia = opensim.Inertia(0.0103, 0.0104, 0.0124)
        skull = opensim.Body(
            "skull",
            skull_mass,
            opensim.Vec3(*skull_center),
            skull_inertia,
        )

        _remove_segment_mass(
            torso,
            skull_mass + neck_mass,
            (skull_mass * skull_center + neck_mass * neck_center)
            / (skull_mass + neck_mass),
        )
        self.addBody(neck)
        self.addBody(skull)
        neck.attachGeometry(neck_mesh)
        # Map the universal joint's X/Y axes to anatomical flexion/lateral flexion.
        neck_joint_orientation = opensim.Vec3(
            -math.pi / 2,
            0.0,
            -math.pi / 2,
        )
        base_neck_joint = opensim.UniversalJoint(
            "torso_neck",
            torso,
            opensim.Vec3(0.0, 0.445, 0.0),
            neck_joint_orientation,
            neck,
            opensim.Vec3(0.0, 0.445, 0.0),
            neck_joint_orientation,
        )
        base_neck_coordinates = (
            ("neck_base_flexion", -0.7, 0.85),
            ("neck_base_lateral_flexion", -0.5, 0.5),
        )
        for index, (name, low, high) in enumerate(base_neck_coordinates):
            coordinate = base_neck_joint.getCoordinate(index)
            coordinate.setName(name)
            coordinate.setRangeMin(low)
            coordinate.setRangeMax(high)
            coordinate.set_clamped(True)
            coordinate.set_locked(False)
        self.addJoint(base_neck_joint)
        head_joint = opensim.BallJoint(
            "neck_skull",
            neck,
            opensim.Vec3(0.0, 0.49, 0.0),
            opensim.Vec3(0),
            skull,
            opensim.Vec3(0.0, 0.49, 0.0),
            opensim.Vec3(0),
        )
        neck_coordinates = (
            ("neck_lateral_flexion", -0.5, 0.5),
            ("neck_rotation", -1.22, 1.22),
            ("neck_flexion", -0.7, 0.85),
        )
        for index, (name, low, high) in enumerate(neck_coordinates):
            coordinate = head_joint.getCoordinate(index)
            coordinate.setName(name)
            coordinate.setRangeMin(low)
            coordinate.setRangeMax(high)
            coordinate.set_clamped(True)
            coordinate.set_locked(False)
        self.addJoint(head_joint)

        for mesh in head_meshes:
            head_mesh = opensim.Mesh(mesh.get_mesh_file())
            head_mesh.setName(f"skull_{mesh.getName()}")
            head_mesh.set_scale_factors(mesh.get_scale_factors())
            head_mesh.setColor(mesh.getColor())
            skull.attachGeometry(head_mesh)

        for marker_name, location in (
            ("REYE", (0.075, 0.56, 0.03)),
            ("LEYE", (0.075, 0.56, -0.03)),
        ):
            self.addMarker(opensim.Marker(marker_name, skull, opensim.Vec3(*location)))

        baseline_palm_length = self._baseline.values["palmlength"] * 1e-3
        for side, marker_name in (("r", "RPALM"), ("l", "LPALM")):
            hand = self.getBodySet().get(f"hand_{side}")
            self.addMarker(
                opensim.Marker(
                    marker_name,
                    hand,
                    opensim.Vec3(0.0, -baseline_palm_length / 2, 0.0),
                )
            )

    def _unlock_coordinates(self) -> None:
        """Unlock the coordinates that the bundled model locks by default."""
        coordinates = self.getCoordinateSet()
        for index in range(coordinates.getSize()):
            coordinates.get(index).set_locked(False)

    def update_state(self) -> None:
        """Realize the model through position, velocity, and dynamics stages.

        The method first realizes position and velocity, equilibrates muscles
        when the model contains any, and then realizes dynamics.
        """
        self.realizePosition(self.state)
        self.realizeVelocity(self.state)
        if self.getMuscles().getSize():
            self.equilibrateMuscles(self.state)
        self.realizeDynamics(self.state)

    def _sync_coordinate_defaults(self) -> None:
        coordinates = self.getCoordinateSet()
        for index in range(coordinates.getSize()):
            coordinate = coordinates.get(index)
            coordinate.setDefaultValue(coordinate.getValue(self.state))
            coordinate.setDefaultSpeedValue(coordinate.getSpeedValue(self.state))

        constraints = self.getConstraintSet()
        for index in range(constraints.getSize()):
            coupler = self.opensim.CoordinateCouplerConstraint.safeDownCast(
                constraints.get(index)
            )
            if coupler is None:
                continue
            independent_names = coupler.getIndependentCoordinateNames()
            independent_values = self.opensim.Vector(independent_names.getSize(), 0.0)
            for coordinate_index in range(independent_names.getSize()):
                independent = coordinates.get(independent_names.get(coordinate_index))
                independent_values.set(coordinate_index, independent.getDefaultValue())
            dependent = coordinates.get(coupler.getDependentCoordinateName())
            dependent.setDefaultValue(
                coupler.getFunction().calcValue(independent_values)
            )

    def reinitialize(self) -> None:
        """Rebuild the OpenSim system while preserving current coordinates.

        Current coordinate values are copied to their defaults, dependent
        coordinate defaults are recalculated from their coupler functions,
        model connections are finalized, and a fresh system state is created.
        """
        self._sync_coordinate_defaults()
        self.finalizeConnections()
        self.state = self.initSystem()

    def scale_bodies(self, factors: dict[str, tuple[float, float, float]]) -> None:
        """Apply OpenSim's native body scaling and preserve the current state.

        A factor for a hand body is also applied to its finger bodies unless
        a factor for a finger body is supplied explicitly. After scaling,
        each palm marker is repositioned at half the resolved palm length
        from the wrist along the hand's local negative y axis.

        Parameters
        ----------
        factors : dict[str, tuple[float, float, float]]
            Body names mapped to positive, unitless ``(x, y, z)`` scale
            factors. Names not present in this model are ignored.

        Raises
        ------
        ValueError
            If a scale factor for an existing body is not positive and finite.
        """
        # Propagate each hand scale to its fingers unless explicitly overridden.
        factors = dict(factors)
        for side in ("r", "l"):
            hand_name = f"hand_{side}"
            if hand_name in factors:
                for segment in MOVING_HAND_SEGMENTS:
                    factors.setdefault(f"{segment}_{side}", factors[hand_name])

        body_set = self.getBodySet()
        scale_set = self.opensim.ScaleSet()
        for body_name, axes in factors.items():
            if not body_set.contains(body_name):
                continue
            if len(axes) != 3 or any(
                not math.isfinite(value) or value <= 0 for value in axes
            ):
                raise ValueError(f"Invalid scale factor for body {body_name!r}: {axes}")
            scale = self.opensim.Scale()
            scale.setSegmentName(body_name)
            scale.setScaleFactors(self.opensim.Vec3(*axes))
            scale.setApply(True)
            scale_set.adoptAndAppend(scale)
        self.scale(self.state, scale_set, True)
        for side in ("left", "right"):
            marker = self.getMarkerSet().get("LPALM" if side == "left" else "RPALM")
            marker.set_location(
                self.opensim.Vec3(
                    0.0,
                    -float(getattr(self, f"{side}_palm_length")) / 2,
                    0.0,
                )
            )
        self._sync_coordinate_defaults()
        self.state = self.initSystem()

    def add_geometry_directory(self, directory: str | Path) -> None:
        """Register a directory containing geometry files for this model.

        Parameters
        ----------
        directory : str or pathlib.Path
            Directory to resolve and register with OpenSim's geometry search
            path.
        """
        path = Path(directory).expanduser().resolve()
        register_geometry_search_path(self.opensim, path)
        self._geometry_dirs.append(path)

    @property
    def geometry_directories(self) -> tuple[Path, ...]:
        """Return the geometry directories registered for this user.

        Returns
        -------
        tuple of pathlib.Path
            Resolved directories registered for mesh lookup, in registration
            order.
        """
        return tuple(self._geometry_dirs)

    @property
    def visualizer(self) -> Any | None:
        """Return the active VTK visualizer, if one is displaying this user.

        Returns
        -------
        VTKVisualizer or None
            Active visualizer, or ``None`` when the user is not shown.
        """
        return self._visualizer

    def show(
        self,
        motion: Any = None,
        *,
        loop: bool = False,
        fps: float = 30.0,
    ) -> Any:
        """Open the interactive viewer for this user.

        Parameters
        ----------
        motion : str, pathlib.Path, opensim.TimeSeriesTable, or None, default=None
            Optional motion table or path to a ``.mot``/``.sto`` file.
        loop : bool, default=False
            Whether playback repeats after the final sample.
        fps : float, default=30.0
            Playback refresh and animation-export frame rate.

        Returns
        -------
        PlayerWindow
            Handle used to inspect playback state or close the viewer.
        """
        from ...viewer import VTKViewer

        return VTKViewer().show(self, motion, loop=loop, fps=fps)

    def to_osim(
        self,
        file: str | Path,
        *,
        with_meshes: bool = True,
        mesh_dir: str = "Geometry",
        overwrite: bool = True,
    ) -> Path:
        """Export the scaled model in its current posture to an OpenSim file.

        OpenSim stores coordinate defaults in the model, but live coordinate
        values in its ``State``. This method synchronizes the current values
        before serialization, so the exported model opens in the current
        posture. When requested, referenced meshes are copied beside the
        model and their exported paths are made relative to ``mesh_dir``.

        Parameters
        ----------
        file : str or pathlib.Path
            Destination ``.osim`` file.
        with_meshes : bool, default=True
            Copy referenced mesh files beside the model.
        mesh_dir : str, default="Geometry"
            Relative folder name for copied mesh files.
        overwrite : bool, default=True
            Whether an existing destination file may be replaced.

        Returns
        -------
        pathlib.Path
            The destination path.

        Raises
        ------
        FileExistsError
            If the destination exists and ``overwrite`` is false.
        ValueError
            If ``mesh_dir`` is not a safe relative path or mesh filenames
            collide in the destination folder.
        FileNotFoundError
            If any referenced mesh cannot be located.
        RuntimeError
            If OpenSim cannot serialize the model.
        """
        destination = Path(file).expanduser()
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        mesh_subdirectory = Path(mesh_dir)
        if with_meshes and (
            not mesh_dir
            or mesh_subdirectory.is_absolute()
            or ".." in mesh_subdirectory.parts
        ):
            raise ValueError("mesh_dir must be a safe relative directory path")

        self._sync_coordinate_defaults()
        mesh_files = get_referenced_meshes(self) if with_meshes else []
        if with_meshes:
            self._export_mesh_files(destination.parent / mesh_subdirectory, mesh_files)

        try:
            if with_meshes:
                for mesh, filename in mesh_files:
                    mesh.set_mesh_file(str(mesh_subdirectory / Path(filename).name))
            self.printToXML(str(destination))
        finally:
            for mesh, filename in mesh_files:
                mesh.set_mesh_file(filename)
        return destination

    def _export_mesh_files(
        self, destination_dir: Path, mesh_files: list[tuple[Any, str]]
    ) -> None:
        sources: dict[str, Path] = {}
        missing: list[str] = []
        search_directories = (
            *self._geometry_dirs,
            DEFAULT_MODEL_PATH.parent,
            MESHES_DIR,
        )
        for _, filename in mesh_files:
            source = next(
                (
                    directory / filename
                    for directory in search_directories
                    if (directory / filename).is_file()
                ),
                None,
            )
            if source is None:
                missing.append(filename)
                continue
            destination_name = Path(filename).name
            existing = sources.get(destination_name)
            if existing is not None and existing.resolve() != source.resolve():
                raise ValueError(
                    f"Mesh filename collision for {destination_name!r}: "
                    f"{existing} and {source}"
                )
            sources[destination_name] = source
        if missing:
            raise FileNotFoundError(
                "Could not locate referenced mesh file(s): "
                + ", ".join(sorted(missing))
            )
        if not sources:
            return
        destination_dir.mkdir(parents=True, exist_ok=True)
        for filename, source in sources.items():
            destination = destination_dir / filename
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)

    def _set_coordinate_degrees(self, name: str, degrees: float) -> None:
        if not math.isfinite(degrees):
            raise ValueError("coordinate angle must be finite")
        self.getCoordinateSet().get(name).setValue(
            self.state, math.radians(degrees), False
        )

    def _coordinate_degrees(self, name: str) -> float:
        value = self.getCoordinateSet().get(name).getValue(self.state)
        return math.degrees(value)

    def _expand_bilateral_bodies(self, factors: dict[str, tuple[float, float, float]]):
        expanded: dict[str, tuple[float, float, float]] = {}
        body_set = self.getBodySet()
        for body, values in factors.items():
            for candidate in (body, f"{body}_r", f"{body}_l"):
                if body_set.contains(candidate):
                    expanded[candidate] = values
        return expanded
