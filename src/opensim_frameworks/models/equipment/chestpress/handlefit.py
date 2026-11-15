"""Handle geometry and grip measurements used by the arm fit."""

from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from opensim_frameworks.utils.geometry import (
    body_transform_array,
    enclosing_cylinder,
    get_ls_line,
    get_mesh_vertices_in_ground,
)
from opensim_frameworks.utils.kinematics import marker_position_array
from opensim_frameworks import SHOULDER_GIRDLE_COORDINATES

__all__ = ["fit_handle", "mirror_arm"]

_ARM_COORDINATE_PREFIXES = (
    "arm_",
    "elbow_flex_",
    "pro_sup_",
    "wrist_",
    "index_",
    "middle_",
    "ring_",
    "little_",
    "thumb_",
)
_FIT_COORDINATE_PREFIXES = ("arm_", "elbow_flex_", "pro_sup_", "wrist_")


def mirror_arm(user, to_mirror: Literal["left", "right"]) -> None:
    """Copy one arm and shoulder girdle to the opposite side.

    Parameters
    ----------
    user : opensim_frameworks.User
        User whose arm coordinates are mirrored.
    to_mirror : {'left', 'right'}
        Source side to copy to the opposite arm.

    Raises
    ------
    ValueError
        If ``to_mirror`` is not ``"left"`` or ``"right"``.
    RuntimeError
        If source or target arm coordinates are missing.
    """
    if to_mirror not in ("left", "right"):
        raise ValueError("to_mirror must be 'left' or 'right'")

    source_suffix = "_r" if to_mirror == "right" else "_l"
    target_suffix = "_l" if to_mirror == "right" else "_r"
    coordinates = user.getCoordinateSet()
    names = {coordinates.get(index).getName() for index in range(coordinates.getSize())}
    source_names = sorted(
        name
        for name in names
        if name.endswith(source_suffix)
        and (
            name.startswith(_ARM_COORDINATE_PREFIXES)
            or name[: -len(source_suffix)] in SHOULDER_GIRDLE_COORDINATES
        )
    )
    if not source_names:
        raise RuntimeError(f"No {to_mirror} arm coordinates were found")

    for source_name in source_names:
        target_name = source_name[: -len(source_suffix)] + target_suffix
        if target_name not in names:
            raise RuntimeError(f"Missing mirrored arm coordinate {target_name}")
        source = coordinates.get(source_name)
        target = coordinates.get(target_name)
        default_locked = target.get_locked()
        state_locked = target.getLocked(user.state)
        target.set_locked(False)
        target.setLocked(user.state, False)
        try:
            target.setValue(user.state, source.getValue(user.state), False)
        finally:
            target.set_locked(default_locked)
            target.setLocked(user.state, state_locked)
    user.realizePosition(user.state)


def fit_handle(
    user,
    equipment,
    handle_name: str,
    side: Literal["left", "right"],
    *,
    wrist_extension_deg: float = 45.0,
    palm_handle_distance_m: float = 0.002,
    palm_position_handle: npt.ArrayLike | None = None,
    palm_contact_weight: float = 1e6,
    palm_facing_weight: float = 1000.0,
    knuckle_weight: float = 10000.0,
    knuckle_end_weight: float = 1e8,
    plane_distance_weight: float = 1e6,
    handle_end_weight: float = 1e6,
    pro_sup_weight: float = 100.0,
    wrist_dev_weight: float = 1000.0,
    wrist_extension_weight: float = 1000.0,
    protraction_weight: float = 5.0,
    scapular_elevation_weight: float = 1000.0,
    wrist_flexion_weight: float = 10000.0,
    arm_weight: float = 0.05,
) -> tuple[float | None, np.ndarray | None]:
    """Fit a user's arm and wrist against a handle.

    The user and equipment must share a Ground reference. The fitted side's
    arm, wrist and shoulder girdle are optimized; finger coordinates are left
    unchanged. The handle meshes are approximated by an enclosing cylinder.
    Retraction, low shoulder elevation, arm-plane alignment, knuckle-line
    alignment and palmar orientation are weighted objectives. The palm marker
    must remain outside the cylinder, and each MCP joint center must project
    strictly between its end disks. Scapular proximity is optimized against
    the rib mesh. Scapula intersections with ribs or spine and remaining
    separation violations are reported as warnings.

    The weighted solution is checked after optimization and after any scapular
    correction. An invalid solution is rejected without a constrained retry,
    even if another feasible solution might exist.

    Parameters
    ----------
    user : opensim_frameworks.User
        Scaled user to fit.
    equipment : opensim_frameworks.Equipment
        Equipment providing the handle.
    handle_name : str
        Name of the equipment body and its center marker.
    side : {'left', 'right'}
        Side being fitted (for the arm and shoulder prefix matching).
    wrist_extension_deg : float, default=45.0
        Target wrist extension angle, in degrees.
    palm_handle_distance_m : float, default=0.002
        Target distance in metres between the palm marker and the enclosing
        cylindrical approximation of the handle surface.
    palm_position_handle : array-like or None, default=None
        Target position of the palm marker relative to the center marker,
        expressed along the handle body's local axes. If None, the fit
        optimizes for contact.
    palm_contact_weight : float, default=1e6
        Weight for minimizing the palm-to-handle distance.
    palm_facing_weight : float, default=1000.0
        Weight for orienting the palmar normal toward the handle axis.
    knuckle_weight : float, default=10000.0
        Weight for aligning the least-squares line through the four MCP
        joint centers with the handle axis.
    knuckle_end_weight : float, default=1e8
        Weight penalizing MCP joint projections outside the handle ends.
        A final validity check rejects projections at or beyond an end.
    plane_distance_weight : float, default=1e6
        Weight for the elbow's distance from the plane through the shoulder
        and palm that is parallel to the handle axis.
    handle_end_weight : float, default=1e6
        Weight for restricting the hand to within the handle ends.
    pro_sup_weight : float, default=100.0
        Weight for neutral forearm rotation (90-degree pronation/supination).
    wrist_dev_weight : float, default=1000.0
        Weight for neutral wrist radial/ulnar deviation.
    wrist_extension_weight : float, default=1000.0
        Weight for the wrist-extension target.
    protraction_weight : float, default=5.0
        Weight penalizing protraction relative to the coordinate's range
        minimum, favoring a retracted shoulder.
    scapular_elevation_weight : float, default=1000.0
        Weight penalizing positive scapular elevation.
    wrist_flexion_weight : float, default=10000.0
        Weight strongly penalizing positive wrist flexion.
    arm_weight : float, default=0.05
        Weight for the "press posture" tie-break (favors internal rotation).

    Returns
    -------
    tuple of (float or None, np.ndarray or None)
        Fitted elbow flexion angle in degrees (or None if no valid fit was
        found), and palm position relative to the handle's center marker,
        expressed along its body's local axes (or None if no valid fit).

    Raises
    ------
    ValueError
        If ``side`` is not ``"left"`` or ``"right"``, or if arm coordinate
        prefixes, handle marker, or valid handle mesh are missing, or if the
        target distance, knuckle-end weight or palm position is invalid.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    if not math.isfinite(palm_handle_distance_m) or palm_handle_distance_m < 0:
        raise ValueError("palm_handle_distance_m must be finite and nonnegative")
    if not math.isfinite(knuckle_end_weight) or knuckle_end_weight < 0:
        raise ValueError("knuckle_end_weight must be finite and nonnegative")

    # Seleziono solo braccio, polso e cintura scapolare: le dita si chiudono
    # dopo il fit e non devono modificare la retta delle nocche durante il fit.
    suffix = "r" if side == "right" else "l"
    coordinates = user.getCoordinateSet()
    names = {coordinates.get(index).getName() for index in range(coordinates.getSize())}
    arm_names = sorted(
        name
        for name in names
        if name.endswith("_" + suffix) and name.startswith(_FIT_COORDINATE_PREFIXES)
    )
    if not arm_names:
        raise ValueError(f"No {side} arm coordinates found in user model")

    shoulder_names = [
        name + "_" + suffix
        for name in SHOULDER_GIRDLE_COORDINATES
        if name + "_" + suffix in names
    ]
    if len(shoulder_names) != len(SHOULDER_GIRDLE_COORDINATES):
        raise ValueError(f"Incomplete shoulder-girdle coordinates for side {side}")

    # Verifico il marker centrale e ricavo asse, raggio ed estremita' dalle
    # mesh, nel sistema locale del corpo che ruota con la leva.
    handle = equipment.getBodySet().get(handle_name)
    marker_set = equipment.getMarkerSet()
    if marker_set.getIndex(handle_name) < 0:
        raise ValueError(f"Marker {handle_name} not found in equipment model")
    handle_marker = marker_set.get(handle_name)
    if (
        handle_marker.getParentFrame().findBaseFrame().getAbsolutePathString()
        != handle.getAbsolutePathString()
    ):
        raise ValueError(f"Marker {handle_name} is not attached to its handle body")

    equipment.realizePosition(equipment.state)
    handle_frame = body_transform_array(handle, equipment.state)
    mesh_directories = list(equipment.geometry_directories)
    if equipment.mesh_path is not None:
        mesh_directories.append(equipment.mesh_path)
    vertices = get_mesh_vertices_in_ground(
        equipment,
        equipment.state,
        handle_name,
        mesh_directories,
    )
    vertices_local = (vertices - handle_frame[:3, 3]) @ handle_frame[:3, :3]
    cylinder_axis, cylinder_radius, end_min, end_max = enclosing_cylinder(
        vertices_local
    )
    axis_origin, axis_direction = cylinder_axis[:3], cylinder_axis[3:]
    min_axial = np.dot(end_min - axis_origin, axis_direction)
    max_axial = np.dot(end_max - axis_origin, axis_direction)
    target = None
    if palm_position_handle is not None:
        target = np.asarray(palm_position_handle, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("palm_position_handle must contain three finite values")

    # Recupero il marker del palmo e i centri articolari usati nella funzione
    # obiettivo; tutti i punti saranno letti nello stesso Ground.
    marker_name = side[0].upper() + "PALM"
    user_model = user
    user_marker = None
    for i in range(user_model.getMarkerSet().getSize()):
        if user_model.getMarkerSet().get(i).getName() == marker_name:
            user_marker = user_model.getMarkerSet().get(i)
            break

    if user_marker is None:
        raise ValueError(f"Marker {marker_name} not found in user model")
    def _apply_coordinates(coord_values: npt.ArrayLike) -> None:
        values = np.asarray(coord_values, dtype=float)
        for coord_name, value in zip(arm_names, values):
            coordinates.get(coord_name).setValue(user.state, float(value), False)
        user.set_scapular_girdle_values(
            suffix, values[len(arm_names):], check_contact=False
        )

    def _surface_gap() -> float:
        palm = marker_position_array(user_marker, user.state)
        frame = body_transform_array(handle, equipment.state)
        local = frame[:3, :3].T @ (palm - frame[:3, 3])
        delta = local - axis_origin
        radial = delta - np.dot(delta, axis_direction) * axis_direction
        return float(np.linalg.norm(radial) - cylinder_radius)

    def _knuckle_end_gaps(knuckles_ground: np.ndarray, frame: np.ndarray) -> np.ndarray:
        # Proietto ogni nocca sull'asse del cilindro nel riferimento locale.
        knuckles_local = (knuckles_ground - frame[:3, 3]) @ frame[:3, :3]
        axial = (knuckles_local - axis_origin) @ axis_direction
        return np.concatenate((axial - min_axial, max_axial - axial))

    def _constraints_satisfied() -> bool:
        frame = body_transform_array(handle, equipment.state)
        knuckles = np.asarray([
            user.joint_center(f"{finger}_mcp_{suffix}")
            for finger in ("index", "middle", "ring", "little")
        ])
        return _surface_gap() >= -1e-9 and bool(
            np.all(_knuckle_end_gaps(knuckles, frame) > 0.0)
        )

    def _objective(coord_values: npt.ArrayLike) -> np.ndarray:
        """Evaluate weighted residuals for the arm fit."""
        # Aggiorno la postura e calcolo la distanza radiale dal cilindro.
        _apply_coordinates(coord_values)
        equipment.realizePosition(equipment.state)

        palm_marker_pos = marker_position_array(user_marker, user.state)
        handle_frame = body_transform_array(handle, equipment.state)
        rotation = handle_frame[:3, :3]
        palm_in_body = rotation.T @ (palm_marker_pos - handle_frame[:3, 3])
        center = marker_position_array(handle_marker, equipment.state)
        palm_in_handle = rotation.T @ (palm_marker_pos - center)

        axial_position = np.dot(palm_in_body - axis_origin, axis_direction)
        radial = palm_in_body - axis_origin - axial_position * axis_direction
        surface_gap = np.linalg.norm(radial) - cylinder_radius
        residuals = [
            math.sqrt(palm_contact_weight) * (surface_gap - palm_handle_distance_m)
        ]

        if target is not None:
            residuals.extend(
                math.sqrt(palm_contact_weight / 10.0) * (palm_in_handle - target)
            )

        # Regolarizzo polso, avambraccio e rotazione dell'omero. Nel modello
        # la flessione del polso e' positiva e l'estensione e' negativa.
        wrist_ext = coordinates.get(f"wrist_flex_{suffix}").getValue(user.state)
        residuals.append(
            math.sqrt(wrist_extension_weight)
            * (wrist_ext + math.radians(wrist_extension_deg))
        )
        residuals.append(math.sqrt(wrist_flexion_weight) * max(wrist_ext, 0.0))

        wrist_dev = coordinates.get(f"wrist_dev_{suffix}").getValue(user.state)
        residuals.append(math.sqrt(wrist_dev_weight) * wrist_dev)

        pro_sup = coordinates.get(f"pro_sup_{suffix}").getValue(user.state)
        residuals.append(math.sqrt(pro_sup_weight) * (pro_sup - math.pi / 2))

        arm_rot = coordinates.get(f"arm_rot_{suffix}").getValue(user.state)
        residuals.append(math.sqrt(arm_weight) * (arm_rot - math.pi / 2))

        # L'utente valuta retrazione, elevazione e contatto scapolo-toracico.
        residuals.extend(user.scapular_fit_residuals(
            suffix, protraction_weight, scapular_elevation_weight
        ))

        # Ricavo la retta ai minimi quadrati delle nocche e la confronto con
        # l'asse dell'impugnatura, senza distinguere i due versi della retta.
        shoulder = np.asarray(user.joint_center(f"acromial_{suffix}"))
        elbow = np.asarray(user.joint_center(f"elbow_{suffix}"))
        wrist = np.asarray(user.joint_center(f"radius_hand_{suffix}"))
        knuckles = [
            np.asarray(user.joint_center(f"{finger}_mcp_{suffix}"))
            for finger in ("index", "middle", "ring", "little")
        ]
        handle_axis_ground = rotation @ axis_direction
        knuckle_axis = get_ls_line(*knuckles)[3:]
        residuals.extend(
            math.sqrt(knuckle_weight) * np.cross(knuckle_axis, handle_axis_ground)
        )

        # Penalizzo soltanto le proiezioni delle nocche oltre gli estremi.
        knuckle_end_gaps = _knuckle_end_gaps(np.asarray(knuckles), handle_frame)
        residuals.extend(math.sqrt(knuckle_end_weight) * np.minimum(knuckle_end_gaps, 0.0))

        # Il piano contiene spalla e palmo ed e' parallelo all'asse; la
        # distanza del gomito da quel piano misura il criterio richiesto.
        plane_normal = np.cross(handle_axis_ground, shoulder - palm_marker_pos)
        plane_norm = np.linalg.norm(plane_normal)
        if plane_norm > 1e-12:
            plane_distance = np.dot(plane_normal, elbow - palm_marker_pos) / plane_norm
        else:
            # Se spalla e palmo sono sull'asse, esiste sempre un piano che
            # contiene anche il gomito.
            plane_distance = 0.0
        residuals.append(math.sqrt(plane_distance_weight) * plane_distance)

        end_distance = max(min_axial - axial_position, axial_position - max_axial, 0.0)
        residuals.append(math.sqrt(handle_end_weight) * end_distance)

        # Oriento il lato palmare verso l'asse, evitando la soluzione con il
        # dorso rivolto all'impugnatura pur avendo le nocche parallele.
        longitudinal = np.mean(knuckles, axis=0) - wrist
        across = knuckles[0] - knuckles[-1]
        palmar = np.cross(longitudinal, across)
        if side == "right":
            palmar = -palmar
        palmar /= np.linalg.norm(palmar)
        radial_ground = rotation @ radial
        radial_length = np.linalg.norm(radial_ground)
        radial_direction = radial_ground / max(radial_length, 1e-12)
        residuals.extend(math.sqrt(palm_facing_weight) * (palmar + radial_direction))

        return np.asarray(residuals)

    x0 = np.array(
        [
            coordinates.get(name).getValue(user.state)
            for name in arm_names + shoulder_names
        ]
    )
    bounds = (
        [coordinates.get(name).getRangeMin() for name in arm_names + shoulder_names],
        [coordinates.get(name).getRangeMax() for name in arm_names + shoulder_names],
    )

    # Risolvo i residui pesati, incluse le penalita' per le nocche fuori sede.
    result = least_squares(
        _objective,
        x0,
        bounds=bounds,
        max_nfev=1000,
        diff_step=1e-3,
    )

    if result.success:
        _apply_coordinates(result.x)
        if not _constraints_satisfied():
            _apply_coordinates(x0)
            warnings.warn(
                "No valid arm fit satisfying the palm and knuckle handle constraints",
                RuntimeWarning,
            )
            return None, None

    if result.success and _constraints_satisfied():
        # L'utente gestisce la correzione e la segnalazione del contatto
        # scapolare; se la cintura cambia, riadatto soltanto braccio e polso.
        if user.resolve_scapula_intersection(suffix, x0[len(arm_names):]):
            shoulder_values = np.array(
                [coordinates.get(name).getValue(user.state) for name in shoulder_names]
            )
            arm_result = least_squares(
                lambda values: _objective(np.concatenate((values, shoulder_values))),
                result.x[:len(arm_names)],
                bounds=(bounds[0][:len(arm_names)], bounds[1][:len(arm_names)]),
                max_nfev=1000,
                diff_step=1e-3,
            )
            if arm_result.success:
                _apply_coordinates(np.concatenate((arm_result.x, shoulder_values)))
            else:
                _apply_coordinates(np.concatenate((result.x[:len(arm_names)], shoulder_values)))
                warnings.warn("Arm refit after scapula correction did not converge", RuntimeWarning)
        if not _constraints_satisfied():
            _apply_coordinates(x0)
            warnings.warn(
                "Arm fit violates the palm or knuckle handle constraints",
                RuntimeWarning,
            )
            return None, None
        # Restituisco il gomito e la posizione del palmo rispetto al marker
        # centrale, mantenendo gli assi locali dell'impugnatura.

        elbow_flex_deg = math.degrees(
            coordinates.get(f"elbow_flex_{suffix}").getValue(user.state)
        )

        equipment.realizePosition(equipment.state)
        palm_marker_pos = marker_position_array(user_marker, user.state)
        handle_frame = body_transform_array(handle, equipment.state)
        center = marker_position_array(handle_marker, equipment.state)
        palm_in_handle = handle_frame[:3, :3].T @ (palm_marker_pos - center)

        return elbow_flex_deg, palm_in_handle

    for coord_name, value in zip(arm_names + shoulder_names, x0):
        coordinates.get(coord_name).setValue(user.state, float(value), False)
    user.realizePosition(user.state)
    return None, None
