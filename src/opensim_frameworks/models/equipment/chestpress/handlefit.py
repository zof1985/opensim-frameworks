"""Handle geometry and grip measurements used by the arm fit."""

from __future__ import annotations

import functools
import inspect
import linecache
import logging
import math
import sys
import threading
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Literal, ParamSpec

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
_PALM_GAP_TOLERANCE_M = 0.001
_P = ParamSpec("_P")
_FIT_LOG_LOCK = threading.RLock()
_FitMessage = dict[str, str | int | None]


@contextmanager
def _capture_messages() -> Generator[list[_FitMessage]]:
    messages: list[_FitMessage] = []
    thread_id = threading.get_ident()
    with _FIT_LOG_LOCK:
        previous_factory = logging.getLogRecordFactory()
        previous_showwarning = warnings.showwarning

        def record_factory(*factory_args, **factory_kwargs):
            record = previous_factory(*factory_args, **factory_kwargs)
            if threading.get_ident() == thread_id:
                messages.append(
                    {
                        "kind": "log",
                        "level": record.levelname,
                        "category": None,
                        "message": record.getMessage(),
                        "filename": record.pathname,
                        "line": record.lineno,
                        "function": record.funcName,
                        "source": linecache.getline(
                            record.pathname, record.lineno
                        ).strip(),
                        "logger": record.name,
                    }
                )
            return record

        def showwarning(message, category, filename, lineno, file=None, line=None):
            if threading.get_ident() != thread_id:
                previous_showwarning(message, category, filename, lineno, file, line)
                return
            frame = inspect.currentframe()
            try:
                frame = frame.f_back if frame is not None else None
                while frame is not None and frame.f_globals.get("__name__") == "warnings":
                    frame = frame.f_back
                source_file = frame.f_code.co_filename if frame is not None else filename
                source_line = frame.f_lineno if frame is not None else lineno
                messages.append(
                    {
                        "kind": "warning",
                        "level": "WARNING",
                        "category": category.__name__,
                        "message": str(message),
                        "filename": source_file,
                        "line": source_line,
                        "function": frame.f_code.co_name if frame is not None else None,
                        "source": linecache.getline(source_file, source_line).strip(),
                        "logger": None,
                    }
                )
            finally:
                del frame

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = showwarning
            logging.setLogRecordFactory(record_factory)
            try:
                yield messages
            finally:
                logging.setLogRecordFactory(previous_factory)


def _capture_fit_messages(
    function: Callable[_P, tuple[float | None, np.ndarray | None]],
) -> Callable[_P, tuple[float | None, np.ndarray | None, list[_FitMessage]]]:
    @functools.wraps(function)
    def wrapped(
        *args: _P.args, **kwargs: _P.kwargs
    ) -> tuple[float | None, np.ndarray | None, list[_FitMessage]]:
        with _capture_messages() as messages:
            completed = False
            try:
                elbow, palm = function(*args, **kwargs)
                completed = True
            finally:
                error = sys.exc_info()[1] if not completed else None
                if error is not None:
                    BaseException.__setattr__(error, "handle_fit_log", messages)
        return elbow, palm, messages

    result_type = tuple[float | None, np.ndarray | None, list[_FitMessage]]
    wrapped.__annotations__ = {**wrapped.__annotations__, "return": result_type}
    setattr(
        wrapped,
        "__signature__",
        inspect.signature(function).replace(return_annotation=result_type),
    )
    return wrapped


def mirror_arm(user, to_mirror: Literal["left", "right"]) -> None:
    """Mirror the upper-limb and shoulder-girdle coordinates.

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

    Notes
    -----
    The model's corresponding right and left coordinates use the same sign.
    The mirrored scapular pose is checked for contact but is not refitted to
    the opposite side's shoulder map.
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

    target_names = [
        source_name[: -len(source_suffix)] + target_suffix
        for source_name in source_names
    ]
    for target_name in target_names:
        if target_name not in names:
            raise RuntimeError(f"Missing mirrored arm coordinate {target_name}")

    for source_name, target_name in zip(source_names, target_names):
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
    user.scapula_contact_report(target_suffix[1:])


@_capture_fit_messages
def fit_handle(
    user,
    equipment,
    handle_name: str,
    side: Literal["left", "right"],
    *,
    palm_contact_weight: float = 1e7,
    knuckle_weight: float = 10000.0,
    knuckle_end_weight: float = 1e8,
    plane_distance_weight: float = 1e6,
    pro_sup_weight: float = 100000.0,
    wrist_dev_weight: float = 1000.0,
    protraction_weight: float = 5.0,
    scapular_elevation_weight: float = 1000.0,
    wrist_flexion_weight: float = 10000.0,
    arm_weight: float = 100000.0,
) -> tuple[float | None, np.ndarray | None]:
    """Fit a user's arm and wrist against a handle.

    The user and equipment must share a Ground reference. The fitted side's
    arm, wrist and two shoulder coordinates are optimized; finger coordinates
    are left unchanged. User derives the scapular coordinates from the shoulder
    angles. The handle meshes are approximated by an enclosing cylinder.
    Retraction, low shoulder elevation, arm-plane alignment and knuckle-line
    alignment are weighted objectives. Forearm rotation
    is encouraged to remain between 75 and 115 degrees and external upper-arm
    rotation is penalized. The desired pose has the palm marker on the
    cylindrical surface, facing down, and each MCP joint center projecting strictly
    between its end disks. Scapular proximity is optimized
    at three costal-surface points against the rib mesh. A soft penalty uses
    collision risk interpolated from the shoulder map. Neither these unsigned
    distances nor the interpolated risk guarantee contact or exclude mesh
    intersections.

    The final weighted solution is retained even if it violates the palm or
    knuckle criteria. A palm gap differing from its target by more than 1 mm is
    reported when the contact objective is enabled. Palm orientation and
    position relative to the handle axis are reported without being optimized.
    Warnings and enabled
    Python logging records generated on the fitting thread are returned in the
    log. Warnings are not displayed; logging retains its existing handlers.
    Native OpenSim output is excluded.

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
    palm_contact_weight : float, default=1e7
        Weight for matching the palm-to-axis distance to the radius of the
        enclosing cylindrical approximation of the handle.
    knuckle_weight : float, default=10000.0
        Weight for aligning the least-squares line through the four MCP
        joint centers with the handle axis.
    knuckle_end_weight : float, default=1e8
        Weight penalizing MCP joint projections outside the handle ends.
        Projections at or beyond an end are reported by a warning.
    plane_distance_weight : float, default=1e6
        Weight for the elbow's distance from the plane through the shoulder
        and palm that is parallel to the handle axis.
    pro_sup_weight : float, default=100000.0
        Weight penalizing forearm rotation outside 75-115 degrees.
    wrist_dev_weight : float, default=1000.0
        Weight for neutral wrist radial/ulnar deviation.
    protraction_weight : float, default=5.0
        Weight penalizing protraction relative to the coordinate's range
        minimum, favoring a retracted shoulder.
    scapular_elevation_weight : float, default=1000.0
        Weight penalizing positive scapular elevation.
    wrist_flexion_weight : float, default=10000.0
        Weight strongly penalizing positive wrist flexion.
    arm_weight : float, default=100000.0
        Weight penalizing negative shoulder axial rotation, corresponding to
        external rotation in the model's anatomical reference pose.

    Returns
    -------
    tuple of (float or None, np.ndarray or None, list of dict)
        Fitted elbow flexion angle in degrees (or None if the optimizer did not
        converge), and palm position relative to the handle's center marker,
        expressed along its body's local axes (or None if the optimizer did not
        converge), followed by the messages in emission order. Each message
        contains ``kind``, ``level``, ``category``, ``message``, ``filename``,
        ``line``, ``function``, ``source`` and ``logger``.

    Raises
    ------
    ValueError
        If ``side`` is not ``"left"`` or ``"right"``, or if arm coordinate
        prefixes, handle marker, or valid handle mesh are missing, or if the
        knuckle-end weight is invalid.

    Notes
    -----
    Message capture temporarily changes Python's process-wide warning and
    logging hooks. Concurrent calls to this function are serialized, but other
    code changing those hooks concurrently is not supported. Exceptions are
    propagated with any messages collected so far in ``handle_fit_log``.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    if not math.isfinite(knuckle_end_weight) or knuckle_end_weight < 0:
        raise ValueError("knuckle_end_weight must be finite and nonnegative")

    # Seleziono braccio, polso e i due comandi della spalla; la scapola viene
    # posizionata da User e le dita si chiudono dopo il fit.
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

    shoulder_names = [f"{name}_{suffix}" for name in ("sc_protraction", "sc_elevation")]
    if not all(name + "_" + suffix in names for name in SHOULDER_GIRDLE_COORDINATES):
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
        user.set_shoulder_girdle(suffix, *values[len(arm_names) :], check_contact=False)

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

    def _palmar_direction(knuckles: np.ndarray, wrist: np.ndarray) -> np.ndarray:
        longitudinal = np.mean(knuckles, axis=0) - wrist
        across = knuckles[0] - knuckles[-1]
        palmar = np.cross(longitudinal, across)
        if side == "right":
            palmar = -palmar
        length = np.linalg.norm(palmar)
        if length <= 1e-12:
            raise ValueError(f"{side} palm landmarks do not define an orientation")
        return palmar / length

    def _objective(coord_values: npt.ArrayLike) -> np.ndarray:
        """Evaluate weighted residuals for the arm fit."""
        # Aggiorno la postura e calcolo la distanza radiale dal cilindro.
        _apply_coordinates(coord_values)
        equipment.realizePosition(equipment.state)

        palm_marker_pos = marker_position_array(user_marker, user.state)
        handle_frame = body_transform_array(handle, equipment.state)
        rotation = handle_frame[:3, :3]
        palm_in_body = rotation.T @ (palm_marker_pos - handle_frame[:3, 3])
        axial_position = np.dot(palm_in_body - axis_origin, axis_direction)
        radial = palm_in_body - axis_origin - axial_position * axis_direction
        surface_gap = np.linalg.norm(radial) - cylinder_radius
        residuals = [math.sqrt(palm_contact_weight) * surface_gap]

        # Regolarizzo polso, avambraccio e rotazione dell'omero. Nel modello
        # la flessione del polso e' positiva e l'estensione e' negativa.
        wrist_flex = coordinates.get(f"wrist_flex_{suffix}").getValue(user.state)
        residuals.append(
            math.sqrt(wrist_flexion_weight) * abs(wrist_flex - math.radians(-45.0))
        )

        wrist_dev = coordinates.get(f"wrist_dev_{suffix}").getValue(user.state)
        residuals.append(math.sqrt(wrist_dev_weight) * wrist_dev)

        pro_sup = coordinates.get(f"pro_sup_{suffix}").getValue(user.state)
        pro_sup_deviation = min(pro_sup - math.radians(75.0), 0.0) + max(
            pro_sup - math.radians(115.0), 0.0
        )
        residuals.append(math.sqrt(pro_sup_weight) * pro_sup_deviation)

        arm_rot = coordinates.get(f"arm_rot_{suffix}").getValue(user.state)
        residuals.append(math.sqrt(arm_weight) * min(arm_rot, 0.0))

        # L'utente valuta retrazione, elevazione e contatto scapolo-toracico.
        residuals.extend(
            user.scapular_fit_residuals(
                suffix, protraction_weight, scapular_elevation_weight
            )
        )

        # Ricavo la retta ai minimi quadrati delle nocche e la confronto con
        # l'asse dell'impugnatura, senza distinguere i due versi della retta.
        shoulder = np.asarray(user.joint_center(f"acromial_{suffix}"))
        elbow = np.asarray(user.joint_center(f"elbow_{suffix}"))
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
        residuals.extend(
            math.sqrt(knuckle_end_weight) * np.minimum(knuckle_end_gaps, 0.0)
        )

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

        return np.asarray(residuals)

    initial_girdle = np.array(
        [
            coordinates.get(f"{name}_{suffix}").getValue(user.state)
            for name in SHOULDER_GIRDLE_COORDINATES
        ]
    )
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

    # Risolvo i residui pesati con la scala del Jacobiano e passo numerico automatico.
    result = least_squares(
        _objective,
        x0,
        bounds=bounds,
        max_nfev=1000,
        x_scale="jac",
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
    )

    if result.success:
        _apply_coordinates(result.x)
        # Segnalo i problemi scapolari senza modificare la posa ottimizzata.
        user.scapula_contact_report(suffix)
        # Segnalo i criteri geometrici non rispettati senza annullare la posa.
        violations = []
        surface_gap = _surface_gap()
        if surface_gap < -1e-9:
            violations.append(f"palm inside cylinder by {-surface_gap * 1000:.3f} mm")
        elif palm_contact_weight > 0 and surface_gap > _PALM_GAP_TOLERANCE_M:
            violations.append(
                f"palm gap {surface_gap * 1000:.3f} mm differs from target 0.000 mm"
            )
        pro_sup_deg = math.degrees(
            coordinates.get(f"pro_sup_{suffix}").getValue(user.state)
        )
        if pro_sup_weight > 0 and not 75.0 - 1e-6 <= pro_sup_deg <= 115.0 + 1e-6:
            violations.append(
                f"forearm rotation {pro_sup_deg:.2f} deg outside 75-115 deg"
            )
        arm_rot_deg = math.degrees(
            coordinates.get(f"arm_rot_{suffix}").getValue(user.state)
        )
        if arm_weight > 0 and arm_rot_deg < -1e-6:
            violations.append(f"external upper-arm rotation {-arm_rot_deg:.2f} deg")
        frame = body_transform_array(handle, equipment.state)
        fingers = ("index", "middle", "ring", "little")
        knuckles = np.asarray(
            [user.joint_center(f"{finger}_mcp_{suffix}") for finger in fingers]
        )
        wrist = np.asarray(user.joint_center(f"radius_hand_{suffix}"))
        palmar = _palmar_direction(knuckles, wrist)
        if palmar[1] >= 0.0:
            violations.append("palm does not face downward")
        palm = marker_position_array(user_marker, user.state)
        local = frame[:3, :3].T @ (palm - frame[:3, 3])
        radial = local - axis_origin
        radial -= np.dot(radial, axis_direction) * axis_direction
        if (frame[:3, :3] @ radial)[1] <= 0.0:
            violations.append("palm is not above handle axis")
        gaps = _knuckle_end_gaps(knuckles, frame)
        nearest_ends = np.minimum(gaps[: len(fingers)], gaps[len(fingers) :])
        outside = [
            f"{finger} ({max(0.0, -gap) * 1000:.3f} mm)"
            for finger, gap in zip(fingers, nearest_ends)
            if gap <= 0.0
        ]
        if outside:
            violations.append(
                f"knuckles at or beyond handle ends: {', '.join(outside)}"
            )
        if violations:
            warnings.warn(
                f"{side} handle fit: {'; '.join(violations)}",
                RuntimeWarning,
                stacklevel=2,
            )
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

    for coord_name, value in zip(arm_names, x0[: len(arm_names)]):
        coordinates.get(coord_name).setValue(user.state, float(value), False)
    user.set_scapular_girdle_values(suffix, initial_girdle, check_contact=False)
    user.realizePosition(user.state)
    warnings.warn(
        f"{side} handle fit did not converge: {result.message}; nfev={result.nfev}, "
        f"cost={result.cost:.6g}, optimality={result.optimality:.6g}",
        RuntimeWarning,
        stacklevel=2,
    )
    return None, None
