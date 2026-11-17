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
from typing import Any, Literal, ParamSpec

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

__all__ = [
    "lock_hands_to_handles",
    "fit_handle",
    "handle_fit_failure_message",
    "mirror_arm",
    "mirror_arm_with_log",
]

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
_MIRRORED_CONTACT_COORDINATE_PREFIXES = (
    "elbow_flex_",
    "pro_sup_",
    "wrist_",
)
_WRIST_AXIS_DISTANCE_TOLERANCE_M = 0.001
_KNUCKLE_CENTER_DISTANCE_TOLERANCE_M = 0.001
_PALM_POSITION_TOLERANCE_M = 1e-5
_KNUCKLE_POSITION_TOLERANCE_M = 0.005
_PALM_HANDLE_SURFACE_TOLERANCE_M = 0.01
_PALM_ABOVE_AXIS_WEIGHT = 1e9
_PALM_ABOVE_AXIS_MIN_HEIGHT_M = 0.001
_KNUCKLE_POSITION_WEIGHT = 1e4
_P = ParamSpec("_P")
_FIT_LOG_LOCK = threading.RLock()
_FitMessage = dict[str, str | int | None]


class _HandleContactFitError(RuntimeError):
    """Raised when a fitted handle contact cannot be maintained."""


class _PalmPositionFitError(_HandleContactFitError):
    """Raised when the fitted palm marker cannot reach its handle point."""


class _ScapulaIntersectionFitError(_HandleContactFitError):
    """Raised when an exact scapula collision cannot be resolved."""


def lock_hands_to_handles(
    framework: Any,
    *,
    mirror_left_from_right: bool = False,
    fit_log: list[dict[str, Any]] | None = None,
) -> None:
    """Keep each palm marker and MCP knuckle centres at fitted handle points.

    The first call records the palm marker and four MCP joint centres in each
    handle frame. Subsequent calls fit those points, while leaving finger
    articulation unconstrained by this helper. When ``mirror_left_from_right``
    is true, only the right arm is fitted and then mirrored to the left.

    Parameters
    ----------
    framework : opensim_frameworks.Framework
        Framework containing ``user`` and ``equipment`` member models.
    mirror_left_from_right : bool, default=False
        Fit the right hand and mirror its arm pose to the left instead of
        fitting each side independently.
    fit_log : list of dict, optional
        Log to which mirrored-arm warnings are appended.

    Raises
    ------
    ValueError
        If a marker is missing or is not attached to its corresponding hand.
    RuntimeError
    If the hand points cannot be maintained or a scapula intersection
    cannot be resolved.
    """
    user = framework.user
    equipment = framework.equipment
    user.realizePosition(user.state)
    equipment.realizePosition(equipment.state)
    target_locations = getattr(framework, "_chestpress_palm_handle_targets", None)
    if target_locations is None:
        target_locations = {}
        object.__setattr__(
            framework, "_chestpress_palm_handle_targets", target_locations
        )

    for side, suffix in (("right", "r"), ("left", "l")):
        marker_name = "RPALM" if suffix == "r" else "LPALM"
        marker_set = user.getMarkerSet()
        if marker_set.getIndex(marker_name) < 0:
            raise ValueError(f"Marker {marker_name} not found in user model")
        marker = marker_set.get(marker_name)
        hand = user.getBodySet().get(f"hand_{suffix}")
        handle = equipment.getBodySet().get(f"{side}_handle")
        hand_path = hand.getAbsolutePathString()
        if (
            marker.getParentFrame().findBaseFrame().getAbsolutePathString()
            != hand_path
        ):
            raise ValueError(f"Marker {marker_name} is not attached to hand_{suffix}")
        _resolve_scapula_intersection(user, suffix, report_if_clear=False)
        handle_location = target_locations.get(side)
        if handle_location is None:
            handle_transform = body_transform_array(handle, equipment.state)
            rotation = handle_transform[:3, :3]
            origin = handle_transform[:3, 3]
            palm_position = marker_position_array(marker, user.state)
            knuckle_positions = np.asarray(
                [
                    user.joint_center(f"{finger}_mcp_{suffix}")
                    for finger in ("index", "middle", "ring", "little")
                ]
            )
            handle_location = {
                "palm": tuple(float(value) for value in rotation.T @ (palm_position - origin)),
                "knuckles": tuple(
                    tuple(float(value) for value in rotation.T @ (point - origin))
                    for point in knuckle_positions
                ),
            }
            target_locations[side] = handle_location

        if mirror_left_from_right and side == "left":
            continue

        for _ in range(3):
            _fit_palm_marker_to_handle(
                user,
                equipment,
                marker,
                handle,
                side,
                suffix,
                handle_location["palm"],
                handle_location["knuckles"],
            )
            if not _resolve_scapula_intersection(
                user, suffix, report_if_clear=False
            ):
                break
        else:
            handle_transform = body_transform_array(handle, equipment.state)
            palm_target = (
                handle_transform[:3, :3] @ np.asarray(handle_location["palm"])
                + handle_transform[:3, 3]
            )
            knuckle_targets = (
                handle_transform[:3, :3]
                @ np.asarray(handle_location["knuckles"]).T
            ).T + handle_transform[:3, 3]
            palm_error = np.linalg.norm(
                marker_position_array(marker, user.state) - palm_target
            )
            knuckle_errors = np.linalg.norm(
                np.asarray(
                    [
                        user.joint_center(f"{finger}_mcp_{suffix}")
                        for finger in ("index", "middle", "ring", "little")
                    ]
                )
                - knuckle_targets,
                axis=1,
            )
            if (
                palm_error > _PALM_POSITION_TOLERANCE_M
                or np.max(knuckle_errors) > _KNUCKLE_POSITION_TOLERANCE_M
            ):
                raise _PalmPositionFitError(
                    f"the {side} hand cannot retain its handle points while "
                    "resolving the scapula intersection: palm error "
                    f"{palm_error:.6g} m, max knuckle error "
                    f"{np.max(knuckle_errors):.6g} m"
                )

    if mirror_left_from_right:
        if fit_log:
            mirror_arm_with_log(user, fit_log)
        else:
            mirror_arm(user, "right")

        handle_location = target_locations["left"]
        marker = user.getMarkerSet().get("LPALM")
        handle = equipment.getBodySet().get("left_handle")
        _fit_palm_marker_to_handle(
            user,
            equipment,
            marker,
            handle,
            "left",
            "l",
            handle_location["palm"],
            handle_location["knuckles"],
            coordinate_prefixes=_MIRRORED_CONTACT_COORDINATE_PREFIXES,
        )
        handle_transform = body_transform_array(handle, equipment.state)
        palm_target = (
            handle_transform[:3, :3] @ np.asarray(handle_location["palm"])
            + handle_transform[:3, 3]
        )
        knuckle_targets = (
            handle_transform[:3, :3]
            @ np.asarray(handle_location["knuckles"]).T
        ).T + handle_transform[:3, 3]
        palm_error = np.linalg.norm(
            marker_position_array(marker, user.state) - palm_target
        )
        knuckle_errors = np.linalg.norm(
            np.asarray(
                [
                    user.joint_center(f"{finger}_mcp_l")
                    for finger in ("index", "middle", "ring", "little")
                ]
            )
            - knuckle_targets,
            axis=1,
        )
        if (
            palm_error > _PALM_POSITION_TOLERANCE_M
            or np.max(knuckle_errors) > _KNUCKLE_POSITION_TOLERANCE_M
        ):
            raise _PalmPositionFitError(
                "the mirrored left hand cannot retain its handle points: "
                f"palm error {palm_error:.6g} m, max knuckle error "
                f"{np.max(knuckle_errors):.6g} m"
            )


def _palm_height_above_handle_axis(
    palm_ground: np.ndarray,
    frame: np.ndarray,
    axis_origin: np.ndarray,
    axis_direction: np.ndarray,
) -> float:
    palm_local = frame[:3, :3].T @ (palm_ground - frame[:3, 3])
    delta = palm_local - axis_origin
    radial = delta - np.dot(delta, axis_direction) * axis_direction
    return float((frame[:3, :3] @ radial)[1])
def _fit_palm_marker_to_handle(
    user: Any,
    equipment: Any,
    marker: Any,
    handle: Any,
    side: str,
    suffix: str,
    handle_location: tuple[float, float, float],
    knuckle_locations: tuple[tuple[float, float, float], ...],
    coordinate_prefixes: tuple[str, ...] = _FIT_COORDINATE_PREFIXES,
) -> None:
    handle_transform = body_transform_array(handle, equipment.state)
    target_position = (
        handle_transform[:3, :3] @ np.asarray(handle_location)
        + handle_transform[:3, 3]
    )
    handle_rotation = handle_transform[:3, :3]
    target_knuckles = (
        handle_rotation @ np.asarray(knuckle_locations).T
    ).T + handle_transform[:3, 3]

    def current_knuckles() -> np.ndarray:
        return np.asarray(
            [
                user.joint_center(f"{finger}_mcp_{suffix}")
                for finger in ("index", "middle", "ring", "little")
            ]
        )

    if (
        np.linalg.norm(marker_position_array(marker, user.state) - target_position)
        <= _PALM_POSITION_TOLERANCE_M
        and np.max(np.linalg.norm(current_knuckles() - target_knuckles, axis=1))
        <= _KNUCKLE_POSITION_TOLERANCE_M
    ):
        return

    coordinates = user.getCoordinateSet()
    coordinate_names = {
        coordinates.get(index).getName()
        for index in range(coordinates.getSize())
    }
    arm_names = sorted(
        name
        for name in coordinate_names
        if name.endswith(f"_{suffix}")
        and name.startswith(coordinate_prefixes)
    )
    variable_names = [
        name
        for name in arm_names
        if not coordinates.get(name).getLocked(user.state)
        and coordinates.get(name).getRangeMin()
        < coordinates.get(name).getRangeMax()
    ]
    if not variable_names:
        raise _PalmPositionFitError(
            f"no movable {side} arm coordinates can satisfy the palm position"
        )

    initial_values = np.asarray(
        [coordinates.get(name).getValue(user.state) for name in variable_names],
        dtype=float,
    )
    lower_bounds = np.asarray(
        [coordinates.get(name).getRangeMin() for name in variable_names],
        dtype=float,
    )
    upper_bounds = np.asarray(
        [coordinates.get(name).getRangeMax() for name in variable_names],
        dtype=float,
    )

    def apply_values(values: npt.ArrayLike) -> None:
        for name, value in zip(variable_names, values, strict=True):
            coordinates.get(name).setValue(user.state, float(value), False)
        user.realizePosition(user.state)

    def residual(values: npt.ArrayLike) -> np.ndarray:
        apply_values(values)
        return np.concatenate(
            (
                1e5 * (marker_position_array(marker, user.state) - target_position),
                math.sqrt(_KNUCKLE_POSITION_WEIGHT)
                * (current_knuckles() - target_knuckles).ravel(),
            )
        )

    result = least_squares(
        residual,
        initial_values,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        max_nfev=1000,
        x_scale="jac",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    apply_values(result.x)
    position_error = float(
        np.linalg.norm(marker_position_array(marker, user.state) - target_position)
    )
    knuckle_errors = np.linalg.norm(current_knuckles() - target_knuckles, axis=1)
    if (
        position_error > _PALM_POSITION_TOLERANCE_M
        or np.max(knuckle_errors) > _KNUCKLE_POSITION_TOLERANCE_M
    ):
        raise _PalmPositionFitError(
            f"could not fit the {side} palm and MCP knuckles to their handle "
            f"positions: palm error {position_error:.6g} m, max knuckle error "
            f"{np.max(knuckle_errors):.6g} m; {result.message}"
        )


def _resolve_scapula_intersection(
    user: Any, side: str, *, report_if_clear: bool = True
) -> bool:
    user.realizePosition(user.state)
    if not user._scapula_mesh_contact(side).intersects():
        if report_if_clear:
            user.scapula_contact_report(side)
        return False

    reference_pose = user.shoulder_girdle_reference_pose()
    reference = tuple(
        math.radians(reference_pose[name]) for name in SHOULDER_GIRDLE_COORDINATES
    )
    if not user.resolve_scapula_intersection(side, reference):
        raise _ScapulaIntersectionFitError(
            f"the {side} scapula intersects the ribs and cannot be moved "
            "to a collision-free reference pose"
        )
    return True


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


def handle_fit_failure_message(
    phase: str, angle_deg: float, messages: list[dict[str, str | int | None]]
) -> str:
    prefix = "right handle fit did not converge: "
    diagnostic = next(
        (
            message["message"][len(prefix) :]
            for message in messages
            if isinstance(message.get("message"), str)
            and message["message"].startswith(prefix)
        ),
        None,
    )
    summary = (
        f"Handle fit failed during the {phase} phase at "
        f"{angle_deg:.2f} degrees"
    )
    if diagnostic is None:
        return f"{summary}; optimizer failure details are missing from the fit log"
    return f"{summary}; optimizer: {diagnostic}"


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
    The mirrored scapular pose is checked for mesh intersection and is moved
    toward the collision-free reference pose when necessary.
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
    _resolve_scapula_intersection(user, target_suffix[1:])


def mirror_arm_with_log(user: Any, fit_log: list[dict[str, Any]]) -> None:
    with _capture_messages() as messages:
        completed = False
        try:
            mirror_arm(user=user, to_mirror="right")
            completed = True
        finally:
            fit_log[-1]["messages"].extend(messages)
            error = sys.exc_info()[1] if not completed else None
            if error is not None:
                BaseException.__setattr__(error, "handle_fit_log", fit_log)


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
    Retraction, low shoulder elevation, wrist-to-axis distance, arm-plane
    alignment and knuckle-line alignment are weighted objectives. Forearm
    rotation is encouraged to remain between 75 and 115 degrees and external
    upper-arm rotation is penalized. The wrist-to-axis target is the enclosing
    cylinder radius plus half the palm-marker-to-wrist-center distance, and
    the palm marker is fitted to the cylinder surface. The desired pose has
    the palm above the handle axis. The index MCP center is
    targeted to a distance from the fitted handle center of half the endpoint
    distance minus half the index-to-little MCP distance. The little MCP may
    coincide with an endpoint. Palm orientation and deviation from the index
    target are reported, not imposed as hard constraints. Scapular proximity
    is optimized at three costal-surface points against the rib mesh. A soft
    strong soft penalty uses collision risk interpolated from the shoulder
    map, followed by an exact collision check and repair against the rib and
    spine meshes.

    The final weighted solution is retained even if it violates the wrist or
    knuckle criteria. A wrist-to-axis distance differing from its target by
    more than 1 mm is reported when the contact objective is enabled. A palm
    marker more than 10 mm from the fitted handle-cylinder surface is rejected.
    Palm orientation is reported without being optimized.
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
        Weight for matching the wrist-center-to-axis distance to the enclosing
        cylinder radius plus half the palm-marker-to-wrist-center distance.
    knuckle_weight : float, default=10000.0
        Weight for aligning the least-squares line through the four MCP
        joint centers with the handle axis.
    knuckle_end_weight : float, default=1e8
        Weight penalizing the index MCP center's deviation from its target
        distance to the fitted handle center: half the distance between
        cylinder endpoints minus half the Euclidean distance between the index
        and little MCP centers.
    plane_distance_weight : float, default=1e6
        Weight for the elbow's distance from the plane through the shoulder
        and wrist joint center that is parallel to the handle axis.
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

    def _wrist_axis_distance(
        palm_ground: np.ndarray,
        wrist_ground: np.ndarray,
        frame: np.ndarray,
    ) -> tuple[float, float]:
        wrist_local = frame[:3, :3].T @ (wrist_ground - frame[:3, 3])
        delta = wrist_local - axis_origin
        radial = delta - np.dot(delta, axis_direction) * axis_direction
        wrist_axis_distance = float(np.linalg.norm(radial))
        target_distance = cylinder_radius + 0.5 * float(
            np.linalg.norm(palm_ground - wrist_ground)
        )
        return wrist_axis_distance, target_distance

    def _palm_axis_distance(palm_ground: np.ndarray, frame: np.ndarray) -> float:
        palm_local = frame[:3, :3].T @ (palm_ground - frame[:3, 3])
        delta = palm_local - axis_origin
        radial = delta - np.dot(delta, axis_direction) * axis_direction
        return float(np.linalg.norm(radial))

    def _index_knuckle_center_distance(
        knuckles_ground: np.ndarray, frame: np.ndarray
    ) -> tuple[float, float]:
        knuckles_local = (knuckles_ground - frame[:3, 3]) @ frame[:3, :3]
        handle_center = 0.5 * (end_min + end_max)
        index_distance = float(np.linalg.norm(knuckles_local[0] - handle_center))
        target_distance = 0.5 * float(np.linalg.norm(end_max - end_min)) - 0.5 * float(
            np.linalg.norm(knuckles_ground[0] - knuckles_ground[-1])
        )
        return index_distance, target_distance

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
        # Aggiorno la postura e calcolo la distanza radiale del polso.
        _apply_coordinates(coord_values)
        equipment.realizePosition(equipment.state)

        palm_marker_pos = marker_position_array(user_marker, user.state)
        handle_frame = body_transform_array(handle, equipment.state)
        rotation = handle_frame[:3, :3]
        wrist = np.asarray(user.joint_center(f"radius_hand_{suffix}"))
        wrist_axis_distance, target_wrist_axis_distance = _wrist_axis_distance(
            palm_marker_pos, wrist, handle_frame
        )
        residuals = [
            math.sqrt(palm_contact_weight)
            * (wrist_axis_distance - target_wrist_axis_distance),
            math.sqrt(palm_contact_weight)
            * (_palm_axis_distance(palm_marker_pos, handle_frame) - cylinder_radius),
            math.sqrt(_PALM_ABOVE_AXIS_WEIGHT)
            * min(
                _palm_height_above_handle_axis(
                    palm_marker_pos, handle_frame, axis_origin, axis_direction
                )
                - _PALM_ABOVE_AXIS_MIN_HEIGHT_M,
                0.0,
            ),
        ]

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

        # Confronto la nocca indice con la distanza target dal centro impugnatura.
        index_center_distance, target_knuckle_center_distance = (
            _index_knuckle_center_distance(np.asarray(knuckles), handle_frame)
        )
        residuals.append(
            math.sqrt(knuckle_end_weight)
            * (index_center_distance - target_knuckle_center_distance)
        )

        # Il piano contiene spalla e centro del polso ed e' parallelo all'asse.
        plane_normal = np.cross(handle_axis_ground, shoulder - wrist)
        plane_norm = np.linalg.norm(plane_normal)
        if plane_norm > 1e-12:
            plane_distance = np.dot(plane_normal, elbow - wrist) / plane_norm
        else:
            # Se spalla e polso sono sull'asse, il gomito appartiene al piano.
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
        try:
            _resolve_scapula_intersection(user, suffix)
        except _ScapulaIntersectionFitError as error:
            for coord_name, value in zip(arm_names, x0[: len(arm_names)]):
                coordinates.get(coord_name).setValue(user.state, float(value), False)
            user.set_scapular_girdle_values(
                suffix, initial_girdle, check_contact=False
            )
            user.realizePosition(user.state)
            warnings.warn(
                f"{side} handle fit did not converge: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None, None
        # Segnalo i criteri geometrici non rispettati senza annullare la posa.
        violations = []
        frame = body_transform_array(handle, equipment.state)
        palm = marker_position_array(user_marker, user.state)
        palm_height = _palm_height_above_handle_axis(
            palm, frame, axis_origin, axis_direction
        )
        if palm_height <= _PALM_ABOVE_AXIS_MIN_HEIGHT_M:
            for coord_name, value in zip(arm_names, x0[: len(arm_names)]):
                coordinates.get(coord_name).setValue(user.state, float(value), False)
            user.set_scapular_girdle_values(
                suffix, initial_girdle, check_contact=False
            )
            user.realizePosition(user.state)
            warnings.warn(
                f"{side} handle fit did not converge: palm marker is "
                f"{palm_height * 1000:.3f} mm above the handle axis; "
                f"minimum is {_PALM_ABOVE_AXIS_MIN_HEIGHT_M * 1000:.3f} mm",
                RuntimeWarning,
                stacklevel=2,
            )
            return None, None
        wrist = np.asarray(user.joint_center(f"radius_hand_{suffix}"))
        wrist_axis_distance, target_wrist_axis_distance = _wrist_axis_distance(
            palm, wrist, frame
        )
        if (
            palm_contact_weight > 0
            and abs(wrist_axis_distance - target_wrist_axis_distance)
            > _WRIST_AXIS_DISTANCE_TOLERANCE_M
        ):
            violations.append(
                f"wrist-to-axis distance {wrist_axis_distance * 1000:.3f} mm "
                f"differs from target {target_wrist_axis_distance * 1000:.3f} mm"
            )
        palm_surface_error = abs(
            _palm_axis_distance(palm, frame) - cylinder_radius
        )
        if palm_contact_weight > 0 and palm_surface_error > _PALM_HANDLE_SURFACE_TOLERANCE_M:
            for coord_name, value in zip(arm_names, x0[: len(arm_names)]):
                coordinates.get(coord_name).setValue(user.state, float(value), False)
            user.set_scapular_girdle_values(
                suffix, initial_girdle, check_contact=False
            )
            user.realizePosition(user.state)
            warnings.warn(
                f"{side} handle fit did not converge: palm marker is "
                f"{palm_surface_error * 1000:.3f} mm from the handle surface",
                RuntimeWarning,
                stacklevel=2,
            )
            return None, None
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
        fingers = ("index", "middle", "ring", "little")
        knuckles = np.asarray(
            [user.joint_center(f"{finger}_mcp_{suffix}") for finger in fingers]
        )

        palmar = _palmar_direction(knuckles, wrist)
        if palmar[1] >= 0.0:
            violations.append("palm does not face downward")
        local = frame[:3, :3].T @ (palm - frame[:3, 3])
        index_center_distance, target_knuckle_center_distance = (
            _index_knuckle_center_distance(knuckles, frame)
        )
        if (
            abs(index_center_distance - target_knuckle_center_distance)
            > _KNUCKLE_CENTER_DISTANCE_TOLERANCE_M
        ):
            violations.append(
                f"index knuckle-to-handle-center distance "
                f"{index_center_distance * 1000:.3f} mm differs from target "
                f"{target_knuckle_center_distance * 1000:.3f} mm"
            )
        if violations:
            warnings.warn(
                f"{side} handle fit: {'; '.join(violations)}",
                RuntimeWarning,
                stacklevel=2,
            )
        # Restituisco la posizione del palmo negli assi locali dell'impugnatura.

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
