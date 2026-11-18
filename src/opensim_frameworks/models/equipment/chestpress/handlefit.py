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
    "palm_axis_distance_penalty",
    "wrist_deviation_penalty",
    "forearm_supination_penalty",
    "external_shoulder_rotation_penalty",
    "scapular_retraction_penalty",
    "scapular_depression_penalty",
    "knuckle_alignment_penalty",
    "index_knuckle_endpoint_penalty",
    "little_knuckle_endpoint_penalty",
    "shoulder_palm_plane_penalty",
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
_PALM_POSITION_TOLERANCE_M = 0.001
_P = ParamSpec("_P")
_FIT_LOG_LOCK = threading.RLock()
_FitMessage = dict[str, str | int | None]


class _PalmPositionFitError(RuntimeError):
    """Raised when the fitted palm marker cannot reach its handle point."""


def palm_axis_distance_penalty(
    palm_ground: npt.ArrayLike,
    handle_frame: npt.NDArray[np.float64],
    axis_origin: npt.ArrayLike,
    axis_direction: npt.ArrayLike,
    radius_m: float,
    weight: float,
) -> float:
    """Return the weighted residual for palm distance from the handle axis.

    Parameters
    ----------
    palm_ground : array-like
        Palm marker in Ground coordinates, in metres.
    handle_frame : numpy.ndarray
        Handle body transform in Ground.
    axis_origin, axis_direction : array-like
        Handle-axis origin and unit direction in handle-local coordinates.
    radius_m : float
        Enclosing-cylinder radius, in metres.
    weight : float
        Nonnegative loss weight.

    Returns
    -------
    float
        Normalized weighted difference between palm-axis distance and radius.
    """
    palm_local = handle_frame[:3, :3].T @ (
        np.asarray(palm_ground) - handle_frame[:3, 3]
    )
    delta = palm_local - np.asarray(axis_origin)
    direction = np.asarray(axis_direction)
    radial = delta - np.dot(delta, direction) * direction
    return weight * (float(np.linalg.norm(radial)) - radius_m)


def wrist_deviation_penalty(wrist_deviation_rad: float, weight: float) -> float:
    """Return the weighted residual for neutral wrist radial/ulnar deviation."""
    return weight * wrist_deviation_rad


def forearm_supination_penalty(pro_supination_rad: float, weight: float) -> float:
    """Return the weighted residual for forearm supination near 90 degrees."""
    return weight * abs(pro_supination_rad - math.radians(90.0))


def external_shoulder_rotation_penalty(
    shoulder_rotation_rad: float, weight: float
) -> float:
    """Return the one-sided weighted residual for external shoulder rotation."""
    return weight * abs(shoulder_rotation_rad - math.radians(-90.0))


def scapular_retraction_penalty(protraction_rad: float, weight: float) -> float:
    """Return the weighted residual for scapular retraction."""
    return weight * protraction_rad


def scapular_depression_penalty(elevation_rad: float, weight: float) -> float:
    """Return the one-sided weighted residual for positive scapular elevation."""
    return weight * elevation_rad


def knuckle_alignment_penalty(
    index_knuckle: npt.ArrayLike,
    little_knuckle: npt.ArrayLike,
    handle_axis_ground: npt.ArrayLike,
    weight: float,
) -> npt.NDArray[np.float64]:
    """Align the little-to-index knuckle vector with endpoint 1-to-2.

    Parameters
    ----------
    index_knuckle, little_knuckle : array-like
        Index and little MCP centres in Ground coordinates.
    handle_axis_ground : array-like
        Unit vector from handle endpoint 1 to endpoint 2 in Ground coordinates.
    weight : float
        Nonnegative loss weight.

    Returns
    -------
    numpy.ndarray
        Three perpendicular-alignment residuals and one same-direction
        residual.
    """
    direction = np.asarray(index_knuckle, dtype=float) - np.asarray(
        little_knuckle, dtype=float
    )
    length = float(np.linalg.norm(direction))
    if length > 1e-12:
        direction /= length
    else:
        direction = np.zeros(3)
    handle_axis = np.asarray(handle_axis_ground, dtype=float)
    return weight * np.concatenate(
        (
            np.cross(direction, handle_axis),
            np.array([1.0 - float(np.dot(direction, handle_axis))]),
        )
    )


def index_knuckle_endpoint_penalty(
    projection_distance_m: float, endpoint_limit_m: float, weight: float
) -> float:
    """Return the one-sided weighted index MCP endpoint residual."""
    return weight * max(projection_distance_m - endpoint_limit_m, 0.0)


def little_knuckle_endpoint_penalty(
    projection_distance_m: float, endpoint_limit_m: float, weight: float
) -> float:
    """Return the one-sided weighted little MCP endpoint residual."""
    return weight * max(projection_distance_m - endpoint_limit_m, 0.0)


def shoulder_palm_plane_penalty(
    shoulder: npt.ArrayLike,
    elbow: npt.ArrayLike,
    palm: npt.ArrayLike,
    handle_axis_ground: npt.ArrayLike,
    weight: float,
) -> float:
    """Return the weighted sine of the handle axis's angle from the plane.

    The plane is defined by the shoulder, elbow, and palm marker. A zero
    residual means the handle axis lies in that plane.

    Parameters
    ----------
    shoulder, elbow, palm : array-like
        Points in Ground coordinates, in metres.
    handle_axis_ground : array-like
        Unit vector from handle endpoint 1 to endpoint 2 in Ground coordinates.
    weight : float
        Nonnegative loss weight.

    Returns
    -------
    float
        Signed sine residual normalized by the sine of the one-degree
        tolerance.
    """
    shoulder_point = np.asarray(shoulder, dtype=float)
    elbow_point = np.asarray(elbow, dtype=float)
    palm_point = np.asarray(palm, dtype=float)
    axis = np.asarray(handle_axis_ground, dtype=float)
    plane_normal = np.cross(elbow_point - shoulder_point, palm_point - shoulder_point)
    normal_length = float(np.linalg.norm(plane_normal))
    if normal_length > 1e-12:
        deviation = float(np.dot(plane_normal, axis) / normal_length)
    else:
        deviation = 0.0
    return weight * deviation


def lock_hands_to_handles(
    framework: Any,
    *,
    mirror_left_from_right: bool = False,
    fit_log: list[dict[str, Any]] | None = None,
) -> None:
    """Keep palm markers at their fitted points on the handles best-effort.

    The first call records each palm marker in its handle frame. Subsequent
    calls fit that anchor without changing finger articulation. When
    ``mirror_left_from_right`` is true, only the right arm is fitted and then
    mirrored to the left. Any residual is logged without stopping simulation.
    The 1 mm tolerance controls the optimizer's early exit and diagnostic level;
    it is not a hard acceptance threshold.

    Parameters
    ----------
    framework : opensim_frameworks.Framework
        Framework containing ``user`` and ``equipment`` member models.
    mirror_left_from_right : bool, default=False
        Fit the right hand and mirror its arm pose to the left instead of
        fitting each side independently.
    fit_log : list of dict, optional
        Log to which anchor residuals and mirrored-arm warnings are appended.
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

    for side in ("right", "left"):
        handle = equipment.getBodySet().get(f"{side}_handle")
        handle_location = target_locations.get(side)
        if handle_location is None:
            if mirror_left_from_right and side == "left":
                continue
            handle_transform = body_transform_array(handle, equipment.state)
            rotation = handle_transform[:3, :3]
            origin = handle_transform[:3, 3]
            marker_name = "RPALM" if side == "right" else "LPALM"
            markers = user.getMarkerSet()
            if markers.getIndex(marker_name) < 0:
                raise ValueError(f"Marker {marker_name} not found in user model")
            palm_position = marker_position_array(markers.get(marker_name), user.state)
            handle_location = {
                "palm": tuple(
                    float(value) for value in rotation.T @ (palm_position - origin)
                ),
            }
            target_locations[side] = handle_location

        if mirror_left_from_right and side == "left":
            continue

        error, converged, message = _fit_palm_marker_to_handle(
            user,
            equipment,
            handle,
            side,
            handle_location["palm"],
        )
        _report_palm_anchor_residual(side, error, converged, message, fit_log)

    if mirror_left_from_right:
        if fit_log:
            mirror_arm_with_log(user, fit_log)
        else:
            mirror_arm(user, "right")

        handle = equipment.getBodySet().get("left_handle")
        handle_location = target_locations.get("left")
        if handle_location is None:
            handle_transform = body_transform_array(handle, equipment.state)
            rotation = handle_transform[:3, :3]
            origin = handle_transform[:3, 3]
            markers = user.getMarkerSet()
            if markers.getIndex("LPALM") < 0:
                raise ValueError("Marker LPALM not found in user model")
            marker = markers.get("LPALM")
            handle_location = {
                "palm": tuple(
                    float(value)
                    for value in rotation.T
                    @ (marker_position_array(marker, user.state) - origin)
                ),
            }
            target_locations["left"] = handle_location
        error, converged, message = _fit_palm_marker_to_handle(
            user,
            equipment,
            handle,
            "left",
            handle_location["palm"],
            coordinate_prefixes=_MIRRORED_CONTACT_COORDINATE_PREFIXES,
        )
        _report_palm_anchor_residual("left", error, converged, message, fit_log)


def _report_palm_anchor_residual(
    side: str,
    error: float,
    converged: bool,
    optimizer_message: str,
    fit_log: list[dict[str, Any]] | None,
) -> None:
    marker_name = "RPALM" if side == "right" else "LPALM"
    level = "INFO" if converged and error <= _PALM_POSITION_TOLERANCE_M else "WARNING"
    message = (
        f"{side} {marker_name} handle-anchor residual={error * 1000:.3f} mm; "
        f"optimizer_converged={converged}; {optimizer_message}"
    )
    entry = {
        "kind": "contact_fit",
        "level": level,
        "category": None,
        "message": message,
    }
    if fit_log:
        fit_log[-1].setdefault("messages", []).append(entry)
    elif level == "WARNING":
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def _handle_cylinder_geometry(equipment: Any, handle_name: str):
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
    frame = body_transform_array(handle, equipment.state)
    directories = list(equipment.geometry_directories)
    if equipment.mesh_path is not None:
        directories.append(equipment.mesh_path)
    vertices = get_mesh_vertices_in_ground(
        equipment, equipment.state, handle_name, directories
    )
    vertices_local = (vertices - frame[:3, 3]) @ frame[:3, :3]
    _, radius, first, second = enclosing_cylinder(vertices_local)
    endpoints_ground = np.asarray([first, second]) @ frame[:3, :3].T + frame[:3, 3]
    # Endpoint 1 is the more lateral (lower-Z) end, defining the directed axis.
    if endpoints_ground[0, 2] > endpoints_ground[1, 2]:
        first, second = second, first
    axis = second - first
    axis_length = float(np.linalg.norm(axis))
    if axis_length <= 1e-12:
        raise ValueError(f"Handle {handle_name} has a degenerate cylinder axis")
    direction = axis / axis_length
    origin = 0.5 * (first + second)
    return handle, handle_marker, frame, origin, direction, radius, first, second


def _fit_palm_marker_to_handle(
    user: Any,
    equipment: Any,
    handle: Any,
    side: str,
    handle_location: tuple[float, float, float],
    coordinate_prefixes: tuple[str, ...] = _FIT_COORDINATE_PREFIXES,
) -> tuple[float, bool, str]:
    suffix = "r" if side == "right" else "l"
    marker_name = "RPALM" if side == "right" else "LPALM"
    markers = user.getMarkerSet()
    if markers.getIndex(marker_name) < 0:
        raise ValueError(f"Marker {marker_name} not found in user model")
    marker = markers.get(marker_name)
    handle_transform = body_transform_array(handle, equipment.state)
    target_position = (
        handle_transform[:3, :3] @ np.asarray(handle_location) + handle_transform[:3, 3]
    )
    position_error = float(
        np.linalg.norm(marker_position_array(marker, user.state) - target_position)
    )
    if position_error <= _PALM_POSITION_TOLERANCE_M:
        return position_error, True, "already within tolerance"

    coordinates = user.getCoordinateSet()
    coordinate_names = {
        coordinates.get(index).getName() for index in range(coordinates.getSize())
    }
    arm_names = sorted(
        name
        for name in coordinate_names
        if name.endswith(f"_{suffix}") and name.startswith(coordinate_prefixes)
    )
    girdle_names = tuple(f"{name}_{suffix}" for name in SHOULDER_GIRDLE_COORDINATES)
    has_girdle_coordinates = all(name in coordinate_names for name in girdle_names)
    variable_names = [
        name
        for name in sorted(
            arm_names + (list(girdle_names) if has_girdle_coordinates else [])
        )
        if not coordinates.get(name).getLocked(user.state)
        and coordinates.get(name).getRangeMin() < coordinates.get(name).getRangeMax()
    ]
    if not variable_names:
        return (
            position_error,
            False,
            f"no movable {side} arm or shoulder coordinates",
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
            if not has_girdle_coordinates or name not in girdle_names:
                coordinates.get(name).setValue(user.state, float(value), False)
        if has_girdle_coordinates:
            user.set_scapular_girdle_values(
                suffix,
                [coordinates.get(name).getValue(user.state) for name in girdle_names],
                check_contact=False,
            )
        else:
            user.realizePosition(user.state)

    def residual(values: npt.ArrayLike) -> np.ndarray:
        apply_values(values)
        return (
            marker_position_array(marker, user.state) - target_position
        ) / _PALM_POSITION_TOLERANCE_M

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
    return position_error, bool(result.success), str(result.message)


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
                while (
                    frame is not None and frame.f_globals.get("__name__") == "warnings"
                ):
                    frame = frame.f_back
                source_file = (
                    frame.f_code.co_filename if frame is not None else filename
                )
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
        f"Handle fit failed during the {phase} phase at " f"{angle_deg:.2f} degrees"
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
    The mirrored scapular pose is checked for mesh intersection; any issue is
    reported without changing the mirrored pose.
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
    palm_contact_weight: float = 10.0,
    knuckle_weight: float = 5.0,
    knuckle_end_weight: float = 10.0,
    plane_distance_weight: float = 1.0,
    pro_sup_weight: float = 1.0,
    wrist_dev_weight: float = 0.2,
    protraction_weight: float = 1.0,
    scapular_elevation_weight: float = 1.0,
    arm_weight: float = 1.0,
    fit_shoulder_girdle: bool = True,
) -> tuple[float | None, np.ndarray | None]:
    """Fit a user's arm and wrist against a handle.

    The user and equipment must share a Ground reference. The fitted side's
    arm, wrist and optionally two shoulder coordinates are optimized; finger
    coordinates are left unchanged. User derives the scapular coordinates from
    the shoulder angles. The handle meshes are approximated by an enclosing
    cylinder.
    Retraction, low shoulder elevation, palm-to-axis distance, arm-plane
    alignment and knuckle-line alignment are weighted objectives. Forearm
    rotation is encouraged toward 90 degrees and external upper-arm rotation
    is penalized. Wrist flexion or extension has no preferred target. The
    palm-to-axis distance target is the enclosing-cylinder radius. The
    little-to-index MCP vector is aligned in the same direction as endpoint 1
    (lower Ground Z) to endpoint 2 (higher Ground Z). The index MCP
    projection is limited by the distance from the handle centre to the second
    endpoint minus half the MCP span; the little MCP projection is limited by
    the distance from the centre to the first endpoint. Palm orientation is
    reported, not optimized; palm height relative to the axis has no preferred
    target. Wrist radial/ulnar deviation has a separate zero-deviation
    objective. Scapular proximity is optimized at three
    costal-surface points against the rib mesh, and a soft penalty uses
    collision risk interpolated from the shoulder map. The post-fit contact
    report is diagnostic and does not reject the fitted pose.

    The weighted residuals are normalized by millimetre- and degree-scale
    tolerances, so weights close to one can be used without letting units
    distort the optimization. No individual residual threshold rejects a
    converged solution. The log reports every weighted normalized residual
    and its squared loss contribution. If the optimizer does not converge,
    its best-attempt residuals and convergence details are logged before the
    initial pose is restored.
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
    palm_contact_weight : float, default=10.0
        Weight for matching the palm-marker-to-axis distance to the cylinder
        radius.
    knuckle_weight : float, default=5.0
        Weight for aligning the directed vector from little to index MCP with
        the directed handle axis.
    knuckle_end_weight : float, default=10.0
        Weight for keeping the index and little MCP projections within their
        specified endpoint limits.
    plane_distance_weight : float, default=1.0
        Weight for keeping the handle axis in the plane defined by the
        shoulder, elbow and palm marker.
    pro_sup_weight : float, default=1.0
        Weight penalizing forearm rotation away from 90 degrees.
    wrist_dev_weight : float, default=0.2
        Weight for neutral wrist radial/ulnar deviation.
    protraction_weight : float, default=1.0
        Weight penalizing protraction relative to the coordinate's range
        minimum, favoring a retracted shoulder.
    scapular_elevation_weight : float, default=1.0
        Weight penalizing positive scapular elevation.
    arm_weight : float, default=1.0
        Weight penalizing negative shoulder axial rotation, corresponding to
        external rotation in the model's anatomical reference pose.
    fit_shoulder_girdle : bool, default=True
        Whether to include the shoulder-girdle coordinates in the fit.
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

    (
        handle,
        handle_marker,
        handle_frame,
        axis_origin,
        axis_direction,
        cylinder_radius,
        end_min,
        end_max,
    ) = _handle_cylinder_geometry(equipment, handle_name)
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

    fit_shoulder = fit_shoulder_girdle

    def _apply_coordinates(coord_values: npt.ArrayLike) -> None:
        values = np.asarray(coord_values, dtype=float)
        for coord_name, value in zip(arm_names, values[: len(arm_names)]):
            coordinates.get(coord_name).setValue(user.state, float(value), False)
        shoulder_values = (
            values[len(arm_names) :]
            if fit_shoulder
            else [coordinates.get(name).getValue(user.state) for name in shoulder_names]
        )
        user.set_shoulder_girdle(suffix, *shoulder_values, check_contact=False)

    def _knuckle_endpoint_distances(
        knuckles_ground: np.ndarray, frame: np.ndarray
    ) -> tuple[float, float, float, float]:
        knuckles_local = (knuckles_ground - frame[:3, 3]) @ frame[:3, :3]
        center = 0.5 * (end_min + end_max)
        projections = (knuckles_local - center) @ axis_direction
        knuckle_span = float(np.linalg.norm(knuckles_ground[0] - knuckles_ground[-1]))
        half_length = 0.5 * float(np.linalg.norm(end_max - end_min))
        return (
            float(abs(projections[0])),
            max(half_length - 0.5 * knuckle_span, 0.0),
            float(abs(projections[-1])),
            half_length,
        )

    def _objective(coord_values: npt.ArrayLike) -> np.ndarray:
        """Evaluate weighted residuals for the arm fit."""
        # Aggiorno la postura prima di calcolare le penalità geometriche.
        _apply_coordinates(coord_values)
        equipment.realizePosition(equipment.state)

        palm_marker_pos = marker_position_array(user_marker, user.state)
        handle_frame = body_transform_array(handle, equipment.state)
        rotation = handle_frame[:3, :3]
        residuals = [
            palm_axis_distance_penalty(
                palm_marker_pos,
                handle_frame,
                axis_origin,
                axis_direction,
                cylinder_radius,
                palm_contact_weight,
            ),
        ]
        wrist_dev = coordinates.get(f"wrist_dev_{suffix}").getValue(user.state)
        residuals.append(wrist_deviation_penalty(wrist_dev, wrist_dev_weight))

        pro_sup = coordinates.get(f"pro_sup_{suffix}").getValue(user.state)
        residuals.append(forearm_supination_penalty(pro_sup, pro_sup_weight))

        arm_rot = coordinates.get(f"arm_rot_{suffix}").getValue(user.state)
        residuals.append(external_shoulder_rotation_penalty(arm_rot, arm_weight))

        protraction = coordinates.get(f"sc_protraction_{suffix}")
        elevation = coordinates.get(f"sc_elevation_{suffix}")
        residuals.extend(
            [
                scapular_retraction_penalty(
                    protraction.getValue(user.state),
                    protraction_weight,
                ),
                scapular_depression_penalty(
                    elevation.getValue(user.state),
                    scapular_elevation_weight,
                ),
            ]
        )
        scapular_residuals = user.scapular_fit_residuals(suffix, 0.0, 0.0)
        residuals.extend(scapular_residuals[2:])

        shoulder = np.asarray(user.joint_center(f"acromial_{suffix}"))
        elbow = np.asarray(user.joint_center(f"elbow_{suffix}"))
        knuckles = [
            np.asarray(user.joint_center(f"{finger}_mcp_{suffix}"))
            for finger in ("index", "middle", "ring", "little")
        ]
        handle_axis_ground = rotation @ axis_direction
        residuals.extend(
            knuckle_alignment_penalty(
                knuckles[0],
                knuckles[-1],
                handle_axis_ground,
                knuckle_weight,
            )
        )

        index_distance, index_limit, little_distance, little_limit = (
            _knuckle_endpoint_distances(np.asarray(knuckles), handle_frame)
        )
        residuals.extend(
            [
                index_knuckle_endpoint_penalty(
                    index_distance, index_limit, knuckle_end_weight
                ),
                little_knuckle_endpoint_penalty(
                    little_distance, little_limit, knuckle_end_weight
                ),
            ]
        )

        residuals.append(
            shoulder_palm_plane_penalty(
                shoulder,
                elbow,
                palm_marker_pos,
                handle_axis_ground,
                plane_distance_weight,
            )
        )

        return np.asarray(residuals)

    initial_girdle = np.array(
        [
            coordinates.get(f"{name}_{suffix}").getValue(user.state)
            for name in SHOULDER_GIRDLE_COORDINATES
        ]
    )
    fit_coordinate_names = arm_names + shoulder_names if fit_shoulder else arm_names
    x0 = np.array(
        [coordinates.get(name).getValue(user.state) for name in fit_coordinate_names]
    )
    initial_x0 = x0.copy()
    bounds = (
        [coordinates.get(name).getRangeMin() for name in fit_coordinate_names],
        [coordinates.get(name).getRangeMax() for name in fit_coordinate_names],
    )

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
    loss_component_names = (
        "palm_axis_distance",
        "wrist_deviation",
        "forearm_supination",
        "external_shoulder_rotation",
        "scapular_protraction",
        "scapular_elevation",
        "scapular_contact_AI",
        "scapular_contact_TS",
        "scapular_contact_AA",
        "scapular_collision_risk",
        "knuckle_alignment_x",
        "knuckle_alignment_y",
        "knuckle_alignment_z",
        "knuckle_alignment_direction",
        "index_endpoint_projection",
        "little_endpoint_projection",
        "shoulder_palm_plane",
    )

    def report_loss(values: npt.ArrayLike) -> None:
        residuals = _objective(values)
        contributions = "; ".join(
            f"{name}: residual={residual:.6g}, loss={residual**2:.6g}"
            for name, residual in zip(loss_component_names, residuals, strict=True)
        )
        warnings.warn(
            f"{side} handle fit loss (weighted normalized residuals): "
            f"{contributions}",
            RuntimeWarning,
            stacklevel=2,
        )

    if not result.success:
        for coord_name, value in zip(arm_names, initial_x0[: len(arm_names)]):
            coordinates.get(coord_name).setValue(user.state, float(value), False)
        user.set_scapular_girdle_values(suffix, initial_girdle, check_contact=False)
        user.realizePosition(user.state)
        warnings.warn(
            f"{side} handle fit did not converge: {result.message}; nfev={result.nfev}, "
            f"cost={result.cost:.6g}, optimality={result.optimality:.6g}",
            RuntimeWarning,
            stacklevel=2,
        )
    report_loss(result.x)
    user.scapula_contact_report(suffix)
    elbow_flex_deg = math.degrees(
        coordinates.get(f"elbow_flex_{suffix}").getValue(user.state)
    )

    equipment.realizePosition(equipment.state)
    palm_marker_pos = marker_position_array(user_marker, user.state)
    handle_frame = body_transform_array(handle, equipment.state)
    center = marker_position_array(handle_marker, equipment.state)
    palm_in_handle = handle_frame[:3, :3].T @ (palm_marker_pos - center)

    return elbow_flex_deg, palm_in_handle
