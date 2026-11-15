"""Low-level image and animation rendering operations for the player."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

__all__ = ["save_image", "render_motion_animation"]


def save_image(visualizer: Any, path: str | Path, *, dpi: int = 300) -> None:
    """Save ``visualizer``'s current frame as a PNG."""
    from PIL import Image

    Image.fromarray(visualizer.capture_frame()).save(str(path), dpi=(dpi, dpi))


def render_motion_animation(
    model: Any,
    data: "MotionData",
    visualizer: Any,
    apply_time: Callable[[Any, "MotionData", float], None],
    path: str | Path,
    *,
    fps: float = 30.0,
) -> None:
    """Render a loaded player motion to MP4 at ``fps``.

    The caller is responsible for restoring the model's previous posture.
    """
    import imageio

    duration = data.end_time - data.start_time
    frame_count = max(1, round(duration * fps))
    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    try:
        for index in range(frame_count + 1):
            time = data.start_time + duration * index / frame_count
            apply_time(model, data, time)
            writer.append_data(visualizer.capture_frame())
    finally:
        writer.close()


def _set_pose(model: Any, data: Any, time: float) -> None:
    """Pose ``model`` at ``time`` of ``data`` without redrawing the scene."""
    native_model = getattr(model, "model", model)
    state = getattr(model, "state", None)
    if state is None:
        state = native_model.getWorkingState()
    coordinate_set = native_model.getCoordinateSet()
    for name, value in data.values_at(time).items():
        coordinate_name = (
            name.rsplit("/", 2)[-2] if name.endswith("/value") else name
        )
        if not coordinate_set.contains(coordinate_name):
            continue
        coordinate = coordinate_set.get(coordinate_name)
        was_locked = coordinate.getLocked(state)
        if was_locked:
            coordinate.setLocked(state, False)
        try:
            coordinate.setValue(state, value, False)
        finally:
            if was_locked:
                coordinate.setLocked(state, True)
    native_model.realizePosition(state)
