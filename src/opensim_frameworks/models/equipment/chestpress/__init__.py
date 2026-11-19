"""Base chest-press equipment model with user fitting and simulation."""

from .chestpress import ChestPress
from .handlefit import *
from .userpositioning import position_user

__all__ = [
    "ChestPress",
    "fit_handle",
    "handle_fit_failure_message",
    "lock_hands_to_handles",
    "mirror_arm",
    "mirror_arm_with_log",
    "mirror_finger_coordinates",
    "position_user",
    "palm_axis_distance_penalty",
    "wrist_deviation_penalty",
    "wrist_flexion_penalty",
    "forearm_supination_penalty",
    "external_shoulder_rotation_penalty",
    "scapular_retraction_penalty",
    "scapular_depression_penalty",
    "knuckle_alignment_penalty",
    "index_knuckle_endpoint_penalty",
    "little_knuckle_endpoint_penalty",
    "shoulder_palm_plane_penalty",
    "shoulder_wrist_axis_distance_penalty",
]
