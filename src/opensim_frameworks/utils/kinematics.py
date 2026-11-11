"""Closed-form forward kinematics of a chain of OpenSim joints.

``Model.realizePosition`` evaluates the whole model and, through the Python
bindings, costs a few hundred microseconds per call whatever is needed. A
solver that moves a handful of coordinates of one limb and needs the pose of
one body can instead evaluate the joints of that limb alone, for many
coordinate vectors at once (for instance a finite-difference Jacobian in a
single call). :class:`KinematicChain` does that for the joint types of the
bundled models and refuses, explicitly, anything it cannot reproduce.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .opensim import import_opensim, rotation_to_numpy, vec3_to_numpy

__all__ = ["KinematicChain"]

_IDENTITY = np.eye(3)


def _axis_rotations(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Return the rotations about a fixed unit axis (Rodrigues formula).

    Parameters
    ----------
    axis : numpy.ndarray
        Unit vector, shape ``(3,)``.
    angles : numpy.ndarray
        Angles in radians, shape ``(N,)``.

    Returns
    -------
    numpy.ndarray
        Rotation matrices, shape ``(N, 3, 3)``.
    """
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    cosine = np.cos(angles)[:, None, None]
    sine = np.sin(angles)[:, None, None]
    return cosine * _IDENTITY + sine * skew + (1.0 - cosine) * np.outer(axis, axis)


class _Function:
    """Affine function ``scale * (slope * q + intercept)`` of one coordinate."""

    def __init__(self, coordinate: int | None, slope: float, intercept: float):
        self.coordinate = coordinate
        self.slope = slope
        self.intercept = intercept

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """Return the function for ``values`` of shape ``(N, n_coordinates)``."""
        if self.coordinate is None:
            return np.full(len(values), self.intercept)
        return self.slope * values[:, self.coordinate] + self.intercept


class _Joint:
    """Motion of one joint from the parent offset frame to the child one."""

    def __init__(
        self,
        kind: str,
        parent_rotation: np.ndarray,
        parent_position: np.ndarray,
        child_rotation: np.ndarray,
        child_position: np.ndarray,
        axes: list[tuple[np.ndarray, _Function]],
    ):
        self.kind = kind
        self.parent_rotation = parent_rotation
        self.parent_position = parent_position
        # inverse of the child offset frame: child body -> child frame
        self.child_rotation_t = child_rotation.T
        self.child_position = child_position
        self.axes = axes

    def body_in_parent(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the child body pose in the parent body, for each row.

        Parameters
        ----------
        values : numpy.ndarray
            Coordinate values, shape ``(N, n_coordinates)``.

        Returns
        -------
        rotation, position : numpy.ndarray
            Shapes ``(N, 3, 3)`` and ``(N, 3)``.
        """
        count = len(values)
        motion_rotation = np.broadcast_to(_IDENTITY, (count, 3, 3))
        motion_position = np.zeros((count, 3))
        if self.kind == "pin":
            axis, function = self.axes[0]
            motion_rotation = _axis_rotations(axis, function(values))
        elif self.kind == "universal":
            (first, f_first), (second, f_second) = self.axes
            motion_rotation = _axis_rotations(first, f_first(values)) @ _axis_rotations(
                second, f_second(values)
            )
        elif self.kind == "custom":
            for axis, function in self.axes[:3]:
                motion_rotation = motion_rotation @ _axis_rotations(
                    axis, function(values)
                )
            for axis, function in self.axes[3:]:
                motion_position = motion_position + function(values)[:, None] * axis
        # X_PB = X_P,Pf X_Pf,Cf(q) X_C,Cf^-1
        rotation = (self.parent_rotation @ motion_rotation) @ self.child_rotation_t
        child_origin = motion_position - np.einsum(
            "nij,j->ni", motion_rotation @ self.child_rotation_t, self.child_position
        )
        position = self.parent_position + child_origin @ self.parent_rotation.T
        return rotation, position


class KinematicChain:
    """Poses of the bodies of an OpenSim model along a chain of joints.

    The chain goes from a ``root`` body to an ``end`` body through the joints
    that connect them. Coordinate values are given in model units (radians
    or metres); all poses are relative to the root body, so nothing outside
    the chain (for instance the pose of the root in the ground) enters.

    Parameters
    ----------
    model : opensim.Model
        Model whose system is initialised (``model.state`` is not needed
        after construction).
    root : str
        Name of the body the poses refer to.
    end : str
        Name of the last body of the chain; ``root`` must be one of its
        ancestors.

    Attributes
    ----------
    coordinate_names : tuple of str
        Coordinates that move the chain, in the order of the columns of the
        ``values`` arrays (root to end, joint by joint).
    body_names : tuple of str
        Bodies of the chain, from ``root`` (first) to ``end`` (last).

    Raises
    ------
    ValueError
        If ``end`` is not a descendant of ``root``, or a joint or a joint
        function of the chain is not supported. Supported: ``WeldJoint``,
        ``PinJoint``, ``UniversalJoint`` and ``CustomJoint`` whose transform
        axes use constant, linear or multiplied linear/constant functions of
        at most one coordinate.

    Notes
    -----
    The result is meant to be compared once with
    ``Body.getTransformInGround`` (see :meth:`max_error`): the conventions
    for the custom joints (body-fixed rotations in order, translation in the
    parent offset frame) are those of the bundled models. Coupled
    coordinates (constraints) are not enforced: each coordinate is an
    independent input.
    """

    def __init__(self, model: Any, root: str, end: str) -> None:
        self._model = model
        bodies = model.getBodySet()
        for name in (root, end):
            if not bodies.contains(name):
                raise ValueError(f"The model has no body {name!r}")
        joints_by_child = {}
        joint_set = model.getJointSet()
        for index in range(joint_set.getSize()):
            joint = joint_set.get(index)
            joints_by_child[joint.getChildFrame().findBaseFrame().getName()] = joint

        # joints from the end body back to the root
        path = []
        name = end
        while name != root:
            joint = joints_by_child.get(name)
            if joint is None:
                raise ValueError(f"{end!r} is not a descendant of {root!r}")
            path.append(joint)
            name = joint.getParentFrame().findBaseFrame().getName()
        path.reverse()

        self.body_names = (root, *[j.getChildFrame().findBaseFrame().getName() for j in path])
        self.coordinate_names: tuple[str, ...] = ()
        names: list[str] = []
        self._joints = [self._read_joint(joint, names) for joint in path]
        self.coordinate_names = tuple(names)

    def _coordinate_index(self, names: list[str], coordinate: str) -> int:
        """Return the column of a coordinate, adding it if it is new."""
        if coordinate not in names:
            names.append(coordinate)
        return names.index(coordinate)

    def _function(self, function: Any, coordinates: list[str], names: list[str]):
        """Reduce an OpenSim function to an affine function of one coordinate."""
        osim = import_opensim()
        kind = function.getConcreteClassName()
        if kind == "Constant":
            return _Function(None, 0.0, osim.Constant.safeDownCast(function).getValue())
        if kind == "LinearFunction":
            linear = osim.LinearFunction.safeDownCast(function)
            if len(coordinates) != 1:
                raise ValueError("A linear function needs exactly one coordinate")
            index = self._coordinate_index(names, coordinates[0])
            return _Function(index, linear.getSlope(), linear.getIntercept())
        if kind == "MultiplierFunction":
            multiplier = osim.MultiplierFunction.safeDownCast(function)
            inner = self._function(multiplier.getFunction(), coordinates, names)
            scale = multiplier.getScale()
            return _Function(inner.coordinate, scale * inner.slope, scale * inner.intercept)
        raise ValueError(f"Unsupported joint function {kind!r}")

    def _read_joint(self, joint: Any, names: list[str]) -> _Joint:
        """Read the parameters of one joint of the chain."""
        parent = joint.getParentFrame().findTransformInBaseFrame()
        child = joint.getChildFrame().findTransformInBaseFrame()
        kind = joint.getConcreteClassName()
        coordinates = [
            joint.get_coordinates(i).getName() for i in range(joint.numCoordinates())
        ]
        axes: list[tuple[np.ndarray, _Function]] = []
        x_axis, y_axis, z_axis = np.eye(3)
        if kind == "WeldJoint":
            motion = "weld"
        elif kind == "PinJoint":
            motion = "pin"
            index = self._coordinate_index(names, coordinates[0])
            axes = [(z_axis, _Function(index, 1.0, 0.0))]
        elif kind == "UniversalJoint":
            motion = "universal"
            axes = [
                (x_axis, _Function(self._coordinate_index(names, coordinates[0]), 1.0, 0.0)),
                (y_axis, _Function(self._coordinate_index(names, coordinates[1]), 1.0, 0.0)),
            ]
        elif kind == "CustomJoint":
            motion = "custom"
            transform = import_opensim().CustomJoint.safeDownCast(joint).get_SpatialTransform()
            for index in range(6):
                axis = transform.getTransformAxis(index)
                used = axis.getCoordinateNames()
                axes.append(
                    (
                        vec3_to_numpy(axis.getAxis()),
                        self._function(
                            axis.getFunction(),
                            [used.getValue(i) for i in range(used.size())],
                            names,
                        ),
                    )
                )
        else:
            raise ValueError(f"Unsupported joint type {kind!r} ({joint.getName()!r})")
        return _Joint(
            motion,
            rotation_to_numpy(parent.R()),
            vec3_to_numpy(parent.p()),
            rotation_to_numpy(child.R()),
            vec3_to_numpy(child.p()),
            axes,
        )

    def poses(self, values: np.ndarray | Sequence[Sequence[float]]):
        """Return the pose of every body of the chain, relative to the root.

        Parameters
        ----------
        values : array_like
            Coordinate values, shape ``(N, len(coordinate_names))``.

        Returns
        -------
        rotations : numpy.ndarray
            Shape ``(N, len(body_names), 3, 3)``: ``rotations[n, b]`` maps
            vectors of body ``b`` to the root body frame.
        positions : numpy.ndarray
            Shape ``(N, len(body_names), 3)``: origins of the bodies in the
            root body frame.

        Raises
        ------
        ValueError
            If ``values`` does not have one column per coordinate.
        """
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.coordinate_names):
            raise ValueError(
                f"values must have shape (N, {len(self.coordinate_names)})"
            )
        count = len(values)
        rotation = np.broadcast_to(_IDENTITY, (count, 3, 3))
        position = np.zeros((count, 3))
        rotations = [rotation]
        positions = [position]
        for joint in self._joints:
            step_rotation, step_position = joint.body_in_parent(values)
            position = position + np.einsum("nij,nj->ni", rotation, step_position)
            rotation = rotation @ step_rotation
            rotations.append(rotation)
            positions.append(position)
        return np.stack(rotations, axis=1), np.stack(positions, axis=1)

    def max_error(self, state: Any) -> tuple[float, float]:
        """Compare the chain with the model at the current coordinate values.

        Parameters
        ----------
        state : opensim.State
            State of the model given to the constructor, realized to
            Position by this method.

        Returns
        -------
        position_error : float
            Largest distance between the origin of a body of the chain here
            and in ``Model.getTransformInGround`` (metres).
        rotation_error : float
            Largest element difference of the body rotations.
        """
        model = self._model
        model.realizePosition(state)
        coordinates = model.getCoordinateSet()
        values = np.array(
            [[coordinates.get(name).getValue(state) for name in self.coordinate_names]]
        )
        rotations, positions = self.poses(values)
        bodies = model.getBodySet()
        root = bodies.get(self.body_names[0]).getTransformInGround(state)
        root_rotation = rotation_to_numpy(root.R())
        root_position = vec3_to_numpy(root.p())
        position_error = rotation_error = 0.0
        for index, name in enumerate(self.body_names):
            frame = bodies.get(name).getTransformInGround(state)
            expected_position = vec3_to_numpy(frame.p())
            expected_rotation = rotation_to_numpy(frame.R())
            position_error = max(
                position_error,
                float(
                    np.linalg.norm(
                        root_position + root_rotation @ positions[0, index] - expected_position
                    )
                ),
            )
            rotation_error = max(
                rotation_error,
                float(
                    np.abs(root_rotation @ rotations[0, index] - expected_rotation).max()
                ),
            )
        return position_error, rotation_error
