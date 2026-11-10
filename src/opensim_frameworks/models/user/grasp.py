"""Closing the fingers of a :class:`User` hand on a handle.

The procedure only needs the handle (a body and one of its meshes) and a hand
that is already placed on it: the placement of shoulder, elbow and wrist is
left to the caller, since it depends on the equipment. It is exposed on
:class:`User` through :class:`_GraspMixin` as ``grasp_handle``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.spatial import cKDTree

from ...utils.collision import collision_detector, mesh_matrix, mesh_polydata
from ...utils.constants import MESHES_DIR
from ...utils.opensim import read_vtp_vertices, rotation_to_numpy, vec3_to_numpy
from .fingers import _resolve_body_mesh_path

__all__ = []

#: Fingers in the order they are closed, with the finger whose mesh each one
#: must not overlap (``None`` for the first one) and its metacarpal number.
_GRASP_ORDER = ("middle", "ring", "index", "little")
_REFERENCE_FINGER = {"middle": None, "ring": "middle", "index": "middle", "little": "ring"}
_METACARPAL = {"index": 2, "middle": 3, "ring": 4, "little": 5}
_SEGMENTS = ("proximal", "medial", "distal")
#: (joint, segments it moves) in the order each finger is closed.
_STAGES = (
    ("mcp", ("proximal", "medial", "distal")),
    ("pip", ("medial", "distal")),
    ("dip", ("distal",)),
)


class _GraspMixin:
    """Closing the fingers on a handle."""

    def _body_vertices_in_ground(self, body_name: str) -> np.ndarray:
        """Return the scaled vertices of the meshes of a body, in Ground.

        Parameters
        ----------
        body_name : str
            Name of a body of this user.

        Returns
        -------
        numpy.ndarray, shape (n, 3)
            Vertices in Ground coordinates, in metres.

        Raises
        ------
        ValueError
            If the body has no mesh or a mesh is invalid.
        FileNotFoundError
            If a mesh cannot be found in the geometry directories.
        """
        self.realizePosition(self.state)
        body = self.getBodySet().get(body_name)
        arrays = []
        for index in range(body.getPropertyByName("attached_geometry").size()):
            mesh = self.opensim.Mesh.safeDownCast(body.get_attached_geometry(index))
            if mesh is None:
                continue
            path = _resolve_body_mesh_path(
                Path(mesh.get_mesh_file()), self.geometry_directories, "Hand"
            )
            scale = np.array([mesh.get_scale_factors().get(i) for i in range(3)])
            vertices = read_vtp_vertices(path)
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or not np.all(np.isfinite(vertices))
                or not np.all(np.isfinite(scale))
                or np.any(scale <= 0)
            ):
                raise ValueError(f"Invalid mesh geometry for body {body_name!r}")
            frame = mesh.getFrame().getTransformInGround(self.state)
            arrays.append(
                (vertices * scale) @ rotation_to_numpy(frame.R()).T
                + vec3_to_numpy(frame.p())
            )
        if not arrays:
            raise ValueError(f"No mesh is attached to body {body_name!r}")
        return np.vstack(arrays)

    def grasp_handle(
        self,
        hand: Literal["left", "right"],
        handle_body: Any,
        handle_mesh: Any,
        *,
        palm_margin_m: float = 2e-3,
    ) -> dict[str, dict[str, Any]]:
        """Close the four fingers of a hand on a handle, one finger at a time.

        The hand must already be placed on the handle, with the palm on it.
        The fingers are closed in this order: middle, ring, index, little.
        Each finger follows its own sequence, starting from the pose reached
        at the previous step:

        1. the proximal phalanx is flexed (MCP) up to its maximum flexion or
           until the finger touches the handle;
        2. the medial phalanx is flexed (PIP) up to its maximum flexion or
           until the moving segments touch the handle;
        3. the distal phalanx is flexed (DIP) up to its maximum flexion or
           until it touches the handle or the palm.

        The handle is its real mesh (exact triangle-surface contact). The
        thumb is not moved and is ignored.

        The middle finger is flexed without abduction/adduction. The ring and
        the index fingers use the MCP abduction/adduction needed not to
        overlap the middle finger, and the little finger the one needed not to
        overlap the ring finger. After each step the pose must also keep the
        bone meshes of adjacent segments of the same finger apart. When a
        pose overlaps, the abduction/adduction nearest to the current one
        that separates the meshes is used; only if none does, the flexion is
        reduced to the largest value without overlap.

        Only the fingers move, and their coordinates stay locked at the pose
        found.

        Parameters
        ----------
        hand : {'left', 'right'}
            Hand to close.
        handle_body : opensim.Body
            Body of a separately initialized model (for example a piece of
            equipment). Its pose is read from its model's own working state.
        handle_mesh : opensim.Mesh
            Mesh attached to `handle_body` that the fingers touch. A relative
            mesh file is searched in the user's geometry directories and in
            the bundled primitive meshes.
        palm_margin_m : float, default=2e-3
            Distance, in metres, below which a finger vertex is considered to
            touch the palm.

        Returns
        -------
        dict
            Per finger: ``mcp``, ``pip`` and ``dip`` entries with
            ``angle_deg`` and ``stopped_by`` (``"handle"``, ``"palm"``,
            ``"range"``, ``"bone"`` or ``"finger"`` when the flexion had to be
            reduced to avoid an overlap with adjacent bones or with the
            reference finger, or ``"blocked"`` when the segments already touch
            before moving), and ``abduction_deg`` (MCP abduction/adduction
            used).

        Raises
        ------
        ValueError
            If `hand` is invalid or `handle_mesh` is not attached to
            `handle_body`.
        TypeError
            If `handle_body` or `handle_mesh` have the wrong type.

        Notes
        -----
        A finger stops on the palm at step 3; at steps 1 and 2 the palm is a
        limit only, so that no segment penetrates it. The palm is evaluated
        with the distance between mesh vertices of the medial and distal
        phalanges and of the palm, minus `palm_margin_m`, so it is
        approximate (bones only, no soft tissue); the proximal phalanx is
        excluded since it sits next to the palm at the MCP. Overlaps between
        bones are exact triangle-surface checks. The metacarpals are never
        moved. The finger coordinates are left locked at the pose found.
        """
        if hand not in ("left", "right"):
            raise ValueError("hand must be 'left' or 'right'")
        if not isinstance(handle_body, self.opensim.Body):
            raise TypeError("handle_body must be an opensim.Body")
        if not isinstance(handle_mesh, self.opensim.Mesh):
            raise TypeError("handle_mesh must be an opensim.Mesh")
        if not handle_body.hasModel():
            raise ValueError("handle_body is not connected to an initialized model")
        if handle_mesh.getFrame().findBaseFrame().getName() != handle_body.getName():
            raise ValueError("handle_mesh is not attached to handle_body")

        side = hand
        suffix = "l" if side == "left" else "r"
        # le mesh relative si cercano tra quelle dell'utente e le primitive della libreria
        directories = [*self.geometry_directories, MESHES_DIR]
        handle_model = handle_body.getModel()
        handle_state = handle_model.getWorkingState()
        handle_model.realizePosition(handle_state)
        self.realizePosition(self.state)
        user_hand = self.getBodySet().get(f"hand_{suffix}")

        def transform(body: Any) -> tuple[np.ndarray, np.ndarray]:
            frame = body.getTransformInGround(self.state)
            return rotation_to_numpy(frame.R()), vec3_to_numpy(frame.p())

        # vertici di dita e palmo nei rispettivi sistemi locali, calcolati una volta
        def local_vertices(name: str) -> np.ndarray:
            rotation, position = transform(self.getBodySet().get(name))
            return (self._body_vertices_in_ground(name) - position) @ rotation

        palm_tree = cKDTree(local_vertices(f"hand_{suffix}"))
        bodies = {
            (finger, segment): self.getBodySet().get(f"{finger}_{segment}_{suffix}")
            for finger in _GRASP_ORDER
            for segment in _SEGMENTS
        }
        vertices = {
            key: local_vertices(body.getName()) for key, body in bodies.items()
        }

        # controllo esatto delle intersezioni tra mesh ossee
        def mesh_of(body_name: str, stem: str) -> Any:
            body = self.getBodySet().get(body_name)
            for index in range(body.getPropertyByName("attached_geometry").size()):
                mesh = self.opensim.Mesh.safeDownCast(body.get_attached_geometry(index))
                if mesh is not None and mesh.get_mesh_file() == f"{stem}_{suffix}vs.vtp":
                    return mesh
            raise ValueError(f"Mesh {stem!r} not found on body {body_name!r}")

        def matrix(mesh: Any) -> Any:
            # la mesh dell'impugnatura appartiene al suo modello, le altre all'utente
            state = handle_state if mesh is handle_mesh else self.state
            return mesh_matrix(mesh, state)

        def detector_for(first: Any, second: Any) -> tuple[Any, Any, Any]:
            """Collision detector between two meshes.

            Each mesh is an ``opensim.Mesh`` or a (body name, mesh stem) pair.
            """
            meshes = [
                item if isinstance(item, self.opensim.Mesh) else mesh_of(*item)
                for item in (first, second)
            ]
            detector = collision_detector(
                *(mesh_polydata(mesh, directories) for mesh in meshes)
            )
            return meshes[0], meshes[1], detector

        def finger_mesh(finger: str, segment: str) -> tuple[str, str]:
            return f"{finger}_{segment}_{suffix}", f"{finger}_{segment}"

        def both(first: Any, second: Any) -> list:
            """Detectors for a pair of meshes, in both orders.

            The VTK collision filter is not symmetric: a contact can be found
            with one mesh as first input and missed with the other, so every
            pair is tested in both orders.
            """
            return [detector_for(first, second), detector_for(second, first)]

        # mesh reale dell'impugnatura contro ogni segmento di ogni dito
        handle_pairs = {
            (finger, segment): both(finger_mesh(finger, segment), handle_mesh)
            for finger in _GRASP_ORDER
            for segment in _SEGMENTS
        }
        # ossa adiacenti dello stesso dito
        own = {
            finger: [
                *both(
                    (f"hand_{suffix}", f"metacarpal{_METACARPAL[finger]}"),
                    finger_mesh(finger, "proximal"),
                ),
                *both(finger_mesh(finger, "proximal"), finger_mesh(finger, "medial")),
                *both(finger_mesh(finger, "medial"), finger_mesh(finger, "distal")),
            ]
            for finger in _GRASP_ORDER
        }
        # ossa del dito di riferimento contro quelle del dito considerato
        neighbor = {
            finger: [
                entry
                for a in _SEGMENTS
                for b in _SEGMENTS
                for entry in both(finger_mesh(other, a), finger_mesh(finger, b))
            ]
            for finger, other in _REFERENCE_FINGER.items()
            if other is not None
        }

        def collide(entries: list) -> bool:
            """True if any pair of meshes touches or crosses."""
            self.realizePosition(self.state)
            for first, second, detector in entries:
                detector.SetMatrix(0, matrix(first))
                detector.SetMatrix(1, matrix(second))
                detector.Update()
                if detector.GetNumberOfContacts() > 0:
                    return True
            return False

        def touches_handle(finger: str, moving: tuple[str, ...]) -> bool:
            return any(collide(handle_pairs[(finger, segment)]) for segment in moving)

        def bones_free(finger: str) -> bool:
            return not collide(own[finger])

        def fingers_apart(finger: str) -> bool:
            return finger not in neighbor or not collide(neighbor[finger])

        def palm_clearance(finger: str, moving: tuple[str, ...]) -> float:
            """Smallest clearance from the palm in metres (infinite if none).

            The palm is a limit at every step (no segment may penetrate it).
            The proximal phalanx is excluded: it sits next to the palm at the
            MCP joint.
            """
            self.realizePosition(self.state)
            hand_rotation, hand_position = transform(user_hand)
            palm = np.inf
            for segment in moving:
                if segment == "proximal":
                    continue
                rotation, position = transform(bodies[(finger, segment)])
                ground = vertices[(finger, segment)] @ rotation.T + position
                distance, _ = palm_tree.query((ground - hand_position) @ hand_rotation)
                palm = min(palm, float(distance.min() - palm_margin_m))
            return palm

        def acceptable(finger: str) -> bool:
            """No overlap and no penetration of handle or palm, whole finger."""
            return (
                bones_free(finger)
                and fingers_apart(finger)
                and not touches_handle(finger, _SEGMENTS)
                and palm_clearance(finger, _SEGMENTS) >= 0.0
            )

        abductions = {
            finger: self.finger_coordinate(side, finger, "mcp", "abduction")
            for finger in _GRASP_ORDER
        }
        for finger in _GRASP_ORDER:
            for joint, _ in _STAGES:
                self.set_finger_coordinate_locked(side, finger, joint, "flexion", False)
            self.set_finger_coordinate_locked(side, finger, "mcp", "abduction", False)
        self.realizePosition(self.state)

        def settle(
            finger: str, coordinate: Any, start: float, stop: float
        ) -> float | None:
            """Largest flexion in [start, stop] with an acceptable pose.

            For the fingers other than the middle one, at each flexion the
            abduction nearest to the current one that makes the pose
            acceptable is used. Returns ``None`` if no flexion qualifies.
            """
            abduction = abductions[finger]
            base = abduction.getValue(self.state)
            if finger == "middle":
                candidates = [base]
            else:
                samples = np.arange(
                    abduction.getRangeMin(),
                    abduction.getRangeMax() + 1e-9,
                    math.radians(1.0),
                )
                candidates = [base] + sorted(samples, key=lambda a: abs(a - base))
            value = stop
            while value >= start - 1e-12:
                coordinate.setValue(self.state, value, False)
                for candidate in candidates:
                    abduction.setValue(self.state, float(candidate), False)
                    if acceptable(finger):
                        return value
                value -= math.radians(1.0)
            abduction.setValue(self.state, base, False)
            return None

        report: dict[str, dict[str, Any]] = {}
        for finger in _GRASP_ORDER:
            report[finger] = {}
            for joint, moving in _STAGES:
                coordinate = self.finger_coordinate(side, finger, joint, "flexion")
                current = coordinate.getValue(self.state)
                limit = coordinate.getRangeMax()

                def free(value: float) -> bool:
                    coordinate.setValue(self.state, value, False)
                    return palm_clearance(finger, moving) >= 0.0 and not touches_handle(
                        finger, moving
                    )

                if not free(current):
                    coordinate.setValue(self.state, current, False)
                    report[finger][joint] = {
                        "angle_deg": math.degrees(current),
                        "stopped_by": "blocked",
                    }
                    continue
                # ricerca del primo contatto: scansione a passi di 0.5 gradi, poi bisezione
                free_value, contact_value = current, None
                for value in np.arange(current, limit, math.radians(0.5))[1:]:
                    if free(float(value)):
                        free_value = float(value)
                    else:
                        contact_value = float(value)
                        break
                if contact_value is None and free(limit):
                    free_value = limit
                elif contact_value is None:
                    contact_value = limit
                if contact_value is not None:
                    for _ in range(40):
                        middle = (free_value + contact_value) / 2
                        if free(middle):
                            free_value = middle
                        else:
                            contact_value = middle
                if contact_value is None:
                    stopped = "range"
                else:
                    # cosa si tocca alla prima posa in contatto: impugnatura o palmo
                    coordinate.setValue(self.state, contact_value, False)
                    stopped = "handle" if touches_handle(finger, moving) else "palm"
                coordinate.setValue(self.state, free_value, False)

                # nessuna sovrapposizione con le ossa adiacenti o con il dito di
                # riferimento: prima con l'abduzione, poi riducendo la flessione
                if not acceptable(finger):
                    cause = "bone" if not bones_free(finger) else "finger"
                    settled = settle(finger, coordinate, current, free_value)
                    if settled is None:
                        free_value = current
                        stopped = cause
                    elif settled < free_value - 1e-9:
                        free_value = settled
                        stopped = cause
                    coordinate.setValue(self.state, free_value, False)
                report[finger][joint] = {
                    "angle_deg": math.degrees(free_value),
                    "stopped_by": stopped,
                }
            report[finger]["abduction_deg"] = math.degrees(
                abductions[finger].getValue(self.state)
            )
        # le dita restano nella posa trovata
        for finger in _GRASP_ORDER:
            for joint, _ in _STAGES:
                self.set_finger_coordinate_locked(side, finger, joint, "flexion", True)
            self.set_finger_coordinate_locked(side, finger, "mcp", "abduction", True)
        return report
