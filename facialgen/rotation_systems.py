from __future__ import annotations

from typing import List

import numpy as np
import scipy.sparse as sp

Dart = tuple[int, int]
RotationSystem = dict[int, list[int]]
PiMap = dict[int, dict[int, int]]


def build_rotation_from_curvature_signs(
    A: sp.spmatrix,
    curvature: np.ndarray,
    signs: np.ndarray | List[int] | List[bool],
) -> RotationSystem:
    """
    Build a rotation system from adjacency, curvature, and sign assignments.
    """
    A = sp.csr_matrix(A)
    n = A.shape[0]

    curvature = np.asarray(curvature, dtype=float)
    if curvature.shape != (n,):
        raise ValueError(f"curvature must have shape ({n},), got {curvature.shape}")

    signs = np.asarray(signs)
    if signs.shape != (n,):
        raise ValueError(f"signs must have shape ({n},), got {signs.shape}")

    if signs.dtype == bool:
        signs_pm = np.where(signs, 1, -1)
    else:
        signs_pm = np.where(signs > 0, 1, -1)

    rotation: RotationSystem = {}

    for u in range(n):
        nbrs = A.indices[A.indptr[u]:A.indptr[u + 1]]
        nbrs = np.asarray(nbrs, dtype=int)

        if len(nbrs) == 0:
            rotation[u] = []
            continue

        if signs_pm[u] > 0:
            order = sorted(nbrs, key=lambda v: (curvature[v], v))
        else:
            order = sorted(nbrs, key=lambda v: (-curvature[v], v))

        rotation[u] = list(order)

    return rotation


def random_rotation_system(
    A: sp.spmatrix,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> RotationSystem:
    """Sample a random rotation system by shuffling neighbors at each vertex."""
    A = sp.csr_matrix(A)
    n = A.shape[0]

    if rng is None:
        rng = np.random.default_rng(seed)

    rotation: RotationSystem = {}
    for u in range(n):
        nbrs = np.asarray(A.indices[A.indptr[u]:A.indptr[u + 1]], dtype=int)
        if nbrs.size == 0:
            rotation[u] = []
            continue

        order = nbrs[rng.permutation(nbrs.size)]
        rotation[u] = order.tolist()

    return rotation


def build_pi_from_rotation(rotation: RotationSystem) -> PiMap:
    """Build pi[u][v] = next neighbor after v in rotation[u]."""
    pi: PiMap = {}

    for u, nbrs in rotation.items():
        k = len(nbrs)
        mp = {}
        for i, v in enumerate(nbrs):
            mp[v] = nbrs[(i + 1) % k]
        pi[u] = mp

    return pi


def build_rotation_from_pi(pi: PiMap) -> RotationSystem:
    """Reconstruct a rotation system from a pi-map."""
    rotation: RotationSystem = {}

    for u, mp in pi.items():
        if not mp:
            rotation[u] = []
            continue

        keys = set(mp.keys())
        vals = set(mp.values())
        if keys != vals:
            raise ValueError(
                f"pi[{u}] must permute its neighbor set. "
                f"Got keys={sorted(keys)}, values={sorted(vals)}"
            )

        start = min(keys)
        order = [start]
        cur = start

        while True:
            nxt = mp[cur]
            if nxt == start:
                break
            if nxt in order:
                raise ValueError(
                    f"pi[{u}] is not a single cyclic order on the neighbors of {u}."
                )
            order.append(nxt)
            cur = nxt

        if set(order) != keys:
            raise ValueError(
                f"pi[{u}] is not a single cyclic order on the neighbors of {u}."
            )

        rotation[u] = order

    return rotation


def list_all_darts_from_rotation(rotation: RotationSystem) -> List[Dart]:
    """List all darts (u,v) from a rotation system."""
    darts: List[Dart] = []
    for u, nbrs in rotation.items():
        for v in nbrs:
            darts.append((u, v))
    return darts


def list_all_darts_from_pi(pi: PiMap) -> List[Dart]:
    """List all darts (u,v) encoded by a pi-map."""
    darts: List[Dart] = []
    for u, mp in pi.items():
        for v in mp:
            darts.append((u, v))
    return darts


def facial_successor(dart: Dart, pi: PiMap) -> Dart:
    """Face permutation (u,v) -> (pi_u(v), u)."""
    u, v = dart
    return (pi[u][v], u)


def enumerate_facial_walks_from_pi(
    pi: PiMap,
    *,
    rng: np.random.Generator | None = None,
) -> List[List[Dart]]:
    """Enumerate all facial walks from a pi-map."""
    darts = list_all_darts_from_pi(pi)
    if rng is not None and len(darts) > 1:
        order = rng.permutation(len(darts))
        darts = [darts[i] for i in order.tolist()]

    visited: set[Dart] = set()
    faces: List[List[Dart]] = []

    for start in darts:
        if start in visited:
            continue

        face: List[Dart] = []
        cur = start

        while cur not in visited:
            u, v = cur
            if u not in pi or v not in pi[u]:
                raise ValueError(f"Dart {cur} is not defined in pi.")

            visited.add(cur)
            face.append(cur)
            cur = facial_successor(cur, pi)

        faces.append(face)

    return faces


def enumerate_facial_walks_from_rotation(
    rotation: RotationSystem,
    *,
    rng: np.random.Generator | None = None,
) -> List[List[Dart]]:
    """Enumerate all facial walks from a rotation system."""
    pi = build_pi_from_rotation(rotation)
    return enumerate_facial_walks_from_pi(pi, rng=rng)


def dart_face_to_vertex_sequence(face: List[Dart]) -> List[int]:
    """Convert a dart face to a second-order vertex sequence."""
    if not face:
        return []

    u0, u1 = face[0]
    return [u0, u1] + [u for (u, _) in face[1:]]


def check_facial_walks_from_pi(pi: PiMap, faces: List[List[Dart]]) -> None:
    """Sanity checks for faces produced from a pi-map."""
    all_darts = list_all_darts_from_pi(pi)
    dart_set = set(all_darts)

    seen: List[Dart] = []
    for face in faces:
        seen.extend(face)

    if len(seen) != len(set(seen)):
        raise ValueError("Some dart appears more than once across faces.")

    if set(seen) != dart_set:
        missing = dart_set - set(seen)
        extra = set(seen) - dart_set
        raise ValueError(
            f"Faces do not partition the dart set. "
            f"Missing={len(missing)}, Extra={len(extra)}"
        )

    if sum(len(face) for face in faces) != len(all_darts):
        raise ValueError("Total face length does not equal number of darts.")


def facial_walks_from_pi(
    pi: PiMap,
    *,
    return_rotation: bool = False,
    return_vertex_faces: bool = False,
    rng: np.random.Generator | None = None,
):
    """Compute facial walks from a pi-map."""
    faces = enumerate_facial_walks_from_pi(pi, rng=rng)
    check_facial_walks_from_pi(pi, faces)

    outs = [faces]

    if return_rotation:
        rotation = build_rotation_from_pi(pi)
        outs.append(rotation)

    if return_vertex_faces:
        vertex_faces = [dart_face_to_vertex_sequence(face) for face in faces]
        outs.append(vertex_faces)

    return outs[0] if len(outs) == 1 else tuple(outs)


def facial_walks_from_curvature_signs(
    A: sp.spmatrix,
    curvature: np.ndarray,
    signs: np.ndarray | List[int] | List[bool],
    *,
    return_rotation: bool = False,
    return_vertex_faces: bool = False,
    rng: np.random.Generator | None = None,
):
    """Build curvature-sign rotation then compute facial walks."""
    rotation = build_rotation_from_curvature_signs(A, curvature, signs)
    pi = build_pi_from_rotation(rotation)
    result = facial_walks_from_pi(
        pi,
        return_rotation=return_rotation,
        return_vertex_faces=return_vertex_faces,
        rng=rng,
    )

    if not return_rotation:
        return result

    result_items = list(result if isinstance(result, tuple) else (result,))
    result_items[1] = rotation
    return tuple(result_items)
