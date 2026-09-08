"""Geometric building blocks: wavevector sets and n-dimensional motions.

A *wavevector set* is an ``(m, n)`` array of plane-wave wavevectors.  The
functions below produce the classical symmetric configurations (simplex,
cross-polytope, hypercube, planar rings) plus random ones.  Rotations in
SO(n) are represented as ``(n, n)`` orthogonal matrices; the "twist"
between two wave sets is such a rotation, which in n dimensions rotates in
up to ``n // 2`` mutually orthogonal planes at once.
"""
from __future__ import annotations

from itertools import product

import numpy as np
from scipy.linalg import expm, logm


# ----------------------------------------------------------------------------
# wavevector sets
# ----------------------------------------------------------------------------
def unit_rows(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=float)
    return K / np.linalg.norm(K, axis=-1, keepdims=True)


def simplex(n: int) -> np.ndarray:
    """``n + 1`` unit vectors at the vertices of a regular simplex in R^n."""
    E = np.eye(n + 1) - 1.0 / (n + 1)
    _, _, Vt = np.linalg.svd(E)
    K = E @ Vt[:n].T
    return unit_rows(K)


def cross_polytope(n: int) -> np.ndarray:
    """``2n`` unit vectors ``+-e_i`` (the vertices of the cross-polytope)."""
    return np.concatenate([np.eye(n), -np.eye(n)])


def hypercube(n: int) -> np.ndarray:
    """``2^n`` unit vectors ``(+-1, ..., +-1) / sqrt(n)``."""
    return np.array(list(product([-1.0, 1.0], repeat=n))) / np.sqrt(n)


def ring(n: int, m: int, plane=(0, 1), phase: float = 0.0) -> np.ndarray:
    """``m`` unit vectors equally spaced on a great circle in a 2-plane."""
    K = np.zeros((m, n))
    ang = phase + 2 * np.pi * np.arange(m) / m
    K[:, plane[0]] = np.cos(ang)
    K[:, plane[1]] = np.sin(ang)
    return K


def random_sphere(n: int, m: int, rng=None) -> np.ndarray:
    """``m`` random unit vectors in R^n."""
    rng = np.random.default_rng(rng)
    return unit_rows(rng.standard_normal((m, n)))


# ----------------------------------------------------------------------------
# rotations, reflections, screws
# ----------------------------------------------------------------------------
def plane_rotation(n: int, i: int, j: int, angle: float) -> np.ndarray:
    """Rotation by ``angle`` in the coordinate plane ``(i, j)`` of R^n."""
    R = np.eye(n)
    c, s = np.cos(angle), np.sin(angle)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R


def rotation_from_angles(n: int, angles) -> np.ndarray:
    """Simultaneous rotation by ``angles[k]`` in the planes ``(2k, 2k+1)``.

    This is the normal form of an element of SO(n): every rotation is such a
    multi-plane rotation in a suitable orthonormal basis.
    """
    R = np.eye(n)
    for k, ang in enumerate(angles):
        if 2 * k + 1 >= n:
            raise ValueError("too many angles for dimension n")
        R = R @ plane_rotation(n, 2 * k, 2 * k + 1, ang)
    return R


def rotation_from_bivector(B: np.ndarray) -> np.ndarray:
    """``exp(B)`` for an antisymmetric generator ``B`` (a bivector)."""
    B = np.asarray(B, dtype=float)
    return expm(B)


def rotation_generator(R: np.ndarray) -> np.ndarray:
    """Antisymmetric generator ``log(R)`` of a rotation (principal branch)."""
    L = logm(np.asarray(R, dtype=float))
    L = np.real(L)
    return 0.5 * (L - L.T)


def rotation_planes(R: np.ndarray, tol: float = 1e-10):
    """Decompose ``R`` into rotation angles and oriented planes.

    Returns ``(angles, frames)`` with ``angles`` descending and ``frames`` of
    shape ``(k, n, 2)`` such that ``R`` rotates by ``angles[j]`` in the
    oriented plane spanned by ``frames[j][:, 0] -> frames[j][:, 1]``.
    """
    B = rotation_generator(R)
    n = B.shape[0]
    S = -B @ B
    w, V = np.linalg.eigh(S)
    order = np.argsort(-w)
    w, V = w[order], V[:, order]
    angles, frames = [], []
    found = []  # orthonormal vectors already assigned to planes
    for idx in range(n):
        if w[idx] <= tol:
            break
        u = V[:, idx].copy()
        for f in found:
            u -= (u @ f) * f
        nu = np.linalg.norm(u)
        if nu < 1e-8:
            continue
        u /= nu
        ang = np.sqrt(max(w[idx], 0.0))
        v = B @ u / ang
        angles.append(ang)
        frames.append(np.stack([u, v], axis=-1))
        found.extend([u, v])
    return np.array(angles), np.array(frames)


def rotation_taking(a, b) -> np.ndarray:
    """The minimal rotation (in the plane of ``a`` and ``b``) taking the
    direction of ``a`` to the direction of ``b``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    n = len(a)
    c = float(np.clip(a @ b, -1.0, 1.0))
    w = b - c * a
    nw = np.linalg.norm(w)
    if nw < 1e-12:
        if c > 0:
            return np.eye(n)
        # antipodal: rotate by pi in a plane containing a
        e = np.zeros(n)
        e[np.argmin(np.abs(a))] = 1.0
        w = e - (e @ a) * a
        w /= np.linalg.norm(w)
        return np.eye(n) - 2.0 * np.outer(a, a) - 2.0 * np.outer(w, w) + 0 * np.eye(n) if n == 2 else \
            np.eye(n) + (-1 - 1) * (np.outer(a, a) + np.outer(w, w))
    v = w / nw
    theta = np.arctan2(nw, c)
    return (np.eye(n) + np.sin(theta) * (np.outer(v, a) - np.outer(a, v))
            + (np.cos(theta) - 1.0) * (np.outer(a, a) + np.outer(v, v)))


def random_rotation(n: int, rng=None) -> np.ndarray:
    """Haar-random element of SO(n)."""
    rng = np.random.default_rng(rng)
    Q, Rr = np.linalg.qr(rng.standard_normal((n, n)))
    Q = Q * np.sign(np.diag(Rr))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def reflection(n: int, normal) -> np.ndarray:
    """Householder reflection through the hyperplane orthogonal to ``normal``."""
    m = np.asarray(normal, dtype=float)
    m = m / np.linalg.norm(m)
    return np.eye(n) - 2.0 * np.outer(m, m)


def screw_chirality(ea, R: np.ndarray, d) -> np.ndarray:
    """Configuration chirality ``log(R) ^ d`` of the screw motion (R, d).

    ``log(R)`` is a bivector (the rotation planes weighted by their angles)
    and ``d`` the displacement.  The wedge is a 3-vector: in three
    dimensions it is the pseudoscalar ``angle * (axis . d)`` -- the sign of
    the screw -- while in n >= 4 dimensions it is an oriented 3-volume
    element, i.e. a genuinely multi-component chirality.

    Note: the field chirality ``A ^ F`` of two Gaussian volumes related by
    the screw ``(R, d)`` is proportional to *minus* this 3-vector (see
    ``analysis.screw_prediction``); the sign is the orientation convention
    of the Berry curvature.
    """
    B = rotation_generator(R)
    # The generator that rotates e_i toward e_j has matrix entry B_ji = +1,
    # so the geometric bivector (e_i ^ e_j <-> "e_i turns toward e_j") has
    # components B_ji = -B_ij for i < j.
    b2 = ea.from_antisymmetric(B.T)
    return ea.wedge(b2, 2, np.asarray(d, dtype=float), 1)


def orthonormal_frame(*vectors) -> np.ndarray:
    """Gram-Schmidt orthonormalisation; returns an ``(n, k)`` frame whose
    j-th column has positive overlap with the j-th input vector."""
    cols = []
    for v in vectors:
        u = np.asarray(v, dtype=float).copy()
        for c in cols:
            u -= (u @ c) * c
        nu = np.linalg.norm(u)
        if nu < 1e-12:
            raise ValueError("vectors are linearly dependent")
        cols.append(u / nu)
    return np.array(cols).T


def complete_frame(*vectors, n: int) -> np.ndarray:
    """Orthonormal basis of R^n whose first columns span the given vectors."""
    cols = list(orthonormal_frame(*vectors).T) if vectors else []
    for i in range(n):
        if len(cols) == n:
            break
        u = np.zeros(n)
        u[i] = 1.0
        for c in cols:
            u -= (u @ c) * c
        nu = np.linalg.norm(u)
        if nu > 1e-8:
            cols.append(u / nu)
    return np.array(cols).T
