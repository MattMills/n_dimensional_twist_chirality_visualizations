"""Grids on 2- and 3-dimensional affine subspaces of R^n, and small helpers
for looking at fields on them."""
from __future__ import annotations

import numpy as np

from .geometry import orthonormal_frame


def plane_grid(origin, u, v, extent, res: int = 200):
    """Grid on the plane ``origin + s u + t v`` for ``s, t in [-extent, extent]``.

    ``u, v`` are orthonormalised (keeping ``u``'s direction).  Returns
    ``(X (res*res, n), S (res, res), T (res, res), frame (n, 2))``.
    """
    origin = np.asarray(origin, dtype=float)
    frame = orthonormal_frame(u, v)
    ext = np.broadcast_to(np.asarray(extent, dtype=float), (2,))
    s = np.linspace(-ext[0], ext[0], res)
    tt = np.linspace(-ext[1], ext[1], res)
    S, T = np.meshgrid(s, tt, indexing="xy")
    X = origin[None, :] + S.reshape(-1, 1) * frame[:, 0][None, :] + T.reshape(-1, 1) * frame[:, 1][None, :]
    return X, S, T, frame


def volume_grid(origin, u, v, w, extent, res: int = 40):
    """Grid on the 3-space ``origin + s u + t v + r w``."""
    origin = np.asarray(origin, dtype=float)
    frame = orthonormal_frame(u, v, w)
    ext = np.broadcast_to(np.asarray(extent, dtype=float), (3,))
    axes = [np.linspace(-e, e, res) for e in ext]
    S, T, R = np.meshgrid(*axes, indexing="ij")
    coords = np.stack([S.ravel(), T.ravel(), R.ravel()], -1)
    X = origin[None, :] + coords @ frame.T
    return X, (S, T, R), frame, axes


def vortex_charges(phase: np.ndarray) -> np.ndarray:
    """Winding numbers of a 2-D phase grid around each plaquette.

    ``phase`` has shape ``(ny, nx)``; the result has shape ``(ny-1, nx-1)``
    with integer charges (zero almost everywhere, +-1 at phase singularities).
    """
    def wrap(d):
        return (d + np.pi) % (2 * np.pi) - np.pi
    d1 = wrap(phase[:-1, 1:] - phase[:-1, :-1])
    d2 = wrap(phase[1:, 1:] - phase[:-1, 1:])
    d3 = wrap(phase[1:, :-1] - phase[1:, 1:])
    d4 = wrap(phase[:-1, :-1] - phase[1:, :-1])
    return np.rint((d1 + d2 + d3 + d4) / (2 * np.pi)).astype(int)


def natural_plane(system, a: int = 0, b: int = 1):
    """A plane through the midpoint of two volumes containing their
    displacement and (the tangential part of) their mean wavevector."""
    va, vb = system[a], system[b]
    d = vb.center - va.center
    kbar = va.k.mean(0) + vb.k.mean(0)
    if np.linalg.norm(d) < 1e-12:
        d = np.eye(system.n)[0]
    tangential = kbar - (kbar @ d) * d / (d @ d)
    if np.linalg.norm(tangential) < 1e-8:
        tangential = np.eye(system.n)[1] if system.n > 1 else d
    origin = 0.5 * (va.center + vb.center)
    return origin, d, tangential
