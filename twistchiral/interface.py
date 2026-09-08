"""Interfaces and junctions between interfering volumes.

The *interface* ``Sigma_ab`` between volumes ``a`` and ``b`` is the
equal-amplitude hypersurface ``|psi_a| = |psi_b|``.  It is where the fringe
visibility of the pair is exactly one, and (in Bloch-sphere language for a
pair) it is the pre-image of the equator.  With more volumes present we keep
only the part of ``Sigma_ab`` where ``a`` and ``b`` dominate all others, so
the interfaces tile the space like the walls of an "amplitude Voronoi"
diagram; walls meet at triple junctions (codimension 2), quadruple junctions
(codimension 3) and so on.

All samplers project random seed points onto the required equal-amplitude
sets with (Gauss-)Newton iterations on the smooth log-amplitude ratios.
"""
from __future__ import annotations

import numpy as np

from .waves import WaveSystem


def log_intensities(system: WaveSystem, X, t: float = 0.0, eps: float = 1e-300):
    """``ln|psi_a|^2`` and its gradient for every volume."""
    Psi, dPsi = system.evaluate(X, t)
    I = np.abs(Psi) ** 2 + eps                                    # (P, N)
    grad = 2.0 * (Psi.conj()[:, None, :] * dPsi).real / I[:, None, :]   # (P, n, N)
    return np.log(I), grad, Psi, dPsi


def log_amplitude_ratio(system: WaveSystem, X, a: int, b: int, t: float = 0.0):
    """``h = ln|psi_a|^2 - ln|psi_b|^2`` and ``grad h``.

    ``h > 0`` inside the domain of ``a``; the pair's Bloch polar angle is
    ``cos(theta) = tanh(h/2)`` and the fringe visibility ``V = sech(h/2)``.
    """
    L, G, _, _ = log_intensities(system, X, t)
    return L[:, a] - L[:, b], G[:, :, a] - G[:, :, b]


def visibility(system: WaveSystem, X, a: int, b: int, t: float = 0.0) -> np.ndarray:
    h, _ = log_amplitude_ratio(system, X, a, b, t)
    return 1.0 / np.cosh(0.5 * h)


def bloch_vector(system: WaveSystem, X, a: int, b: int, t: float = 0.0) -> np.ndarray:
    """Bloch vector of the pair ``(psi_a, psi_b)``: ``(2Re, 2Im, |a|^2-|b|^2)/rho``."""
    Psi, _ = system.evaluate(X, t)
    pa, pb = Psi[:, a], Psi[:, b]
    rho = np.abs(pa) ** 2 + np.abs(pb) ** 2
    cross = np.conj(pa) * pb
    return np.stack([2 * cross.real, 2 * cross.imag, np.abs(pa) ** 2 - np.abs(pb) ** 2], -1) / np.maximum(rho, 1e-300)[:, None]


def project_to_interface(system: WaveSystem, X, a: int = 0, b: int = 1, t: float = 0.0,
                         iters: int = 30, tol: float = 1e-9, max_step: float | None = None):
    """Newton-project points onto ``|psi_a| = |psi_b|``.

    Returns ``(X_on, h, normal, converged)`` where ``normal`` is the unit
    gradient of ``h`` (pointing into the domain of ``a``).
    """
    X = np.array(X, dtype=float, copy=True)
    if max_step is None:
        w = system.widths()
        max_step = 0.5 * float(np.min(w[np.isfinite(w)])) if np.any(np.isfinite(w)) else 1.0
    for _ in range(iters):
        h, gh = log_amplitude_ratio(system, X, a, b, t)
        g2 = (gh * gh).sum(-1)
        step = -h[:, None] * gh / np.maximum(g2, 1e-300)[:, None]
        sn = np.linalg.norm(step, axis=-1)
        scale = np.minimum(1.0, max_step / np.maximum(sn, 1e-300))
        X += step * scale[:, None]
        if np.max(np.abs(h)) < tol:
            break
    h, gh = log_amplitude_ratio(system, X, a, b, t)
    normal = gh / np.maximum(np.linalg.norm(gh, axis=-1, keepdims=True), 1e-300)
    converged = np.abs(h) < 1e-6
    return X, h, normal, converged


def project_to_junction(system: WaveSystem, X, members, t: float = 0.0,
                        iters: int = 40, tol: float = 1e-9, max_step: float | None = None):
    """Gauss-Newton projection onto the k-fold junction ``|psi_a1| = ... = |psi_ak|``."""
    members = list(members)
    X = np.array(X, dtype=float, copy=True)
    if max_step is None:
        w = system.widths()
        max_step = 0.5 * float(np.min(w[np.isfinite(w)])) if np.any(np.isfinite(w)) else 1.0
    a0 = members[0]
    for _ in range(iters):
        L, G, _, _ = log_intensities(system, X, t)
        h = np.stack([L[:, a0] - L[:, b] for b in members[1:]], -1)          # (P, k-1)
        J = np.stack([G[:, :, a0] - G[:, :, b] for b in members[1:]], 1)      # (P, k-1, n)
        step = -np.einsum("pnk,pk->pn", np.linalg.pinv(J), h)
        sn = np.linalg.norm(step, axis=-1)
        scale = np.minimum(1.0, max_step / np.maximum(sn, 1e-300))
        X += step * scale[:, None]
        if np.max(np.abs(h)) < tol:
            break
    L, G, _, _ = log_intensities(system, X, t)
    h = np.stack([L[:, a0] - L[:, b] for b in members[1:]], -1)
    converged = np.all(np.abs(h) < 1e-6, axis=-1)
    return X, converged


def dominance_mask(system: WaveSystem, X, members, t: float = 0.0, slack: float = 0.0) -> np.ndarray:
    """True where every member's intensity beats every non-member's (log slack)."""
    L, _, _, _ = log_intensities(system, X, t)
    members = list(members)
    others = [c for c in range(system.N) if c not in members]
    if not others:
        return np.ones(X.shape[0], dtype=bool)
    return L[:, members].min(-1) >= L[:, others].max(-1) - slack


def dominant_volume(system: WaveSystem, X, t: float = 0.0) -> np.ndarray:
    L, _, _, _ = log_intensities(system, X, t)
    return L.argmax(-1)


def seed_points(system: WaveSystem, count: int, rng=None, pad: float = 2.0, t: float = 0.0,
                box=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    lo, hi = system.bounding_box(pad, t) if box is None else box
    return rng.uniform(lo, hi, size=(count, system.n))


def sample_interface(system: WaveSystem, a: int = 0, b: int = 1, count: int = 4000, rng=None,
                     t: float = 0.0, pad: float = 2.0, require_dominant: bool = True,
                     box=None, slack: float = 0.0) -> dict:
    """Points on the interface ``Sigma_ab`` (restricted to where a, b dominate)."""
    rng = np.random.default_rng(rng)
    X0 = seed_points(system, count, rng, pad, t, box)
    X, h, normal, ok = project_to_interface(system, X0, a, b, t)
    keep = ok
    if require_dominant and system.N > 2:
        keep = keep & dominance_mask(system, X, (a, b), t, slack)
    return {"X": X[keep], "normal": normal[keep], "h": h[keep], "seeds": X0[keep]}


def sample_junction(system: WaveSystem, members, count: int = 4000, rng=None, t: float = 0.0,
                    pad: float = 2.0, require_dominant: bool = True, box=None, slack: float = 0.0) -> dict:
    """Points on the k-fold junction of the given member volumes."""
    rng = np.random.default_rng(rng)
    X0 = seed_points(system, count, rng, pad, t, box)
    X, ok = project_to_junction(system, X0, members, t)
    keep = ok
    if require_dominant and len(members) < system.N:
        keep = keep & dominance_mask(system, X, members, t, slack)
    return {"X": X[keep], "seeds": X0[keep]}


def sample_bulk(system: WaveSystem, count: int = 20000, rng=None, t: float = 0.0, pad: float = 2.0,
                box=None) -> np.ndarray:
    """Uniform points in the bounding box (for bulk statistics)."""
    return seed_points(system, count, rng, pad, t, box)
