"""Wave volumes: localised superpositions of plane waves in R^n.

A :class:`WaveVolume` is

    psi(x, t) = G(x, t) * sum_m c_m exp(i (k_m . x - omega_m t))

with a Gaussian envelope ``G`` (the "volume") centred at ``center + velocity t``.
Everything is evaluated analytically, including the gradient, so the
quantum-geometric quantities built downstream involve no finite differences.

A :class:`WaveSystem` is an ordered collection of such volumes.  Evaluated at
a point cloud it returns the spinor field ``Psi(x) in C^N`` (one component per
volume) and its gradient ``dPsi(x) in C^{n x N}``.  The interference geometry
of the system is the geometry of the map ``x -> [Psi(x)] in CP^{N-1}``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np


class _Precision:
    """Inverse covariance of the envelope, stored as compactly as possible:
    a scalar (isotropic width), a vector (axis-aligned widths) or a full
    matrix.  All operations are O(n) unless the matrix form is used."""

    def __init__(self, width, n: int):
        w = np.asarray(width, dtype=float)
        self.n = n
        if w.ndim == 0:
            self.kind, self.value = "scalar", 1.0 / float(w) ** 2
        elif w.ndim == 1:
            if w.shape[0] != n:
                raise ValueError("width vector must have length n")
            self.kind, self.value = "diag", 1.0 / w**2
        else:
            if w.shape != (n, n):
                raise ValueError("width matrix must be (n, n)")
            self.kind, self.value = "full", np.linalg.inv(w)   # a matrix is a covariance

    def apply_rows(self, Y: np.ndarray) -> np.ndarray:
        """``Y @ L`` for row vectors ``Y (P, n)``."""
        if self.kind == "scalar":
            return Y * self.value
        if self.kind == "diag":
            return Y * self.value[None, :]
        return Y @ self.value

    def apply(self, c: np.ndarray) -> np.ndarray:
        """``L @ c`` for one vector."""
        if self.kind == "scalar":
            return c * self.value
        if self.kind == "diag":
            return c * self.value
        return self.value @ c

    def log_eigenvalues(self) -> np.ndarray:
        if self.kind == "scalar":
            return np.full(self.n, np.log(self.value))
        if self.kind == "diag":
            return np.log(self.value)
        return np.log(np.linalg.eigvalsh(self.value))

    def equals(self, other: "_Precision") -> bool:
        if self.kind == other.kind:
            return bool(np.allclose(self.value, other.value))
        return bool(np.allclose(self.dense(), other.dense()))

    def dense(self) -> np.ndarray:
        if self.kind == "scalar":
            return np.eye(self.n) * self.value
        if self.kind == "diag":
            return np.diag(self.value)
        return self.value


def _as_precision(width, n: int):
    """Turn a width specification into a compact precision object."""
    if width is None:
        return None
    return _Precision(width, n)


def dispersion_frequencies(K: np.ndarray, dispersion, speed: float = 1.0) -> np.ndarray:
    """Angular frequencies for each wavevector under a dispersion relation."""
    kn = np.linalg.norm(K, axis=-1)
    if dispersion is None or dispersion == "none":
        return np.zeros_like(kn)
    if dispersion == "linear":  # non-dispersive waves, omega = c |k|
        return speed * kn
    if dispersion == "quadratic":  # Schroedinger-like, omega = |k|^2 / 2
        return 0.5 * kn**2
    if callable(dispersion):
        return np.asarray(dispersion(K), dtype=float)
    return np.asarray(dispersion, dtype=float)


@dataclass
class WaveVolume:
    """A localised superposition of plane waves."""

    k: np.ndarray                        # (m, n) wavevectors
    center: np.ndarray | None = None     # (n,)
    width: object = 1.0                  # scalar, (n,) or (n, n) covariance; None = no envelope
    amplitudes: np.ndarray | None = None  # (m,) complex
    dispersion: object = "linear"
    speed: float = 1.0
    velocity: np.ndarray | None = None   # (n,) envelope velocity
    name: str = ""
    _omega: np.ndarray = field(default=None, repr=False)
    _prec: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self.k = np.atleast_2d(np.asarray(self.k, dtype=float))
        n = self.k.shape[1]
        self.center = np.zeros(n) if self.center is None else np.asarray(self.center, dtype=float)
        self.velocity = np.zeros(n) if self.velocity is None else np.asarray(self.velocity, dtype=float)
        if self.amplitudes is None:
            self.amplitudes = np.ones(self.k.shape[0], dtype=complex)
        else:
            self.amplitudes = np.asarray(self.amplitudes, dtype=complex)
            if self.amplitudes.ndim == 0:
                self.amplitudes = np.full(self.k.shape[0], self.amplitudes)
        self._omega = dispersion_frequencies(self.k, self.dispersion, self.speed)
        self._prec = _as_precision(self.width, n)

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        return self.k.shape[1]

    @property
    def m(self) -> int:
        return self.k.shape[0]

    @property
    def omega(self) -> np.ndarray:
        return self._omega

    def center_at(self, t: float) -> np.ndarray:
        return self.center + self.velocity * t

    def sigma(self) -> float:
        """A representative width (geometric mean of the principal widths)."""
        if self._prec is None:
            return np.inf
        return float(np.exp(-0.5 * np.mean(self._prec.log_eigenvalues())))

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, X, t: float = 0.0, relative_envelope: bool = False):
        """Return ``(psi, dpsi)`` at points ``X`` of shape ``(P, n)``.

        With ``relative_envelope=True`` the factor ``exp(-x^T L x / 2)`` that
        is common to all volumes of equal width is dropped, leaving the
        envelope ``exp(x^T L c - c^T L c / 2)``.  Every projective quantity
        (A, g, F, the chirality ladder, interfaces) is unchanged, and nothing
        underflows however large ``n`` is.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        phase = X @ self.k.T - self._omega[None, :] * t          # (P, m)
        E = self.amplitudes[None, :] * np.exp(1j * phase)        # (P, m)
        S = E.sum(-1)                                            # (P,)
        dS = E @ (1j * self.k)                                   # (P, n)
        if self._prec is None:
            return S, dS
        c = self.center_at(t)
        if relative_envelope:
            Lc = self._prec.apply(c)                             # (n,)
            G = np.exp(X @ Lc - 0.5 * (c @ Lc))                  # (P,)
            dG = G[:, None] * Lc[None, :]
        else:
            Y = X - c[None, :]
            LY = self._prec.apply_rows(Y)                        # (P, n)
            G = np.exp(-0.5 * np.einsum("pi,pi->p", Y, LY))     # (P,)
            dG = -G[:, None] * LY                                # (P, n)
        psi = G * S
        dpsi = S[:, None] * dG + G[:, None] * dS
        return psi, dpsi

    def field(self, X, t: float = 0.0) -> np.ndarray:
        return self.evaluate(X, t)[0]

    def local_wavevector(self, X, t: float = 0.0, relative_envelope: bool = False) -> np.ndarray:
        """Phase gradient ``Im(conj(psi) grad psi) / |psi|^2``."""
        psi, dpsi = self.evaluate(X, t, relative_envelope=relative_envelope)
        I = np.abs(psi) ** 2
        return (np.conj(psi)[:, None] * dpsi).imag / np.maximum(I, 1e-300)[:, None]

    # ------------------------------------------------------------------ transformations
    def transformed(self, R: np.ndarray, shift=None) -> "WaveVolume":
        """Apply the affine map ``x -> R x + shift`` to the volume.

        Wavevectors, centre, velocity and the envelope all transform
        covariantly; the field satisfies ``psi'(R x + shift) = psi(x)`` up to
        the plane-wave phase convention.
        """
        R = np.asarray(R, dtype=float)
        n = self.n
        shift = np.zeros(n) if shift is None else np.asarray(shift, dtype=float)
        k = self.k @ R.T
        center = R @ self.center + shift
        velocity = R @ self.velocity
        width = self.width
        if width is not None:
            w = np.asarray(width, dtype=float)
            if w.ndim == 2:
                width = R @ w @ R.T
            elif w.ndim == 1 and not np.allclose(w, w[0]):
                width = R @ np.diag(w**2) @ R.T
        # keep the plane-wave phases anchored at the (moved) centre
        phase_shift = np.exp(-1j * (k @ center - self.k @ self.center))
        return replace(self, k=k, center=center, velocity=velocity, width=width,
                       amplitudes=self.amplitudes * phase_shift)

    def rotated(self, R: np.ndarray, about_center: bool = True) -> "WaveVolume":
        if about_center:
            return self.transformed(R, shift=self.center - np.asarray(R) @ self.center)
        return self.transformed(R)

    def translated(self, shift) -> "WaveVolume":
        return self.transformed(np.eye(self.n), shift=shift)

    def mirrored(self, normal) -> "WaveVolume":
        from .geometry import reflection
        return self.transformed(reflection(self.n, normal))

    def scaled(self, factor: complex) -> "WaveVolume":
        return replace(self, amplitudes=self.amplitudes * factor)

    def with_phase(self, theta: float) -> "WaveVolume":
        return self.scaled(np.exp(1j * theta))


class WaveSystem:
    """An ordered collection of wave volumes interfering simultaneously."""

    def __init__(self, volumes: Sequence[WaveVolume], relative_envelope: bool = False):
        self.volumes = list(volumes)
        if not self.volumes:
            raise ValueError("need at least one volume")
        n = {v.n for v in self.volumes}
        if len(n) != 1:
            raise ValueError("all volumes must live in the same dimension")
        self.relative_envelope = bool(relative_envelope)
        if self.relative_envelope:
            precs = [v._prec for v in self.volumes]
            if any(p is None for p in precs) or any(not p.equals(precs[0]) for p in precs):
                raise ValueError("relative_envelope needs equal Gaussian widths for all volumes")

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        return self.volumes[0].n

    @property
    def N(self) -> int:
        return len(self.volumes)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, i: int) -> WaveVolume:
        return self.volumes[i]

    def centers(self, t: float = 0.0) -> np.ndarray:
        return np.array([v.center_at(t) for v in self.volumes])

    def widths(self) -> np.ndarray:
        return np.array([v.sigma() for v in self.volumes])

    def bounding_box(self, pad: float = 2.5, t: float = 0.0):
        C = self.centers(t)
        w = self.widths()
        w = np.where(np.isfinite(w), w, 1.0)
        lo = (C - pad * w[:, None]).min(0)
        hi = (C + pad * w[:, None]).max(0)
        return lo, hi

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, X, t: float = 0.0):
        """Spinor field ``Psi (P, N)`` and gradient ``dPsi (P, n, N)``."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Psi = np.empty((X.shape[0], self.N), dtype=complex)
        dPsi = np.empty((X.shape[0], self.n, self.N), dtype=complex)
        for a, vol in enumerate(self.volumes):
            psi, dpsi = vol.evaluate(X, t, relative_envelope=self.relative_envelope)
            Psi[:, a] = psi
            dPsi[:, :, a] = dpsi
        return Psi, dPsi

    def total_field(self, X, t: float = 0.0) -> np.ndarray:
        """The physical superposition ``sum_a psi_a``."""
        return self.evaluate(X, t)[0].sum(-1)

    def intensities(self, X, t: float = 0.0) -> np.ndarray:
        return np.abs(self.evaluate(X, t)[0]) ** 2

    # ------------------------------------------------------------------ transformations
    def transformed(self, R, shift=None) -> "WaveSystem":
        return WaveSystem([v.transformed(R, shift) for v in self.volumes], self.relative_envelope)

    def mirrored(self, normal) -> "WaveSystem":
        from .geometry import reflection
        return self.transformed(reflection(self.n, normal))

    def with_relative_phases(self, thetas) -> "WaveSystem":
        thetas = np.broadcast_to(np.asarray(thetas, dtype=float), (self.N,))
        return WaveSystem([v.with_phase(th) for v, th in zip(self.volumes, thetas)], self.relative_envelope)

    def subsystem(self, members) -> "WaveSystem":
        return WaveSystem([self.volumes[a] for a in members], self.relative_envelope)
