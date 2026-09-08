"""Ready-made configurations of interfering volumes."""
from __future__ import annotations

import numpy as np

from . import geometry as geo
from .waves import WaveSystem, WaveVolume

_SETS = {
    "single": lambda n, rng: np.eye(n)[:1],
    "simplex": lambda n, rng: geo.simplex(n),
    "cross": lambda n, rng: geo.cross_polytope(n),
    "hypercube": lambda n, rng: geo.hypercube(n),
    "ring": lambda n, rng: geo.ring(n, 5),
}


def wave_set(n: int, kind="simplex", k_scale: float = 3.0, rng=None, m: int | None = None) -> np.ndarray:
    """A wavevector set: a named polytope, ``'random'`` (m vectors) or an array."""
    rng = np.random.default_rng(rng)
    if isinstance(kind, str):
        if kind == "random":
            K = geo.random_sphere(n, m or 2 * n, rng)
        elif kind in _SETS:
            K = _SETS[kind](n, rng)
        else:
            raise ValueError(f"unknown wave set {kind!r}")
    else:
        K = np.atleast_2d(np.asarray(kind, dtype=float))
    return k_scale * K


def _amplitudes(spec, m: int, rng) -> np.ndarray:
    """Amplitude spec: None (equal), 'random', 'carrier' (1 + weak sidebands),
    a float (sideband weight for the carrier form) or an explicit array."""
    if spec is None:
        return np.ones(m, dtype=complex)
    if isinstance(spec, str) and spec == "random":
        return rng.standard_normal(m) + 1j * rng.standard_normal(m)
    if isinstance(spec, str) and spec == "carrier":
        spec = 0.3
    if np.isscalar(spec):
        amp = np.full(m, float(spec), dtype=complex)
        amp[0] = 1.0
        return amp
    return np.asarray(spec, dtype=complex)


def twisted_pair(n: int, kind="simplex", k_scale: float = 3.0, angles=None, rotation=None,
                 displacement=None, width: float = 1.0, amplitudes=None, rng=None, m: int | None = None,
                 relative_phase: float = 0.0, dispersion="linear") -> WaveSystem:
    """Two volumes: the second is the first *twisted* by a rotation (given as
    multi-plane angles or a matrix) and displaced by ``displacement``.

    The pair is centred on the origin: volume 1 at ``-d/2``, volume 2 at ``+d/2``.
    """
    rng = np.random.default_rng(rng)
    n = int(n)
    K = wave_set(n, kind, k_scale, rng, m)
    amp = _amplitudes(amplitudes, K.shape[0], rng)
    if rotation is None:
        rotation = geo.rotation_from_angles(n, angles if angles is not None else [0.6])
    d = np.zeros(n) if displacement is None else np.asarray(displacement, dtype=float)
    v1 = WaveVolume(k=K, center=-0.5 * d, width=width, amplitudes=amp, dispersion=dispersion, name="volume 1")
    v2 = v1.rotated(rotation).translated(d).with_phase(relative_phase)
    v2.name = "volume 2"
    return WaveSystem([v1, v2])


def cluster(n: int, N: int, kind="simplex", k_scale: float = 3.0, spacing: float = 1.6, width: float = 1.0,
            rng=None, centers=None, rotations="random", amplitudes=None, m: int | None = None,
            velocities=None, dispersion="linear") -> WaveSystem:
    """``N`` volumes at the vertices of a regular simplex (or given centres),
    each carrying its own (randomly rotated) copy of the wave set."""
    rng = np.random.default_rng(rng)
    n = int(n)
    if centers is None:
        if N > n + 1:
            raise ValueError("a regular simplex in R^n has at most n + 1 vertices; pass centres explicitly")
        C = spacing * geo.simplex(n)[:N] if N > 1 else np.zeros((1, n))
        C = C - C.mean(0)
    else:
        C = np.asarray(centers, dtype=float)
    vols = []
    for a in range(N):
        K = wave_set(n, kind, k_scale, rng, m)
        if rotations == "random":
            R = geo.random_rotation(n, rng)
        elif rotations is None:
            R = np.eye(n)
        else:
            R = np.asarray(rotations[a])
        amp = _amplitudes(amplitudes if (amplitudes is None or isinstance(amplitudes, str) or np.isscalar(amplitudes))
                          else amplitudes[a], K.shape[0], rng)
        vel = None if velocities is None else np.asarray(velocities[a], dtype=float)
        vols.append(WaveVolume(k=K @ R.T, center=C[a], width=width, amplitudes=amp, velocity=vel,
                               dispersion=dispersion, name=f"volume {a + 1}"))
    return WaveSystem(vols)
