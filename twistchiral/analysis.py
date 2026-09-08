"""Characterisation of the twist-chirality structure of interfaces."""
from __future__ import annotations

import numpy as np

from .exterior import ExteriorAlgebra
from .interface import log_amplitude_ratio, sample_bulk, sample_interface
from .qgt import chirality_ladder, ladder_relative_norms, quantum_geometry, rotation_planes
from .waves import WaveSystem


# ----------------------------------------------------------------------------
# evaluation helpers
# ----------------------------------------------------------------------------
def geometry_at(system: WaveSystem, X, t: float = 0.0, ea: ExteriorAlgebra | None = None,
                max_degree: int | None = None) -> dict:
    """Quantum geometry + chirality ladder at the points ``X``."""
    ea = ea or ExteriorAlgebra(system.n)
    Psi, dPsi = system.evaluate(X, t)
    q = quantum_geometry(Psi, dPsi)
    ladder = chirality_ladder(ea, q["A"], q["F"], max_degree)
    q["ladder"] = ladder
    q["norms"] = {p: ea.norm(c) for p, c in ladder.items()}
    q["relative"] = ladder_relative_norms(ea, ladder)
    q["Psi"], q["dPsi"], q["X"], q["ea"] = Psi, dPsi, np.asarray(X), ea
    return q


def analytic_two_volume_chirality(system: WaveSystem, X, t: float = 0.0, a: int = 0, b: int = 1,
                                  ea: ExteriorAlgebra | None = None) -> dict:
    """Closed form for a pair of volumes:

        F  = (V^2 / 2) grad ln(A_b/A_a) ^ (k_b - k_a)
        C3 = (V^2 / 2) k_a ^ k_b ^ grad ln(A_a/A_b)

    with ``V`` the fringe visibility and ``k_a, k_b`` the local wavevectors.
    Exact wherever both fields are non-zero (the local wavevector is then
    well defined).
    """
    ea = ea or ExteriorAlgebra(system.n)
    rel = getattr(system, "relative_envelope", False)
    ka = system[a].local_wavevector(X, t, relative_envelope=rel)
    kb = system[b].local_wavevector(X, t, relative_envelope=rel)
    h, gh = log_amplitude_ratio(system, X, a, b, t)   # h = 2 ln(A_a/A_b)
    grad_ln_ratio = 0.5 * gh                           # grad ln(A_a/A_b)
    V2 = 1.0 / np.cosh(0.5 * h) ** 2
    F = 0.5 * V2[:, None] * ea.wedge(-grad_ln_ratio, 1, kb - ka, 1)
    C3 = 0.5 * V2[:, None] * ea.wedge_vectors(ka, kb, grad_ln_ratio)
    return {"F": F, "C3": C3, "V2": V2, "k_a": ka, "k_b": kb}


def screw_prediction(system: WaveSystem, X, t: float = 0.0, a: int = 0, b: int = 1,
                     ea: ExteriorAlgebra | None = None) -> np.ndarray:
    """Chirality of two single-plane-wave Gaussian volumes of equal width:

        A ^ F = -(V^2 / (2 sigma^2)) (k_a ^ k_b) ^ d,   d = x_b - x_a.

    The 3-vector ``(k_a ^ k_b) ^ d`` is the *screw chirality* of the
    configuration (rotate ``k_a`` into ``k_b`` while advancing from volume
    ``a`` to volume ``b``).  With the standard orientation of the Berry
    curvature the Chern-Simons density carries the opposite sign; the
    magnitude and the oriented 3-plane are what matter.
    """
    ea = ea or ExteriorAlgebra(system.n)
    va, vb = system[a], system[b]
    if va.m != 1 or vb.m != 1:
        raise ValueError("screw prediction needs single-plane-wave volumes")
    sigma = va.sigma()
    d = vb.center_at(t) - va.center_at(t)
    h, _ = log_amplitude_ratio(system, X, a, b, t)
    V2 = 1.0 / np.cosh(0.5 * h) ** 2
    screw = ea.wedge_vectors(va.k, vb.k, d[None])
    return -0.5 * V2[:, None] / sigma**2 * screw


# ----------------------------------------------------------------------------
# statistics of chirality directions
# ----------------------------------------------------------------------------
def direction_spectrum(vectors: np.ndarray, weights=None, eps: float = 1e-300) -> dict:
    """Second-moment spectrum of the *directions* of a vector field sample.

    Unit vectors ``u_i`` are formed, then the eigenvalues of ``M = sum w_i u_i
    u_i^T / sum w_i`` are returned (they sum to one).  The participation
    ratio ``1 / sum lambda^2`` is the *effective dimension* of the set of
    directions: 1 if all vectors are (anti)parallel -- a binary chirality --
    and ``m`` for an isotropic cloud in ``R^m``.  The centred covariance
    spectrum measures the dimension of the *variation* instead.
    """
    v = np.asarray(vectors, dtype=float)
    nrm = np.linalg.norm(v, axis=-1)
    keep = nrm > eps
    v = v[keep] / nrm[keep][:, None]
    if weights is None:
        w = np.ones(len(v))
    else:
        w = np.asarray(weights, dtype=float)[keep]
    w = w / max(w.sum(), eps)
    M = np.einsum("p,pi,pj->ij", w, v, v)
    lam = np.sort(np.linalg.eigvalsh(M))[::-1]
    lam = np.clip(lam, 0, None)
    mean = np.einsum("p,pi->i", w, v)
    C = M - np.outer(mean, mean)
    mu = np.clip(np.sort(np.linalg.eigvalsh(C))[::-1], 0, None)
    return {
        "eigenvalues": lam,
        "effective_dimension": 1.0 / max((lam**2).sum(), eps),
        "centered_eigenvalues": mu,
        "variation_dimension": (mu.sum() ** 2) / max((mu**2).sum(), eps),
        "mean_direction": mean,
        "alignment": float(np.linalg.norm(mean)),
        "count": int(len(v)),
    }


def sign_balance(values: np.ndarray) -> dict:
    """Fractions of positive / negative values (for pseudoscalar chirality)."""
    v = np.asarray(values)
    pos = float((v > 0).mean())
    return {"positive": pos, "negative": 1.0 - pos, "mean_sign": float(np.sign(v).mean())}


# ----------------------------------------------------------------------------
# localisation around an interface
# ----------------------------------------------------------------------------
def localization_profile(system: WaveSystem, a: int = 0, b: int = 1, count: int = 40000, rng=None,
                         t: float = 0.0, bins: int = 41, h_max: float = 8.0, pad: float = 1.5,
                         weighted: bool = True) -> dict:
    """Mean chirality densities as a function of the log-amplitude ratio ``h``.

    ``h = ln(|psi_a|^2/|psi_b|^2)`` is the natural transverse coordinate of
    the interface (the interface is ``h = 0``).  For a pair the prediction is
    ``|F|, |A^F| ~ sech^2(h/2)``.
    """
    ea = ExteriorAlgebra(system.n)
    X = sample_bulk(system, count, rng, t, pad)
    q = geometry_at(system, X, t, ea)
    h, _ = log_amplitude_ratio(system, X, a, b, t)
    edges = np.linspace(-h_max, h_max, bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(h, edges) - 1
    out = {"h": centers, "profiles": {}, "weighted_profiles": {}, "median_profiles": {}}
    for p, nrm in q["norms"].items():
        prof = np.full(bins, np.nan)
        wprof = np.full(bins, np.nan)
        mprof = np.full(bins, np.nan)
        for i in range(bins):
            sel = idx == i
            if sel.any():
                prof[i] = nrm[sel].mean()
                wprof[i] = (nrm[sel] * q["rho"][sel]).mean()
                mprof[i] = np.median(nrm[sel])
        out["profiles"][p] = prof
        out["weighted_profiles"][p] = wprof
        out["median_profiles"][p] = mprof
    out["sech2"] = 1.0 / np.cosh(0.5 * centers) ** 2
    out["samples"] = {"h": h, "norms": q["norms"], "rho": q["rho"]}
    return out


# ----------------------------------------------------------------------------
# invariance tests
# ----------------------------------------------------------------------------
def relative_phase_invariance(system: WaveSystem, X, t: float = 0.0, thetas=(0.7, 2.1, 4.4)) -> dict:
    """Maximum relative change of ``F`` and ``A^F`` when the volumes acquire
    arbitrary constant relative phases.  The geometry is exactly invariant;
    the returned numbers should be at machine precision."""
    ea = ExteriorAlgebra(system.n)
    base = geometry_at(system, X, t, ea, max_degree=3)
    worst = {"F": 0.0, "C3": 0.0, "intensity": 0.0}
    I0 = np.abs(system.total_field(X, t)) ** 2
    rng = np.random.default_rng(0)
    for th in thetas:
        phases = rng.uniform(0, 2 * np.pi, system.N) * 0 + np.arange(system.N) * th
        shifted = system.with_relative_phases(phases)
        q = geometry_at(shifted, X, t, ea, max_degree=3)
        worst["F"] = max(worst["F"], np.abs(q["F"] - base["F"]).max() / (np.abs(base["F"]).max() + 1e-300))
        if 3 in base["ladder"]:
            worst["C3"] = max(worst["C3"], np.abs(q["ladder"][3] - base["ladder"][3]).max() / (np.abs(base["ladder"][3]).max() + 1e-300))
        I1 = np.abs(shifted.total_field(X, t)) ** 2
        worst["intensity"] = max(worst["intensity"], np.abs(I1 - I0).max() / (I0.max() + 1e-300))
    return worst


def mirror_transform_of_form(ea: ExteriorAlgebra, R: np.ndarray, comp: np.ndarray, p: int) -> np.ndarray:
    """Push a p-form field forward through the linear map ``R`` (pointwise)."""
    # exterior power of R acting on components: (R^p)_{IJ} = det R[I, J]
    n = ea.n
    basis = ea.basis[p]
    Rp = np.zeros((len(basis), len(basis)))
    R = np.asarray(R, dtype=float)
    for i, I in enumerate(basis):
        for j, J in enumerate(basis):
            Rp[i, j] = np.linalg.det(R[np.ix_(I, J)])
    return comp @ Rp.T


def mirror_comparison(system: WaveSystem, normal, X, t: float = 0.0) -> dict:
    """Compare the chirality of the mirror-image system with the pushed-forward
    chirality of the original.

    For the 3-form ``C3`` the two agree exactly (it is a tensor); the point is
    that in n = 3 the pseudoscalar flips sign under the mirror, whereas in
    n >= 4 the mirror rotates the chirality direction instead of negating it.
    """
    from .geometry import reflection
    ea = ExteriorAlgebra(system.n)
    Rm = reflection(system.n, normal)
    mirrored = system.mirrored(normal)
    Xm = X @ Rm.T
    q0 = geometry_at(system, X, t, ea, max_degree=3)
    q1 = geometry_at(mirrored, Xm, t, ea, max_degree=3)
    C0, C1 = q0["ladder"][3], q1["ladder"][3]
    pushed = mirror_transform_of_form(ea, Rm, C0, 3)
    out = {
        "tensor_error": float(np.abs(C1 - pushed).max() / (np.abs(C0).max() + 1e-300)),
        "C_original": C0, "C_mirror": C1,
    }
    if system.n == 3:
        out["sign_flip_error"] = float(np.abs(C1 + C0).max() / (np.abs(C0).max() + 1e-300))
    else:
        u0, u1 = ea.unit(C0), ea.unit(C1)
        cos = (u0 * u1).sum(-1)
        out["direction_cosines"] = cos
        out["negated_fraction"] = float((cos < -0.999).mean())
    return out


# ----------------------------------------------------------------------------
# rank / ladder scans
# ----------------------------------------------------------------------------
def ladder_existence(system: WaveSystem, X, t: float = 0.0, threshold: float = 1e-6) -> dict:
    """Which rungs of the chirality ladder are non-zero at the points ``X``.

    Returns the median relative norm of each rung and the highest degree whose
    median relative norm exceeds ``threshold``."""
    ea = ExteriorAlgebra(system.n)
    q = geometry_at(system, X, t, ea)
    med = {p: float(np.median(r)) for p, r in q["relative"].items()}
    alive = [p for p, m in med.items() if m > threshold]
    return {"median_relative": med, "max_degree": max(alive) if alive else 0,
            "predicted_max_degree": min(system.n, 2 * system.N - 1)}


def twist_rank_map(system: WaveSystem, X, t: float = 0.0):
    """Rotation rates of the twist form at the points ``X`` (descending)."""
    Psi, dPsi = system.evaluate(X, t)
    q = quantum_geometry(Psi, dPsi)
    rates, frames = rotation_planes(q["F"])
    return rates, frames, q


def interface_chirality_summary(system: WaveSystem, a: int = 0, b: int = 1, count: int = 6000, rng=None,
                                t: float = 0.0, pad: float = 2.0) -> dict:
    """One-call characterisation of the interface between two volumes."""
    ea = ExteriorAlgebra(system.n)
    samp = sample_interface(system, a, b, count, rng, t, pad)
    X = samp["X"]
    q = geometry_at(system, X, t, ea)
    out = {"X": X, "normal": samp["normal"], "geometry": q, "count": len(X)}
    if 3 in q["ladder"]:
        C3 = q["ladder"][3]
        out["C3"] = C3
        out["spectrum"] = direction_spectrum(C3, weights=q["rho"])
        if system.n == 3:
            out["signs"] = sign_balance(C3[:, 0])
    rates, _ = rotation_planes(q["F"])
    out["rates"] = rates
    return out
