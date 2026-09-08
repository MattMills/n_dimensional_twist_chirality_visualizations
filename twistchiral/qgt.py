"""Quantum-geometric tensor of an interfering wave system.

Write the N interfering volumes as one spinor field ``Psi(x) in C^N``.  Its
projective class ``[Psi(x)] in CP^{N-1}`` encodes exactly the interference
data -- the relative amplitudes and relative phases of the volumes -- and is
blind to the overall amplitude and phase.  Pulling the Fubini-Study geometry
of ``CP^{N-1}`` back to R^n gives

* the Berry connection ``A_i = Im(Psi^+ d_i Psi) / rho``            (1-form),
* the quantum geometric tensor ``Q_ij = d_iPsi^+ P_perp d_jPsi / rho``,
* the quantum metric ``g = Re Q``  (how fast the interference texture changes),
* the Berry curvature ``F = 2 Im Q = dA``  -- the **twist** 2-form,

and from these the **chirality ladder** of Chern-Simons and Chern forms

    A,  F,  A^F,  F^F,  A^F^F,  F^F^F, ...

of degrees 1, 2, 3, 4, 5, 6, ...  The degree-p rung is an oriented
p-volume element (a p-vector) at every point of R^n.  Only when p = n is it a
pseudoscalar, i.e. a *binary* chirality; for p < n it is a genuinely
multi-component, direction-valued chirality.  Because ``F`` is pulled back
from a 2(N-1)-dimensional space its rank is at most ``2(N-1)``, so the ladder
of N simultaneously interfering volumes ends at degree ``min(n, 2N-1)``.
"""
from __future__ import annotations

import numpy as np

from .exterior import ExteriorAlgebra


def quantum_geometry(Psi: np.ndarray, dPsi: np.ndarray, floor: float = 0.0) -> dict:
    """Berry connection, quantum metric and curvature of a spinor field.

    Parameters
    ----------
    Psi : (P, N) complex
    dPsi : (P, n, N) complex, ``dPsi[p, i, a] = d_i psi_a(x_p)``
    floor : points with ``rho <= floor`` get zero geometry.

    Returns a dict with ``rho (P,)``, ``A (P, n)``, ``g (P, n, n)``,
    ``F (P, n, n)`` (antisymmetric), ``Q (P, n, n)`` (Hermitian) and
    ``A_raw = Im(Psi^+ dPsi)`` (the unnormalised current).
    """
    Psi = np.asarray(Psi)
    dPsi = np.asarray(dPsi)
    rho = (np.abs(Psi) ** 2).sum(-1)
    safe = rho > floor
    inv = np.where(safe, 1.0 / np.where(safe, rho, 1.0), 0.0)
    current = np.einsum("pa,pia->pi", Psi.conj(), dPsi)          # Psi^+ d_i Psi
    A = current.imag * inv[:, None]
    M = np.einsum("pia,pja->pij", dPsi.conj(), dPsi)               # d_iPsi^+ d_jPsi
    w = current.conj()                                             # d_iPsi^+ Psi
    Q = (M - w[:, :, None] * w.conj()[:, None, :] * inv[:, None, None]) * inv[:, None, None]
    g = Q.real
    F = 2.0 * Q.imag
    return {"rho": rho, "A": A, "g": g, "F": F, "Q": Q, "A_raw": current.imag}


def berry_connection_only(Psi: np.ndarray, dPsi: np.ndarray, floor: float = 0.0) -> np.ndarray:
    rho = (np.abs(Psi) ** 2).sum(-1)
    inv = np.where(rho > floor, 1.0 / np.where(rho > floor, rho, 1.0), 0.0)
    return np.einsum("pa,pia->pi", Psi.conj(), dPsi).imag * inv[:, None]


def chirality_ladder(ea: ExteriorAlgebra, A: np.ndarray, F: np.ndarray, max_degree: int | None = None) -> dict:
    """All Chern-Simons / Chern forms ``A, F, A^F, F^F, ...`` up to degree n.

    ``F`` may be given as an antisymmetric matrix field ``(P, n, n)`` or as
    2-form components ``(P, C(n,2))``.  Returns ``{degree: components}``.
    """
    n = ea.n
    F2 = F if F.shape[-1] == ea.dim(2) and F.ndim == 2 else ea.from_antisymmetric(F)
    top = n if max_degree is None else min(n, max_degree)
    forms = {}
    if top >= 1:
        forms[1] = np.asarray(A, dtype=float)
    if top >= 2:
        forms[2] = F2
    Fk, k = F2, 1
    while True:
        if 2 * k + 1 <= top:
            forms[2 * k + 1] = ea.wedge(A, 1, Fk, 2 * k)
        if 2 * k + 2 <= top:
            Fk = ea.wedge(Fk, 2 * k, F2, 2)
            forms[2 * k + 2] = Fk
            k += 1
        else:
            break
    return forms


def ladder_norms(ea: ExteriorAlgebra, ladder: dict) -> dict:
    """Pointwise norms of every rung."""
    return {p: ea.norm(c) for p, c in ladder.items()}


def ladder_relative_norms(ea: ExteriorAlgebra, ladder: dict, eps: float = 1e-300) -> dict:
    """Norm of each rung divided by the product of the norms of its factors.

    A rung that vanishes identically (because the curvature rank is too
    small) gives values at machine precision; a genuinely non-zero rung gives
    O(1) values.  This makes "does the degree-p chirality exist?" a
    dimensionless test.
    """
    nA = ea.norm(ladder[1]) if 1 in ladder else None
    nF = ea.norm(ladder[2]) if 2 in ladder else None
    out = {}
    for p, comp in ladder.items():
        if p == 1:
            out[p] = np.ones_like(nA)
            continue
        k = p // 2
        denom = np.maximum(nF, eps) ** k
        if p % 2 == 1:
            denom = denom * np.maximum(nA, eps)
        out[p] = ea.norm(comp) / denom
    return out


def rotation_planes(F: np.ndarray, tol: float = 0.0):
    """Rotation rates and oriented planes of the antisymmetric field ``F``.

    Returns ``rates (P, n//2)`` (descending) and ``frames (P, n//2, n, 2)``
    with ``F u = rate * v`` and ``F v = -rate * u`` for each ``(u, v)`` pair.
    """
    F = np.asarray(F, dtype=float)
    P, n, _ = F.shape
    S = -F @ F
    w, V = np.linalg.eigh(S)                      # ascending
    w = w[:, ::-1]
    V = V[:, :, ::-1]
    k = n // 2
    rates = np.zeros((P, k))
    frames = np.zeros((P, k, n, 2))
    used = np.zeros((P, n, 0))
    for j in range(k):
        # pick the j-th independent eigenvector: Gram-Schmidt against found planes
        u = V[:, :, 2 * j].copy()
        if used.shape[-1]:
            proj = np.einsum("pnm,pn->pm", used, u)
            u = u - np.einsum("pnm,pm->pn", used, proj)
        nu = np.linalg.norm(u, axis=-1)
        u = u / np.maximum(nu, 1e-300)[:, None]
        Fu = np.einsum("pij,pj->pi", F, u)
        rate = np.linalg.norm(Fu, axis=-1)
        v = Fu / np.maximum(rate, 1e-300)[:, None]
        rate = np.where(rate > tol, rate, 0.0)
        rates[:, j] = rate
        frames[:, j, :, 0] = u
        frames[:, j, :, 1] = v
        used = np.concatenate([used, u[:, :, None], v[:, :, None]], axis=-1)
    return rates, frames


def twist_rank(F: np.ndarray, rtol: float = 1e-6):
    """Number of rotation planes with rate above ``rtol`` times the largest."""
    rates, _ = rotation_planes(F)
    top = rates[:, :1]
    return (rates > rtol * np.maximum(top, 1e-300)).sum(-1), rates


def chirality_axis_4d(ea: ExteriorAlgebra, C3: np.ndarray) -> np.ndarray:
    """In four dimensions the chirality 3-form is dual to a vector.  Returns ``*C3``."""
    if ea.n != 4:
        raise ValueError("chirality axis vector only exists in n = 4")
    return ea.hodge(C3, 3)


def pairwise_decomposition(Psi: np.ndarray, dPsi: np.ndarray) -> dict:
    """Exact decomposition of the N-volume twist into pairwise twists.

    With ``rho_a = |psi_a|^2``, ``rho_ab = rho_a + rho_b`` and ``rho = sum_a rho_a``,

        F = sum_{a<b} (rho_ab / rho)^2 F_ab,

    where ``F_ab`` is the Berry curvature of the pair ``(psi_a, psi_b)`` on
    its own.  Since every ``F_ab`` has rank 2, ``F_ab ^ F_ab = 0`` and the
    second Chern form is purely made of cross terms between *different*
    pairs -- each involving at least three volumes:

        F ^ F = 2 sum_{(ab) < (cd)} w_ab w_cd F_ab ^ F_cd,   w_ab = (rho_ab / rho)^2.

    Returns the weights, the pair curvatures and the reconstruction.
    """
    Psi = np.asarray(Psi)
    dPsi = np.asarray(dPsi)
    N = Psi.shape[1]
    rho_a = np.abs(Psi) ** 2
    rho = rho_a.sum(-1)
    pairs = [(a, b) for a in range(N) for b in range(a + 1, N)]
    weights, F_pairs = {}, {}
    recon = np.zeros((Psi.shape[0], dPsi.shape[1], dPsi.shape[1]))
    for a, b in pairs:
        qab = quantum_geometry(Psi[:, [a, b]], dPsi[:, :, [a, b]])
        w = ((rho_a[:, a] + rho_a[:, b]) / np.maximum(rho, 1e-300)) ** 2
        weights[(a, b)] = w
        F_pairs[(a, b)] = qab["F"]
        recon += w[:, None, None] * qab["F"]
    return {"pairs": pairs, "weights": weights, "F_pairs": F_pairs, "F_reconstructed": recon}
