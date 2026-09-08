"""Generic-n machinery: twist and chirality without exterior components.

For N interfering volumes the projector onto the complement of Psi in C^N has
rank N - 1.  Writing it as ``sum_c chi_c chi_c^+`` with an orthonormal basis
``chi_c`` of that complement, the quantum geometric tensor factorises,

    Q = Z^+ Z / rho,      Z_{c,i} = chi_c^+ d_i Psi   (an (N-1) x n complex matrix),

so with ``a_c = Re Z_c`` and ``b_c = Im Z_c``

    F = (2 / rho) sum_c a_c ^ b_c .

The twist therefore lives in the subspace ``W = span{a_c, b_c}`` of dimension
at most ``2(N-1)`` whatever ``n`` is, and its rotation planes come from a
``2(N-1) x 2(N-1)`` eigenproblem.  Writing ``F = sum_j lambda_j v_j ^ u_j`` in
orthonormal planes, every rung of the chirality ladder has an exact norm

    |F^k|^2       = (k!)^2 e_k(lambda^2)
    |A ^ F^k|^2   = (k!)^2 [ |A_res|^2 e_k(lambda^2) + sum_j p_j e_k^{(-j)}(lambda^2) ]

with ``e_k`` the elementary symmetric polynomials, ``p_j`` the squared
projection of ``A`` on plane ``j``, ``A_res`` the residual of ``A`` outside
all planes and ``e^{(-j)}`` the leave-one-out polynomials.  Inner products of chiralities at different points are Gram
determinants of the spanning vectors, which gives the direction spectrum by a
kernel (P x P) eigenproblem.  Everything is O(n) per point.
"""
from __future__ import annotations

from math import factorial

import numpy as np

from .qgt import rotation_planes


# ----------------------------------------------------------------------------
# factorised geometry
# ----------------------------------------------------------------------------
def complement_basis(Psi: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the orthogonal complement of ``Psi`` in C^N.

    Returns ``chi`` of shape ``(P, N, N-1)`` (Householder construction)."""
    Psi = np.asarray(Psi, dtype=complex)
    P, N = Psi.shape
    norm = np.linalg.norm(Psi, axis=1)
    p0 = np.abs(Psi[:, 0])
    phase = np.where(p0 > 0, Psi[:, 0] / np.where(p0 > 0, p0, 1.0), 1.0)
    v = Psi.copy()
    v[:, 0] += phase * norm
    vv = (np.abs(v) ** 2).sum(1)
    H = np.eye(N, dtype=complex)[None] - 2.0 * v[:, :, None] * np.conj(v)[:, None, :] / np.maximum(vv, 1e-300)[:, None, None]
    return H[:, :, 1:]


def factored_geometry(Psi: np.ndarray, dPsi: np.ndarray, floor: float = 1e-290) -> dict:
    """Berry connection and the factorised twist ``F = sum_j lambda_j v_j ^ u_j``.

    Returns ``rho (P,)``, ``A (P, n)``, ``rates (P, r)`` (descending, r = N-1),
    ``U, V (P, r, n)`` orthonormal plane frames with ``F u = rate v``,
    ``projA (P, r)`` the squared projections of ``A`` on each plane, ``A2 (P,)``,
    and the raw factors ``a, b (P, r, n)``.
    """
    Psi = np.asarray(Psi)
    dPsi = np.asarray(dPsi)
    P, N = Psi.shape
    n = dPsi.shape[1]
    rho = (np.abs(Psi) ** 2).sum(1)
    safe = rho > floor
    inv = np.where(safe, 1.0 / np.where(safe, rho, 1.0), 0.0)
    A = np.einsum("pa,pia->pi", Psi.conj(), dPsi).imag * inv[:, None]
    A2 = (A * A).sum(1)
    r = N - 1
    if r == 0:
        z = np.zeros((P, 0, n))
        return {"rho": rho, "A": A, "A2": A2, "rates": np.zeros((P, 0)), "U": z, "V": z, "projA": np.zeros((P, 0)),
                "Ares2": A2, "a": z, "b": z}
    chi = complement_basis(Psi)                                   # (P, N, r)
    Z = np.einsum("pac,pia->pci", chi.conj(), dPsi)               # (P, r, n)
    a, b = Z.real, Z.imag
    w = 2 * r
    B = np.empty((P, n, w))
    B[:, :, 0::2] = np.transpose(a, (0, 2, 1))
    B[:, :, 1::2] = np.transpose(b, (0, 2, 1))
    Qw, Rw = np.linalg.qr(B)                                      # (P, n, w), (P, w, w)
    ahat = Rw[:, :, 0::2]                                          # (P, w, r): coordinates of a_c in the Q basis
    bhat = Rw[:, :, 1::2]
    FW = 2.0 * inv[:, None, None] * (np.einsum("pwc,pvc->pwv", ahat, bhat) - np.einsum("pwc,pvc->pwv", bhat, ahat))
    rates, frames = rotation_planes(FW)                            # (P, r), (P, r, w, 2)
    U = np.einsum("pnw,pjw->pjn", Qw, frames[..., 0])
    V = np.einsum("pnw,pjw->pjn", Qw, frames[..., 1])
    cu = np.einsum("pjn,pn->pj", U, A)
    cv = np.einsum("pjn,pn->pj", V, A)
    projA = cu**2 + cv**2
    Ares = A - np.einsum("pj,pjn->pn", cu, U) - np.einsum("pj,pjn->pn", cv, V)
    Ares2 = (Ares * Ares).sum(1)
    return {"rho": rho, "A": A, "A2": A2, "rates": rates, "U": U, "V": V, "projA": projA, "Ares2": Ares2,
            "a": a, "b": b}


def reconstruct_F(geo: dict) -> np.ndarray:
    """Antisymmetric matrix field from the factorisation (for checks)."""
    rates, U, V = geo["rates"], geo["U"], geo["V"]
    return np.einsum("pj,pjn,pjm->pnm", rates, V, U) - np.einsum("pj,pjn,pjm->pnm", rates, U, V)


# ----------------------------------------------------------------------------
# ladder norms from the rotation rates
# ----------------------------------------------------------------------------
def elementary_symmetric(x: np.ndarray, kmax: int) -> np.ndarray:
    """``e_k(x_1..x_r)`` for ``k = 0..kmax``; ``x`` has shape ``(P, r)``."""
    P, r = x.shape
    e = np.zeros((P, kmax + 1))
    e[:, 0] = 1.0
    for j in range(r):
        for k in range(min(kmax, j + 1), 0, -1):
            e[:, k] += x[:, j] * e[:, k - 1]
    return e


def leave_one_out_symmetric(x: np.ndarray, m: int) -> np.ndarray:
    """``e_m`` of the variables with ``x_j`` removed, for every ``j``: shape ``(P, r)``.

    Computed with the forward recurrence on the remaining variables (all
    terms non-negative for ``x >= 0``); the division-free downdate
    ``e_m - x_j e_{m-1}`` cancels catastrophically when ``x_j`` dominates."""
    P, r = x.shape
    out = np.zeros((P, r))
    for j in range(r):
        others = np.delete(x, j, axis=1)
        out[:, j] = elementary_symmetric(others, m)[:, m] if others.shape[1] >= m else 0.0
    return out


def ladder_norms_factored(geo: dict, n: int, max_degree: int | None = None) -> dict:
    """Pointwise norms of ``A, F, A^F, F^F, ...`` from the factorisation.

    Uses the manifestly non-negative form

        |A ^ F^k|^2 = (k!)^2 [ |A_res|^2 e_k + sum_j p_j e_k^{(-j)} ],

    where ``A_res`` is the residual of ``A`` outside all twist planes.
    """
    lam2 = geo["rates"] ** 2
    Ares2, pA = geo["Ares2"], geo["projA"]
    r = lam2.shape[1]
    top = n if max_degree is None else min(n, max_degree)
    kmax = min(r, n // 2)
    e = elementary_symmetric(lam2, kmax)
    norms = {1: np.sqrt(geo["A2"])} if top >= 1 else {}
    for k in range(1, kmax + 1):
        fk = float(factorial(k))
        if 2 * k <= top:
            norms[2 * k] = fk * np.sqrt(np.maximum(e[:, k], 0.0))
        if 2 * k + 1 <= top:
            eloo = leave_one_out_symmetric(lam2, k)
            val = Ares2 * e[:, k] + (pA * eloo).sum(1)
            norms[2 * k + 1] = fk * np.sqrt(np.maximum(val, 0.0))
    return norms


def predicted_max_degree(n: int, N: int) -> int:
    return min(n, 2 * N - 1)


# ----------------------------------------------------------------------------
# kernels: inner products of chiralities without components
# ----------------------------------------------------------------------------
def _det3(G):
    return (G[0][0] * (G[1][1] * G[2][2] - G[1][2] * G[2][1])
            - G[0][1] * (G[1][0] * G[2][2] - G[1][2] * G[2][0])
            + G[0][2] * (G[1][0] * G[2][1] - G[1][1] * G[2][0]))


def chirality3_kernel(geo1: dict, geo2: dict | None = None, reflect_normal=None) -> np.ndarray:
    """``K[p, q] = <C3_p, C3_q>`` for ``C3 = A ^ F = sum_j lambda_j A ^ v_j ^ u_j``.

    With ``reflect_normal = m`` the first argument is reflected through the
    hyperplane orthogonal to ``m`` first (``<R C3_p, C3_q>``)."""
    geo2 = geo1 if geo2 is None else geo2
    A1, U1, V1, L1 = geo1["A"], geo1["U"], geo1["V"], geo1["rates"]
    A2, U2, V2, L2 = geo2["A"], geo2["U"], geo2["V"], geo2["rates"]
    m = None if reflect_normal is None else np.asarray(reflect_normal, dtype=float) / np.linalg.norm(reflect_normal)

    def dot(X, Y):
        D = X @ Y.T
        if m is not None:
            D = D - 2.0 * np.outer(X @ m, Y @ m)
        return D

    K = np.zeros((A1.shape[0], A2.shape[0]))
    for j in range(L1.shape[1]):
        X = (A1, V1[:, j], U1[:, j])
        for j2 in range(L2.shape[1]):
            Y = (A2, V2[:, j2], U2[:, j2])
            G = [[dot(X[a], Y[b]) for b in range(3)] for a in range(3)]
            K += L1[:, j][:, None] * L2[:, j2][None, :] * _det3(G)
    return K


def kernel_direction_spectrum(K: np.ndarray, weights=None, eps: float = 1e-300) -> dict:
    """Second-moment spectrum of unit vectors given only their Gram matrix."""
    d = np.sqrt(np.clip(np.diag(K), 0, None))
    keep = d > eps * max(d.max(), eps)
    K = K[np.ix_(keep, keep)]
    d = d[keep]
    Khat = K / np.outer(d, d)
    w = np.ones(len(d)) if weights is None else np.asarray(weights, dtype=float)[keep]
    w = w / w.sum()
    sw = np.sqrt(w)
    M = sw[:, None] * Khat * sw[None, :]
    mu = np.clip(np.linalg.eigvalsh(M), 0, None)[::-1]
    mu = mu / max(mu.sum(), eps)
    Hc = np.eye(len(d)) - np.outer(np.ones(len(d)), w)
    Kc = Hc @ Khat @ Hc.T
    Mc = sw[:, None] * Kc * sw[None, :]
    nu = np.clip(np.linalg.eigvalsh(Mc), 0, None)[::-1]
    # Population estimate without the sample-size ceiling: for independent
    # samples E[(c_p . c_q)^2] = tr(M^2) = sum mu^2, so 1 / mean_{p != q} Khat^2
    # estimates the participation ratio of the population second-moment matrix.
    off = Khat[~np.eye(len(d), dtype=bool)]
    pop = 1.0 / max(float(np.mean(off**2)), eps) if off.size else 1.0
    return {"eigenvalues": mu, "effective_dimension": 1.0 / max((mu**2).sum(), eps),
            "centered_eigenvalues": nu, "variation_dimension": (nu.sum() ** 2) / max((nu**2).sum(), eps),
            "population_effective_dimension": pop, "count": int(len(d))}


def mirror_cosines(geo: dict, normal) -> np.ndarray:
    """Cosine between the mirror image of the chirality 3-vector and itself, per point."""
    A, U, V, L = geo["A"], geo["U"], geo["V"], geo["rates"]
    m = np.asarray(normal, dtype=float)
    m = m / np.linalg.norm(m)
    P = A.shape[0]
    num = np.zeros(P)
    den = np.zeros(P)
    for j in range(L.shape[1]):
        X = (A, V[:, j], U[:, j])
        for j2 in range(L.shape[1]):
            Y = (A, V[:, j2], U[:, j2])
            G = [[(X[a] * Y[b]).sum(1) for b in range(3)] for a in range(3)]
            Gm = [[G[a][b] - 2.0 * (X[a] @ m) * (Y[b] @ m) for b in range(3)] for a in range(3)]
            num += L[:, j] * L[:, j2] * _det3(Gm)
            den += L[:, j] * L[:, j2] * _det3(G)
    return num / np.maximum(den, 1e-300)


def expected_mirror_cosine(p: int, n: int) -> float:
    """Mean cosine between a degree-p form and its image under a random mirror: ``1 - 2p/n``."""
    return 1.0 - 2.0 * p / n


# ----------------------------------------------------------------------------
# convenience
# ----------------------------------------------------------------------------
def geometry_factored(system, X, t: float = 0.0, max_degree: int | None = None) -> dict:
    """Factorised geometry + ladder norms at the points ``X`` (any n)."""
    Psi, dPsi = system.evaluate(X, t)
    geo = factored_geometry(Psi, dPsi)
    geo["norms"] = ladder_norms_factored(geo, system.n, max_degree)
    geo["X"] = np.asarray(X)
    return geo


def screw_saturation(k1: np.ndarray, k2: np.ndarray, d: np.ndarray) -> np.ndarray:
    """``|k1 ^ k2 ^ d| / (|k1| |k2| |d|)`` via the 3x3 Gram determinant (any n)."""
    X = np.stack([k1, k2, d], axis=-2)                     # (..., 3, n)
    G = X @ np.swapaxes(X, -1, -2)
    det = np.linalg.det(G)
    nrm = np.prod(np.linalg.norm(X, axis=-1), axis=-1)
    return np.sqrt(np.clip(det, 0, None)) / np.maximum(nrm, 1e-300)


# ----------------------------------------------------------------------------
# large-n helpers
# ----------------------------------------------------------------------------
def geometry_factored_batched(system, X, t: float = 0.0, max_degree: int | None = None, batch: int = 1500,
                              keep_frames: bool = True) -> dict:
    """Same as :func:`geometry_factored`, evaluated in batches to bound memory."""
    X = np.asarray(X)
    parts = []
    for s in range(0, len(X), batch):
        Psi, dPsi = system.evaluate(X[s:s + batch], t)
        geo = factored_geometry(Psi, dPsi)
        geo["norms"] = ladder_norms_factored(geo, system.n, max_degree)
        geo["intensity"] = np.abs(Psi.sum(1)) ** 2
        if not keep_frames:
            for key in ("U", "V", "a", "b", "A"):
                geo.pop(key, None)
        parts.append(geo)
    out = {}
    for key in parts[0]:
        if key == "norms":
            out[key] = {p: np.concatenate([g["norms"][p] for g in parts]) for p in parts[0]["norms"]}
        else:
            out[key] = np.concatenate([g[key] for g in parts])
    out["X"] = X
    return out


def random_frame_image(K: np.ndarray, rng=None) -> np.ndarray:
    """Image of the wavevector set ``K`` under a Haar-random rotation, without
    forming the ``n x n`` matrix: ``R K^T = (R Q_K) R_K`` with ``R Q_K`` a
    random orthonormal frame."""
    rng = np.random.default_rng(rng)
    K = np.atleast_2d(np.asarray(K, dtype=float))
    m, n = K.shape
    Qk, Rk = np.linalg.qr(K.T)                      # (n, min(m,n)), (min(m,n), m)
    G = rng.standard_normal((n, Qk.shape[1]))
    Qr, Rr = np.linalg.qr(G)
    Qr = Qr * np.sign(np.diag(Rr))[None, :]
    return (Qr @ Rk).T


def seed_near(center, count: int, n: int, rng=None, spread: float | None = None) -> np.ndarray:
    """Gaussian seeds around ``center`` with a per-coordinate spread small
    enough that the envelopes do not underflow in high dimension."""
    rng = np.random.default_rng(rng)
    if spread is None:
        spread = min(1.0, np.sqrt(300.0 / n))
    return np.asarray(center, dtype=float)[None, :] + spread * rng.standard_normal((count, n))
