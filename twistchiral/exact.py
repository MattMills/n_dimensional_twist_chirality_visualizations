"""Exact integer twin of the library: no floating point anywhere.

Model
-----
Positions live on the lattice ``x = (pi/2) m`` with ``m`` an integer vector,
wavevectors are integer vectors ``K``, so every plane-wave phase is a power of
``i``: ``exp(i K . x) = i^(K . m)`` -- a Gaussian integer.  A *packet* is a
carrier plus side-bands with Gaussian-integer amplitudes; its beat pattern is
the envelope (period 4 in every direction, so the whole system lives on the
torus ``(Z/4Z)^n``).  A *twist* is a signed permutation matrix, the exact
rotations of the lattice.

Because the derivative of ``i^(K . m)`` with respect to the physical position
is ``i K`` times the wave, all first derivatives are Gaussian integers as
well, and every quantity of the continuum theory evaluated at a lattice point
is a rational function of Gaussian integers.  Clearing the projective
denominators,

    A~ = rho A          (integer vector),
    F~ = rho^2 F        (integer antisymmetric matrix),
    A~ ^ F~ = rho^3 A^F,   F~ ^ F~ = rho^4 F^F, ...

so the whole chirality ladder is made of Python integers.  Ranks are exact
(fraction-free Bareiss elimination), norms are kept squared (integers), and
statistics are exact rationals (``fractions.Fraction``) built from Gram
determinants.  Nothing in this module imports or produces a float.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product
from math import comb

import numpy as np

from .exterior import ExteriorAlgebra

I0 = np.int64  # only used for shapes / small index arrays, never for field values


# ----------------------------------------------------------------------------
# Gaussian-integer arrays: a pair (re, im) of object arrays holding Python ints
# ----------------------------------------------------------------------------
def ints(a) -> np.ndarray:
    """Object array of Python ints."""
    arr = np.array(a, dtype=object)
    it = np.nditer(arr, flags=["refs_ok", "multi_index"], op_flags=["readwrite"])
    for v in it:
        v[...] = int(v.item())
    return arr


def zeros(shape) -> np.ndarray:
    out = np.empty(shape, dtype=object)
    out.fill(0)
    return out


def gmul(a, b):
    ar, ai = a
    br, bi = b
    return ar * br - ai * bi, ar * bi + ai * br


def gconj(a):
    return a[0], -a[1]


def gadd(a, b):
    return a[0] + b[0], a[1] + b[1]


def gscale(a, s):
    return a[0] * s, a[1] * s


def gabs2(a):
    return a[0] * a[0] + a[1] * a[1]


def ipow(e) -> tuple:
    """``i**e`` for an object array of integers ``e``: returns (re, im)."""
    e = np.asarray(e, dtype=object)
    r = zeros(e.shape)
    im = zeros(e.shape)
    flat_e = e.reshape(-1)
    fr, fi = r.reshape(-1), im.reshape(-1)
    for idx in range(flat_e.shape[0]):
        k = int(flat_e[idx]) % 4
        if k == 0:
            fr[idx] = 1
        elif k == 1:
            fi[idx] = 1
        elif k == 2:
            fr[idx] = -1
        else:
            fi[idx] = -1
    return r, im


# ----------------------------------------------------------------------------
# packets and systems on the lattice
# ----------------------------------------------------------------------------
class Packet:
    """``psi(m) = sum_s c_s i^(K_s . (m - mu))`` with integer ``K_s``, Gaussian-
    integer ``c_s = (re, im)`` and integer centre ``mu``."""

    def __init__(self, K, amplitudes, center=None, name: str = ""):
        self.K = ints(np.atleast_2d(np.array(K, dtype=object)))           # (m, n)
        self.n = self.K.shape[1]
        self.m = self.K.shape[0]
        self.c = [(int(a), int(b)) for a, b in amplitudes]
        if len(self.c) != self.m:
            raise ValueError("one amplitude per wavevector")
        self.center = ints(np.zeros(self.n, dtype=object)) if center is None else ints(np.array(center, dtype=object))
        self.name = name

    @classmethod
    def carrier_with_sidebands(cls, carrier, sideband_offsets, carrier_amp=(1, 0), sideband_amp=(1, 0), center=None):
        K = [list(carrier)] + [[c + o for c, o in zip(carrier, off)] for off in sideband_offsets]
        amps = [carrier_amp] + [sideband_amp] * len(sideband_offsets)
        return cls(K, amps, center)

    def evaluate(self, M):
        """Field ``psi`` (re, im) of shape (P,) and ``T`` (re, im) of shape (P, n)
        with ``d_j psi = i T_j`` (physical derivative)."""
        M = ints(np.atleast_2d(np.array(M, dtype=object)))
        P, n = M.shape
        Y = M - self.center[None, :]
        psi = (zeros(P), zeros(P))
        T = (zeros((P, n)), zeros((P, n)))
        for s in range(self.m):
            e = (Y * self.K[s][None, :]).sum(axis=1)            # K_s . (m - mu)
            ph = ipow(e)
            term = gmul(ph, (np.full(P, self.c[s][0], dtype=object), np.full(P, self.c[s][1], dtype=object)))
            psi = gadd(psi, term)
            for j in range(n):
                kj = self.K[s][j]
                if kj != 0:
                    T[0][:, j] = T[0][:, j] + term[0] * kj
                    T[1][:, j] = T[1][:, j] + term[1] * kj
        return psi, T

    def twisted(self, R, center=None) -> "Packet":
        """Apply an integer orthogonal matrix (signed permutation) to the wavevectors."""
        R = ints(np.array(R, dtype=object))
        K = np.array([[sum(int(R[i][j]) * int(k[j]) for j in range(self.n)) for i in range(self.n)] for k in self.K], dtype=object)
        return Packet(K, self.c, self.center if center is None else center, self.name)

    def with_phase(self, quarter_turns: int) -> "Packet":
        """Multiply the packet by ``i**quarter_turns`` (the exact phase group)."""
        ph = [(1, 0), (0, 1), (-1, 0), (0, -1)][quarter_turns % 4]
        c = [(a * ph[0] - b * ph[1], a * ph[1] + b * ph[0]) for a, b in self.c]
        return Packet(self.K, c, self.center, self.name)


class LatticeSystem:
    def __init__(self, packets):
        self.packets = list(packets)
        self.n = self.packets[0].n
        self.N = len(self.packets)

    def evaluate(self, M):
        M = ints(np.atleast_2d(np.array(M, dtype=object)))
        P, n = M.shape
        Psi = (zeros((P, self.N)), zeros((P, self.N)))
        T = (zeros((P, n, self.N)), zeros((P, n, self.N)))
        for a, pk in enumerate(self.packets):
            psi, Ta = pk.evaluate(M)
            Psi[0][:, a], Psi[1][:, a] = psi[0], psi[1]
            T[0][:, :, a], T[1][:, :, a] = Ta[0], Ta[1]
        return Psi, T

    def subsystem(self, members) -> "LatticeSystem":
        return LatticeSystem([self.packets[a] for a in members])


def torus_points(n: int, period: int = 4) -> np.ndarray:
    """All lattice points of ``(Z/period)^n`` as an object array (period^n, n)."""
    return ints(np.array(list(product(range(period), repeat=n)), dtype=object))


def lcg_points(n: int, count: int, seed: int = 1, period: int = 4) -> np.ndarray:
    """Deterministic pseudo-random lattice points (a linear congruential
    generator) -- exact integers, for high n where the torus is too large."""
    a, c, mod = 6364136223846793005, 1442695040888963407, 1 << 64
    s = seed
    out = zeros((count, n))
    for p in range(count):
        for j in range(n):
            s = (a * s + c) % mod
            out[p, j] = (s >> 33) % period
    return out


def signed_permutation(n: int, perm, signs) -> np.ndarray:
    """Integer orthogonal matrix ``R e_j = signs[j] e_{perm[j]}``."""
    R = zeros((n, n))
    for j in range(n):
        R[perm[j], j] = int(signs[j])
    return R


def plane_rotation_90(n: int, i: int, j: int) -> np.ndarray:
    """Exact 90-degree rotation taking ``e_i`` to ``e_j``."""
    perm = list(range(n))
    signs = [1] * n
    perm[i], perm[j] = j, i
    signs[j] = -1
    return signed_permutation(n, perm, signs)


# ----------------------------------------------------------------------------
# exact quantum geometry
# ----------------------------------------------------------------------------
def exact_geometry(Psi, T) -> dict:
    """Integer-cleared Berry connection and curvature.

    ``rho = |Psi|^2``, ``A~ = rho A = Im(Psi^+ dPsi)``, ``F~ = rho^2 F`` and
    ``g~ = rho^2 g`` with ``dPsi_j = i T_j``.  All entries are Python ints.
    """
    Pr, Pi = Psi
    Tr, Ti = T
    P, n, N = Tr.shape
    rho = (Pr * Pr + Pi * Pi).sum(axis=1)                                  # (P,)
    # current_j = Psi^+ d_j Psi = sum_a conj(psi_a) i T_aj = i * sum_a conj(psi_a) T_aj
    cr = zeros((P, n)); ci = zeros((P, n))
    for a in range(N):
        pa = (Pr[:, a], Pi[:, a])
        for j in range(n):
            t = (Tr[:, j, a], Ti[:, j, a])
            prod = gmul(gconj(pa), t)
            cr[:, j] = cr[:, j] + prod[0]
            ci[:, j] = ci[:, j] + prod[1]
    # i * (cr + i ci) = -ci + i cr  ->  Im = cr, Re = -ci
    Atil = cr.copy()                                                        # rho A
    # w_j = d_jPsi^+ Psi = conj(current_j) = (-ci) - i cr
    wr, wi = -ci, -cr
    # M_jk = sum_a conj(i T_aj)(i T_ak) = sum_a conj(T_aj) T_ak
    Mr = zeros((P, n, n)); Mi = zeros((P, n, n))
    for a in range(N):
        for j in range(n):
            tj = (Tr[:, j, a], Ti[:, j, a])
            for k in range(n):
                tk = (Tr[:, k, a], Ti[:, k, a])
                prod = gmul(gconj(tj), tk)
                Mr[:, j, k] = Mr[:, j, k] + prod[0]
                Mi[:, j, k] = Mi[:, j, k] + prod[1]
    # rho^2 Q_jk = rho M_jk - w_j conj(w_k)
    Qr = zeros((P, n, n)); Qi = zeros((P, n, n))
    for j in range(n):
        for k in range(n):
            ww = gmul((wr[:, j], wi[:, j]), gconj((wr[:, k], wi[:, k])))
            Qr[:, j, k] = rho * Mr[:, j, k] - ww[0]
            Qi[:, j, k] = rho * Mi[:, j, k] - ww[1]
    Ftil = 2 * Qi                                                            # rho^2 F
    gtil = Qr                                                                # rho^2 g
    return {"rho": rho, "A": Atil, "F": Ftil, "g": gtil}


def pair_factors(Psi, T) -> dict:
    """For N = 2: ``F~ = 2 x ^ y`` with integer vectors ``x, y`` (the exact
    rank-2 factorisation), plus the local-wavevector numerators ``u_a = rho_a k_a``
    and the envelope gradients ``grad rho_a``."""
    Pr, Pi = Psi
    Tr, Ti = T
    P, n, N = Tr.shape
    if N != 2:
        raise ValueError("pair_factors needs exactly two packets")
    p1, p2 = (Pr[:, 0], Pi[:, 0]), (Pr[:, 1], Pi[:, 1])
    x = zeros((P, n)); y = zeros((P, n))
    u = [zeros((P, n)), zeros((P, n))]
    grad = [zeros((P, n)), zeros((P, n))]
    for j in range(n):
        t1 = (Tr[:, j, 0], Ti[:, j, 0])
        t2 = (Tr[:, j, 1], Ti[:, j, 1])
        z = gadd(gscale(gmul(p2, t1), -1), gmul(p1, t2))      # -psi_2 T_1 + psi_1 T_2
        x[:, j], y[:, j] = z[0], z[1]
        for a, (pa, ta) in enumerate(((p1, t1), (p2, t2))):
            pt = gmul(gconj(pa), ta)
            u[a][:, j] = pt[0]                                   # Re(conj(psi) T) = rho_a k_a
            grad[a][:, j] = -2 * pt[1]                           # d_j rho_a = -2 Im(conj(psi) T)
    rho = [gabs2(p1), gabs2(p2)]
    return {"x": x, "y": y, "u": u, "grad_rho": grad, "rho_a": rho}


# ----------------------------------------------------------------------------
# exact exterior algebra (object dtype)
# ----------------------------------------------------------------------------
class IntExterior(ExteriorAlgebra):
    """Exterior algebra with Python-integer components."""

    def _scatter(self, p, q):
        I_idx, J_idx, scatter = self._wedge_table(p, q)
        return I_idx, J_idx, ints(scatter)

    def wedge(self, a, p: int, b, q: int) -> np.ndarray:  # type: ignore[override]
        a = np.asarray(a, dtype=object)
        b = np.asarray(b, dtype=object)
        lead = np.broadcast_shapes(a.shape[:-1], b.shape[:-1])
        r = p + q
        if r > self.n:
            return zeros(lead + (0,))
        I_idx, J_idx, scatter = self._scatter(p, q)
        prod = a[..., I_idx] * b[..., J_idx]
        return np.dot(prod, scatter)

    def wedge_vectors(self, *vectors) -> np.ndarray:  # type: ignore[override]
        out = np.asarray(vectors[0], dtype=object)
        p = 1
        for v in vectors[1:]:
            out = self.wedge(out, p, v, 1)
            p += 1
        return out

    def wedge_power(self, a, p: int, k: int) -> np.ndarray:  # type: ignore[override]
        out = np.asarray(a, dtype=object)
        deg = p
        for _ in range(k - 1):
            if deg + p > self.n:
                return zeros(out.shape[:-1] + (0,))
            out = self.wedge(out, deg, a, p)
            deg += p
        return out

    def hodge(self, a, p: int) -> np.ndarray:  # type: ignore[override]
        a = np.asarray(a, dtype=object)
        src, dst, sgn = self._hodge_table(p)
        out = zeros(a.shape[:-1] + (self.dim(self.n - p),))
        out[..., dst] = a[..., src] * ints(sgn)
        return out

    def interior(self, v, a, p: int) -> np.ndarray:  # type: ignore[override]
        v = np.asarray(v, dtype=object)
        a = np.asarray(a, dtype=object)
        V_idx, A_idx, scatter = self._interior_table(p)
        return np.dot(v[..., V_idx] * a[..., A_idx], ints(scatter))

    def from_antisymmetric(self, M) -> np.ndarray:  # type: ignore[override]
        M = np.asarray(M, dtype=object)
        I = [ij[0] for ij in self.basis[2]]
        J = [ij[1] for ij in self.basis[2]]
        return M[..., I, J]

    @staticmethod
    def norm2(a) -> np.ndarray:
        a = np.asarray(a, dtype=object)
        if a.shape[-1] == 0:
            return zeros(a.shape[:-1])
        return (a * a).sum(axis=-1)


def exact_ladder(ea: IntExterior, Atil, Ftil, max_degree: int | None = None) -> dict:
    """Integer rungs ``A~, F~, A~^F~, F~^F~, ...`` (rung of degree p equals rho^p
    times the continuum rung)."""
    n = ea.n
    F2 = ea.from_antisymmetric(Ftil)
    top = n if max_degree is None else min(n, max_degree)
    forms = {}
    if top >= 1:
        forms[1] = np.asarray(Atil, dtype=object)
    if top >= 2:
        forms[2] = F2
    Fk, k = F2, 1
    while True:
        if 2 * k + 1 <= top:
            forms[2 * k + 1] = ea.wedge(Atil, 1, Fk, 2 * k)
        if 2 * k + 2 <= top:
            Fk = ea.wedge(Fk, 2 * k, F2, 2)
            forms[2 * k + 2] = Fk
            k += 1
        else:
            break
    return forms


# ----------------------------------------------------------------------------
# exact linear algebra
# ----------------------------------------------------------------------------
def bareiss_rank(M) -> int:
    """Exact rank of an integer matrix by fraction-free elimination."""
    A = [[int(v) for v in row] for row in M]
    rows, cols = len(A), len(A[0]) if A else 0
    rank = 0
    prev = 1
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        piv = next((i for i in range(r, rows) if A[i][c] != 0), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        for i in range(r + 1, rows):
            for j in range(c + 1, cols):
                A[i][j] = (A[r][c] * A[i][j] - A[i][c] * A[r][j]) // prev
            A[i][c] = 0
        prev = A[r][c]
        r += 1
        rank += 1
    return rank


def gram_det(vectors) -> int:
    """Exact Gram determinant of a list of integer vectors (each an object array)."""
    k = len(vectors)
    G = [[int((vectors[i] * vectors[j]).sum()) for j in range(k)] for i in range(k)]
    return int_det(G)


def int_det(G) -> int:
    """Exact determinant of an integer matrix (Bareiss)."""
    A = [[int(v) for v in row] for row in G]
    m = len(A)
    sign, prev = 1, 1
    for c in range(m - 1):
        piv = next((i for i in range(c, m) if A[i][c] != 0), None)
        if piv is None:
            return 0
        if piv != c:
            A[c], A[piv] = A[piv], A[c]
            sign = -sign
        for i in range(c + 1, m):
            for j in range(c + 1, m):
                A[i][j] = (A[c][c] * A[i][j] - A[i][c] * A[c][j]) // prev
        prev = A[c][c]
    return sign * A[m - 1][m - 1]


# ----------------------------------------------------------------------------
# exact identities
# ----------------------------------------------------------------------------
def closed_form_identity(geo: dict, pf: dict, ea: IntExterior) -> np.ndarray:
    """``rho_1 rho_2 (A~ ^ F~) == rho (u_1 ^ u_2 ^ (rho_2 grad rho_1 - rho_1 grad rho_2))``
    checked exactly at every point; returns a boolean array."""
    rho = geo["rho"]
    r1, r2 = pf["rho_a"]
    lhs = ea.wedge(geo["A"], 1, ea.from_antisymmetric(geo["F"]), 2) * (r1 * r2)[:, None]
    w = pf["grad_rho"][0] * r2[:, None] - pf["grad_rho"][1] * r1[:, None]
    rhs = ea.wedge_vectors(pf["u"][0], pf["u"][1], w) * rho[:, None]
    return np.array([bool(np.all(lhs[p] == rhs[p])) for p in range(lhs.shape[0])])


def factor_identity(geo: dict, pf: dict, ea: IntExterior) -> np.ndarray:
    """``F~ == 2 x ^ y`` exactly (N = 2)."""
    F2 = ea.from_antisymmetric(geo["F"])
    xy = ea.wedge(pf["x"], 1, pf["y"], 1) * 2
    return np.array([bool(np.all(F2[p] == xy[p])) for p in range(F2.shape[0])])


def pairwise_identity(system: LatticeSystem, M) -> np.ndarray:
    """``F~ == sum_{a<b} F~_ab`` exactly (integer-cleared pairwise additivity)."""
    Psi, T = system.evaluate(M)
    F = exact_geometry(Psi, T)["F"]
    total = zeros(F.shape)
    for a, b in combinations(range(system.N), 2):
        Pab = (Psi[0][:, [a, b]], Psi[1][:, [a, b]])
        Tab = (T[0][:, :, [a, b]], T[1][:, :, [a, b]])
        total = total + exact_geometry(Pab, Tab)["F"]
    return np.array([bool(np.all(F[p] == total[p])) for p in range(F.shape[0])])


def phase_invariance(system: LatticeSystem, M, ea: IntExterior) -> dict:
    """Multiplying any packet by a power of ``i`` leaves ``F~`` and ``A~ ^ F~`` unchanged."""
    base = exact_geometry(*system.evaluate(M))
    C0 = ea.wedge(base["A"], 1, ea.from_antisymmetric(base["F"]), 2)
    ok_F, ok_C, changed = True, True, False
    for a in range(system.N):
        for q in (1, 2, 3):
            pk = list(system.packets)
            pk[a] = pk[a].with_phase(q)
            g = exact_geometry(*LatticeSystem(pk).evaluate(M))
            C = ea.wedge(g["A"], 1, ea.from_antisymmetric(g["F"]), 2)
            ok_F &= bool(np.all(g["F"] == base["F"]))
            ok_C &= bool(np.all(C == C0))
            # the physical intensity |sum psi|^2 does change
            Psi = LatticeSystem(pk).evaluate(M)[0]
            tot = (Psi[0].sum(axis=1), Psi[1].sum(axis=1))
            Psi0 = system.evaluate(M)[0]
            tot0 = (Psi0[0].sum(axis=1), Psi0[1].sum(axis=1))
            changed |= bool(np.any(gabs2(tot) != gabs2(tot0)))
    return {"F_invariant": ok_F, "chirality_invariant": ok_C, "intensity_changes": changed}


def mirror_identity(ea: IntExterior, C, p: int) -> dict:
    """For a degree-p form: ``sum_i |iota_{e_i} C|^2 == p |C|^2`` exactly, hence the
    mean cosine over the n coordinate mirrors is exactly ``1 - 2p/n``.  Returns
    the per-mirror cosines (Fractions) of the first form and the identity check."""
    n = ea.n
    C = np.asarray(C, dtype=object)
    norm2 = ea.norm2(C)
    total = zeros(norm2.shape)
    cos_first = []
    for i in range(n):
        e = zeros((n,))
        e[i] = 1
        contr = ea.interior(e, C, p)
        c2 = ea.norm2(contr)
        total = total + c2
        if norm2.shape and norm2[0] != 0:
            cos_first.append(1 - Fraction(2 * int(c2[0]), int(norm2[0])))
    ok = bool(np.all(total == p * norm2))
    mean = sum(cos_first, Fraction(0)) / n if cos_first else None
    return {"identity": ok, "cosines_first_point": cos_first, "mean_cosine": mean,
            "predicted": 1 - Fraction(2 * p, n)}


def saturation_design_average(k, n: int) -> dict:
    """Exact average of ``s^2 = |k ^ r ^ d|^2 / (|k|^2 |r|^2 |d|^2)`` over the
    coordinate design ``r, d in {+-e_i}``: equals ``1 - 3/n + 2/n^2``."""
    k = ints(np.array(k, dtype=object))
    k2 = int((k * k).sum())
    total = Fraction(0)
    count = 0
    for i in range(n):
        for j in range(n):
            for si in (1, -1):
                for sj in (1, -1):
                    r = zeros((n,)); r[i] = si
                    d = zeros((n,)); d[j] = sj
                    total += Fraction(gram_det([k, r, d]), k2)
                    count += 1
    return {"average": total / count, "predicted": 1 - Fraction(3, n) + Fraction(2, n * n)}


# ----------------------------------------------------------------------------
# exact statistics
# ----------------------------------------------------------------------------
def direction_statistics(C) -> dict:
    """Exact participation ratios of the directions of integer vectors ``C (P, D)``.

    ``cos^2_pq = <C_p, C_q>^2 / (|C_p|^2 |C_q|^2)`` are exact rationals; the
    sample effective dimension is ``P^2 / sum_pq cos^2`` and the population
    estimate ``P(P-1) / sum_{p != q} cos^2``.
    """
    C = np.asarray(C, dtype=object)
    n2 = (C * C).sum(axis=1)
    keep = [p for p in range(C.shape[0]) if n2[p] != 0]
    C = C[keep]
    n2 = n2[keep]
    P = C.shape[0]
    G = np.dot(C, C.T)
    total = Fraction(0)
    off = Fraction(0)
    for p in range(P):
        for q in range(P):
            c2 = Fraction(int(G[p, q]) ** 2, int(n2[p]) * int(n2[q]))
            total += c2
            if p != q:
                off += c2
    return {"count": P,
            "effective_dimension": Fraction(P * P) / total if total else None,
            "population_effective_dimension": (Fraction(P * (P - 1)) / off) if off else None}


def dominance(rho_matrix) -> np.ndarray:
    """Index of the dominant packet at each point (exact integer comparison; ties -> lowest index)."""
    R = np.asarray(rho_matrix, dtype=object)
    return np.array([max(range(R.shape[1]), key=lambda a: (int(R[p, a]), -a)) for p in range(R.shape[0])])


def visibility_band(rho1, rho2, threshold: Fraction = Fraction(1, 2)) -> np.ndarray:
    """Points where the pair visibility squared ``4 rho1 rho2 / (rho1 + rho2)^2 >= threshold``."""
    out = []
    for a, b in zip(rho1, rho2):
        a, b = int(a), int(b)
        out.append(a + b > 0 and Fraction(4 * a * b, (a + b) ** 2) >= threshold)
    return np.array(out, dtype=bool)


def junction_cubes(labels_by_point: dict, n: int, period: int = 4, min_labels: int = 3) -> int:
    """Number of unit hypercubes of the torus whose vertices carry at least
    ``min_labels`` distinct dominant packets."""
    count = 0
    for base in product(range(period), repeat=n):
        seen = set()
        for corner in product((0, 1), repeat=n):
            pt = tuple((b + c) % period for b, c in zip(base, corner))
            seen.add(labels_by_point[pt])
        if len(seen) >= min_labels:
            count += 1
    return count


# ----------------------------------------------------------------------------
# decomposable 3-vectors without components (any n)
# ----------------------------------------------------------------------------
def _gram_cross(U, V):
    """3x3 cross-Gram matrices of two triples of integer vectors: shape (3, 3) of ints."""
    return [[int((U[a] * V[b]).sum()) for b in range(3)] for a in range(3)]


def decomposable_inner(U, V) -> int:
    """``<u1^u2^u3, v1^v2^v3> = det[u_a . v_b]`` exactly."""
    return int_det(_gram_cross(U, V))


def decomposable_equal(U, su: int, V, sv: int) -> bool:
    """Exact test ``su (u1^u2^u3) == sv (v1^v2^v3)`` via norms and inner product."""
    nu = su * su * decomposable_inner(U, U)
    nv = sv * sv * decomposable_inner(V, V)
    uv = su * sv * decomposable_inner(U, V)
    return nu == nv == uv


def closed_form_identity_highn(geo: dict, pf: dict) -> np.ndarray:
    """The closed form ``rho_1 rho_2 (A~^F~) == rho (u_1^u_2^w)`` for N = 2 in any n,
    using ``A~ ^ F~ = 2 A~ ^ x ^ y`` and Gram determinants only."""
    rho = geo["rho"]
    r1, r2 = pf["rho_a"]
    w = pf["grad_rho"][0] * r2[:, None] - pf["grad_rho"][1] * r1[:, None]
    out = []
    for p in range(rho.shape[0]):
        U = (geo["A"][p], pf["x"][p], pf["y"][p])
        V = (pf["u"][0][p], pf["u"][1][p], w[p])
        out.append(decomposable_equal(U, 2 * int(r1[p]) * int(r2[p]), V, int(rho[p])))
    return np.array(out, dtype=bool)


def decomposable_direction_statistics(triples) -> dict:
    """Exact direction statistics of decomposable 3-vectors given by triples of
    integer vectors: ``cos^2_pq = det[G_pq]^2 / (det[G_pp] det[G_qq])``."""
    P = len(triples)
    norms = [decomposable_inner(t, t) for t in triples]
    keep = [p for p in range(P) if norms[p] != 0]
    total = Fraction(0)
    off = Fraction(0)
    for p in keep:
        for q in keep:
            c2 = Fraction(decomposable_inner(triples[p], triples[q]) ** 2, norms[p] * norms[q])
            total += c2
            if p != q:
                off += c2
    Pk = len(keep)
    return {"count": Pk,
            "effective_dimension": Fraction(Pk * Pk) / total if total else None,
            "population_effective_dimension": Fraction(Pk * (Pk - 1)) / off if off else None}


def decomposable_mirror_mean(triples, n: int) -> dict:
    """Exact per-point mean over the n coordinate mirrors of the cosine between a
    decomposable 3-vector and its mirror image: ``1 - 2 |proj e_i|^2`` averaged,
    which equals ``1 - 6/n`` identically."""
    means = []
    for U in triples:
        G = _gram_cross(U, U)
        detG = int_det(G)
        if detG == 0:
            continue
        # adjugate of the 3x3 Gram matrix
        adj = [[0] * 3 for _ in range(3)]
        for a in range(3):
            for b in range(3):
                minor = [[G[r][c] for c in range(3) if c != a] for r in range(3) if r != b]
                adj[a][b] = (-1) ** (a + b) * (minor[0][0] * minor[1][1] - minor[0][1] * minor[1][0])
        total = Fraction(0)
        for i in range(n):
            g = [int(U[a][i]) for a in range(3)]
            quad = sum(g[a] * adj[a][b] * g[b] for a in range(3) for b in range(3))
            total += 1 - Fraction(2 * quad, detG)
        means.append(total / n)
    return {"mean_cosines": means, "all_equal_predicted": all(m == 1 - Fraction(6, n) for m in means),
            "predicted": 1 - Fraction(6, n)}
