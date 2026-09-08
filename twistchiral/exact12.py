"""Exact twin of the library with 12th roots of unity.

Same model as :mod:`twistchiral.exact` (lattice positions, integer
wavevectors, carrier + side-band packets, signed-permutation twists, cleared
projective denominators) but with phases ``zeta^(K . m)``, ``zeta = exp(i pi/6)``.
The lattice spacing is now ``pi/6`` and the torus is ``(Z/12)^n``, so the beat
envelopes are three times wider and the phase group has 12 elements.

``Q(zeta_12) = Q(i, sqrt 3)``: its real subfield is ``Q(sqrt 3)``, not ``Q``,
so real quantities are elements ``a + b sqrt 3``.  Every phase is scaled to
``2 zeta^e`` (a harmless global factor), which makes all field values, all
derivatives and everything built from them live in the ring ``Z[sqrt 3]`` of
integer pairs.  Sign tests, equality tests and the fraction-free (Bareiss)
elimination are exact in that ring; division appears only in the final
statistics, where results are elements of ``Q(sqrt 3)`` with rational
coefficients.  No floating point anywhere.  Orders 4 and 6 are the special
cases ``zeta_4 = zeta_12^3`` and ``zeta_6 = zeta_12^2``.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product
from math import isqrt

import numpy as np

from .exterior import ExteriorAlgebra


# ----------------------------------------------------------------------------
# the ring Z[sqrt 3] (and the field Q(sqrt 3) with Fraction coefficients)
# ----------------------------------------------------------------------------
class R3:
    """``a + b sqrt(3)`` with integer (or Fraction) coefficients."""

    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        self.a = a
        self.b = b

    # -- coercion
    @staticmethod
    def of(x) -> "R3":
        if isinstance(x, R3):
            return x
        if isinstance(x, (int, Fraction)):
            return R3(x, 0)
        if isinstance(x, np.integer):
            return R3(int(x), 0)
        raise TypeError(f"cannot coerce {type(x)} to R3")

    # -- ring operations
    def __add__(self, o):
        o = R3.of(o)
        return R3(self.a + o.a, self.b + o.b)

    __radd__ = __add__

    def __sub__(self, o):
        o = R3.of(o)
        return R3(self.a - o.a, self.b - o.b)

    def __rsub__(self, o):
        return R3.of(o) - self

    def __neg__(self):
        return R3(-self.a, -self.b)

    def __mul__(self, o):
        o = R3.of(o)
        return R3(self.a * o.a + 3 * self.b * o.b, self.a * o.b + self.b * o.a)

    __rmul__ = __mul__

    def conj(self) -> "R3":
        return R3(self.a, -self.b)

    def norm(self):
        """``(a + b sqrt3)(a - b sqrt3) = a^2 - 3 b^2`` (zero only for zero)."""
        return self.a * self.a - 3 * self.b * self.b

    def __truediv__(self, o):
        """Exact division in Q(sqrt 3) (Fraction coefficients)."""
        o = R3.of(o)
        num = self * o.conj()
        d = o.norm()
        return R3(Fraction(num.a, d), Fraction(num.b, d))

    def __floordiv__(self, o):
        """Exact division in Z[sqrt 3]; raises if not divisible."""
        o = R3.of(o)
        num = self * o.conj()
        d = o.norm()
        if num.a % d or num.b % d:
            raise ArithmeticError("inexact division in Z[sqrt3]")
        return R3(num.a // d, num.b // d)

    # -- comparisons (exact: sign of a + b sqrt3)
    def sign(self) -> int:
        a, b = self.a, self.b
        if a == 0 and b == 0:
            return 0
        if a >= 0 and b >= 0:
            return 1
        if a <= 0 and b <= 0:
            return -1
        # a and b of opposite signs: compare a^2 with 3 b^2
        if a > 0:
            return 1 if a * a > 3 * b * b else -1
        return -1 if a * a > 3 * b * b else 1

    def is_zero(self) -> bool:
        return self.a == 0 and self.b == 0

    def __eq__(self, o):
        try:
            o = R3.of(o)
        except TypeError:
            return NotImplemented
        return self.a == o.a and self.b == o.b

    def __ne__(self, o):
        r = self.__eq__(o)
        return r if r is NotImplemented else not r

    def __lt__(self, o):
        return (self - R3.of(o)).sign() < 0

    def __le__(self, o):
        return (self - R3.of(o)).sign() <= 0

    def __gt__(self, o):
        return (self - R3.of(o)).sign() > 0

    def __ge__(self, o):
        return (self - R3.of(o)).sign() >= 0

    def __hash__(self):
        return hash((self.a, self.b))

    def __bool__(self):
        return not self.is_zero()

    # -- exact rendering
    def floor_scaled(self, digits: int) -> int:
        """``floor((a + b sqrt3) * 10^digits)`` computed with integer arithmetic."""
        X = Fraction(self.a) * 10**digits
        Y = Fraction(self.b) * 10**digits
        D = X.denominator * Y.denominator // _gcd(X.denominator, Y.denominator)
        Xn = X.numerator * (D // X.denominator)
        Yn = Y.numerator * (D // Y.denominator)
        if Yn >= 0:
            f = isqrt(3 * Yn * Yn)
        else:
            f = -isqrt(3 * Yn * Yn) - 1
        return (Xn + f) // D

    def decimal(self, digits: int = 4) -> str:
        """Exact decimal expansion truncated toward zero (integer arithmetic only)."""
        if self.sign() < 0:
            return "-" + (-self).decimal(digits)
        s = self.floor_scaled(digits)
        q, r = divmod(s, 10**digits)
        return f"{q}.{r:0{digits}d}"

    def __repr__(self):
        return f"({self.a} + {self.b}√3)"


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def r3s(a) -> np.ndarray:
    """Object array of R3 built from an array-like of ints/R3."""
    arr = np.array(a, dtype=object)
    flat = arr.reshape(-1)
    out = np.empty(flat.shape, dtype=object)
    for i in range(flat.shape[0]):
        out[i] = R3.of(flat[i])
    return out.reshape(arr.shape)


def zeros(shape) -> np.ndarray:
    out = np.empty(shape, dtype=object)
    flat = out.reshape(-1)
    for i in range(flat.shape[0]):
        flat[i] = R3(0, 0)
    return out


# ----------------------------------------------------------------------------
# complex numbers over Q(zeta_12) as (re, im) pairs of R3 arrays
# ----------------------------------------------------------------------------
def gmul(x, y):
    xr, xi = x
    yr, yi = y
    return xr * yr - xi * yi, xr * yi + xi * yr


def gconj(x):
    return x[0], -x[1]


def gadd(x, y):
    return x[0] + y[0], x[1] + y[1]


def gscale(x, s):
    return x[0] * s, x[1] * s


def gabs2(x):
    return x[0] * x[0] + x[1] * x[1]


# 2 zeta^e for e = 0..11: 2cos(e pi/6), 2sin(e pi/6) in Z[sqrt3]
_COS2 = [R3(2, 0), R3(0, 1), R3(1, 0), R3(0, 0), R3(-1, 0), R3(0, -1), R3(-2, 0), R3(0, -1), R3(-1, 0), R3(0, 0), R3(1, 0), R3(0, 1)]
_SIN2 = [R3(0, 0), R3(1, 0), R3(0, 1), R3(2, 0), R3(0, 1), R3(1, 0), R3(0, 0), R3(-1, 0), R3(0, -1), R3(-2, 0), R3(0, -1), R3(-1, 0)]


def zpow2(e, order: int = 12):
    """``2 * zeta_order ** e`` as (re, im) object arrays of R3 for an object array of ints ``e``."""
    e = np.asarray(e, dtype=object)
    step = 12 // order
    r = np.empty(e.shape, dtype=object)
    im = np.empty(e.shape, dtype=object)
    fr, fi, fe = r.reshape(-1), im.reshape(-1), e.reshape(-1)
    for idx in range(fe.shape[0]):
        k = (int(fe[idx]) * step) % 12
        fr[idx] = _COS2[k]
        fi[idx] = _SIN2[k]
    return r, im


# ----------------------------------------------------------------------------
# packets and systems
# ----------------------------------------------------------------------------
class Packet:
    """``psi(m) = sum_s c_s (2 zeta^(K_s . (m - mu)))`` with integer ``K_s``,
    Gaussian-integer amplitudes ``c_s = (re, im)`` and integer centre ``mu``."""

    def __init__(self, K, amplitudes, center=None, order: int = 12, name: str = ""):
        self.K = np.array(np.atleast_2d(np.array(K, dtype=object)), dtype=object)
        self.n = self.K.shape[1]
        self.m = self.K.shape[0]
        self.c = [(R3.of(a), R3.of(b)) for a, b in amplitudes]
        self.center = np.array([0] * self.n, dtype=object) if center is None else np.array(center, dtype=object)
        self.order = order
        self.name = name

    @classmethod
    def carrier_with_sidebands(cls, carrier, sideband_offsets, carrier_amp=(1, 0), sideband_amp=(1, 0), center=None, order: int = 12):
        K = [list(carrier)] + [[c + o for c, o in zip(carrier, off)] for off in sideband_offsets]
        amps = [carrier_amp] + [sideband_amp] * len(sideband_offsets)
        return cls(K, amps, center, order)

    def evaluate(self, M):
        M = np.array(np.atleast_2d(np.array(M, dtype=object)), dtype=object)
        P, n = M.shape
        Y = M - self.center[None, :]
        psi = (zeros(P), zeros(P))
        T = (zeros((P, n)), zeros((P, n)))
        for s in range(self.m):
            e = (Y * self.K[s][None, :]).sum(axis=1)
            ph = zpow2(e, self.order)
            cr = np.empty(P, dtype=object); cr.fill(self.c[s][0])
            ci = np.empty(P, dtype=object); ci.fill(self.c[s][1])
            term = gmul(ph, (cr, ci))
            psi = gadd(psi, term)
            for j in range(n):
                kj = int(self.K[s][j])
                if kj != 0:
                    T[0][:, j] = T[0][:, j] + term[0] * kj
                    T[1][:, j] = T[1][:, j] + term[1] * kj
        return psi, T

    def twisted(self, R, center=None) -> "Packet":
        R = np.array(R, dtype=object)
        K = np.array([[sum(int(R[i][j]) * int(k[j]) for j in range(self.n)) for i in range(self.n)] for k in self.K], dtype=object)
        return Packet(K, self.c, self.center if center is None else center, self.order, self.name)

    def scaled_by_phase(self, q: int) -> "Packet":
        """Multiply the amplitudes by ``2 zeta^q`` (use with :meth:`LatticeSystem.with_phase`)."""
        ph = (_COS2[(q * (12 // self.order)) % 12], _SIN2[(q * (12 // self.order)) % 12])
        c = [gmul((a, b), ph) for a, b in self.c]
        return Packet(self.K, c, self.center, self.order, self.name)

    def scaled(self, s: int) -> "Packet":
        return Packet(self.K, [(a * s, b * s) for a, b in self.c], self.center, self.order, self.name)


class LatticeSystem:
    def __init__(self, packets):
        self.packets = list(packets)
        self.n = self.packets[0].n
        self.N = len(self.packets)
        self.order = self.packets[0].order

    def evaluate(self, M):
        M = np.array(np.atleast_2d(np.array(M, dtype=object)), dtype=object)
        P, n = M.shape
        Psi = (zeros((P, self.N)), zeros((P, self.N)))
        T = (zeros((P, n, self.N)), zeros((P, n, self.N)))
        for a, pk in enumerate(self.packets):
            psi, Ta = pk.evaluate(M)
            Psi[0][:, a], Psi[1][:, a] = psi[0], psi[1]
            T[0][:, :, a], T[1][:, :, a] = Ta[0], Ta[1]
        return Psi, T

    def with_phase(self, a: int, q: int) -> "LatticeSystem":
        """Exact relative phase: packet ``a`` gets ``2 zeta^q``, every other packet
        gets the factor 2, so the projective point moves by a pure phase."""
        return LatticeSystem([pk.scaled_by_phase(q) if b == a else pk.scaled(2) for b, pk in enumerate(self.packets)])

    def subsystem(self, members) -> "LatticeSystem":
        return LatticeSystem([self.packets[a] for a in members])


def torus_points(n: int, period: int = 12) -> np.ndarray:
    return np.array(list(product(range(period), repeat=n)), dtype=object)


def lcg_points(n: int, count: int, seed: int = 1, period: int = 12) -> np.ndarray:
    a, c, mod = 6364136223846793005, 1442695040888963407, 1 << 64
    s = seed
    out = np.empty((count, n), dtype=object)
    for p in range(count):
        for j in range(n):
            s = (a * s + c) % mod
            out[p, j] = (s >> 33) % period
    return out


def signed_permutation(n: int, perm, signs) -> np.ndarray:
    R = np.zeros((n, n), dtype=object)
    for j in range(n):
        R[perm[j], j] = int(signs[j])
    return R


def plane_rotation_90(n: int, i: int, j: int) -> np.ndarray:
    perm = list(range(n))
    signs = [1] * n
    perm[i], perm[j] = j, i
    signs[j] = -1
    return signed_permutation(n, perm, signs)


def compose(R2, R1) -> np.ndarray:
    n = R1.shape[0]
    return np.array([[sum(int(R2[i][k]) * int(R1[k][j]) for k in range(n)) for j in range(n)] for i in range(n)], dtype=object)


# ----------------------------------------------------------------------------
# exact quantum geometry (identical algebra to exact.py, now over Z[sqrt3])
# ----------------------------------------------------------------------------
def exact_geometry(Psi, T) -> dict:
    Pr, Pi = Psi
    Tr, Ti = T
    P, n, N = Tr.shape
    rho = (Pr * Pr + Pi * Pi).sum(axis=1)
    cr = zeros((P, n)); ci = zeros((P, n))
    for a in range(N):
        pa = (Pr[:, a], Pi[:, a])
        for j in range(n):
            prod = gmul(gconj(pa), (Tr[:, j, a], Ti[:, j, a]))
            cr[:, j] = cr[:, j] + prod[0]
            ci[:, j] = ci[:, j] + prod[1]
    Atil = cr.copy()
    wr, wi = -ci, -cr
    Mr = zeros((P, n, n)); Mi = zeros((P, n, n))
    for a in range(N):
        for j in range(n):
            tj = (Tr[:, j, a], Ti[:, j, a])
            for k in range(n):
                prod = gmul(gconj(tj), (Tr[:, k, a], Ti[:, k, a]))
                Mr[:, j, k] = Mr[:, j, k] + prod[0]
                Mi[:, j, k] = Mi[:, j, k] + prod[1]
    Qr = zeros((P, n, n)); Qi = zeros((P, n, n))
    for j in range(n):
        for k in range(n):
            ww = gmul((wr[:, j], wi[:, j]), gconj((wr[:, k], wi[:, k])))
            Qr[:, j, k] = rho * Mr[:, j, k] - ww[0]
            Qi[:, j, k] = rho * Mi[:, j, k] - ww[1]
    return {"rho": rho, "A": Atil, "F": Qi * 2, "g": Qr}


def pair_factors(Psi, T) -> dict:
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
        z = gadd(gscale(gmul(p2, t1), -1), gmul(p1, t2))
        x[:, j], y[:, j] = z[0], z[1]
        for a, (pa, ta) in enumerate(((p1, t1), (p2, t2))):
            pt = gmul(gconj(pa), ta)
            u[a][:, j] = pt[0]
            grad[a][:, j] = pt[1] * (-2)
    return {"x": x, "y": y, "u": u, "grad_rho": grad, "rho_a": [gabs2(p1), gabs2(p2)]}


# ----------------------------------------------------------------------------
# exterior algebra and linear algebra over Z[sqrt3]
# ----------------------------------------------------------------------------
class RingExterior(ExteriorAlgebra):
    def _table_obj(self, p, q):
        I_idx, J_idx, scatter = self._wedge_table(p, q)
        return I_idx, J_idx, np.array(scatter.astype(int), dtype=object)

    def wedge(self, a, p: int, b, q: int) -> np.ndarray:  # type: ignore[override]
        a = np.asarray(a, dtype=object)
        b = np.asarray(b, dtype=object)
        lead = np.broadcast_shapes(a.shape[:-1], b.shape[:-1])
        r = p + q
        if r > self.n:
            return zeros(lead + (0,))
        I_idx, J_idx, scatter = self._table_obj(p, q)
        prod = a[..., I_idx] * b[..., J_idx]
        return np.dot(prod, scatter)

    def wedge_vectors(self, *vectors) -> np.ndarray:  # type: ignore[override]
        out = np.asarray(vectors[0], dtype=object)
        p = 1
        for v in vectors[1:]:
            out = self.wedge(out, p, v, 1)
            p += 1
        return out

    def interior(self, v, a, p: int) -> np.ndarray:  # type: ignore[override]
        V_idx, A_idx, scatter = self._interior_table(p)
        return np.dot(np.asarray(v, dtype=object)[..., V_idx] * np.asarray(a, dtype=object)[..., A_idx],
                      np.array(scatter.astype(int), dtype=object))

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


def exact_ladder(ea: RingExterior, Atil, Ftil, max_degree: int | None = None) -> dict:
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


def nonzero(arr) -> bool:
    return any(not R3.of(v).is_zero() for v in np.asarray(arr, dtype=object).reshape(-1))


def bareiss_rank(M) -> int:
    """Exact rank over the integral domain Z[sqrt3] (fraction-free elimination)."""
    A = [[R3.of(v) for v in row] for row in M]
    rows, cols = len(A), len(A[0]) if A else 0
    rank, prev, r = 0, R3(1, 0), 0
    for c in range(cols):
        if r >= rows:
            break
        piv = next((i for i in range(r, rows) if not A[i][c].is_zero()), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        for i in range(r + 1, rows):
            for j in range(c + 1, cols):
                A[i][j] = (A[r][c] * A[i][j] - A[i][c] * A[r][j]) // prev
            A[i][c] = R3(0, 0)
        prev = A[r][c]
        r += 1
        rank += 1
    return rank


def ring_det(G) -> R3:
    A = [[R3.of(v) for v in row] for row in G]
    m = len(A)
    sign, prev = 1, R3(1, 0)
    for c in range(m - 1):
        piv = next((i for i in range(c, m) if not A[i][c].is_zero()), None)
        if piv is None:
            return R3(0, 0)
        if piv != c:
            A[c], A[piv] = A[piv], A[c]
            sign = -sign
        for i in range(c + 1, m):
            for j in range(c + 1, m):
                A[i][j] = (A[c][c] * A[i][j] - A[i][c] * A[c][j]) // prev
        prev = A[c][c]
    return A[m - 1][m - 1] * sign


def gram_det(vectors) -> R3:
    k = len(vectors)
    return ring_det([[(vectors[i] * vectors[j]).sum() for j in range(k)] for i in range(k)])


# ----------------------------------------------------------------------------
# exact identities
# ----------------------------------------------------------------------------
def _all_equal(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=object); b = np.asarray(b, dtype=object)
    return np.array([all(R3.of(x) == R3.of(y) for x, y in zip(a[p].reshape(-1), b[p].reshape(-1))) for p in range(a.shape[0])])


def closed_form_identity(geo: dict, pf: dict, ea: RingExterior) -> np.ndarray:
    rho = geo["rho"]
    r1, r2 = pf["rho_a"]
    lhs = ea.wedge(geo["A"], 1, ea.from_antisymmetric(geo["F"]), 2) * (r1 * r2)[:, None]
    w = pf["grad_rho"][0] * r2[:, None] - pf["grad_rho"][1] * r1[:, None]
    rhs = ea.wedge_vectors(pf["u"][0], pf["u"][1], w) * rho[:, None]
    return _all_equal(lhs, rhs)


def factor_identity(geo: dict, pf: dict, ea: RingExterior) -> np.ndarray:
    return _all_equal(ea.from_antisymmetric(geo["F"]), ea.wedge(pf["x"], 1, pf["y"], 1) * 2)


def pairwise_identity(system: LatticeSystem, M) -> np.ndarray:
    Psi, T = system.evaluate(M)
    F = exact_geometry(Psi, T)["F"]
    total = zeros(F.shape)
    for a, b in combinations(range(system.N), 2):
        Pab = (Psi[0][:, [a, b]], Psi[1][:, [a, b]])
        Tab = (T[0][:, :, [a, b]], T[1][:, :, [a, b]])
        total = total + exact_geometry(Pab, Tab)["F"]
    return _all_equal(F, total)


def phase_invariance(system: LatticeSystem, M, ea: RingExterior) -> dict:
    """A pure relative phase (packet a times zeta^q, others unchanged) must leave
    the projective geometry fixed.  With the 2-scalings of :meth:`with_phase`
    every cleared quantity picks up a global power of 2: F~ scales by 2^4 and
    A~^F~ by 2^6, which is what is checked; the summed intensity changes."""
    base = exact_geometry(*system.evaluate(M))
    C0 = ea.wedge(base["A"], 1, ea.from_antisymmetric(base["F"]), 2)
    Psi0 = system.evaluate(M)[0]
    I0 = gabs2((Psi0[0].sum(axis=1), Psi0[1].sum(axis=1)))
    ok_F = ok_C = True
    changed = False
    for a in range(system.N):
        for q in range(1, system.order):
            shifted = system.with_phase(a, q)
            g = exact_geometry(*shifted.evaluate(M))
            C = ea.wedge(g["A"], 1, ea.from_antisymmetric(g["F"]), 2)
            ok_F &= bool(_all_equal(g["F"], base["F"] * 16).all())
            ok_C &= bool(_all_equal(C, C0 * 64).all())
            Psi = shifted.evaluate(M)[0]
            I1 = gabs2((Psi[0].sum(axis=1), Psi[1].sum(axis=1)))
            changed |= not bool(_all_equal(I1.reshape(-1, 1), (I0 * 4).reshape(-1, 1)).all())
    return {"F_invariant": ok_F, "chirality_invariant": ok_C, "intensity_changes": changed}


def mirror_identity(ea: RingExterior, C, p: int) -> dict:
    n = ea.n
    C = np.asarray(C, dtype=object)
    norm2 = ea.norm2(C)
    total = zeros(norm2.shape)
    cos_first = []
    for i in range(n):
        e = np.array([0] * n, dtype=object)
        e[i] = 1
        c2 = ea.norm2(ea.interior(e, C, p))
        total = total + c2
        if norm2.shape and not R3.of(norm2[0]).is_zero():
            cos_first.append(R3(1, 0) - (R3.of(c2[0]) * 2) / R3.of(norm2[0]))
    ok = bool(_all_equal(total.reshape(-1, 1), (norm2 * p).reshape(-1, 1)).all())
    mean = None
    if cos_first:
        s = R3(0, 0)
        for c in cos_first:
            s = s + c
        mean = s / n
    return {"identity": ok, "mean_cosine": mean, "predicted": R3(1, 0) - R3(Fraction(2 * p, n), 0)}


def saturation_design_average(k, n: int) -> dict:
    k = np.array(k, dtype=object)
    k2 = R3.of((k * k).sum())
    total = R3(0, 0)
    count = 0
    for i in range(n):
        for j in range(n):
            for si in (1, -1):
                for sj in (1, -1):
                    r = np.array([0] * n, dtype=object); r[i] = si
                    d = np.array([0] * n, dtype=object); d[j] = sj
                    total = total + gram_det([k, r, d]) / k2
                    count += 1
    return {"average": total / count, "predicted": R3(1 - Fraction(3, n) + Fraction(2, n * n), 0)}


# ----------------------------------------------------------------------------
# exact statistics
# ----------------------------------------------------------------------------
def direction_statistics(C) -> dict:
    C = np.asarray(C, dtype=object)
    n2 = (C * C).sum(axis=1)
    keep = [p for p in range(C.shape[0]) if not R3.of(n2[p]).is_zero()]
    C = C[keep]
    n2 = n2[keep]
    P = C.shape[0]
    G = np.dot(C, C.T)
    total = R3(0, 0)
    off = R3(0, 0)
    for p in range(P):
        for q in range(P):
            g = R3.of(G[p, q])
            c2 = (g * g) / (R3.of(n2[p]) * R3.of(n2[q]))
            total = total + c2
            if p != q:
                off = off + c2
    return {"count": P,
            "effective_dimension": R3(P * P, 0) / total if not total.is_zero() else None,
            "population_effective_dimension": R3(P * (P - 1), 0) / off if not off.is_zero() else None}


def _cross_gram(U, V):
    return [[(U[a] * V[b]).sum() for b in range(3)] for a in range(3)]


def decomposable_inner(U, V) -> R3:
    return ring_det(_cross_gram(U, V))


def decomposable_equal(U, su, V, sv) -> bool:
    nu = decomposable_inner(U, U) * (su * su)
    nv = decomposable_inner(V, V) * (sv * sv)
    uv = decomposable_inner(U, V) * (su * sv)
    return nu == nv and nv == uv


def closed_form_identity_highn(geo: dict, pf: dict) -> np.ndarray:
    rho = geo["rho"]
    r1, r2 = pf["rho_a"]
    w = pf["grad_rho"][0] * r2[:, None] - pf["grad_rho"][1] * r1[:, None]
    out = []
    for p in range(rho.shape[0]):
        U = (geo["A"][p], pf["x"][p], pf["y"][p])
        V = (pf["u"][0][p], pf["u"][1][p], w[p])
        out.append(decomposable_equal(U, R3.of(r1[p]) * R3.of(r2[p]) * 2, V, R3.of(rho[p])))
    return np.array(out, dtype=bool)


def decomposable_direction_statistics(triples) -> dict:
    norms = [decomposable_inner(t, t) for t in triples]
    keep = [p for p in range(len(triples)) if not norms[p].is_zero()]
    total = R3(0, 0)
    off = R3(0, 0)
    for p in keep:
        for q in keep:
            ip = decomposable_inner(triples[p], triples[q])
            c2 = (ip * ip) / (norms[p] * norms[q])
            total = total + c2
            if p != q:
                off = off + c2
    Pk = len(keep)
    return {"count": Pk,
            "effective_dimension": R3(Pk * Pk, 0) / total if not total.is_zero() else None,
            "population_effective_dimension": R3(Pk * (Pk - 1), 0) / off if not off.is_zero() else None}


def decomposable_mirror_mean(triples, n: int) -> dict:
    means = []
    for U in triples:
        G = _cross_gram(U, U)
        detG = ring_det(G)
        if detG.is_zero():
            continue
        adj = [[None] * 3 for _ in range(3)]
        for a in range(3):
            for b in range(3):
                minor = [[G[r][c] for c in range(3) if c != a] for r in range(3) if r != b]
                adj[a][b] = (minor[0][0] * minor[1][1] - minor[0][1] * minor[1][0]) * ((-1) ** (a + b))
        total = R3(0, 0)
        for i in range(n):
            g = [R3.of(U[a][i]) for a in range(3)]
            quad = R3(0, 0)
            for a in range(3):
                for b in range(3):
                    quad = quad + g[a] * adj[a][b] * g[b]
            total = total + (R3(1, 0) - (quad * 2) / detG)
        means.append(total / n)
    pred = R3(1 - Fraction(6, n), 0)
    return {"mean_cosines": means, "all_equal_predicted": all(m == pred for m in means), "predicted": pred}


def dominance(rho_matrix) -> np.ndarray:
    R = np.asarray(rho_matrix, dtype=object)
    out = []
    for p in range(R.shape[0]):
        best = 0
        for a in range(1, R.shape[1]):
            if R3.of(R[p, a]) > R3.of(R[p, best]):
                best = a
        out.append(best)
    return np.array(out)


def visibility_band(rho1, rho2, threshold=Fraction(1, 2)) -> np.ndarray:
    out = []
    for a, b in zip(rho1, rho2):
        a, b = R3.of(a), R3.of(b)
        s = a + b
        # 4 a b >= threshold (a + b)^2, exact sign test in Z[sqrt3] (scaled by the denominator)
        lhs = a * b * (4 * threshold.denominator)
        rhs = s * s * threshold.numerator
        out.append(s.sign() > 0 and (lhs - rhs).sign() >= 0)
    return np.array(out, dtype=bool)


def junction_cubes(labels_by_point: dict, n: int, period: int = 12, min_labels: int = 3) -> int:
    count = 0
    for base in product(range(period), repeat=n):
        seen = set()
        for corner in product((0, 1), repeat=n):
            seen.add(labels_by_point[tuple((b + c) % period for b, c in zip(base, corner))])
        if len(seen) >= min_labels:
            count += 1
    return count
