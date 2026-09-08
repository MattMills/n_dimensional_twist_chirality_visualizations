"""Exterior algebra on R^n for batched fields.

A p-form (equivalently a p-vector) is stored as an array of shape
``(..., C(n, p))`` whose last axis runs over the sorted index tuples
``(i_1 < i_2 < ... < i_p)`` in lexicographic order.  All operations are
vectorised over the leading axes, so a whole point cloud of forms can be
wedged, dualised or normed in one call.

The basis ``e_I`` is orthonormal, so the norm of a form is the plain
Euclidean norm of its component vector.  For a decomposable p-vector
``v_1 ^ ... ^ v_p`` that norm is the p-volume of the parallelotope
spanned by the vectors.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np


def permutation_sign(seq) -> int:
    """Sign of the permutation that sorts ``seq`` (distinct elements)."""
    seq = list(seq)
    sign = 1
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                sign = -sign
    return sign


class ExteriorAlgebra:
    """Batched exterior algebra of R^n."""

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = int(n)
        self.basis = [list(combinations(range(n), p)) for p in range(n + 1)]
        self.index = [{I: i for i, I in enumerate(B)} for B in self.basis]
        self._wedge_tables: dict[tuple[int, int], tuple] = {}
        self._hodge_tables: dict[int, tuple] = {}
        self._interior_tables: dict[int, tuple] = {}

    # ------------------------------------------------------------------ basics
    def dim(self, p: int) -> int:
        return comb(self.n, p) if 0 <= p <= self.n else 0

    def labels(self, p: int) -> list[str]:
        return ["e" + "".join(str(i) for i in I) if I else "1" for I in self.basis[p]]

    def zeros(self, p: int, *lead) -> np.ndarray:
        return np.zeros(tuple(lead) + (self.dim(p),))

    # ------------------------------------------------------------------ wedge
    def _wedge_table(self, p: int, q: int):
        key = (p, q)
        if key not in self._wedge_tables:
            r = p + q
            rows = []
            for i, I in enumerate(self.basis[p]):
                sI = set(I)
                for j, J in enumerate(self.basis[q]):
                    if sI.isdisjoint(J):
                        K = tuple(sorted(I + J))
                        rows.append((i, j, self.index[r][K], permutation_sign(I + J)))
            if rows:
                I_idx, J_idx, K_idx, S = (np.array(c) for c in zip(*rows))
            else:  # pragma: no cover - r > n handled by caller
                I_idx = J_idx = K_idx = np.zeros(0, dtype=int)
                S = np.zeros(0)
            scatter = np.zeros((len(rows), self.dim(r)))
            scatter[np.arange(len(rows)), K_idx] = S
            self._wedge_tables[key] = (I_idx, J_idx, scatter)
        return self._wedge_tables[key]

    def wedge(self, a, p: int, b, q: int) -> np.ndarray:
        """Wedge product of a p-form ``a`` and a q-form ``b``."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.shape[-1] != self.dim(p) or b.shape[-1] != self.dim(q):
            raise ValueError("component count does not match degree")
        lead = np.broadcast_shapes(a.shape[:-1], b.shape[:-1])
        r = p + q
        if r > self.n:
            return np.zeros(lead + (0,))
        I_idx, J_idx, scatter = self._wedge_table(p, q)
        prod = a[..., I_idx] * b[..., J_idx]
        return prod @ scatter

    def wedge_vectors(self, *vectors) -> np.ndarray:
        """``v_1 ^ v_2 ^ ... ^ v_k`` for 1-forms (vectors)."""
        out = np.asarray(vectors[0], dtype=float)
        p = 1
        for v in vectors[1:]:
            out = self.wedge(out, p, v, 1)
            p += 1
        return out

    def wedge_power(self, a, p: int, k: int) -> np.ndarray:
        """``a ^ a ^ ... ^ a`` (k factors)."""
        if k == 0:
            lead = np.asarray(a).shape[:-1]
            return np.ones(lead + (1,))
        out = np.asarray(a, dtype=float)
        deg = p
        for _ in range(k - 1):
            if deg + p > self.n:
                return np.zeros(out.shape[:-1] + (0,))
            out = self.wedge(out, deg, a, p)
            deg += p
        return out

    # ------------------------------------------------------------------ hodge
    def _hodge_table(self, p: int):
        if p not in self._hodge_tables:
            n = self.n
            src, dst, sgn = [], [], []
            for i, I in enumerate(self.basis[p]):
                Ic = tuple(j for j in range(n) if j not in I)
                src.append(i)
                dst.append(self.index[n - p][Ic])
                sgn.append(permutation_sign(I + Ic))
            self._hodge_tables[p] = (np.array(src), np.array(dst), np.array(sgn, dtype=float))
        return self._hodge_tables[p]

    def hodge(self, a, p: int) -> np.ndarray:
        """Hodge star: p-form -> (n-p)-form, with ``e_I ^ *e_I = e_{1..n}``."""
        a = np.asarray(a, dtype=float)
        src, dst, sgn = self._hodge_table(p)
        out = np.zeros(a.shape[:-1] + (self.dim(self.n - p),))
        out[..., dst] = a[..., src] * sgn
        return out

    # ------------------------------------------------------------------ interior product
    def _interior_table(self, p: int):
        """Table for ``iota_v a`` with ``a`` a p-form: (v index, a index, out index, sign)."""
        if p not in self._interior_tables:
            rows = []
            for i, I in enumerate(self.basis[p]):
                for pos, idx in enumerate(I):
                    J = I[:pos] + I[pos + 1:]
                    rows.append((idx, i, self.index[p - 1][J], (-1) ** pos))
            V_idx, A_idx, O_idx, S = (np.array(c) for c in zip(*rows))
            scatter = np.zeros((len(rows), self.dim(p - 1)))
            scatter[np.arange(len(rows)), O_idx] = S
            self._interior_tables[p] = (V_idx, A_idx, scatter)
        return self._interior_tables[p]

    def interior(self, v, a, p: int) -> np.ndarray:
        """Interior product (contraction) ``iota_v a`` of a vector into a p-form."""
        if p == 0:
            lead = np.broadcast_shapes(np.shape(v)[:-1], np.shape(a)[:-1])
            return np.zeros(lead + (0,))
        v = np.asarray(v, dtype=float)
        a = np.asarray(a, dtype=float)
        V_idx, A_idx, scatter = self._interior_table(p)
        return (v[..., V_idx] * a[..., A_idx]) @ scatter

    # ------------------------------------------------------------------ conversions
    def from_antisymmetric(self, M) -> np.ndarray:
        """Antisymmetric matrix field ``(..., n, n)`` -> 2-form components."""
        M = np.asarray(M)
        I = np.array([ij[0] for ij in self.basis[2]])
        J = np.array([ij[1] for ij in self.basis[2]])
        return M[..., I, J]

    def to_antisymmetric(self, a2) -> np.ndarray:
        """2-form components -> antisymmetric matrix field ``(..., n, n)``."""
        a2 = np.asarray(a2)
        out = np.zeros(a2.shape[:-1] + (self.n, self.n))
        for idx, (i, j) in enumerate(self.basis[2]):
            out[..., i, j] = a2[..., idx]
            out[..., j, i] = -a2[..., idx]
        return out

    # ------------------------------------------------------------------ metrics
    @staticmethod
    def norm(a) -> np.ndarray:
        a = np.asarray(a)
        if a.shape[-1] == 0:
            return np.zeros(a.shape[:-1])
        return np.sqrt((a * a).sum(-1))

    @staticmethod
    def unit(a, eps: float = 1e-300) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        nrm = np.sqrt((a * a).sum(-1, keepdims=True))
        return a / np.maximum(nrm, eps)

    def plane_of_bivector(self, b2) -> np.ndarray:
        """For a (nearly) decomposable bivector, return an orthonormal frame
        ``(..., n, 2)`` of its plane via the dominant singular pair."""
        M = self.to_antisymmetric(b2)
        S = -M @ M
        w, V = np.linalg.eigh(S)
        u = V[..., :, -1]
        Mu = np.einsum("...ij,...j->...i", M, u)
        rate = np.sqrt(np.maximum(w[..., -1], 0))
        v = Mu / np.maximum(rate[..., None], 1e-300)
        return np.stack([u, v], axis=-1)
