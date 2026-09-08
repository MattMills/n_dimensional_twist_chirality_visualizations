import numpy as np
import pytest

from twistchiral.exterior import ExteriorAlgebra, permutation_sign


def test_permutation_sign():
    assert permutation_sign((0, 1, 2)) == 1
    assert permutation_sign((1, 0, 2)) == -1
    assert permutation_sign((2, 0, 1)) == 1


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_wedge_anticommutes_and_triple_product(n):
    ea = ExteriorAlgebra(n)
    rng = np.random.default_rng(n)
    u, v, w = rng.standard_normal((3, 7, n))
    uv = ea.wedge(u, 1, v, 1)
    vu = ea.wedge(v, 1, u, 1)
    assert np.allclose(uv, -vu)
    # |u ^ v|^2 = |u|^2 |v|^2 - (u.v)^2
    assert np.allclose(ea.norm(uv) ** 2, (u * u).sum(-1) * (v * v).sum(-1) - (u * v).sum(-1) ** 2)
    uvw = ea.wedge_vectors(u, v, w)
    # Gram determinant
    G = np.einsum("pi,pj->p", u, u) * 0  # placeholder
    for p in range(7):
        M = np.stack([u[p], v[p], w[p]])
        assert np.isclose(ea.norm(uvw)[p] ** 2, np.linalg.det(M @ M.T))
    if n == 3:
        assert np.allclose(uvw[:, 0], np.linalg.det(np.stack([u, v, w], 1)))


def test_wedge_associative_and_graded_commutative():
    ea = ExteriorAlgebra(5)
    rng = np.random.default_rng(1)
    a = rng.standard_normal((4, ea.dim(1)))
    b = rng.standard_normal((4, ea.dim(2)))
    c = rng.standard_normal((4, ea.dim(2)))
    lhs = ea.wedge(ea.wedge(a, 1, b, 2), 3, c, 2)
    rhs = ea.wedge(a, 1, ea.wedge(b, 2, c, 2), 4)
    assert np.allclose(lhs, rhs)
    # 2-forms commute
    assert np.allclose(ea.wedge(b, 2, c, 2), ea.wedge(c, 2, b, 2))
    # 1-form and 2-form commute
    assert np.allclose(ea.wedge(a, 1, b, 2), ea.wedge(b, 2, a, 1))


@pytest.mark.parametrize("n,p", [(3, 1), (4, 2), (5, 3), (6, 2)])
def test_hodge_is_isometric_involution_up_to_sign(n, p):
    ea = ExteriorAlgebra(n)
    rng = np.random.default_rng(p)
    a = rng.standard_normal((5, ea.dim(p)))
    s = ea.hodge(a, p)
    assert np.allclose(ea.norm(a), ea.norm(s))
    ss = ea.hodge(s, n - p)
    assert np.allclose(ss, (-1) ** (p * (n - p)) * a)
    # a ^ *a = |a|^2 vol
    top = ea.wedge(a, p, s, n - p)
    assert np.allclose(top[:, 0], ea.norm(a) ** 2)


def test_hodge_3d_cross_product():
    ea = ExteriorAlgebra(3)
    u = np.array([[1.0, 0, 0]])
    v = np.array([[0, 1.0, 0]])
    assert np.allclose(ea.hodge(ea.wedge(u, 1, v, 1), 2), [[0, 0, 1.0]])


def test_interior_product():
    ea = ExteriorAlgebra(4)
    rng = np.random.default_rng(3)
    u, v, w = rng.standard_normal((3, 6, 4))
    uvw = ea.wedge_vectors(u, v, w)
    # iota_u (u ^ v ^ w) = |u|^2 v^w - (u.v) u^w + (u.w) u^v
    expect = ((u * u).sum(-1)[:, None] * ea.wedge(v, 1, w, 1)
              - (u * v).sum(-1)[:, None] * ea.wedge(u, 1, w, 1)
              + (u * w).sum(-1)[:, None] * ea.wedge(u, 1, v, 1))
    assert np.allclose(ea.interior(u, uvw, 3), expect)


def test_antisymmetric_roundtrip_and_wedge_power():
    ea = ExteriorAlgebra(6)
    rng = np.random.default_rng(4)
    M = rng.standard_normal((3, 6, 6))
    M = M - np.swapaxes(M, 1, 2)
    b = ea.from_antisymmetric(M)
    assert np.allclose(ea.to_antisymmetric(b), M)
    # F^3 / 3! equals the Pfaffian (top form), |Pf| = sqrt(det)
    top = ea.wedge_power(b, 2, 3) / 6.0
    assert np.allclose(np.abs(top[:, 0]), np.sqrt(np.abs(np.linalg.det(M))))
    # rank-2 bivector squares to zero
    u, v = rng.standard_normal((2, 3, 6))
    assert np.allclose(ea.wedge_power(ea.wedge(u, 1, v, 1), 2, 2), 0)
