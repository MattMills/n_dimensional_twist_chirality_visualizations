import numpy as np
import pytest

from twistchiral import geometry as g
from twistchiral.analysis import (analytic_two_volume_chirality, geometry_at, ladder_existence,
                                  mirror_comparison, relative_phase_invariance)
from twistchiral.exterior import ExteriorAlgebra
from twistchiral.qgt import quantum_geometry, rotation_planes, twist_rank
from twistchiral.waves import WaveSystem, WaveVolume


def make_system(n, N, m=None, rng=0, width=1.0, kscale=3.0):
    rng = np.random.default_rng(rng)
    m = m or n + 1
    vols = []
    for a in range(N):
        K = kscale * g.random_sphere(n, m, rng)
        amp = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        vols.append(WaveVolume(k=K, center=rng.normal(0, 0.7, n), width=width, amplitudes=amp))
    return WaveSystem(vols)


@pytest.mark.parametrize("n,N", [(3, 2), (4, 2), (4, 3), (5, 3)])
def test_curvature_is_curl_of_connection(n, N):
    """F_ij = d_i A_j - d_j A_i, checked by central differences."""
    S = make_system(n, N)
    rng = np.random.default_rng(1)
    X = rng.normal(0, 0.5, (20, n))
    Psi, dPsi = S.evaluate(X)
    q = quantum_geometry(Psi, dPsi)
    eps = 1e-5
    Fnum = np.zeros((20, n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = eps
        Ap, Am = (quantum_geometry(*S.evaluate(X + e))["A"], quantum_geometry(*S.evaluate(X - e))["A"])
        dA_i = (Ap - Am) / (2 * eps)      # d_i A_j
        Fnum[:, i, :] += dA_i
        Fnum[:, :, i] -= dA_i
    assert np.allclose(Fnum, q["F"], atol=1e-5 * np.abs(q["F"]).max() + 1e-7)


@pytest.mark.parametrize("n,N", [(3, 2), (4, 2), (5, 2), (6, 3), (5, 3)])
def test_hermitian_and_rank_bound(n, N):
    S = make_system(n, N, rng=n + N)
    X = np.random.default_rng(2).normal(0, 0.6, (300, n))
    q = quantum_geometry(*S.evaluate(X))
    assert np.allclose(q["Q"], np.conj(np.swapaxes(q["Q"], 1, 2)))
    assert np.allclose(q["F"], -np.swapaxes(q["F"], 1, 2))
    rank, rates = twist_rank(q["F"], rtol=1e-8)
    assert rank.max() <= min(n // 2, N - 1)
    # the metric has the same rank bound
    gr = np.linalg.matrix_rank(q["g"], tol=1e-8 * np.abs(q["g"]).max())
    assert gr.max() <= min(n, 2 * (N - 1))


def test_gradient_is_analytic():
    S = make_system(4, 2, rng=5)
    X = np.random.default_rng(3).normal(0, 0.5, (10, 4))
    Psi, dPsi = S.evaluate(X)
    eps = 1e-6
    for i in range(4):
        e = np.zeros(4)
        e[i] = eps
        num = (S.evaluate(X + e)[0] - S.evaluate(X - e)[0]) / (2 * eps)
        assert np.allclose(num, dPsi[:, i, :], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_closed_form_two_volume_chirality(n):
    S = make_system(n, 2, rng=10 + n)
    X = np.random.default_rng(4).normal(0, 0.6, (400, n))
    q = geometry_at(S, X)
    an = analytic_two_volume_chirality(S, X)
    ea = ExteriorAlgebra(n)
    assert np.allclose(ea.from_antisymmetric(q["F"]), an["F"], atol=1e-9 * np.abs(an["F"]).max())
    assert np.allclose(q["ladder"][3], an["C3"], atol=1e-9 * np.abs(an["C3"]).max())


def test_gaussian_pair_chirality_is_screw_chirality():
    """Single-plane-wave Gaussian volumes: C3 = (V^2 / 2 sigma^2) k1 ^ k2 ^ d."""
    n, sigma = 4, 1.3
    k1 = np.array([2.0, 0.5, 0.0, 0.3])
    R = g.rotation_from_angles(n, [0.7, 1.9])
    k2 = R @ k1
    d = np.array([0.4, -0.2, 0.9, 0.6])
    v1 = WaveVolume(k=k1[None], center=np.zeros(n), width=sigma)
    v2 = WaveVolume(k=k2[None], center=d, width=sigma)
    S = WaveSystem([v1, v2])
    X = np.random.default_rng(6).normal(0, 0.8, (200, n))
    q = geometry_at(S, X)
    ea = ExteriorAlgebra(n)
    h = (-(X - 0) ** 2).sum(-1) / sigma**2 + ((X - d) ** 2).sum(-1) / sigma**2   # ln(A1^2/A2^2)
    V2 = 1 / np.cosh(0.5 * h) ** 2
    # grad ln(A1/A2) = -d / sigma^2, so A ^ F = -(V^2 / 2 sigma^2) k1 ^ k2 ^ d
    expect = -0.5 * V2[:, None] / sigma**2 * ea.wedge_vectors(k1[None], k2[None], d[None])
    assert np.allclose(q["ladder"][3], expect, atol=1e-10)
    from twistchiral.analysis import screw_prediction
    assert np.allclose(q["ladder"][3], screw_prediction(S, X), atol=1e-10)
    # the direction is constant over space: effective dimension 1
    from twistchiral.analysis import direction_spectrum
    assert abs(direction_spectrum(q["ladder"][3])["effective_dimension"] - 1.0) < 1e-8


def test_screw_chirality_sign_convention():
    ea = ExteriorAlgebra(3)
    R = g.plane_rotation(3, 0, 1, 0.5)          # e0 turns toward e1 (counter-clockwise about +e2)
    assert g.screw_chirality(ea, R, [0, 0, 2.0])[0] > 0   # advancing along +e2: right-handed
    assert g.screw_chirality(ea, R, [0, 0, -2.0])[0] < 0
    # the Chern-Simons chirality A ^ F of two single-wave volumes has the
    # OPPOSITE sign to the geometric screw (Berry-curvature orientation):
    # A ^ F = -(V^2 / 2 sigma^2) (k1 ^ k2) ^ d
    k1 = np.array([2.0, 0.0, 0.0])
    v1 = WaveVolume(k=k1[None], center=np.zeros(3), width=1.0)
    v2 = WaveVolume(k=(R @ k1)[None], center=[0, 0, 0.8], width=1.0)
    X = np.array([[0.1, 0.2, 0.4]])
    q = geometry_at(WaveSystem([v1, v2]), X)
    assert q["ladder"][3][0, 0] < 0
    assert np.sign(q["ladder"][3][0, 0]) == -np.sign(g.screw_chirality(ea, R, [0, 0, 0.8])[0])


def test_relative_phase_invariance():
    S = make_system(4, 3, rng=7)
    X = np.random.default_rng(8).normal(0, 0.6, (100, 4))
    res = relative_phase_invariance(S, X)
    assert res["F"] < 1e-10 and res["C3"] < 1e-10
    assert res["intensity"] > 1e-2          # the fringes themselves do move


def test_time_invariance_for_monochromatic_volumes():
    """Equal |k| inside each volume: time only shifts relative phases, so the
    twist and chirality fields are stationary while the fringes move."""
    n = 4
    rng = np.random.default_rng(9)
    # equal |k| inside each volume (one frequency per volume), different between volumes
    vols = [WaveVolume(k=kk * g.random_sphere(n, 5, rng), center=rng.normal(0, 0.5, n), width=1.0,
                       amplitudes=rng.standard_normal(5), dispersion="linear") for kk in (2.5, 3.5)]
    S = WaveSystem(vols)
    X = rng.normal(0, 0.6, (100, n))
    q0 = geometry_at(S, X, t=0.0)
    q1 = geometry_at(S, X, t=1.37)
    assert np.allclose(q0["F"], q1["F"], atol=1e-10 * np.abs(q0["F"]).max())
    assert np.allclose(q0["ladder"][3], q1["ladder"][3], atol=1e-10 * np.abs(q0["ladder"][3]).max())
    I0 = np.abs(S.total_field(X, 0.0)) ** 2
    I1 = np.abs(S.total_field(X, 1.37)) ** 2
    assert not np.allclose(I0, I1)


@pytest.mark.parametrize("n,N", [(3, 2), (4, 2), (4, 3), (5, 2), (5, 3), (6, 3), (6, 4), (7, 4)])
def test_ladder_ends_at_predicted_degree(n, N):
    S = make_system(n, N, rng=100 + 10 * n + N)
    X = np.random.default_rng(5).normal(0, 0.5, (300, n))
    res = ladder_existence(S, X)
    assert res["max_degree"] == res["predicted_max_degree"] == min(n, 2 * N - 1)


def test_mirror_flips_sign_in_3d_but_rotates_direction_in_4d():
    S3 = make_system(3, 2, rng=11)
    X3 = np.random.default_rng(12).normal(0, 0.6, (60, 3))
    r3 = mirror_comparison(S3, [0.3, 1.0, -0.2], X3)
    assert r3["tensor_error"] < 1e-9 and r3["sign_flip_error"] < 1e-9
    S4 = make_system(4, 2, rng=13)
    X4 = np.random.default_rng(14).normal(0, 0.6, (60, 4))
    r4 = mirror_comparison(S4, [0.3, 1.0, -0.2, 0.5], X4)
    assert r4["tensor_error"] < 1e-9
    assert r4["negated_fraction"] < 0.5      # generic directions are rotated, not negated


def test_rotation_planes_reconstruct_F():
    S = make_system(6, 4, rng=15)
    X = np.random.default_rng(16).normal(0, 0.5, (50, 6))
    q = quantum_geometry(*S.evaluate(X))
    rates, frames = rotation_planes(q["F"])
    rec = np.zeros_like(q["F"])
    for j in range(rates.shape[1]):
        u, v = frames[:, j, :, 0], frames[:, j, :, 1]
        rec += rates[:, j, None, None] * (v[:, :, None] * u[:, None, :] - u[:, :, None] * v[:, None, :])
    assert np.allclose(rec, q["F"], atol=1e-9 * np.abs(q["F"]).max())


def test_twist_flux_quantum_is_2pi():
    """Flux of F through a strip crossing the interface and spanning one
    fringe period is exactly 2 pi (half the area of the Bloch sphere)."""
    n = 3
    k1 = np.array([2.0, 0.0, 0.0])
    k2 = np.array([0.0, 2.0, 0.0])
    d = np.array([0.0, 0.0, 1.0])
    S = WaveSystem([WaveVolume(k=k1[None], center=np.zeros(n), width=1.0), WaveVolume(k=k2[None], center=d, width=1.0)])
    # strip in the plane spanned by q = k2 - k1 (fringe direction) and d (across the interface)
    q = k2 - k1
    qh = q / np.linalg.norm(q)
    period = 2 * np.pi / np.linalg.norm(q)
    ns, nt = 400, 4000
    s = np.linspace(0, period, ns, endpoint=False) + 0.5 * period / ns
    tt = np.linspace(-12, 13, nt)
    Sg, Tg = np.meshgrid(s, tt, indexing="ij")
    X = Sg.reshape(-1, 1) * qh[None, :] + Tg.reshape(-1, 1) * d[None, :]
    Psi, dPsi = S.evaluate(X)
    F = quantum_geometry(Psi, dPsi)["F"]
    Fst = np.einsum("i,pij,j->p", qh, F, d)
    flux = Fst.sum() * (period / ns) * (tt[1] - tt[0])
    assert abs(abs(flux) - 2 * np.pi) < 1e-3


def test_twist_is_weighted_sum_of_pairwise_twists():
    from twistchiral.qgt import pairwise_decomposition
    for n, N in ((4, 3), (5, 4), (6, 5)):
        S = make_system(n, N, rng=200 + n)
        X = np.random.default_rng(n).normal(0, 0.5, (80, n))
        Psi, dPsi = S.evaluate(X)
        F = quantum_geometry(Psi, dPsi)["F"]
        dec = pairwise_decomposition(Psi, dPsi)
        assert np.allclose(dec["F_reconstructed"], F, atol=1e-10 * np.abs(F).max())
        # second Chern form = cross terms between different pairs only
        ea = ExteriorAlgebra(n)
        pairs = dec["pairs"]
        cross = np.zeros((80, ea.dim(4)))
        for i, p in enumerate(pairs):
            Fp = ea.from_antisymmetric(dec["F_pairs"][p])
            assert np.allclose(ea.wedge(Fp, 2, Fp, 2), 0, atol=1e-9 * (np.abs(Fp).max() ** 2 + 1e-300))
            for qq in pairs[i + 1:]:
                Fq = ea.from_antisymmetric(dec["F_pairs"][qq])
                cross += 2 * (dec["weights"][p] * dec["weights"][qq])[:, None] * ea.wedge(Fp, 2, Fq, 2)
        F2 = ea.from_antisymmetric(F)
        assert np.allclose(cross, ea.wedge(F2, 2, F2, 2), atol=1e-9 * (np.abs(F2).max() ** 2))


def test_rotation_taking():
    rng = np.random.default_rng(0)
    for n in (3, 4, 6):
        a, b = rng.standard_normal((2, n))
        R = g.rotation_taking(a, b)
        assert np.allclose(R @ R.T, np.eye(n)) and np.isclose(np.linalg.det(R), 1)
        assert np.allclose(R @ (a / np.linalg.norm(a)), b / np.linalg.norm(b))
