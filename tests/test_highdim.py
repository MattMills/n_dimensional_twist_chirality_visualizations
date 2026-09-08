import numpy as np
import pytest

from twistchiral import geometry as g
from twistchiral.analysis import direction_spectrum, geometry_at
from twistchiral.exterior import ExteriorAlgebra
from twistchiral.highdim import (chirality3_kernel, expected_mirror_cosine, factored_geometry, geometry_factored,
                                 kernel_direction_spectrum, ladder_norms_factored, mirror_cosines, reconstruct_F,
                                 screw_saturation, elementary_symmetric, leave_one_out_symmetric)
from twistchiral.qgt import quantum_geometry
from twistchiral.waves import WaveSystem, WaveVolume
from twistchiral.systems import cluster, twisted_pair


def make_system(n, N, m=None, rng=0):
    rng = np.random.default_rng(rng)
    m = m or n + 1
    vols = []
    for a in range(N):
        K = 3.0 * g.random_sphere(n, m, rng)
        amp = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        vols.append(WaveVolume(k=K, center=rng.normal(0, 0.7, n), width=1.0, amplitudes=amp))
    return WaveSystem(vols)


@pytest.mark.parametrize("n,N", [(4, 2), (6, 3), (7, 4), (9, 5)])
def test_factorisation_reproduces_curvature(n, N):
    S = make_system(n, N, rng=n)
    X = np.random.default_rng(1).normal(0, 0.5, (40, n))
    Psi, dPsi = S.evaluate(X)
    q = quantum_geometry(Psi, dPsi)
    geo = factored_geometry(Psi, dPsi)
    assert np.allclose(geo["A"], q["A"])
    assert np.allclose(reconstruct_F(geo), q["F"], atol=1e-9 * np.abs(q["F"]).max())
    # orthonormal frames
    UV = np.concatenate([geo["U"], geo["V"]], axis=1)
    Gm = np.einsum("pjn,pkn->pjk", UV, UV)
    assert np.allclose(Gm, np.eye(UV.shape[1])[None], atol=1e-8)


def test_elementary_symmetric():
    x = np.array([[1.0, 2.0, 3.0]])
    e = elementary_symmetric(x, 3)
    assert np.allclose(e, [[1, 6, 11, 6]])
    assert np.allclose(leave_one_out_symmetric(x, 2), [[6, 3, 2]])
    assert np.allclose(leave_one_out_symmetric(x, 1), [[5, 4, 3]])
    # stability when one variable dominates
    y = np.array([[1e6, 1e-3, 2e-3, 3e-3]])
    loo = leave_one_out_symmetric(y, 3)
    assert np.isclose(loo[0, 0], 1e-3 * 2e-3 * 3e-3, rtol=1e-12)


@pytest.mark.parametrize("n,N", [(5, 2), (7, 3), (8, 4), (9, 5)])
def test_ladder_norms_match_components(n, N):
    S = make_system(n, N, rng=10 + n)
    X = np.random.default_rng(2).normal(0, 0.5, (30, n))
    q = geometry_at(S, X)
    geo = factored_geometry(q["Psi"], q["dPsi"])
    norms = ladder_norms_factored(geo, n)
    for p, nrm in q["norms"].items():
        scale = nrm.max() + 1e-300
        if p in norms:
            assert np.allclose(norms[p], nrm, atol=1e-8 * scale), p
        else:  # rungs beyond the rank vanish
            assert nrm.max() < 1e-8 * (q["norms"][2].max() ** (p // 2)) * (q["norms"][1].max() ** (p % 2))


@pytest.mark.parametrize("n,N", [(6, 2), (6, 3)])
def test_kernel_matches_component_inner_products(n, N):
    S = make_system(n, N, rng=20 + n + N)
    X = np.random.default_rng(3).normal(0, 0.5, (25, n))
    q = geometry_at(S, X, max_degree=3)
    C3 = q["ladder"][3]
    geo = factored_geometry(q["Psi"], q["dPsi"])
    K = chirality3_kernel(geo)
    assert np.allclose(K, C3 @ C3.T, atol=1e-8 * np.abs(C3 @ C3.T).max())
    sp_direct = direction_spectrum(C3, weights=q["rho"])
    sp_kernel = kernel_direction_spectrum(K, weights=q["rho"])
    assert abs(sp_direct["effective_dimension"] - sp_kernel["effective_dimension"]) < 1e-6
    assert abs(sp_direct["variation_dimension"] - sp_kernel["variation_dimension"]) < 1e-6
    m = np.random.default_rng(4).standard_normal(n)
    cos_k = mirror_cosines(geo, m)
    Rm = g.reflection(n, m)
    from twistchiral.analysis import mirror_transform_of_form
    ea = ExteriorAlgebra(n)
    RC = mirror_transform_of_form(ea, Rm, C3, 3)
    cos_d = (RC * C3).sum(1) / (C3 * C3).sum(1)
    assert np.allclose(cos_k, cos_d, atol=1e-8)


def test_mean_mirror_cosine_is_one_minus_six_over_n():
    rng = np.random.default_rng(5)
    for n in (3, 4, 8, 32):
        S = twisted_pair(n, "random", k_scale=3.0, rotation=g.random_rotation(n, rng), displacement=rng.normal(0, 0.4, n),
                         width=1.0, amplitudes="random", rng=int(rng.integers(1 << 30)), m=6)
        X = rng.normal(0, 0.5, (60, n))
        geo = geometry_factored(S, X)
        # exact average over the sphere of normals: E[cos] = 1 - 2p/n  (p = 3)
        cos = np.mean([mirror_cosines(geo, rng.standard_normal(n)) for _ in range(400)], axis=0)
        assert abs(cos.mean() - expected_mirror_cosine(3, n)) < 0.08
        if n == 3:
            assert np.allclose(cos, -1.0)


def test_high_dimension_smoke_and_closed_form():
    n = 512
    rng = np.random.default_rng(6)
    S = twisted_pair(n, "random", k_scale=3.0, rotation=g.random_rotation(n, rng), displacement=rng.normal(0, 0.05, n),
                     width=1.0, amplitudes="random", rng=1, m=5)
    X = rng.normal(0, 0.05, (20, n))
    geo = geometry_factored(S, X)
    assert geo["rates"].shape == (20, 1)
    # closed form |A ^ F| = (V^2/2) |k1 ^ k2 ^ grad ln(A1/A2)| via Gram determinants
    from twistchiral.interface import log_amplitude_ratio
    ka, kb = S[0].local_wavevector(X), S[1].local_wavevector(X)
    h, gh = log_amplitude_ratio(S, X, 0, 1)
    V2 = 1 / np.cosh(0.5 * h) ** 2
    M = np.stack([ka, kb, 0.5 * gh], axis=1)
    vol = np.sqrt(np.clip(np.linalg.det(M @ np.swapaxes(M, 1, 2)), 0, None))
    assert np.allclose(geo["norms"][3], 0.5 * V2 * vol, rtol=1e-8)


def test_screw_saturation_matches_wedge_norm():
    ea = ExteriorAlgebra(5)
    rng = np.random.default_rng(7)
    k1, k2, d = rng.standard_normal((3, 4, 5))
    s = screw_saturation(k1, k2, d)
    direct = ea.norm(ea.wedge_vectors(k1, k2, d)) / (np.linalg.norm(k1, axis=1) * np.linalg.norm(k2, axis=1) * np.linalg.norm(d, axis=1))
    assert np.allclose(s, direct)


def test_rank_bound_at_high_dimension():
    n, N = 64, 5
    S = cluster(n, N, "single", k_scale=3.0, width=1.0, rng=8, centers=np.random.default_rng(9).normal(0, 0.3, (N, n)))
    X = np.random.default_rng(10).normal(0, 0.3, (10, n))
    Psi, dPsi = S.evaluate(X)
    q = quantum_geometry(Psi, dPsi)
    sv = np.linalg.svd(q["F"], compute_uv=False)
    assert np.all((sv > 1e-8 * sv[:, :1]).sum(1) == 2 * (N - 1))
    geo = factored_geometry(Psi, dPsi)
    norms = ladder_norms_factored(geo, n)
    assert max(norms) == min(n, 2 * N - 1)
    assert np.all(norms[2 * N - 1] > 0)


def test_random_frame_image_preserves_gram():
    from twistchiral.highdim import random_frame_image, seed_near
    rng = np.random.default_rng(11)
    K = rng.standard_normal((5, 300))
    K2 = random_frame_image(K, rng)
    assert np.allclose(K2 @ K2.T, K @ K.T)
    assert not np.allclose(K2, K)
    X = seed_near(np.zeros(4096), 10, 4096, rng)
    assert X.shape == (10, 4096) and np.linalg.norm(X, axis=1).max() < 30


def test_relative_envelope_leaves_geometry_unchanged():
    from twistchiral.systems import twisted_pair
    S = twisted_pair(6, "simplex", k_scale=3.0, angles=[0.4, 1.0], displacement=[0.3, 0.2, 0.5, 0.1, 0.4, 0.2], width=1.0, amplitudes=0.3)
    Sr = WaveSystem(S.volumes, relative_envelope=True)
    X = np.random.default_rng(12).normal(0, 1.0, (50, 6))
    q, qr = geometry_at(S, X), geometry_at(Sr, X)
    assert np.allclose(q["A"], qr["A"]) and np.allclose(q["F"], qr["F"])
    assert np.allclose(q["ladder"][3], qr["ladder"][3])
    from twistchiral.interface import log_amplitude_ratio
    assert np.allclose(log_amplitude_ratio(S, X, 0, 1)[0], log_amplitude_ratio(Sr, X, 0, 1)[0])
    # no underflow at n = 16384 with unit spread
    n = 16384
    rng = np.random.default_rng(13)
    big = WaveSystem([WaveVolume(k=3 * g.random_sphere(n, 1, rng), center=rng.normal(0, 0.02, n), width=1.0) for _ in range(2)],
                     relative_envelope=True)
    Xb = rng.normal(0, 1.0, (5, n))
    rho = (np.abs(big.evaluate(Xb)[0]) ** 2).sum(1)
    assert np.all(np.isfinite(rho)) and np.all(rho > 1e-200)
