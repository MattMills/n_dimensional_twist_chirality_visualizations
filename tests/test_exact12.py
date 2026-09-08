from fractions import Fraction

import numpy as np

from twistchiral import exact12 as ex
from twistchiral.qgt import quantum_geometry
from twistchiral.waves import WaveSystem, WaveVolume


def two_packets(n=4, order=12):
    p1 = ex.Packet.carrier_with_sidebands([1, 0, 2, 0][:n], [[1, 0, 0, 0][:n], [0, 1, 0, 0][:n]], (2, 0), (1, 1), center=[0] * n, order=order)
    p2 = p1.twisted(ex.plane_rotation_90(n, 0, 1), center=[3, 0, 7, 2][:n])
    return ex.LatticeSystem([p1, p2])


def float_twin(system, order):
    tau = 2 * np.pi / order
    vols = []
    for pk in system.packets:
        K = np.array([[int(v) for v in row] for row in pk.K], dtype=float)
        amps = np.array([complex(float(a.a) + float(a.b) * 3**0.5, float(b.a) + float(b.b) * 3**0.5) for a, b in pk.c])
        x0 = tau * np.array([int(v) for v in pk.center], dtype=float)
        vols.append(WaveVolume(k=K, center=x0, width=None, amplitudes=2 * amps * np.exp(-1j * (K @ x0)), dispersion="none"))
    return WaveSystem(vols)


def to_float(arr):
    return np.array([[float(ex.R3.of(v).a) + float(ex.R3.of(v).b) * 3**0.5 for v in row] for row in np.asarray(arr, dtype=object).reshape(arr.shape[0], -1)]).reshape(arr.shape)


def test_r3_arithmetic_and_rendering():
    x = ex.R3(1, 1)            # 1 + sqrt3
    y = ex.R3(2, -1)           # 2 - sqrt3
    assert (x * y) == ex.R3(-1, 1)
    assert (x * y) // y == x
    assert ex.R3(0, 1).sign() == 1 and ex.R3(2, -1).sign() == 1 and ex.R3(-2, 1).sign() == -1 and ex.R3(1, -1).sign() == -1
    assert ex.R3(0, 1).decimal(4) == "1.7320"
    assert (ex.R3(1, 0) / ex.R3(0, 1)) == ex.R3(0, Fraction(1, 3))
    assert ex.R3(-1, 0).decimal(2) == "-1.00"


def test_matches_float_pipeline_order_12_and_4():
    for order in (12, 4):
        S = two_packets(4, order)
        M = ex.lcg_points(4, 60, seed=order, period=order)
        Psi, T = S.evaluate(M)
        g = ex.exact_geometry(Psi, T)
        X = (2 * np.pi / order) * np.array([[int(v) for v in row] for row in M], dtype=float)
        q = quantum_geometry(*float_twin(S, order).evaluate(X))
        rho = to_float(g["rho"].reshape(-1, 1))[:, 0]
        ok = rho > 1e-9
        A = to_float(g["A"]) / rho[:, None]
        F = to_float(g["F"]) / rho[:, None, None] ** 2
        assert np.allclose(A[ok], q["A"][ok], atol=1e-9)
        assert np.allclose(F[ok], q["F"][ok], atol=1e-9)


def test_identities_exact_order_12():
    S = two_packets(4)
    M = ex.lcg_points(4, 80, seed=5)
    Psi, T = S.evaluate(M)
    g = ex.exact_geometry(Psi, T)
    pf = ex.pair_factors(Psi, T)
    ea = ex.RingExterior(4)
    assert ex.closed_form_identity(g, pf, ea).all()
    assert ex.factor_identity(g, pf, ea).all()
    assert set(ex.bareiss_rank(g["F"][p].tolist()) for p in range(M.shape[0])) <= {0, 2}
    res = ex.phase_invariance(S, M[:12], ea)
    assert res["F_invariant"] and res["chirality_invariant"] and res["intensity_changes"]
    C = ea.wedge(g["A"], 1, ea.from_antisymmetric(g["F"]), 2)
    keep = [p for p in range(C.shape[0]) if ex.nonzero(C[p])]
    mir = ex.mirror_identity(ea, C[keep[:20]], 3)
    assert mir["identity"] and mir["mean_cosine"] == mir["predicted"]
    sat = ex.saturation_design_average([1, 0, 2, 0], 4)
    assert sat["average"] == sat["predicted"]
    cf = ex.closed_form_identity_highn({"rho": g["rho"], "A": g["A"]}, pf)
    assert cf.all()


def test_pairwise_and_ladder_order_12():
    n = 5
    packets = [ex.Packet.carrier_with_sidebands([1, 0, 0, 0, 0], [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]], (2, 0), (1, 0), center=[0] * n),
               ex.Packet.carrier_with_sidebands([0, 1, 0, 0, 0], [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]], (2, 1), (1, 0), center=[1, 0, 0, 0, 0]),
               ex.Packet.carrier_with_sidebands([0, 0, 1, 0, 1], [[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]], (1, -1), (0, 1), center=[0, 1, 1, 0, 0])]
    S = ex.LatticeSystem(packets)
    M = ex.lcg_points(n, 25, seed=3)
    assert ex.pairwise_identity(S, M[:8]).all()
    g = ex.exact_geometry(*S.evaluate(M))
    lad = ex.exact_ladder(ex.RingExterior(n), g["A"], g["F"])
    top = max(p for p, c in lad.items() if ex.nonzero(c))
    assert top == 5
    assert max(ex.bareiss_rank(g["F"][p].tolist()) for p in range(M.shape[0])) == 4
