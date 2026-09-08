from fractions import Fraction

import numpy as np
import pytest

from twistchiral import exact as ex
from twistchiral.analysis import geometry_at
from twistchiral.exterior import ExteriorAlgebra
from twistchiral.qgt import quantum_geometry
from twistchiral.waves import WaveSystem, WaveVolume

TAU = np.pi / 2


def two_packets(n=4):
    p1 = ex.Packet.carrier_with_sidebands([1, 0, 2, 0][:n], [[1, 0, 0, 0][:n], [0, 1, 0, 0][:n]], (2, 0), (1, 1), center=[0] * n)
    R = ex.plane_rotation_90(n, 0, 1)
    p2 = p1.twisted(R, center=[1, 0, 2, 1][:n]).with_phase(1)
    return ex.LatticeSystem([p1, p2])


def float_twin(system: ex.LatticeSystem) -> WaveSystem:
    vols = []
    for pk in system.packets:
        K = np.array(pk.K.tolist(), dtype=float)
        amps = np.array([a + 1j * b for a, b in pk.c])
        # psi(x) = sum c exp(i K . (x - x0)) with x = TAU m, x0 = TAU mu  ->  phases anchored at the centre
        x0 = TAU * np.array(pk.center.tolist(), dtype=float)
        vols.append(WaveVolume(k=K, center=x0, width=None, amplitudes=amps * np.exp(-1j * (K @ x0)), dispersion="none"))
    return WaveSystem(vols)


def test_exact_matches_float_pipeline():
    S = two_packets(4)
    M = ex.torus_points(4)
    Psi, T = S.evaluate(M)
    g = ex.exact_geometry(Psi, T)
    X = TAU * np.array(M.tolist(), dtype=float)
    q = quantum_geometry(*float_twin(S).evaluate(X))
    rho = np.array([int(r) for r in g["rho"]], dtype=float)
    ok = rho > 0
    A = np.array(g["A"].tolist(), dtype=float) / rho[:, None]
    F = np.array(g["F"].tolist(), dtype=float) / rho[:, None, None] ** 2
    assert np.allclose(A[ok], q["A"][ok], atol=1e-10)
    assert np.allclose(F[ok], q["F"][ok], atol=1e-10)


def test_closed_form_and_factor_identities_hold_everywhere():
    S = two_packets(4)
    M = ex.torus_points(4)
    Psi, T = S.evaluate(M)
    g = ex.exact_geometry(Psi, T)
    pf = ex.pair_factors(Psi, T)
    ea = ex.IntExterior(4)
    assert ex.closed_form_identity(g, pf, ea).all()
    assert ex.factor_identity(g, pf, ea).all()
    # rank of F~ is exactly 2 wherever it is non-zero
    ranks = [ex.bareiss_rank(g["F"][p].tolist()) for p in range(M.shape[0])]
    assert set(ranks) <= {0, 2}


def test_phase_invariance_exact():
    S = two_packets(4)
    M = ex.torus_points(4)[:64]
    res = ex.phase_invariance(S, M, ex.IntExterior(4))
    assert res["F_invariant"] and res["chirality_invariant"] and res["intensity_changes"]


def test_pairwise_additivity_exact():
    n = 4
    p1 = ex.Packet.carrier_with_sidebands([1, 0, 0, 0], [[0, 1, 0, 0]], (2, 0), (1, 0), center=[0, 0, 0, 0])
    p2 = ex.Packet.carrier_with_sidebands([0, 1, 0, 0], [[0, 0, 1, 0]], (2, 1), (1, 0), center=[1, 1, 0, 0])
    p3 = ex.Packet.carrier_with_sidebands([0, 0, 1, 1], [[1, 0, 0, 0]], (1, 1), (0, 1), center=[0, 1, 1, 0])
    S = ex.LatticeSystem([p1, p2, p3])
    M = ex.torus_points(n)[::3]
    assert ex.pairwise_identity(S, M).all()


def test_ladder_degree_exact():
    ea = ExteriorAlgebra(5)
    n = 5
    # side-bands give the packets an envelope; constant-amplitude packets have F = 0
    packets = [ex.Packet.carrier_with_sidebands([1, 0, 0, 0, 0], [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]], (2, 0), (1, 0), center=[0] * n),
               ex.Packet.carrier_with_sidebands([0, 1, 0, 0, 0], [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]], (2, 1), (1, 0), center=[1, 0, 0, 0, 0]),
               ex.Packet.carrier_with_sidebands([0, 0, 1, 0, 1], [[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]], (1, -1), (0, 1), center=[0, 1, 1, 0, 0])]
    S = ex.LatticeSystem(packets)
    M = ex.lcg_points(n, 40, seed=3)
    Psi, T = S.evaluate(M)
    g = ex.exact_geometry(Psi, T)
    iea = ex.IntExterior(n)
    lad = ex.exact_ladder(iea, g["A"], g["F"])
    top = max(p for p, c in lad.items() if any(int(v) != 0 for v in np.asarray(c, dtype=object).reshape(-1)))
    assert top == min(n, 2 * 3 - 1) == 5
    ranks = {ex.bareiss_rank(g["F"][p].tolist()) for p in range(M.shape[0])}
    assert max(ranks) == 4


def test_mirror_and_saturation_exact():
    ea = ex.IntExterior(4)
    S = two_packets(4)
    M = ex.torus_points(4)[:32]
    g = ex.exact_geometry(*S.evaluate(M))
    C = ea.wedge(g["A"], 1, ea.from_antisymmetric(g["F"]), 2)
    keep = [p for p in range(C.shape[0]) if any(int(v) != 0 for v in C[p])]
    res = ex.mirror_identity(ea, C[keep], 3)
    assert res["identity"] and res["mean_cosine"] == res["predicted"] == Fraction(-1, 2)
    sat = ex.saturation_design_average([1, 0, 2, 0], 4)
    assert sat["average"] == sat["predicted"] == 1 - Fraction(3, 4) + Fraction(2, 16)


def test_direction_statistics_exact():
    C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=object)
    st = ex.direction_statistics(C)
    assert st["count"] == 4
    assert st["effective_dimension"] == Fraction(16, 4 + 6 * Fraction(1, 3) * 1)  # 4 diagonal + 6 pairs with cos^2 = 1/3
