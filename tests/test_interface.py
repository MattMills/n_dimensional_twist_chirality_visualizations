import numpy as np

from twistchiral import geometry as g
from twistchiral.interface import (dominance_mask, project_to_interface, project_to_junction,
                                   sample_interface, sample_junction, visibility)
from twistchiral.waves import WaveSystem, WaveVolume


def test_equal_width_gaussians_meet_on_bisector_hyperplane():
    n = 5
    rng = np.random.default_rng(0)
    c1, c2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    v1 = WaveVolume(k=2 * rng.standard_normal((1, n)), center=c1, width=1.0)
    v2 = WaveVolume(k=2 * rng.standard_normal((1, n)), center=c2, width=1.0)
    S = WaveSystem([v1, v2])
    samp = sample_interface(S, count=500, rng=1)
    X = samp["X"]
    d = c2 - c1
    mid = 0.5 * (c1 + c2)
    assert np.allclose((X - mid) @ d, 0, atol=1e-6)
    assert np.allclose(np.abs(samp["normal"] @ d / np.linalg.norm(d)), 1, atol=1e-6)
    assert np.allclose(visibility(S, X, 0, 1), 1.0)


def test_unequal_widths_give_sphere():
    n = 3
    c1, c2 = np.zeros(n), np.array([2.0, 0, 0])
    s1, s2 = 1.0, 1.5
    S = WaveSystem([WaveVolume(k=[[3, 0, 0]], center=c1, width=s1), WaveVolume(k=[[0, 3, 0]], center=c2, width=s2)])
    X = sample_interface(S, count=400, rng=2)["X"]
    # |x-c1|^2/s1^2 = |x-c2|^2/s2^2 is a sphere
    f = ((X - c1) ** 2).sum(-1) / s1**2 - ((X - c2) ** 2).sum(-1) / s2**2
    assert np.allclose(f, 0, atol=1e-6)


def test_junction_and_dominance():
    n = 4
    rng = np.random.default_rng(3)
    # centres at the vertices of a regular simplex: every triple shares a junction
    C = 1.2 * g.simplex(n)[:4]
    vols = [WaveVolume(k=2 * rng.standard_normal((1, n)), center=C[a], width=1.0) for a in range(4)]
    S = WaveSystem(vols)
    j = sample_junction(S, (0, 1, 2), count=800, rng=4)
    X = j["X"]
    assert len(X) > 10
    I = S.intensities(X)
    assert np.allclose(np.log(I[:, 0]) - np.log(I[:, 1]), 0, atol=1e-5)
    assert np.allclose(np.log(I[:, 0]) - np.log(I[:, 2]), 0, atol=1e-5)
    assert np.all(I[:, :3].min(-1) >= I[:, 3] - 1e-9)
    assert dominance_mask(S, X, (0, 1, 2)).all()
