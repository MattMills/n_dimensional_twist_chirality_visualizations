"""Experiment C: how the chirality of a two-volume interface changes with n.

The chirality 3-form has C(n, 3) components.  For n = 3 it is a single
signed number; for n >= 4 the *direction* of the chirality on the interface
spreads over an increasingly large space.  We measure the effective
dimension of that set of directions for several kinds of volume.
"""
from common import Timer, dump, save

import numpy as np
from math import comb

import twistchiral as tc
from twistchiral import viz
from twistchiral.analysis import direction_spectrum, geometry_at, ladder_existence
from twistchiral.exterior import ExteriorAlgebra
from twistchiral.interface import sample_interface


def run(seed: int = 0, ns=(3, 4, 5, 6, 7, 8)) -> dict:
    kinds = {
        "single wave": dict(kind="single"),
        "carrier + side-bands": dict(kind="simplex", amplitudes=0.25),
        "n+1 equal waves (simplex)": dict(kind="simplex"),
        "2n random waves": dict(kind="random", amplitudes="random"),
    }
    rng = np.random.default_rng(seed)
    d_eff = {k: [] for k in kinds}
    d_var = {k: [] for k in kinds}
    spectra = {}
    ladder = []
    res = {"ns": list(ns)}
    with Timer("C1 effective chirality dimension vs n"):
        for n in ns:
            ea = ExteriorAlgebra(n)
            angles = list(0.4 + 0.5 * np.arange(n // 2))
            d = rng.normal(0, 0.5, n)
            d = 1.2 * d / np.linalg.norm(d)
            for key, spec in kinds.items():
                S = tc.twisted_pair(n, k_scale=3.0, angles=angles, displacement=d, width=1.0, rng=seed, m=2 * n, **spec)
                samp = sample_interface(S, count=5000, rng=seed, pad=1.5)
                q = geometry_at(S, samp["X"], ea=ea, max_degree=3)
                sp = direction_spectrum(q["ladder"][3], weights=q["rho"])
                d_eff[key].append(sp["effective_dimension"])
                d_var[key].append(sp["variation_dimension"])
                if key == "n+1 equal waves (simplex)":
                    spectra[f"n = {n}"] = sp
                    ladder.append(ladder_existence(S, samp["X"][:400]))
            print(f"  n={n}: " + ", ".join(f"{k}: {d_eff[k][-1]:.2f}" for k in kinds))
        res["effective_dimension"] = d_eff
        res["variation_dimension"] = d_var
        res["components_C(n,3)"] = [comb(n, 3) for n in ns]
        res["oriented_3plane_dimension_3(n-3)+1"] = [3 * (n - 3) + 1 for n in ns]
        res["ladder_simplex"] = ladder
        fig = viz.plot_effective_dimension(
            list(ns), d_eff, title="Effective dimension of the chirality directions on the interface vs. dimension n")
        fig.axes[0].set_xticklabels([f"n = {n}\n{comb(n, 3)} components" for n in ns])
        save(fig, "C_dimension_sweep_effective_dimension.png")
        fig = viz.plot_direction_spectra(spectra, title="Chirality direction spectra on the interface, n+1 equal waves per volume")
        save(fig, "C_dimension_sweep_spectra.png")
    dump(res, "C_dimension_sweep.json")
    return res


if __name__ == "__main__":
    run()
