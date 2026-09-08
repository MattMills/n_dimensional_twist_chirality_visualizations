"""Experiment A: the twist-chiral interface between two volumes in 3-D.

In three dimensions the chirality 3-form A^F is a pseudoscalar, so the
interface carries a *signed* chirality: right- and left-handed domains
separated by walls on which the two tangential wavevectors align.
"""
from common import FIG_DIR, Timer, dump, save  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt

import twistchiral as tc
from twistchiral import viz
from twistchiral.analysis import (direction_spectrum, geometry_at, localization_profile,
                                  relative_phase_invariance, sign_balance, analytic_two_volume_chirality)
from twistchiral.exterior import ExteriorAlgebra
from twistchiral.interface import sample_interface
from twistchiral.slices import natural_plane


def surface_figure(S, d, name, title_a, title_b, res=64, extent=(2.4, 2.4, 1.6)):
    n = S.n
    frame = tc.geometry.complete_frame(d, n=n)
    surf = viz.interface_surface(S, np.zeros(n), frame[:, 1], frame[:, 2], frame[:, 0], extent=extent, res=res)
    fig = plt.figure(figsize=(12.5, 5.4))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    viz.plot_interface_surface(surf, "chirality", ax=ax1, title=title_a)
    viz.plot_interface_surface(surf, "twist", ax=ax2, title=title_b)
    fig.tight_layout()
    save(fig, name)
    return surf


def run(seed: int = 0) -> dict:
    n = 3
    d = np.array([0.5, 0.3, 1.1])
    res = {"n": n, "N": 2, "twist_angle": 0.7, "displacement": d, "k_scale": 3.0}
    ea = ExteriorAlgebra(n)

    # --- regime 1: a carrier wave with weak side-bands (no zeros inside a volume)
    S_smooth = tc.twisted_pair(n, "simplex", k_scale=3.0, angles=[0.7], displacement=d, width=1.0, amplitudes=0.25)
    # --- regime 2: four equal waves per volume (strongly structured, vortex lines inside each volume)
    S_struct = tc.twisted_pair(n, "simplex", k_scale=3.0, angles=[0.7], displacement=d, width=1.0)

    with Timer("A1 slice panels (both regimes)"):
        origin, u, v = natural_plane(S_smooth)
        fig, _ = viz.plot_slice_panels(S_smooth, origin, u, v, extent=2.6, res=260,
                                       title="Two twisted volumes in 3-D, weakly structured (carrier + 0.25 side-bands)",
                                       axis_labels=("along displacement d", "tangential"))
        save(fig, "A_two_volumes_3d_slice_smooth.png")
        fig, _ = viz.plot_slice_panels(S_struct, origin, u, v, extent=2.6, res=260,
                                       title="Two twisted volumes in 3-D, strongly structured (4 equal waves each)",
                                       axis_labels=("along displacement d", "tangential"))
        save(fig, "A_two_volumes_3d_slice_structured.png")

    with Timer("A2 interface surfaces"):
        surf_s = surface_figure(S_smooth, d, "A_two_volumes_3d_surface_smooth.png",
                                "Weakly structured: one wavy wall, signed chirality A∧F",
                                "Weakly structured: twist |F| on the wall")
        surf_t = surface_figure(S_struct, d, "A_two_volumes_3d_surface_structured.png",
                                "Strongly structured: wall + vortex tubes, signed A∧F",
                                "Strongly structured: twist |F|")
        for key, surf in (("smooth", surf_s), ("structured", surf_t)):
            C = surf["q"]["ladder"][3][:, 0]
            res[f"surface_{key}"] = {"vertices": int(len(C)), "sign_balance": sign_balance(C)}

    with Timer("A3 localisation profiles"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
        for ax, (key, S) in zip(axes, (("smooth", S_smooth), ("structured", S_struct))):
            prof = localization_profile(S, count=80000, rng=seed, bins=45, h_max=8)
            viz.plot_localization_profile(prof, degrees=(2, 3), statistic="median", ax=ax,
                                          title=f"{key.capitalize()} volumes: median density across the interface")
            ok = ~np.isnan(prof["median_profiles"][2])
            res[f"profile_{key}"] = {
                "corr_F_vs_sech2": float(np.corrcoef(prof["median_profiles"][2][ok], prof["sech2"][ok])[0, 1]),
                "corr_C3_vs_sech2": float(np.corrcoef(prof["median_profiles"][3][ok], prof["sech2"][ok])[0, 1]),
            }
        fig.tight_layout()
        save(fig, "A_two_volumes_3d_profile.png")

    with Timer("A4 interface statistics and invariances"):
        for key, S in (("smooth", S_smooth), ("structured", S_struct)):
            samp = sample_interface(S, count=8000, rng=seed)
            X = samp["X"]
            q = geometry_at(S, X, ea=ea)
            an = analytic_two_volume_chirality(S, X, ea=ea)
            nrm = samp["normal"]
            k1, k2 = an["k_a"], an["k_b"]
            k1t = k1 - (k1 * nrm).sum(-1, keepdims=True) * nrm
            k2t = k2 - (k2 * nrm).sum(-1, keepdims=True) * nrm
            pred = ea.wedge(nrm, 1, ea.wedge(k1t, 1, k2t, 1), 2)[:, 0]
            res[f"interface_{key}"] = {
                "points": int(len(X)),
                "closed_form_max_rel_error": float(np.abs(q["ladder"][3] - an["C3"]).max() / np.abs(an["C3"]).max()),
                "sign_balance": sign_balance(q["ladder"][3][:, 0]),
                "sign_agreement_with_n^k1t^k2t": float((np.sign(pred) == np.sign(q["ladder"][3][:, 0])).mean()),
                "relative_phase_invariance": relative_phase_invariance(S, X[:500]),
                "direction_spectrum": {k: v for k, v in direction_spectrum(q["ladder"][3]).items() if k != "mean_direction"},
            }
    dump(res, "A_two_volumes_3d.json")
    return res


if __name__ == "__main__":
    run()
