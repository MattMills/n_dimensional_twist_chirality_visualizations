"""Experiment B: the twist-chiral interface between two volumes in 4-D.

In four dimensions the chirality 3-form A^F is dual to a *vector* tangent to
the three-dimensional interface: chirality is no longer a sign but a
direction field on the interface (the axis about which the tangential
wavevector plane turns).  A mirror image rotates that direction instead of
negating it.
"""
from common import Timer, dump, save

import numpy as np
import matplotlib.pyplot as plt

import twistchiral as tc
from twistchiral import geometry as geo, style, viz
from twistchiral.analysis import (direction_spectrum, geometry_at, mirror_comparison, relative_phase_invariance,
                                  screw_prediction)
from twistchiral.exterior import ExteriorAlgebra
from twistchiral.interface import sample_interface
from twistchiral.qgt import chirality_axis_4d
from twistchiral.slices import natural_plane


def run(seed: int = 0) -> dict:
    n = 4
    d = np.array([0.3, 0.5, 0.4, 0.9])
    angles = [0.5, 1.1]
    ea = ExteriorAlgebra(n)
    res = {"n": n, "N": 2, "twist_angles": angles, "displacement": d}
    pairs = {
        "single wave": tc.twisted_pair(n, "single", k_scale=3.0, angles=angles, displacement=d, width=1.0),
        "carrier + side-bands": tc.twisted_pair(n, "simplex", k_scale=3.0, angles=angles, displacement=d, width=1.0, amplitudes=0.25),
        "5 equal waves": tc.twisted_pair(n, "simplex", k_scale=3.0, angles=angles, displacement=d, width=1.0),
        "8 random waves": tc.twisted_pair(n, "random", k_scale=3.0, angles=angles, displacement=d, width=1.0,
                                          amplitudes="random", rng=seed, m=8),
    }

    with Timer("B1 slice panels"):
        S = pairs["carrier + side-bands"]
        origin, u, v = natural_plane(S)
        fig, _ = viz.plot_slice_panels(S, origin, u, v, extent=2.6, res=260,
                                       title="Two twisted volumes in 4-D (carrier + side-bands): slice through both centres",
                                       axis_labels=("along displacement d", "tangential"))
        save(fig, "B_two_volumes_4d_slice.png")

    with Timer("B2 chirality-axis field on the 3-D interface"):
        frame = geo.complete_frame(d, n=n)          # columns: d-hat, then 3 tangential directions
        fig = plt.figure(figsize=(13, 5.8))
        for i, key in enumerate(("carrier + side-bands", "5 equal waves")):
            S = pairs[key]
            # a regular lattice of seeds in the mean tangent hyperplane, projected onto the interface
            g = np.linspace(-1.8, 1.8, 7)
            G = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
            X0 = G @ frame[:, 1:].T
            X, h, normal, ok = tc.project_to_interface(S, X0)
            X, normal = X[ok], normal[ok]
            q = geometry_at(S, X, ea=ea, max_degree=3)
            axis = chirality_axis_4d(ea, q["ladder"][3])
            P3 = X @ frame[:, 1:]
            V3 = axis @ frame[:, 1:]
            tangency = np.abs((axis * normal).sum(-1)) / np.maximum(np.linalg.norm(axis, axis=-1), 1e-300)
            res[f"axis_tangency_median_{key}"] = float(np.median(tangency))
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            idx = np.arange(len(P3))
            c = ea.norm(q["ladder"][3])[idx]
            norm = plt.Normalize(0, np.percentile(c, 97))
            cols = style.seq_cmap(0.3 + 0.7 * np.clip(norm(c), 0, 1))
            Vn = V3[idx] / np.maximum(np.linalg.norm(V3[idx], axis=-1, keepdims=True), 1e-300)
            ax.quiver(P3[idx, 0], P3[idx, 1], P3[idx, 2], Vn[:, 0], Vn[:, 1], Vn[:, 2], length=0.42, normalize=True,
                      colors=cols, linewidth=1.2, arrow_length_ratio=0.3)
            ax.view_init(elev=24, azim=-50)
            ax.set_facecolor(style.SURFACE)
            for axis_ in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis_.set_pane_color((1, 1, 1, 0))
                axis_._axinfo["grid"]["color"] = style.GRID
            ax.tick_params(colors=style.MUTED, labelsize=7.5)
            ax.set_xlabel("interface coord. 1"); ax.set_ylabel("interface coord. 2"); ax.set_zlabel("interface coord. 3")
            ax.set_title(f"Chirality axis ★(A∧F) on the 3-D interface: {key}", loc="left", fontsize=10.5)
            style.colorbar(fig, plt.cm.ScalarMappable(norm=norm, cmap=style.seq_cmap), ax, "|A∧F|")
        fig.tight_layout()
        save(fig, "B_two_volumes_4d_axis_field.png")

    with Timer("B3 direction spectra and closed form"):
        spectra = {}
        for key, S in pairs.items():
            samp = sample_interface(S, count=6000, rng=seed, pad=1.6)
            q = geometry_at(S, samp["X"], ea=ea, max_degree=3)
            spectra[key] = direction_spectrum(q["ladder"][3], weights=q["rho"])
            res[f"spectrum_{key}"] = {k: v for k, v in spectra[key].items() if k != "mean_direction"}
            if key == "single wave":
                pred = screw_prediction(S, samp["X"])
                res["single_wave_screw_prediction_error"] = float(np.abs(q["ladder"][3] - pred).max() / np.abs(pred).max())
                res["relative_phase_invariance"] = relative_phase_invariance(S, samp["X"][:400])
        fig = viz.plot_direction_spectra(spectra, title="Spectrum of chirality directions on the 4-D interface (4 components)")
        save(fig, "B_two_volumes_4d_spectra.png")

    with Timer("B4 mirror image: sign flip (3-D) vs. rotated direction (4-D)"):
        S3 = tc.twisted_pair(3, "simplex", k_scale=3.0, angles=[0.7], displacement=[0.5, 0.3, 1.1], width=1.0)
        S4 = pairs["5 equal waves"]
        X3 = sample_interface(S3, count=3000, rng=seed)["X"]
        X4 = sample_interface(S4, count=3000, rng=seed, pad=1.6)["X"]
        m3 = mirror_comparison(S3, [0.3, 1.0, -0.2], X3)
        m4 = mirror_comparison(S4, [0.3, 1.0, -0.2, 0.5], X4)
        ea3 = ExteriorAlgebra(3)
        cos3 = (ea3.unit(m3["C_original"]) * ea3.unit(m3["C_mirror"])).sum(-1)
        cos4 = m4["direction_cosines"]
        res["mirror_3d"] = {"tensor_error": m3["tensor_error"], "sign_flip_error": m3["sign_flip_error"]}
        res["mirror_4d"] = {"tensor_error": m4["tensor_error"], "negated_fraction": m4["negated_fraction"],
                            "mean_direction_cosine": float(np.mean(cos4))}
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        bins = np.linspace(-1, 1, 41)
        for ax, cos, lab in zip(axes, (cos3, cos4), ("n = 3: mirror image negates the chirality (cosine = −1)",
                                                     "n = 4: mirror image rotates the chirality direction")):
            ax.hist(cos, bins=bins, color=style.series(0), edgecolor=style.SURFACE, linewidth=0.6)
            ax.set_xlabel("cosine between mirrored and original chirality direction")
            ax.set_ylabel("interface samples")
            ax.set_title(lab, loc="left")
            style.tidy(ax)
        fig.tight_layout()
        save(fig, "B_two_volumes_4d_mirror.png")

    with Timer("B5 twist-angle sweep"):
        sigma = 1.0
        beta = 0.6
        k1 = 3.0 * np.array([np.cos(beta), 0.0, np.sin(beta), 0.0])
        a1 = np.linspace(0, np.pi, 41)
        a2 = np.linspace(0, np.pi, 41)
        Z = np.zeros((len(a2), len(a1)))
        for i, x2 in enumerate(a2):
            for j, x1 in enumerate(a1):
                R = geo.rotation_from_angles(n, [x1, x2])
                Z[i, j] = ea.norm(ea.wedge_vectors(k1[None], (R @ k1)[None], d[None]))[0] / (2 * sigma**2)
        a1c = np.linspace(0, np.pi, 17)
        a2c = np.linspace(0, np.pi, 17)
        Zs = np.zeros((len(a2c), len(a1c)))
        for i, x2 in enumerate(a2c):
            for j, x1 in enumerate(a1c):
                S = tc.twisted_pair(n, "simplex", k_scale=3.0, angles=[x1, x2], displacement=d, width=1.0, amplitudes=0.25)
                samp = sample_interface(S, count=700, rng=seed, pad=1.4)
                q = geometry_at(S, samp["X"], ea=ea, max_degree=3)
                Zs[i, j] = np.median(ea.norm(q["ladder"][3]))
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
        viz.plot_heatmap(a1, a2, Z, "twist angle α₁ in plane (e₀,e₁)", "twist angle α₂ in plane (e₂,e₃)",
                         "Single-wave pair: |A∧F| on the interface = |k₁∧R(α)k₁∧d| / 2σ²", ax=axes[0], cbar_label="|A∧F|")
        viz.plot_heatmap(a1c, a2c, Zs, "twist angle α₁ in plane (e₀,e₁)", "twist angle α₂ in plane (e₂,e₃)",
                         "Carrier + side-bands: median |A∧F| over the interface", ax=axes[1], cbar_label="median |A∧F|")
        for ax in axes:
            ax.set_xticks([0, np.pi / 2, np.pi]); ax.set_xticklabels(["0", "π/2", "π"])
            ax.set_yticks([0, np.pi / 2, np.pi]); ax.set_yticklabels(["0", "π/2", "π"])
        fig.tight_layout()
        save(fig, "B_two_volumes_4d_twist_sweep.png")
        res["twist_sweep_single_wave"] = {"max": float(Z.max()), "argmax_angles": [float(a1[Z.argmax() % len(a1)]), float(a2[Z.argmax() // len(a1)])]}
        sub = np.round(np.linspace(0, 40, 17)).astype(int)
        res["twist_sweep_structured_corr_with_single"] = float(np.corrcoef(Zs.ravel(), Z[np.ix_(sub, sub)].ravel())[0, 1])
    dump(res, "B_two_volumes_4d.json")
    return res


if __name__ == "__main__":
    run()
