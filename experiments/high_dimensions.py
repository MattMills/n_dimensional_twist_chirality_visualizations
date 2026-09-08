"""Experiment E: pushing the dimension as high as the machine allows.

Everything here uses the component-free machinery of ``twistchiral.highdim``:
the twist is factorised into at most N-1 rotation planes, ladder norms come
from the rotation rates, and direction statistics come from Gram-determinant
kernels.  Cost is linear in n per point, so n is limited only by memory.
"""
import os
import time
from math import comb

from common import Timer, dump, save

import numpy as np
import matplotlib.pyplot as plt

import twistchiral as tc
from twistchiral import geometry as geo, style, viz
from twistchiral.highdim import (chirality3_kernel, expected_mirror_cosine, factored_geometry,
                                 geometry_factored_batched, kernel_direction_spectrum, ladder_norms_factored,
                                 mirror_cosines, random_frame_image, screw_saturation, seed_near)
from twistchiral.interface import log_amplitude_ratio, project_to_interface
from twistchiral.slices import plane_grid
from twistchiral.waves import WaveSystem, WaveVolume

N_LIST = [int(x) for x in os.environ.get("TC_NS", "3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096,6144,8192,12288,16384,24576,32768,49152,65536").split(",")]
STRUCT_CAP = int(os.environ.get("TC_STRUCT_CAP", "2048"))     # largest n for volumes with ~n waves
POINTS = int(os.environ.get("TC_POINTS", "1200"))


def make_pair(n, kind, rng, k_scale=3.0, width=1.0, d_norm=1.2):
    """Two volumes: a wave set and its Haar-random twisted copy, displaced by a random d."""
    d = rng.standard_normal(n)
    d = d_norm * d / np.linalg.norm(d)
    if kind == "single wave":
        K = k_scale * geo.random_sphere(n, 1, rng)
        amp = np.ones(1, dtype=complex)
    elif kind == "carrier + 7 side-bands":
        K = k_scale * geo.random_sphere(n, 8, rng)
        amp = np.full(8, 0.25, dtype=complex); amp[0] = 1.0
    elif kind == "n+1 equal waves (simplex)":
        K = k_scale * geo.simplex(n)
        amp = np.ones(n + 1, dtype=complex)
    elif kind == "2n random waves":
        K = k_scale * geo.random_sphere(n, 2 * n, rng)
        amp = rng.standard_normal(2 * n) + 1j * rng.standard_normal(2 * n)
    else:
        raise ValueError(kind)
    K2 = random_frame_image(K, rng)
    v1 = WaveVolume(k=K, center=-0.5 * d, width=width, amplitudes=amp)
    v2 = WaveVolume(k=K2, center=0.5 * d, width=width, amplitudes=amp)
    # the common Gaussian factor is dropped: seeds can have unit spread in
    # every coordinate at any n without underflow (projective geometry unchanged)
    return WaveSystem([v1, v2], relative_envelope=True), d


def interface_points(S, count, rng, spread=1.0):
    n = S.n
    X0 = seed_near(np.zeros(n), count, n, rng, spread)
    X, h, normal, ok = project_to_interface(S, X0, 0, 1, iters=40)
    rho = (np.abs(S.evaluate(X)[0]) ** 2).sum(1)
    keep = ok & np.isfinite(rho) & (rho > 1e-200)
    return X[keep], normal[keep]


ALL_PARTS = ("E1", "E2", "E3", "E4", "E5", "E6")


def run(seed: int = 0, parts=ALL_PARTS) -> dict:
    import json
    from common import RES_DIR
    rng = np.random.default_rng(seed)
    kinds = ["single wave", "carrier + 7 side-bands", "n+1 equal waves (simplex)", "2n random waves"]
    res = {"n": N_LIST, "points": POINTS, "struct_cap": STRUCT_CAP}
    prev = os.path.join(RES_DIR, "E_high_dimensions.json")
    if parts != ALL_PARTS and os.path.exists(prev):
        res.update(json.load(open(prev)))
    d_eff = {k: [] for k in kinds}
    d_pop = {k: [] for k in kinds}
    d_var = {k: [] for k in kinds}
    mirror = {k: [] for k in kinds}
    seconds = {k: [] for k in kinds}
    saturation_cfg, saturation_field, closed_form_err = [], [], []
    kept = {k: [] for k in kinds}

    if "E1" in parts:
      with Timer("E1 two-volume interface statistics vs n"):
        for n in N_LIST:
            P = POINTS if n < 8192 else (max(300, POINTS // 2) if n < 24576 else max(200, POINTS // 4))
            for kind in kinds:
                if kind in ("n+1 equal waves (simplex)", "2n random waves") and n > STRUCT_CAP:
                    for dct in (d_eff, d_pop, d_var, mirror, seconds, kept):
                        dct[kind].append(np.nan)
                    continue
                t0 = time.time()
                S, d = make_pair(n, kind, rng)
                X, normal = interface_points(S, P, rng)
                g = geometry_factored_batched(S, X, max_degree=3)
                K = chirality3_kernel(g)
                sp = kernel_direction_spectrum(K)          # unweighted: uniform over interface samples
                d_eff[kind].append(sp["effective_dimension"])
                d_pop[kind].append(sp["population_effective_dimension"])
                d_var[kind].append(sp["variation_dimension"])
                normals = rng.standard_normal((96, n))
                mirror[kind].append(float(np.mean([mirror_cosines(g, m).mean() for m in normals])))
                seconds[kind].append((time.time() - t0) * 1000.0 / max(len(X), 1))
                kept[kind].append(int(len(X)))
                if kind == "single wave":
                    k1, k2 = S[0].k[0], S[1].k[0]
                    sat_cfg = screw_saturation(k1[None], k2[None], d[None])[0]
                    h, gh = log_amplitude_ratio(S, X, 0, 1)
                    V2 = 1 / np.cosh(0.5 * h) ** 2
                    pred = 0.5 * V2 / S[0].sigma() ** 2 * np.linalg.norm(k1) * np.linalg.norm(k2) * np.linalg.norm(d)
                    ratio = g["norms"][3] / pred
                    saturation_cfg.append(float(sat_cfg))
                    saturation_field.append(float(np.median(ratio)))
                    # closed form via Gram determinants
                    ka, kb = S[0].local_wavevector(X, relative_envelope=True), S[1].local_wavevector(X, relative_envelope=True)
                    M = np.stack([ka, kb, 0.5 * gh], axis=1)
                    vol = np.sqrt(np.clip(np.linalg.det(M @ np.swapaxes(M, 1, 2)), 0, None))
                    closed_form_err.append(float(np.abs(g["norms"][3] - 0.5 * V2 * vol).max() / (0.5 * V2 * vol).max()))
            print(f"  n={n:6d}: " + ", ".join(f"{k.split()[0]}={d_eff[k][-1]:.1f}/{d_pop[k][-1]:.0f}" for k in kinds)
                  + f"  mirror(single)={mirror['single wave'][-1]:+.3f} (pred {expected_mirror_cosine(3, n):+.3f})"
                  + f"  sat={saturation_field[-1]:.3f}  ms/pt={seconds['single wave'][-1]:.2f}")
        res.update({"effective_dimension": d_eff, "population_effective_dimension": d_pop,
                    "variation_dimension": d_var, "mirror_mean_cosine": mirror,
                    "ms_per_point": seconds, "points_kept": kept, "saturation_configuration": saturation_cfg,
                    "saturation_field_median": saturation_field, "closed_form_max_rel_error": closed_form_err,
                    "components_C(n,3)": [comb(n, 3) for n in N_LIST]})

    if "E2" in parts:
      with Timer("E2 random-twist saturation statistics"):
        sat_mean_sq = []
        for n in N_LIST:
            vals = []
            for _ in range(300):
                k1 = rng.standard_normal(n)
                k2 = random_frame_image(k1[None], rng)[0]
                d = rng.standard_normal(n)
                vals.append(screw_saturation(k1[None], k2[None], d[None])[0] ** 2)
            sat_mean_sq.append(float(np.mean(vals)))
        res["saturation_mean_square"] = sat_mean_sq
        res["saturation_mean_square_predicted"] = [1 - 3 / n + 2 / n**2 for n in N_LIST]

    if "E3" in parts:
      with Timer("E3 figure: two-volume statistics vs n"):
        ns = np.array(res["n"])
        if "E1" not in parts:
            d_eff, d_pop, mirror, seconds = (res["effective_dimension"], res["population_effective_dimension"],
                                              res["mirror_mean_cosine"], res["ms_per_point"])
            saturation_field, sat_mean_sq = res["saturation_field_median"], res["saturation_mean_square"]
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
        ax = axes[0, 0]
        for i, k in enumerate(kinds):
            y = np.array(d_pop[k])
            ok = ~np.isnan(y)
            ax.plot(ns[ok], y[ok], color=style.series(i), marker="o", markersize=3.5, label=k)
        ax.plot(ns, [comb(n, 3) for n in ns], color=style.BASELINE, linewidth=2.2, label="C(n,3) components")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("dimension n"); ax.set_ylabel("effective dimension of chirality directions")
        ax.set_title("Chirality directions on the interface (pair-coherence estimate)", loc="left")
        style.tidy(ax); ax.legend(loc="upper left", fontsize=8)
        ax = axes[0, 1]
        for i, k in enumerate(kinds):
            y = np.array(mirror[k]); ok = ~np.isnan(y)
            ax.plot(ns[ok], y[ok], color=style.series(i), marker="o", markersize=3.5, label=k)
        ax.plot(ns, [expected_mirror_cosine(3, n) for n in ns], color=style.BASELINE, linewidth=2.2, label="1 − 6/n (exact mean)")
        ax.set_xscale("log")
        ax.set_xlabel("dimension n"); ax.set_ylabel("mean cosine, mirror image vs original")
        ax.set_title("A generic mirror stops reversing the chirality", loc="left")
        ax.axhline(0, color=style.GRID, linewidth=0.8)
        style.tidy(ax); ax.legend(loc="lower right", fontsize=8)
        ax = axes[1, 0]
        ax.plot(ns, np.sqrt(sat_mean_sq), color=style.series(0), marker="o", markersize=3.5, label="random twist, √E[s²]")
        ax.plot(ns, saturation_field, color=style.series(1), marker="o", markersize=3.5, label="|A∧F| / (V²|k|²|d| / 2σ²) on the interface")
        ax.plot(ns, np.sqrt(res["saturation_mean_square_predicted"]), color=style.BASELINE, linewidth=2.2, label="√(1 − 3/n + 2/n²)")
        ax.set_xscale("log"); ax.set_ylim(0, 1.05)
        ax.set_xlabel("dimension n"); ax.set_ylabel("twist saturation  s = |k₁∧k₂∧d| / |k₁||k₂||d|")
        ax.set_title("Generic twists become maximal", loc="left")
        style.tidy(ax); ax.legend(loc="lower right", fontsize=8)
        ax = axes[1, 1]
        for i, k in enumerate(kinds):
            y = np.array(seconds[k]); ok = ~np.isnan(y)
            ax.plot(ns[ok], y[ok], color=style.series(i), marker="o", markersize=3.5, label=k)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("dimension n"); ax.set_ylabel("milliseconds per interface point")
        ax.set_title("Cost of the full characterisation (4 cores)", loc="left")
        style.tidy(ax); ax.legend(loc="upper left", fontsize=8)
        fig.suptitle(f"Two twisted volumes, n = {N_LIST[0]} … {N_LIST[-1]}", x=0.01, ha="left", fontsize=12.5)
        fig.tight_layout()
        save(fig, "E_highdim_two_volume_sweep.png")

    if "E4" in parts:
      with Timer("E4 ladder and twist isotropy for N volumes at high n"):
        ns_l = [16, 32, 64, 128, 256]
        Ns_l = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 129]
        table = {}
        iso = {}
        rank_ok = True
        for n in ns_l:
            row = []
            isorow = []
            for N in Ns_l:
                if N > n + 1:
                    row.append(np.nan); isorow.append(np.nan); continue
                # tight cluster of centres: at generic points near the centroid every
                # amplitude is comparable, which is all the ladder theorem needs
                centers = 0.08 * rng.standard_normal((N, n)) / np.sqrt(n / 8)
                vols = [WaveVolume(k=3.0 * geo.random_sphere(n, 1, rng), center=centers[a], width=1.0) for a in range(N)]
                S = WaveSystem(vols)
                Xj = seed_near(np.zeros(n), 40, n, rng, spread=0.05 / np.sqrt(n / 8))
                Psi, dPsi = S.evaluate(Xj)
                g = factored_geometry(Psi, dPsi)
                # structural criterion: the rung of degree 2k needs k rotation planes with
                # non-zero rate, the rung of degree 2k+1 additionally a component of A
                # outside them (norm-based thresholds fail for ~100 planes because of
                # the k! and product-of-rates scaling of the top rungs)
                rates = g["rates"]
                r_eff = int(np.median((rates > 1e-8 * rates[:, :1]).sum(1)))
                a_out = np.median(g["Ares2"] + (g["projA"][:, r_eff:].sum(1) if rates.shape[1] > r_eff else 0.0)) > 1e-12 * np.median(g["A2"])
                deg = min(n, 2 * r_eff + (1 if a_out else 0))
                row.append(deg)
                if deg != min(n, 2 * N - 1):
                    rank_ok = False
                    print(f"  MISMATCH n={n} N={N}: measured {deg} predicted {min(n, 2*N-1)}")
                r_eff = min(N - 1, n // 2)
                isorow.append(float(np.median(rates[:, r_eff - 1] / rates[:, 0])) if r_eff >= 1 else np.nan)
            table[n] = row
            iso[n] = isorow
            print(f"  n={n}: top degree by N = {row}")
        res["ladder_high_n"] = {"ns": ns_l, "Ns": Ns_l, "max_degree": table, "all_match": rank_ok,
                               "rate_ratio_min_over_max": iso}
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
        ax = axes[0]
        for i, n in enumerate(ns_l):
            y = np.array(table[n], dtype=float); ok = ~np.isnan(y)
            ax.plot(np.array(Ns_l)[ok], y[ok], color=style.series(i), marker="o", markersize=3.5, label=f"n = {n}")
            ax.axhline(n, color=style.series(i), linewidth=0.6, alpha=0.4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("simultaneously interfering volumes N"); ax.set_ylabel("highest non-zero chirality degree")
        ax.set_title("Ladder degree min(n, 2N − 1): binary chirality needs N ≥ ⌊n/2⌋ + 1", loc="left")
        style.tidy(ax); ax.legend(loc="upper left", fontsize=8)
        ax = axes[1]
        for i, n in enumerate(ns_l):
            y = np.array(iso[n], dtype=float); ok = ~np.isnan(y)
            ax.plot(np.array(Ns_l)[ok], y[ok], color=style.series(i), marker="o", markersize=3.5, label=f"n = {n}")
        ax.set_xscale("log"); ax.set_ylim(0, 1.05)
        ax.set_xlabel("simultaneously interfering volumes N"); ax.set_ylabel("λ_min / λ_max of the twist at the N-fold junction")
        ax.set_title("Twist isotropy at junctions", loc="left")
        style.tidy(ax); ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        save(fig, "E_highdim_ladder.png")

    if "E5" in parts:
      with Timer("E5 slices of the interface at n = 8, 64, 512, 4096"):
        ns_s = [8, 64, 512, 4096]
        fig, axes = plt.subplots(len(ns_s), 4, figsize=(15.5, 3.75 * len(ns_s)))
        for i, n in enumerate(ns_s):
            S, d = make_pair(n, "carrier + 7 side-bands", np.random.default_rng(100 + i))
            S = WaveSystem(S.volumes, relative_envelope=False)      # physical intensity in the slice
            kc = S[0].k[0]
            tang = kc - (kc @ d) * d / (d @ d)
            X, Sg, Tg, frame = plane_grid(np.zeros(n), d, tang, 2.4, 150)
            g = geometry_factored_batched(S, X, max_degree=3, batch=1500, keep_frames=False)
            h, _ = log_amplitude_ratio(S, X, 0, 1)
            ext = [Sg.min(), Sg.max(), Tg.min(), Tg.max()]
            panels = [("|Σψ|²", g["intensity"], None, style.seq_cmap),
                      ("visibility V", (1 / np.cosh(0.5 * h)), (0, 1), style.seq_cmap),
                      ("twist |F|", g["norms"][2], "pct", style.seq_cmap),
                      ("chirality |A∧F|", g["norms"][3], "pct", style.seq_cmap)]
            for j, (lab, arr, lim, cm) in enumerate(panels):
                ax = axes[i, j]
                arr2 = arr.reshape(Sg.shape)
                if lim == "pct":
                    vmin, vmax = 0, np.percentile(arr2, 99)
                elif lim is None:
                    vmin, vmax = None, None
                else:
                    vmin, vmax = lim
                im = ax.imshow(arr2, origin="lower", extent=ext, cmap=cm, vmin=vmin, vmax=vmax, aspect="equal")
                ax.contour(Sg, Tg, h.reshape(Sg.shape), levels=[0.0], colors=[style.INK], linewidths=0.9, alpha=0.8)
                style.image_axis(ax, f"n = {n}:  {lab}")
                style.colorbar(fig, im, ax)
            axes[i, 0].set_ylabel("tangential (carrier)")
        for ax in axes[-1]:
            ax.set_xlabel("along displacement d")
        fig.suptitle("The same slice (plane of d and the carrier) through the interface as n grows: carrier + 7 random side-bands",
                     x=0.01, ha="left", fontsize=12.5)
        fig.tight_layout()
        save(fig, "E_highdim_slices.png")

    if "E6" in parts:
      with Timer("E6 localisation profile at n = 4096"):
        n = 4096
        prof = {}
        for kind in ("single wave", "carrier + 7 side-bands"):
            S, d = make_pair(n, kind, np.random.default_rng(7))
            X, normal = interface_points(S, 500, np.random.default_rng(8), spread=0.25)
            offs = np.linspace(-2.6, 2.6, 27)
            Xall = (X[:, None, :] + offs[None, :, None] * normal[:, None, :]).reshape(-1, n)
            g = geometry_factored_batched(S, Xall, max_degree=3, batch=1500, keep_frames=False)
            h, _ = log_amplitude_ratio(S, Xall, 0, 1)
            edges = np.linspace(-8, 8, 41)
            cen = 0.5 * (edges[1:] + edges[:-1])
            idx = np.digitize(h, edges) - 1
            m = np.array([np.median(g["norms"][3][idx == i]) if (idx == i).any() else np.nan for i in range(40)])
            prof[kind] = m / np.nanmax(m)
        fig, ax = viz.plot_lines(cen, prof, "h = ln(|ψ₁|² / |ψ₂|²)", "median |A∧F|, normalised",
                                 title="n = 4096: the interface profile is still sech²(h/2)",
                                 reference=("sech²(h/2)", 1 / np.cosh(0.5 * cen) ** 2), direct_labels=False)
        save(fig, "E_highdim_profile.png")
        okk = ~np.isnan(prof["single wave"])
        res["profile_corr_n4096"] = {k: float(np.corrcoef(v[okk], (1 / np.cosh(0.5 * cen) ** 2)[okk])[0, 1]) for k, v in prof.items()}
    dump(res, "E_high_dimensions.json")
    return res


if __name__ == "__main__":
    run()
