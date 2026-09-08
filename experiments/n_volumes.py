"""Experiment D: N volumes interfering simultaneously.

With N volumes the interference map goes to CP^{N-1}, whose real dimension
is 2(N-1).  The twist form can therefore have up to N-1 independent
rotation planes, and the chirality ladder A, F, A^F, F^F, ... extends to
degree min(n, 2N-1).  Pairwise interference can never produce a second
rotation plane: the higher rungs (F^F and beyond) are irreducibly
multi-body and live at the junctions where three or more volumes meet.
"""
from common import Timer, dump, save

import numpy as np
import matplotlib.pyplot as plt

import twistchiral as tc
from twistchiral import geometry as geo, style, viz
from twistchiral.analysis import geometry_at, ladder_existence
from twistchiral.exterior import ExteriorAlgebra
from twistchiral.interface import dominant_volume, project_to_junction, sample_bulk, sample_junction
from twistchiral.qgt import quantum_geometry, rotation_planes
from twistchiral.slices import plane_grid
from twistchiral.waves import WaveSystem, WaveVolume


def three_volumes_4d(seed=0, spacing=1.7, amplitudes=0.25):
    n = 4
    ang = 2 * np.pi * np.arange(3) / 3 + 0.3
    C = np.zeros((3, n))
    C[:, 0] = spacing / np.sqrt(3) * np.cos(ang)
    C[:, 1] = spacing / np.sqrt(3) * np.sin(ang)
    return tc.cluster(n, 3, "simplex", k_scale=3.0, width=1.0, rng=seed, centers=C, amplitudes=amplitudes)


def run(seed: int = 0) -> dict:
    res = {}
    n = 4
    ea = ExteriorAlgebra(n)
    S3 = three_volumes_4d(seed)

    with Timer("D1 three volumes in 4-D: cells, twist rank, second Chern density"):
        junc = sample_junction(S3, (0, 1, 2), count=6000, rng=seed, pad=1.2)
        J = junc["X"]
        near_plane = J[np.linalg.norm(J[:, 2:], axis=-1) < 0.08] if len(J) else None
        fig, f = viz.plot_cells_and_junctions(S3, np.zeros(n), np.eye(n)[0], np.eye(n)[1], extent=2.3, res=240,
                                             title="Three volumes in 4-D (slice through the three centres)", junctions=near_plane)
        save(fig, "D_three_volumes_4d_cells.png")
        res["triple_junction_points"] = int(len(J))

    with Timer("D2 localisation of F∧F at the triple junction"):
        X = sample_bulk(S3, count=30000, rng=seed, pad=1.2)
        Xj, ok = project_to_junction(S3, X, (0, 1, 2))
        dist = np.linalg.norm(Xj - X, axis=-1)
        q = geometry_at(S3, X, ea=ea)
        rates, _ = rotation_planes(q["F"])
        ratio = rates[:, 1] / np.maximum(rates[:, 0], 1e-300)
        edges = np.linspace(0, 2.0, 26)
        cen = 0.5 * (edges[1:] + edges[:-1])
        idx = np.digitize(dist, edges) - 1
        prof = {}
        for key, arr in (("|F∧F|  (second Chern density)", q["norms"][4]), ("|F|  (twist)", q["norms"][2]),
                         ("λ₂/λ₁  (second twist plane)", ratio)):
            m = np.array([np.median(arr[(idx == i) & ok]) if ((idx == i) & ok).any() else np.nan for i in range(len(cen))])
            prof[key] = m / np.nanmax(m)
        fig, ax = viz.plot_lines(cen, prof, "distance to the triple junction", "median, normalised to peak",
                                 title="The second Chern density F∧F lives at the triple junction; the twist F does not",
                                 direct_labels=False)
        save(fig, "D_three_volumes_4d_junction_profile.png")
        res["junction_profile"] = {"distance": cen, **{k: v for k, v in prof.items()}}

    with Timer("D3 pairwise decomposition of the twist; F∧F is made of pair-pair cross terms"):
        from twistchiral.qgt import pairwise_decomposition
        Xg, Sg, Tg, frame = plane_grid(np.zeros(n), np.eye(n)[0], np.eye(n)[1], 2.3, 200)
        Psi, dPsi = S3.evaluate(Xg)
        F_full = quantum_geometry(Psi, dPsi)["F"]
        dec = pairwise_decomposition(Psi, dPsi)
        res["pairwise_identity_max_rel_error"] = float(np.abs(dec["F_reconstructed"] - F_full).max() / np.abs(F_full).max())
        pairs = dec["pairs"]
        F2 = {p: ea.from_antisymmetric(dec["F_pairs"][p]) for p in pairs}
        terms = {}
        for i, p in enumerate(pairs):
            for qq in pairs[i + 1:]:
                terms[(p, qq)] = 2 * (dec["weights"][p] * dec["weights"][qq])[:, None] * ea.wedge(F2[p], 2, F2[qq], 2)
        total = sum(terms.values())
        Ffull2 = ea.from_antisymmetric(F_full)
        res["FF_cross_term_identity_max_rel_error"] = float(np.abs(total - ea.wedge(Ffull2, 2, Ffull2, 2)).max()
                                                            / np.abs(ea.wedge(Ffull2, 2, Ffull2, 2)).max())
        res["max_pairwise_F^F"] = float(max(np.abs(ea.wedge(F2[p], 2, F2[p], 2)).max() for p in pairs))
        dom = dominant_volume(S3, Xg).reshape(Sg.shape)
        ext = [Sg.min(), Sg.max(), Tg.min(), Tg.max()]
        fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9))
        vmax = np.percentile(np.abs(total), 99.5)
        for ax, (key, arr) in zip(axes, list(terms.items()) + [("total", total)]):
            im = ax.imshow(np.abs(arr[:, 0]).reshape(Sg.shape), origin="lower", extent=ext, cmap=style.seq_cmap, vmin=0,
                           vmax=vmax, aspect="equal")
            ax.contour(Sg, Tg, dom, levels=np.arange(3) + 0.5, colors=[style.INK], linewidths=0.7, alpha=0.6)
            if key == "total":
                style.image_axis(ax, "|F∧F|  (sum of the three cross terms)")
            else:
                (a, b), (c, d_) = key
                style.image_axis(ax, f"|2 w{a+1}{b+1} w{c+1}{d_+1} F{a+1}{b+1}∧F{c+1}{d_+1}|")
            style.colorbar(fig, im, ax)
        fig.suptitle("Second Chern density of three volumes = cross terms between different pairs (each involves all three volumes)",
                     x=0.01, ha="left", fontsize=11.5)
        fig.tight_layout()
        save(fig, "D_three_volumes_4d_pair_cross_terms.png")

    with Timer("D4 chirality ladder vs (n, N)"):
        ns = [3, 4, 5, 6, 7, 8]
        Ns = [2, 3, 4, 5, 6]
        table = np.zeros((len(ns), len(Ns)), dtype=int)
        rng = np.random.default_rng(seed)
        for i, nn in enumerate(ns):
            for j, NN in enumerate(Ns):
                centers = rng.normal(0, 0.45, (NN, nn))
                S = tc.cluster(nn, NN, "single", k_scale=3.0, width=1.0, rng=int(rng.integers(1 << 30)), centers=centers)
                X = rng.normal(0, 0.35, (300, nn))
                le = ladder_existence(S, X)
                table[i, j] = le["max_degree"]
                assert le["max_degree"] == le["predicted_max_degree"], (nn, NN, le)
        res["ladder_table"] = {"ns": ns, "Ns": Ns, "max_degree": table, "predicted": [[min(nn, 2 * NN - 1) for NN in Ns] for nn in ns]}
        fig = viz.plot_ladder_heatmap(ns, Ns, table, title="Highest non-vanishing chirality form for N simultaneous volumes in n dimensions")
        save(fig, "D_chirality_ladder.png")

    with Timer("D5 simultaneity in time: three converging packets"):
        n = 4
        ang = 2 * np.pi * np.arange(3) / 3 + 0.3
        C0 = np.zeros((3, n))
        C0[:, 0] = 2.4 * np.cos(ang)
        C0[:, 1] = 2.4 * np.sin(ang)
        rng = np.random.default_rng(seed)
        vols = []
        for a in range(3):
            u = -C0[a] / np.linalg.norm(C0[a])                 # carrier points at the meeting point
            K0 = tc.wave_set(n, "simplex", k_scale=3.0 + 0.4 * a)
            R = geo.rotation_taking(K0[0], u) @ geo.plane_rotation(n, 2, 3, 0.9 * a)
            K = K0 @ R.T
            amp = np.full(K.shape[0], 0.25, dtype=complex); amp[0] = 1.0
            # non-dispersive waves (omega = |k|): the envelope rides along the carrier at the wave speed
            vols.append(WaveVolume(k=K, center=C0[a], width=0.9, amplitudes=amp, velocity=u, dispersion="linear", speed=1.0))
        Smov = WaveSystem(vols)
        times = [0.0, 1.2, 2.4]
        fig, axes = plt.subplots(3, len(times), figsize=(4.1 * len(times), 11.4))
        Xg, Sg, Tg, frame = plane_grid(np.zeros(n), np.eye(n)[0], np.eye(n)[1], 3.0, 200)
        ext = [Sg.min(), Sg.max(), Tg.min(), Tg.max()]
        peak, wpeak = [], []
        for j, t in enumerate(times):
            q = geometry_at(Smov, Xg, t=t, ea=ea)
            I = np.abs(Smov.total_field(Xg, t)) ** 2
            dom = dominant_volume(Smov, Xg, t).reshape(Sg.shape)
            panels = [("Interference intensity |Σψ|²", I, None),
                      ("Second Chern density |F∧F| (projective)", q["norms"][4], 99.5),
                      ("Intensity-weighted  ρ·|F∧F|", q["rho"] * q["norms"][4], 99.5)]
            peak.append(float(np.percentile(q["norms"][4], 99.5)))
            wpeak.append(float(np.percentile(q["rho"] * q["norms"][4], 99.5)))
            for i, (lab, arr, pct) in enumerate(panels):
                ax = axes[i, j]
                vmax = np.percentile(arr, pct) if pct else None
                im = ax.imshow(arr.reshape(Sg.shape), origin="lower", extent=ext, cmap=style.seq_cmap, vmin=0, vmax=vmax, aspect="equal")
                ax.contour(Sg, Tg, dom, levels=np.arange(3) + 0.5, colors=[style.INK], linewidths=0.5, alpha=0.35)
                style.image_axis(ax, f"t = {t:.1f}:  {lab}")
                style.colorbar(fig, im, ax)
        fig.suptitle("Three packets converging in 4-D: the three-body chirality density is significant only while all three overlap",
                     x=0.01, ha="left", fontsize=12)
        fig.tight_layout()
        save(fig, "D_three_packets_time.png")
        res["F^F_99.5pct_vs_time"] = {"t": times, "projective": peak, "intensity_weighted": wpeak}
    dump(res, "D_n_volumes.json")
    return res


if __name__ == "__main__":
    run()
