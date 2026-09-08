"""Figures for exploring twist-chirality interfaces."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from . import style  # noqa: E402
from .analysis import geometry_at  # noqa: E402
from .exterior import ExteriorAlgebra  # noqa: E402
from .interface import dominant_volume, log_amplitude_ratio, visibility  # noqa: E402
from .qgt import rotation_planes  # noqa: E402
from .slices import plane_grid, volume_grid, vortex_charges  # noqa: E402

style.apply()


# ----------------------------------------------------------------------------
# field panels on a 2-plane
# ----------------------------------------------------------------------------
def slice_fields(system, origin, u, v, extent, res: int = 220, t: float = 0.0, a: int = 0, b: int = 1):
    """Evaluate the standard set of fields on a 2-plane slice."""
    X, S, T, frame = plane_grid(origin, u, v, extent, res)
    ea = ExteriorAlgebra(system.n)
    q = geometry_at(system, X, t, ea)
    shape = S.shape
    total = system.total_field(X, t)
    out = {
        "S": S, "T": T, "frame": frame, "X": X, "ea": ea, "q": q,
        "intensity": (np.abs(total) ** 2).reshape(shape),
        "phase": np.angle(total).reshape(shape),
        "twist": ea.norm(ea.from_antisymmetric(q["F"])).reshape(shape),
        "rho": q["rho"].reshape(shape),
        "dominant": dominant_volume(system, X, t).reshape(shape),
    }
    if system.N >= 2:
        h, _ = log_amplitude_ratio(system, X, a, b, t)
        out["h"] = h.reshape(shape)
        out["visibility"] = (1 / np.cosh(0.5 * h)).reshape(shape)
    for p, comp in q["ladder"].items():
        if p >= 3:
            out[f"C{p}"] = comp.reshape(shape + (comp.shape[-1],))
            out[f"|C{p}|"] = ea.norm(comp).reshape(shape)
    # in-plane twist component F(u, v)
    Fuv = np.einsum("i,pij,j->p", frame[:, 0], q["F"], frame[:, 1])
    out["twist_in_plane"] = Fuv.reshape(shape)
    return out


def _extent(S, T):
    return [S.min(), S.max(), T.min(), T.max()]


def _clip(arr, pct: float):
    """Robust upper limit for a colour scale (percentile of the finite values)."""
    a = np.asarray(arr)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, pct)) if a.size else 1.0


def plot_slice_panels(system, origin, u, v, extent, res: int = 220, t: float = 0.0, a: int = 0, b: int = 1,
                      title: str | None = None, axis_labels=("s", "t"), show_vortices: bool = True,
                      clip_pct: float = 99.0):
    """Four panels: interference intensity, visibility + interface, twist, chirality.

    Twist and chirality scales are clipped at the ``clip_pct`` percentile: the
    thin interface tubes around vortex lines of the individual volumes carry
    very large (but low-intensity) values that would otherwise hide the wall.
    """
    f = slice_fields(system, origin, u, v, extent, res, t, a, b)
    S, T = f["S"], f["T"]
    ext = _extent(S, T)
    n = system.n
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    ax = axes[0]
    im = ax.imshow(f["intensity"], origin="lower", extent=ext, cmap=style.seq_cmap, aspect="equal")
    style.image_axis(ax, "Interference intensity  |Σψ|²")
    style.colorbar(fig, im, ax)
    if show_vortices:
        ch = vortex_charges(f["phase"])
        iy, ix = np.nonzero(ch)
        xs = S[0, :-1][ix] + 0.5 * (S[0, 1] - S[0, 0])
        ys = T[:-1, 0][iy] + 0.5 * (T[1, 0] - T[0, 0])
        pos = ch[iy, ix] > 0
        ax.scatter(xs[pos], ys[pos], s=9, facecolor=style.RED_STEPS[3], edgecolor=style.SURFACE, linewidth=0.5, label="vortex +1")
        ax.scatter(xs[~pos], ys[~pos], s=9, facecolor=style.SEQ_STEPS[9], edgecolor=style.SURFACE, linewidth=0.5, label="vortex −1")
        if pos.any() or (~pos).any():
            ax.legend(loc="upper right", fontsize=7.5, markerscale=1.2, labelcolor=style.INK_2)
    ax = axes[1]
    im = ax.imshow(f["visibility"], origin="lower", extent=ext, cmap=style.seq_cmap, vmin=0, vmax=1, aspect="equal")
    ax.contour(S, T, f["h"], levels=[0.0], colors=[style.INK], linewidths=1.2)
    style.image_axis(ax, "Fringe visibility  V  (interface: V = 1)")
    style.colorbar(fig, im, ax)
    ax = axes[2]
    im = ax.imshow(f["twist"], origin="lower", extent=ext, cmap=style.seq_cmap, aspect="equal",
                   vmin=0, vmax=_clip(f["twist"], clip_pct))
    ax.contour(S, T, f["h"], levels=[0.0], colors=[style.INK], linewidths=0.8, alpha=0.6)
    style.image_axis(ax, "Twist  |F|  (Berry curvature)")
    style.colorbar(fig, im, ax, f"clipped at {clip_pct:g}th pct")
    ax = axes[3]
    if n == 3 and "C3" in f:
        c = f["C3"][..., 0]
        lim = _clip(np.abs(c), clip_pct) + 1e-300
        im = ax.imshow(c, origin="lower", extent=ext, cmap=style.div_cmap, vmin=-lim, vmax=lim, aspect="equal")
        style.image_axis(ax, "Chirality  A∧F  (pseudoscalar, signed)")
    elif "|C3|" in f:
        im = ax.imshow(f["|C3|"], origin="lower", extent=ext, cmap=style.seq_cmap, aspect="equal",
                       vmin=0, vmax=_clip(f["|C3|"], clip_pct))
        style.image_axis(ax, f"Chirality magnitude  |A∧F|  ({ExteriorAlgebra(n).dim(3)} components)")
    else:
        lim = _clip(np.abs(f["twist_in_plane"]), clip_pct) + 1e-300
        im = ax.imshow(f["twist_in_plane"], origin="lower", extent=ext, cmap=style.div_cmap, aspect="equal",
                       vmin=-lim, vmax=lim)
        style.image_axis(ax, "In-plane twist  F(u, v)")
    ax.contour(S, T, f["h"], levels=[0.0], colors=[style.INK], linewidths=0.8, alpha=0.6)
    style.colorbar(fig, im, ax, f"clipped at {clip_pct:g}th pct")
    for ax in axes:
        ax.set_xlabel(axis_labels[0])
    axes[0].set_ylabel(axis_labels[1])
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=style.INK)
    fig.tight_layout()
    return fig, f


# ----------------------------------------------------------------------------
# interface surface in a 3-slice
# ----------------------------------------------------------------------------
def interface_surface(system, origin, u, v, w, extent, res: int = 48, t: float = 0.0, a: int = 0, b: int = 1):
    """Marching-cubes triangulation of the interface inside a 3-slice, with
    chirality data at the vertices."""
    from skimage import measure
    X, (S, T, R), frame, axes = volume_grid(origin, u, v, w, extent, res)
    h, _ = log_amplitude_ratio(system, X, a, b, t)
    H = h.reshape(S.shape)
    spacing = tuple(float(ax[1] - ax[0]) for ax in axes)
    verts, faces, _, _ = measure.marching_cubes(H, level=0.0, spacing=spacing)
    verts = verts + np.array([ax[0] for ax in axes])[None, :]
    Xv = np.asarray(origin)[None, :] + verts @ frame.T
    q = geometry_at(system, Xv, t, ExteriorAlgebra(system.n), max_degree=3)
    return {"verts": verts, "faces": faces, "X": Xv, "q": q, "frame": frame}


def plot_interface_surface(surf, color_by: str = "chirality", ax=None, title: str | None = None,
                           elev: float = 22, azim: float = -55, cmap=None, symmetric: bool | None = None,
                           clip_pct: float = 97.0):
    """Render a marching-cubes interface coloured by a chirality quantity."""
    verts, faces, q = surf["verts"], surf["faces"], surf["q"]
    ea = q["ea"]
    if color_by == "chirality":
        C = q["ladder"][3]
        vals = C[:, 0] if ea.n == 3 else ea.norm(C)
        symmetric = (ea.n == 3) if symmetric is None else symmetric
        label = "A∧F (signed)" if ea.n == 3 else "|A∧F|"
    elif color_by == "twist":
        vals = ea.norm(ea.from_antisymmetric(q["F"]))
        symmetric = False
        label = "|F|"
    else:
        vals = np.asarray(color_by)
        label = ""
        symmetric = bool(symmetric)
    fv = vals[faces].mean(-1)
    if symmetric:
        lim = _clip(np.abs(fv), clip_pct) + 1e-300
        norm = matplotlib.colors.Normalize(-lim, lim)
        cmap = cmap or style.div_cmap
    else:
        norm = matplotlib.colors.Normalize(0, _clip(fv, clip_pct) + 1e-300)
        cmap = cmap or style.seq_cmap
    colors = cmap(norm(fv))
    own = ax is None
    if own:
        fig = plt.figure(figsize=(6.2, 5.4))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
    poly = Poly3DCollection(verts[faces], facecolors=colors, edgecolor="none", shade=True,
                            lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=65))
    ax.add_collection3d(poly)
    lo, hi = verts.min(0), verts.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=elev, azim=azim)
    ax.set_facecolor(style.SURFACE)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = style.GRID
        axis._axinfo["grid"]["linewidth"] = 0.6
        axis.label.set_color(style.INK_2)
    ax.tick_params(colors=style.MUTED, labelsize=7.5)
    ax.set_xlabel("s"); ax.set_ylabel("t"); ax.set_zlabel("r")
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    style.colorbar(fig, m, ax, label)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, color=style.INK)
    return fig, ax


# ----------------------------------------------------------------------------
# profiles, spectra, ladders
# ----------------------------------------------------------------------------
def plot_localization_profile(profile, degrees=(2, 3), statistic: str = "median", title: str | None = None, ax=None):
    """Rung norms (median / mean / intensity-weighted mean per bin) vs. the
    log-amplitude ratio ``h``, with the sech² law."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
    else:
        fig = ax.figure
    h = profile["h"]
    src = profile[{"median": "median_profiles", "mean": "profiles", "weighted": "weighted_profiles"}[statistic]]
    ax.plot(h, profile["sech2"], color=style.BASELINE, linewidth=2.4, label="sech²(h/2) prediction")
    names = {2: "twist |F|", 3: "chirality |A∧F|", 4: "|F∧F|", 5: "|A∧F∧F|"}
    for i, p in enumerate(degrees):
        if p not in src:
            continue
        y = src[p] / np.nanmax(src[p])
        ax.plot(h, y, color=style.series(i), label=names.get(p, f"degree {p}"))
    ax.axvline(0, color=style.GRID, linewidth=0.8)
    ax.set_xlabel("h = ln(|ψ₁|² / |ψ₂|²)   (interface at h = 0)")
    ax.set_ylabel(f"{statistic} density in bin, normalised to peak")
    ax.set_ylim(0, 1.08)
    style.tidy(ax)
    ax.legend(loc="upper left")
    if title:
        ax.set_title(title, loc="left")
    if own:
        fig.tight_layout()
    return fig, ax


def plot_direction_spectra(spectra: dict, title: str | None = None, max_bars: int = 12):
    """Small multiples of direction-spectrum eigenvalues (one panel per key)."""
    keys = list(spectra)
    fig, axes = plt.subplots(1, len(keys), figsize=(2.6 * len(keys) + 1, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        lam = np.asarray(spectra[key]["eigenvalues"])[:max_bars]
        ax.bar(np.arange(1, len(lam) + 1), lam, width=0.6, color=style.series(0))
        ax.set_xticks(np.arange(1, len(lam) + 1, max(1, len(lam) // 6)))
        ax.set_xlim(0.3, max(len(lam), 4) + 0.7)
        ax.set_title(str(key), loc="left", fontsize=9.5)
        ax.set_xlabel("component")
        ax.text(0.97, 0.92, f"d_eff = {spectra[key]['effective_dimension']:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, color=style.INK_2)
        style.tidy(ax)
    axes[0].set_ylabel("second-moment eigenvalue")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    return fig


def plot_effective_dimension(ns, series_dict: dict, reference: dict | None = None, title: str | None = None):
    """Grouped bars: effective chirality dimension vs. n for several volume types."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    keys = list(series_dict)
    width = 0.8 / len(keys)
    x = np.arange(len(ns))
    for i, key in enumerate(keys):
        vals = np.asarray(series_dict[key])
        bars = ax.bar(x + (i - (len(keys) - 1) / 2) * width, vals, width=width * 0.92, color=style.series(i), label=key)
        if i == int(np.argmax([np.max(series_dict[k]) for k in keys])):   # label only the extreme series
            for b_, v in zip(bars, vals):
                ax.text(b_.get_x() + b_.get_width() / 2, v + 0.08, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=style.INK_2)
    if reference:
        for j, (key, vals) in enumerate(reference.items()):
            ax.plot(x, vals, color=style.INK_2 if j == 0 else style.MUTED, linewidth=1.2, marker="o", markersize=4,
                    markerfacecolor=style.SURFACE, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n = {n}" for n in ns])
    ax.set_ylabel("effective dimension of chirality directions")
    style.tidy(ax)
    ax.legend(loc="upper left")
    if title:
        ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def plot_ladder_heatmap(ns, Ns, measured: np.ndarray, title: str | None = None):
    """Highest non-vanishing chirality degree vs. (n, N); pseudoscalar cells outlined."""
    fig, ax = plt.subplots(figsize=(1.05 * len(Ns) + 2.8, 0.75 * len(ns) + 1.6))
    M = np.asarray(measured, dtype=float)
    im = ax.imshow(M, cmap=style.seq_cmap, vmin=0, vmax=max(ns), aspect="auto", origin="lower")
    for i, n in enumerate(ns):
        for j, N in enumerate(Ns):
            val = int(M[i, j])
            binary = val == n
            color = style.SURFACE if val > 0.55 * max(ns) else style.INK
            ax.text(j, i, f"{val}" + ("★" if binary else ""), ha="center", va="center", fontsize=9.5, color=color)
            if binary:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=style.CATEGORICAL[1], linewidth=1.6))
    ax.set_xticks(range(len(Ns)))
    ax.set_xticklabels([f"N = {N}" for N in Ns])
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([f"n = {n}" for n in ns])
    ax.set_xlabel("simultaneously interfering volumes N")
    ax.set_ylabel("dimension n")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    style.colorbar(fig, im, ax, "highest non-zero chirality degree")
    if title:
        ax.set_title(title, loc="left")
    ax.text(0.0, -0.22, "★ / outlined: degree = n, chirality is a pseudoscalar (binary sign).  Prediction: min(n, 2N − 1).",
            transform=ax.transAxes, fontsize=8, color=style.INK_2, va="top")
    fig.tight_layout()
    return fig


def plot_heatmap(x, y, Z, xlabel: str, ylabel: str, title: str | None = None, cmap=None, symmetric: bool = False,
                 cbar_label: str | None = None, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
    else:
        fig = ax.figure
    Z = np.asarray(Z)
    if symmetric:
        lim = np.abs(Z).max() + 1e-300
        im = ax.imshow(Z, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap or style.div_cmap,
                       vmin=-lim, vmax=lim, aspect="auto")
    else:
        im = ax.imshow(Z, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap or style.seq_cmap,
                       aspect="auto")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for s in ax.spines.values():
        s.set_color(style.GRID)
    ax.tick_params(colors=style.MUTED, labelcolor=style.INK_2)
    style.colorbar(fig, im, ax, cbar_label)
    if title:
        ax.set_title(title, loc="left")
    if own:
        fig.tight_layout()
    return fig, ax


# ----------------------------------------------------------------------------
# many volumes: dominance cells, rank maps, higher Chern densities
# ----------------------------------------------------------------------------
def plot_cells_and_junctions(system, origin, u, v, extent, res: int = 240, t: float = 0.0,
                             title: str | None = None, junctions=None):
    """Dominance cells with interfaces, twist rank ratio, and the F∧F density."""
    X, S, T, frame = plane_grid(origin, u, v, extent, res)
    ext = _extent(S, T)
    ea = ExteriorAlgebra(system.n)
    q = geometry_at(system, X, t, ea)
    dom = dominant_volume(system, X, t).reshape(S.shape)
    rates, _ = rotation_planes(q["F"])
    ratio = (rates[:, 1] / np.maximum(rates[:, 0], 1e-300)).reshape(S.shape) if rates.shape[1] > 1 else np.zeros(S.shape)
    twist = ea.norm(ea.from_antisymmetric(q["F"])).reshape(S.shape)
    panels = 3 + (1 if 4 in q["ladder"] else 0)
    fig, axes = plt.subplots(1, panels, figsize=(3.9 * panels, 3.9))
    ax = axes[0]
    ax.imshow(dom, origin="lower", extent=ext, cmap=style.cat_cmap, vmin=-0.5, vmax=7.5, alpha=0.28, aspect="equal",
              interpolation="nearest")
    ax.contour(S, T, dom, levels=np.arange(system.N) + 0.5, colors=[style.INK], linewidths=1.0)
    for a in range(system.N):
        c = system[a].center_at(t)
        loc = frame.T @ (c - np.asarray(origin))
        ax.scatter([loc[0]], [loc[1]], s=34, facecolor=style.series(a), edgecolor=style.SURFACE, linewidth=1.0)
        ax.annotate(f"volume {a + 1}", loc, xytext=(5, 4), textcoords="offset points", fontsize=8, color=style.INK_2)
    if junctions is not None and len(junctions):
        J = (np.asarray(junctions) - np.asarray(origin)) @ frame
        ax.scatter(J[:, 0], J[:, 1], s=14, facecolor=style.INK, edgecolor=style.SURFACE, linewidth=0.6, zorder=5)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    style.image_axis(ax, "Dominance cells and interfaces")
    ax = axes[1]
    im = ax.imshow(twist, origin="lower", extent=ext, cmap=style.seq_cmap, aspect="equal")
    ax.contour(S, T, dom, levels=np.arange(system.N) + 0.5, colors=[style.INK], linewidths=0.7, alpha=0.6)
    style.image_axis(ax, "Twist  |F|")
    style.colorbar(fig, im, ax)
    ax = axes[2]
    im = ax.imshow(ratio, origin="lower", extent=ext, cmap=style.seq_cmap, vmin=0, vmax=1, aspect="equal")
    ax.contour(S, T, dom, levels=np.arange(system.N) + 0.5, colors=[style.INK], linewidths=0.7, alpha=0.6)
    style.image_axis(ax, "Second twist plane  λ₂ / λ₁")
    style.colorbar(fig, im, ax)
    if panels == 4:
        ax = axes[3]
        c4 = q["norms"][4].reshape(S.shape)
        im = ax.imshow(c4, origin="lower", extent=ext, cmap=style.seq_cmap, aspect="equal")
        ax.contour(S, T, dom, levels=np.arange(system.N) + 0.5, colors=[style.INK], linewidths=0.7, alpha=0.6)
        style.image_axis(ax, "Second Chern density  |F∧F|")
        style.colorbar(fig, im, ax)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    return fig, {"S": S, "T": T, "frame": frame, "q": q, "dominant": dom, "ratio": ratio, "twist": twist}


def plot_quiver_on_cloud(P3, V3, color_vals, title: str | None = None, label: str = "", count: int = 350, rng=None,
                         elev: float = 24, azim: float = -50, length: float = 0.25):
    """3-D quiver of a vector field sampled on a point cloud (both given in 3 coords)."""
    rng = np.random.default_rng(rng)
    idx = rng.choice(len(P3), size=min(count, len(P3)), replace=False)
    P, V, c = P3[idx], V3[idx], np.asarray(color_vals)[idx]
    norm = matplotlib.colors.Normalize(0, np.max(c) + 1e-300)
    cols = style.seq_cmap(0.25 + 0.75 * norm(c))
    Vn = V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-300)
    fig = plt.figure(figsize=(6.4, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.quiver(P[:, 0], P[:, 1], P[:, 2], Vn[:, 0], Vn[:, 1], Vn[:, 2], length=length, normalize=True,
              colors=cols, linewidth=1.1, arrow_length_ratio=0.35)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=4, color=style.MUTED, alpha=0.5)
    ax.view_init(elev=elev, azim=azim)
    ax.set_facecolor(style.SURFACE)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = style.GRID
        axis._axinfo["grid"]["linewidth"] = 0.6
    ax.tick_params(colors=style.MUTED, labelsize=7.5)
    ax.set_xlabel("interface coordinate 1"); ax.set_ylabel("interface coordinate 2"); ax.set_zlabel("interface coordinate 3")
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=style.seq_cmap)
    style.colorbar(fig, m, ax, label)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5)
    return fig, ax


def plot_lines(x, series_dict: dict, xlabel: str, ylabel: str, title: str | None = None, reference=None,
               logy: bool = False, ax=None, direct_labels: bool = True):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
    else:
        fig = ax.figure
    if reference is not None:
        ax.plot(x, reference[1], color=style.BASELINE, linewidth=2.4, label=reference[0])
    for i, (k, y) in enumerate(series_dict.items()):
        ax.plot(x, y, color=style.series(i), label=k)
        if direct_labels:
            ax.annotate(k, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=style.INK_2, va="center")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style.tidy(ax)
    if len(series_dict) + (reference is not None) >= 2:
        ax.legend(loc="best")
    if title:
        ax.set_title(title, loc="left")
    if own:
        fig.tight_layout()
    return fig, ax
