"""Matplotlib styling: one quiet, consistent look for every figure.

Colour roles follow a validated data-viz palette: a single-hue sequential
ramp (blue) for magnitudes, a blue <-> red diverging ramp with a neutral grey
midpoint for signed quantities, a fixed categorical order for identities,
and a perceptually uniform cyclic map (``twilight``) for phases.  Chrome
(axes, grid, ticks) is recessive hairline grey.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
MID_GREY = "#f0efec"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
RED_STEPS = ["#f7c9c8", "#f0a19f", "#e87b79", "#e34948", "#c23736", "#9d2b2a", "#711d1c"]

seq_cmap = LinearSegmentedColormap.from_list("tc_seq", [SURFACE] + SEQ_STEPS)
seq_cmap_r = seq_cmap.reversed()
div_cmap = LinearSegmentedColormap.from_list(
    "tc_div", list(reversed(SEQ_STEPS[::2])) + [MID_GREY] + RED_STEPS[::1])
cat_cmap = ListedColormap(CATEGORICAL, name="tc_cat")
cyclic_cmap = mpl.colormaps["twilight"]


def apply() -> None:
    """Install the house style into matplotlib's rcParams."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "figure.edgecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": "normal",
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "text.color": INK,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
        "font.size": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 5,
        "image.cmap": "tc_seq",
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })
    for cm in (seq_cmap, seq_cmap_r, div_cmap, cat_cmap):
        if cm.name not in mpl.colormaps:
            mpl.colormaps.register(cm)


def series(i: int) -> str:
    """Categorical colour for series ``i`` (fixed order, never cycled)."""
    if i >= len(CATEGORICAL):
        raise ValueError("more than 8 categorical series: fold or facet instead")
    return CATEGORICAL[i]


def tidy(ax, grid: str | None = "y") -> None:
    """Recessive chrome for a cartesian axis."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    if grid:
        ax.grid(True, axis=grid, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def image_axis(ax, title: str | None = None) -> None:
    """Chrome for an image / field panel: no ticks, thin frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_color(GRID)
        side.set_linewidth(0.8)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)


def colorbar(fig, mappable, ax, label: str | None = None):
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_edgecolor(GRID)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8, width=0.6, length=2.5)
    if label:
        cb.set_label(label, color=INK_2, fontsize=8.5)
    return cb
