"""Shared publication style for ProteoEM results figures (matplotlib).

Uses the bundled DejaVu Sans face so figures render identically without any
system font, and the same colour-blind-safe Dark2 accents as Figures 1-5.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# Dark2 accents (match the hand-authored SVG figures)
GREEN = "#1B9E77"   # true origin / primary
ORANGE = "#D95F02"  # competitor / secondary
PURPLE = "#7570B3"  # tertiary
GREY = "#9aa0a6"    # de-emphasised field
DARK = "#444444"    # neutral annotation

# Sequential ramp for the probe-uniqueness strip (pale = shared, dark = splits 50/50)
UNIQ_CMAP = LinearSegmentedColormap.from_list("uniq", ["#f2f2f2", "#7ec9b1", GREEN])

RC = {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 7.2,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,   # embed as TrueType so text stays selectable/editable
    "ps.fonttype": 42,
}


def apply_style() -> None:
    mpl.rcParams.update(RC)
