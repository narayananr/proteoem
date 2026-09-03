"""Shared drawing style for the ProteoEM figures.

One place for the palette, the theme surfaces and the small drawing helpers, so
every figure in this directory reads as part of one set. Both themes are
selected rather than derived by inverting one another.

The categorical pair passes lightness, chroma, colour-vision-deficiency
separation, normal-vision separation and contrast against its own surface in
both light and dark modes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

NA = -1

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "series": ("#2a78d6", "#eb6834"),
        "ink": "#0b0b0b",
        "rule": "#c8c7c2",
        "muted_fill": "#e8e7e3",
        # sequential blue, light -> dark; the lightest step recedes to surface
        "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "series": ("#3987e5", "#d95926"),
        "ink": "#ffffff",
        "rule": "#494845",
        "muted_fill": "#2e2e2b",
        # on a dark surface the near-zero end is the dark step, so it recedes
        "seq": ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#9ec5f4", "#cde2fb"],
    },
}


def on_color(rgba):
    """Black or white text, whichever contrasts with the fill underneath."""
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0b0b0b" if lum > 0.55 else "#ffffff"


def call_cell(ax, x, y, w, h, call, t):
    """A single binary call. Identity never rests on colour alone."""
    if call == 1:
        ax.add_patch(
            Rectangle((x, y), w, h, facecolor=t["ink"], edgecolor=t["ink"], lw=1.2)
        )
    elif call == 0:
        ax.add_patch(
            Rectangle((x, y), w, h, facecolor=t["surface"], edgecolor=t["ink"], lw=1.2)
        )
    else:
        ax.add_patch(
            Rectangle(
                (x, y), w, h,
                facecolor=t["surface"], edgecolor=t["text_secondary"],
                lw=1.2, hatch="////",
            )
        )


def blank(ax, t):
    """A bare 0-100 by 0-100 drawing surface with no axes furniture."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_facecolor(t["surface"])


def panel_title(ax, letter, title, subtitle, t):
    ax.text(
        0, 99, f"{letter}. {title}",
        ha="left", va="top", fontsize=13, fontweight="bold", color=t["text_primary"],
    )
    ax.text(
        0, 90, subtitle,
        ha="left", va="top", fontsize=9.5, color=t["text_secondary"],
    )


def rcparams():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "hatch.linewidth": 1.0,
        "svg.fonttype": "path",
    })


def save(fig, outdir: Path, name: str, t) -> Path:
    """Write both the vector and a 2x raster of one themed figure."""
    stem = Path(outdir) / name
    fig.savefig(f"{stem}.svg", facecolor=t["surface"], format="svg")
    fig.savefig(f"{stem}@2x.png", facecolor=t["surface"], dpi=200)
    plt.close(fig)
    return stem
