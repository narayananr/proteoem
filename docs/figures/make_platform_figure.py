"""Render the platform schematic for the README.

A generic depiction of iterative single-molecule affinity mapping, drawn only
from design features that are publicly described for this class of instrument:
single protein molecules immobilised one per landing pad on a dense array, a
panel of affinity probes applied over repeated cycles, and a yes/no binding
readout per molecule per cycle.

It is not a depiction of any particular commercial instrument, carries no
vendor branding, and states no performance figures. It ends where the method
schematic begins: the per-pad call sequence it produces is the affinity trace
that ``make_method_figure.py`` takes as input, and the traces drawn in panel C
are the same arrays that figure fits.

Usage
-----
    python docs/figures/make_platform_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figstyle import NA, THEMES, blank, call_cell, panel_title, rcparams, save

# The same traces the method figure consumes, so the two figures join up.
from make_method_figure import CYCLE_TO_PROBE, OBSERVATIONS

SEED = 11

# One row of landing pads seen from the side. The first three are the pads that
# panel C follows, so their cycle-1 binding is pinned to the real traces.
OCCUPIED = (True, True, True, True, False, True, True)
BOUND = (True, False, True, False, False, True, False)

CHIP_X0, CHIP_X1 = 5.0, 95.0
CHIP_TOP, CHIP_H = 44.0, 6.0
PAD_X = (12.0, 23.5, 35.0, 46.5, 58.0, 69.5, 81.0)
STALK_H = 27.0


# --------------------------------------------------------------------------
# Protein squiggles
# --------------------------------------------------------------------------
def squiggle(rng, x_anchor, y_anchor, height=STALK_H, n=160):
    """A protein drawn as a smooth random squiggle rising from its pad."""
    k = 9
    knot_y = np.linspace(0.0, height, k)
    knot_x = rng.uniform(-5.2, 5.2, k)
    knot_x[0] = 0.0  # anchored where it meets the chip

    t = np.linspace(0.0, height, n)
    x = np.interp(t, knot_y, knot_x)

    # a little extra curl, then smooth the whole thing
    x = x + 1.6 * np.sin(t / height * rng.uniform(4.0, 7.0) * np.pi + rng.random())
    win = 17
    padded = np.pad(x, (win // 2, win // 2), mode="edge")
    x = np.convolve(padded, np.ones(win) / win, mode="valid")[:n]

    return x_anchor + x, y_anchor + t


def shape_rng(j):
    """Per-pad generator, so a molecule looks the same in every panel.

    Seeding per pad rather than walking one stream means drawing a probe on one
    pad cannot change the shape of any other, which is what made panels A and B
    disagree when the stream was shared.
    """
    return np.random.default_rng([SEED, j])


def probe_rng(j):
    return np.random.default_rng([SEED, 1000 + j])


def draw_probe(ax, xs, ys, t, rng):
    """A short coloured line, the bound probe, meeting the protein."""
    i = int(len(xs) * rng.uniform(0.42, 0.72))
    px, py = xs[i], ys[i]
    side = 1.0 if rng.random() < 0.5 else -1.0
    dx, dy = side * 6.2, rng.uniform(-1.6, 1.6)
    ax.plot([px, px + dx], [py, py + dy], color=t["series"][0], lw=3.4,
            solid_capstyle="round", zorder=4)
    ax.plot([px], [py], marker="o", ms=3.4, color=t["series"][0], zorder=5)


def draw_chip(ax, t):
    ax.add_patch(plt.Rectangle(
        (CHIP_X0, CHIP_TOP - CHIP_H), CHIP_X1 - CHIP_X0, CHIP_H,
        facecolor=t["muted_fill"], edgecolor=t["rule"], lw=1.0, zorder=1,
    ))
    ax.plot([CHIP_X0, CHIP_X1], [CHIP_TOP, CHIP_TOP],
            color=t["text_secondary"], lw=1.4, zorder=2)
    ax.text(CHIP_X1 - 1.5, CHIP_TOP - CHIP_H / 2, "array surface", ha="right",
            va="center", fontsize=8, color=t["text_secondary"], zorder=3)


def draw_row(ax, t, show_binding):
    """The chip, its landing pads, the immobilised proteins and any probes."""
    draw_chip(ax, t)

    for j, x in enumerate(PAD_X):
        # the landing pad itself
        ax.plot([x - 2.4, x + 2.4], [CHIP_TOP, CHIP_TOP],
                color=t["text_primary"], lw=3.0, solid_capstyle="round", zorder=3)

        if not OCCUPIED[j]:
            continue

        rng = shape_rng(j)
        height = STALK_H * rng.uniform(0.82, 1.14)
        xs, ys = squiggle(rng, x, CHIP_TOP, height=height)
        ax.plot(xs, ys, color=t["text_primary"], lw=2.0,
                solid_capstyle="round", zorder=3)
        if show_binding and BOUND[j]:
            draw_probe(ax, xs, ys, t, probe_rng(j))

    for j in range(OBSERVATIONS.shape[0]):
        ax.text(PAD_X[j], CHIP_TOP - CHIP_H - 3.0, f"Pad {j + 1}", ha="center",
                va="top", fontsize=9, color=t["text_secondary"])


def mini_legend(ax, t, items, y=13.0):
    """Small inline key, drawn in the same visual language as the panel."""
    x = 6.0
    for kind, label in items:
        rng = shape_rng(0)
        if kind == "empty":
            ax.plot([x - 2.4, x + 2.4], [y - 4.5, y - 4.5],
                    color=t["text_primary"], lw=3.0, solid_capstyle="round")
        else:
            xs, ys = squiggle(rng, x, y - 4.5, height=9.5, n=60)
            ax.plot(xs, ys, color=t["text_primary"], lw=1.8,
                    solid_capstyle="round")
            ax.plot([x - 2.4, x + 2.4], [y - 4.5, y - 4.5],
                    color=t["text_primary"], lw=3.0, solid_capstyle="round")
            if kind == "probe":
                ax.plot([xs[34], xs[34] + 5.4], [ys[34], ys[34]],
                        color=t["series"][0], lw=3.0, solid_capstyle="round")
        ax.text(x + 9.5, y, label, ha="left", va="center", fontsize=9,
                color=t["text_primary"])
        x += 31.0


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------
def panel_a(ax, t):
    blank(ax, t)
    panel_title(
        ax, "A", "One molecule per landing pad",
        "Protein molecules are immobilised on a dense array,\n"
        "at most one to a pad, and stay there for the whole run.", t,
    )
    draw_row(ax, t, show_binding=False)
    ax.text(6.0, 30.0,
            "Each molecule is read on its own, so the features seen together\n"
            "belong to that one molecule. Some pads stay empty.",
            ha="left", va="top", fontsize=9.5, color=t["text_secondary"])
    mini_legend(ax, t, [("protein", "One protein"), ("empty", "Empty pad")])


def panel_b(ax, t):
    blank(ax, t)
    panel_title(
        ax, "B", "One probe cycle",
        "A probe is washed over the array. It binds some\n"
        "molecules and not others, then is washed off.", t,
    )
    draw_row(ax, t, show_binding=True)
    ax.text(6.0, 30.0,
            "Every occupied pad records one yes or no call for this probe.\n"
            "Probes bind imperfectly, so a call is evidence, not an identity.",
            ha="left", va="top", fontsize=9.5, color=t["text_secondary"])
    mini_legend(ax, t, [("probe", "Probe bound"), ("protein", "No binding")])


def panel_c(ax, t):
    blank(ax, t)
    panel_title(
        ax, "C", "Repeat, and each pad has a trace",
        "The panel is applied cycle after cycle to the same molecules.", t,
    )

    n_mol, n_cyc = OBSERVATIONS.shape
    x0, w, gap = 20.0, 9.0, 2.0
    top, h, vgap = 66.0, 10.0, 3.5

    for c in range(n_cyc):
        x = x0 + c * (w + gap)
        ax.text(x + w / 2, top + 3.5, f"P{CYCLE_TO_PROBE[c]}", ha="center",
                va="bottom", fontsize=9.5, color=t["text_primary"])
        ax.text(x + w / 2, top + 11.0, f"{c + 1}", ha="center", va="bottom",
                fontsize=8.5, color=t["text_secondary"])
    ax.text(x0 - 4, top + 11.0, "cycle", ha="right", va="bottom", fontsize=8.5,
            color=t["text_secondary"])

    for i in range(n_mol):
        y = top - (i + 1) * h - i * vgap
        ax.text(x0 - 4, y + h / 2, f"Pad {i + 1}", ha="right", va="center",
                fontsize=10, color=t["text_primary"])
        for c in range(n_cyc):
            call_cell(ax, x0 + c * (w + gap), y, w, h, OBSERVATIONS[i, c], t)
        ax.text(x0 + n_cyc * (w + gap) - gap + 1.5, y + h / 2, "trace",
                ha="left", va="center", fontsize=9.5, color=t["text_secondary"])

    ax.text(6.0, 23.0,
            "A pad's calls in order are its affinity trace. Traces are the input\n"
            "to the model: the next figure turns them into abundances.",
            ha="left", va="top", fontsize=9.5, color=t["text_secondary"])

    lx, ly = 6.0, 9.0
    for call, label in [(1, "Bound"), (0, "Not bound"), (NA, "No call")]:
        call_cell(ax, lx, ly - 4.0, 7.0, 8.0, call, t)
        ax.text(lx + 9.0, ly, label, ha="left", va="center", fontsize=9,
                color=t["text_primary"])
        lx += 30.0


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def render(mode, outdir):
    t = THEMES[mode]
    rcparams()

    fig = plt.figure(figsize=(16, 6.0), facecolor=t["surface"])
    gs = fig.add_gridspec(1, 3, left=0.012, right=0.988, top=0.95, bottom=0.03,
                          wspace=0.07)
    panel_a(fig.add_subplot(gs[0, 0]), t)
    panel_b(fig.add_subplot(gs[0, 1]), t)
    panel_c(fig.add_subplot(gs[0, 2]), t)

    return save(fig, outdir, f"platform-{mode}", t)


def main():
    outdir = Path(__file__).resolve().parent

    tracked = OBSERVATIONS.shape[0]
    assert len(OCCUPIED) == len(BOUND) == len(PAD_X), "pad tables must line up"
    assert all(OCCUPIED[:tracked]), "the tracked pads must hold a molecule"
    assert not any(b and not o for b, o in zip(BOUND, OCCUPIED)), (
        "an empty pad cannot report binding"
    )
    assert list(BOUND[:tracked]) == [c == 1 for c in OBSERVATIONS[:, 0]], (
        "tracked pads must show the same cycle-1 calls as their traces"
    )
    assert not all(OCCUPIED), "show at least one empty pad"
    print(f"{sum(OCCUPIED)} of {len(PAD_X)} pads occupied, "
          f"{sum(BOUND)} bound this cycle")

    for mode in ("light", "dark"):
        stem = render(mode, outdir)
        print(f"wrote {stem}.svg and {stem}@2x.png")


if __name__ == "__main__":
    main()
