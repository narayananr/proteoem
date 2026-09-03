"""Render the platform schematic for the README.

This is a generic depiction of iterative single-molecule affinity mapping,
drawn only from design features that are publicly described for this class of
instrument: single protein molecules immobilised one per landing pad on a dense
array, a panel of affinity probes applied over repeated cycles, and a yes/no
binding readout per molecule per cycle.

It is not a depiction of any particular commercial instrument, carries no
vendor branding, and states no performance figures. It ends where the method
schematic begins: the per-pad call sequence it produces is the affinity trace
that ``make_method_figure.py`` takes as input, and the traces drawn in panel C
are literally the same arrays that figure uses.

Usage
-----
    python docs/figures/make_platform_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from figstyle import NA, THEMES, blank, call_cell, panel_title, rcparams, save

# The same traces the method figure consumes, so the two figures join up.
from make_method_figure import CYCLE_TO_PROBE, OBSERVATIONS

# Which pads on the illustrative array hold a molecule, and which probe-1 binds.
# Fixed rather than random so the figure is byte-stable across runs.
GRID_COLS, GRID_ROWS = 8, 3
SEED = 7


def array_state():
    """Occupancy of each landing pad, and whether the cycle-1 probe bound there.

    The first three pads of the top row are pinned to the three traces the
    method figure uses, so panels B and C describe the same three molecules
    rather than two unrelated illustrations.
    """
    rng = np.random.default_rng(SEED)
    occupied = rng.random((GRID_ROWS, GRID_COLS)) > 0.18
    bound = occupied & (rng.random((GRID_ROWS, GRID_COLS)) < 0.45)

    tracked = OBSERVATIONS.shape[0]
    occupied[0, :tracked] = True
    bound[0, :tracked] = OBSERVATIONS[:, 0] == 1
    return occupied, bound


def pad(ax, cx, cy, r, state, t):
    """One landing pad. state: 'empty', 'occupied' or 'bound'."""
    if state == "empty":
        ax.add_patch(Circle((cx, cy), r, facecolor=t["surface"],
                            edgecolor=t["rule"], lw=1.1, linestyle=(0, (2, 2))))
        return
    ax.add_patch(Circle((cx, cy), r, facecolor=t["surface"],
                        edgecolor=t["text_secondary"], lw=1.2))
    if state == "bound":
        ax.add_patch(Circle((cx, cy), r * 0.62, facecolor=t["series"][0],
                            edgecolor="none"))
    else:
        ax.add_patch(Circle((cx, cy), r * 0.42, facecolor=t["ink"],
                            edgecolor="none"))


def draw_grid(ax, occupied, bound, t, show_binding):
    x0, y0, dx, dy, r = 13.0, 64.0, 11.0, 13.0, 4.0
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            cx, cy = x0 + j * dx, y0 - i * dy
            if not occupied[i, j]:
                state = "empty"
            elif show_binding and bound[i, j]:
                state = "bound"
            else:
                state = "occupied"
            pad(ax, cx, cy, r, state, t)

    # name the three pads that panel C follows across cycles
    for j in range(OBSERVATIONS.shape[0]):
        ax.text(x0 + j * dx, y0 + r + 2.5, f"Pad {j + 1}", ha="center",
                va="bottom", fontsize=8.5, color=t["text_secondary"])
    return x0, y0, dx, dy, r


def legend(ax, items, t, y=12.0, x=6.0, step=30.0):
    for state, label in items:
        pad(ax, x, y, 3.4, state, t)
        ax.text(x + 6.0, y, label, ha="left", va="center", fontsize=9,
                color=t["text_primary"])
        x += step


def panel_a(ax, occupied, bound, t):
    blank(ax, t)
    panel_title(
        ax, "A", "One molecule per landing pad",
        "Protein molecules are immobilised on a dense array,\n"
        "at most one to a pad, and stay there for the whole run.", t,
    )
    draw_grid(ax, occupied, bound, t, show_binding=False)
    ax.text(6.0, 28.0,
            "Because each molecule is read on its own, the features seen\n"
            "together belong to that one molecule. Some pads stay empty.",
            ha="left", va="top", fontsize=9.5, color=t["text_secondary"])
    legend(ax, [("occupied", "One molecule"), ("empty", "Empty pad")], t)


def panel_b(ax, occupied, bound, t):
    blank(ax, t)
    panel_title(
        ax, "B", "One probe cycle",
        "A probe is washed over the array. It binds some\n"
        "molecules and not others, then is washed off.", t,
    )
    draw_grid(ax, occupied, bound, t, show_binding=True)
    ax.text(6.0, 28.0,
            "Every occupied pad records one yes or no call for this probe.\n"
            "Probes bind imperfectly, so a call is evidence, not an identity.",
            ha="left", va="top", fontsize=9.5, color=t["text_secondary"])
    legend(ax, [("bound", "Probe bound"), ("occupied", "No binding")], t)


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
        ax.text(x0 + n_cyc * (w + gap) - gap + 1.5, y + h / 2, "trace", ha="left",
                va="center", fontsize=9.5, color=t["text_secondary"])

    ax.text(6.0, 23.0,
            "A pad's calls in order are its affinity trace. Traces are the input\n"
            "to the model: the next figure turns them into abundances.",
            ha="left", va="top", fontsize=9.5, color=t["text_secondary"])

    ly = 9.0
    items = [(1, "Bound"), (0, "Not bound"), (NA, "No call")]
    lx = 6.0
    for call, label in items:
        call_cell(ax, lx, ly - 4.0, 7.0, 8.0, call, t)
        ax.text(lx + 9.0, ly, label, ha="left", va="center", fontsize=9,
                color=t["text_primary"])
        lx += 30.0


def render(mode, outdir):
    t = THEMES[mode]
    rcparams()
    occupied, bound = array_state()

    fig = plt.figure(figsize=(16, 6.0), facecolor=t["surface"])
    gs = fig.add_gridspec(1, 3, left=0.012, right=0.988, top=0.95, bottom=0.03,
                          wspace=0.07)
    panel_a(fig.add_subplot(gs[0, 0]), occupied, bound, t)
    panel_b(fig.add_subplot(gs[0, 1]), occupied, bound, t)
    panel_c(fig.add_subplot(gs[0, 2]), t)

    return save(fig, outdir, f"platform-{mode}", t)


def main():
    outdir = Path(__file__).resolve().parent
    occupied, bound = array_state()
    assert occupied.any() and (~occupied).any(), "need both empty and occupied pads"
    assert bound.sum() > 0, "probe must bind somewhere"
    assert (occupied & ~bound).sum() > 0, "some occupied pads must not bind"
    tracked = OBSERVATIONS.shape[0]
    assert (bound[0, :tracked] == (OBSERVATIONS[:, 0] == 1)).all(), (
        "the tracked pads must show the same cycle-1 calls as their traces"
    )
    print(f"array {GRID_ROWS}x{GRID_COLS}: {occupied.sum()} occupied, "
          f"{bound.sum()} bound this cycle")

    for mode in ("light", "dark"):
        stem = render(mode, outdir)
        print(f"wrote {stem}.svg and {stem}@2x.png")


if __name__ == "__main__":
    main()
