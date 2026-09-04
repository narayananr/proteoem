"""Render the ProteoEM method schematic for the README.

Every number drawn by this script is computed by calling ``proteoem.fit_em``.
Nothing quantitative is hardcoded in the plotting code. The assertions below
fail loudly rather than letting a wrong figure be produced.

Usage
-----
    python docs/figures/make_method_figure.py

Writes ``method-light.svg``, ``method-dark.svg`` and 2x PNGs of both into the
directory containing this file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle

from figstyle import NA, THEMES, blank, call_cell, on_color, panel_title, rcparams, save
from proteoem import fit_em

# --------------------------------------------------------------------------
# Data. The observations and cycle_to_probe are the README quick-start values.
# Q is deliberately more overlapping than the quick-start calibration so that
# the posterior split stays visibly graded; with the quick-start Q the third
# probe is uninformative, gets dropped, and the posteriors go effectively hard.
# --------------------------------------------------------------------------
OBSERVATIONS = np.array(
    [
        [1, 0, -1, 1, 0, 0],
        [0, 1, 0, 0, 1, 1],
        [1, 0, 1, 1, 0, 1],
    ],
    dtype=np.int8,
)
CYCLE_TO_PROBE = np.array([0, 1, 2, 0, 1, 2])
Q = np.array(
    [
        [0.65, 0.35, 0.65],
        [0.75, 0.25, 0.75],
    ]
)

PROTEOFORM_NAMES = ("Proteoform A", "Proteoform B")



# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def trace_log_likelihoods(observations, cycle_to_probe, q):
    """Per-molecule log P(trace | origin), with NA cycles marginalized out.

    Computed explicitly here so the figure never has to guess how the library's
    aggregated class rows map back onto molecules.
    """
    n_mol = observations.shape[0]
    n_origin = q.shape[0]
    out = np.zeros((n_mol, n_origin))
    for i in range(n_mol):
        for k in range(n_origin):
            total = 0.0
            for cycle, call in enumerate(observations[i]):
                if call == NA:
                    continue  # marginalized, not recoded as a negative call
                p = q[k, cycle_to_probe[cycle]]
                total += np.log(p) if call == 1 else np.log1p(-p)
            out[i, k] = total
    return out


def compute():
    fit = fit_em(OBSERVATIONS, Q=Q, cycle_to_probe=CYCLE_TO_PROBE)
    agg = fit.aggregated

    assert fit.converged, "EM did not converge; refusing to draw"
    assert agg.n_retained_probes == agg.n_logical_probes, (
        f"{agg.n_logical_probes - agg.n_retained_probes} probe(s) dropped as "
        "uninformative; the figure would misrepresent the panel"
    )
    assert agg.n_unique == OBSERVATIONS.shape[0], (
        "molecules collapsed into fewer aggregated classes; per-molecule rows "
        "cannot be read off the aggregated result"
    )

    weights = np.asarray(fit.weights, dtype=float)
    expected_counts = np.asarray(fit.expected_counts, dtype=float)

    # Per-molecule responsibilities, derived from first principles.
    logl = trace_log_likelihoods(OBSERVATIONS, CYCLE_TO_PROBE, Q)
    logp = logl + np.log(weights)[None, :]
    logp -= logp.max(axis=1, keepdims=True)
    resp = np.exp(logp)
    resp /= resp.sum(axis=1, keepdims=True)

    # Cross-check the explicit computation against the library's own output.
    lib_resp = np.asarray(fit.responsibilities, dtype=float)
    assert lib_resp.shape == resp.shape
    assert np.allclose(np.sort(lib_resp, axis=0), np.sort(resp, axis=0), atol=1e-6), (
        "hand-computed responsibilities disagree with fit.responsibilities"
    )

    assert np.allclose(resp.sum(axis=1), 1.0), "responsibility rows must sum to 1"
    assert np.allclose(resp.sum(axis=0), expected_counts, atol=1e-6), (
        "responsibilities must accumulate to the expected counts"
    )
    assert np.isclose(weights.sum(), 1.0), "weights must sum to 1"
    assert np.isclose(expected_counts.sum(), OBSERVATIONS.shape[0]), (
        "expected counts must total the number of molecules"
    )
    assert np.allclose(weights, expected_counts / expected_counts.sum()), (
        "weights must be the normalized expected counts"
    )
    assert resp.max() <= 0.95, (
        f"most extreme responsibility is {resp.max():.4f}; the split would read "
        "as hard assignment rather than as division"
    )

    return fit, resp, expected_counts, weights


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------
def panel_a(ax, t):
    blank(ax, t)
    panel_title(
        ax, "A", "One molecule, read once",
        "Six probe cycles are applied in order to a single molecule.", t,
    )

    trace = OBSERVATIONS[0]
    x0, w, gap = 12.0, 11.0, 2.5
    y = 55.0
    h = 13.0

    ax.text(x0 - 4, y + h / 2, "Trace", ha="right", va="center",
            fontsize=10.5, color=t["text_primary"])

    for c, call in enumerate(trace):
        x = x0 + c * (w + gap)
        call_cell(ax, x, y, w, h, call, t)
        ax.text(x + w / 2, y + h + 3.5, f"P{CYCLE_TO_PROBE[c]}", ha="center",
                va="bottom", fontsize=10, color=t["text_primary"])
        ax.text(x + w / 2, y + h + 12.5, f"cycle {c + 1}", ha="center", va="bottom",
                fontsize=8.5, color=t["text_secondary"])
        label = "NA" if call == NA else str(int(call))
        ax.text(x + w / 2, y - 5, label, ha="center", va="top", fontsize=10.5,
                color=t["text_secondary"])

    ax.text(x0 + 3 * (w + gap) - gap / 2, y - 17,
            "Each probe is used twice.\n"
            "The six calls in order are this molecule's trace.",
            ha="center", va="top", fontsize=9.5, color=t["text_secondary"])

    # legend
    ly, lh, lw_ = 6.0, 8.0, 8.0
    items = [(1, "Positive"), (0, "Negative"), (NA, "Unavailable")]
    lx = 4.0
    for call, label in items:
        call_cell(ax, lx, ly, lw_, lh, call, t)
        ax.text(lx + lw_ + 2.0, ly + lh / 2, label, ha="left", va="center",
                fontsize=9, color=t["text_primary"])
        lx += lw_ + 26.0


def panel_b(ax, t):
    blank(ax, t)
    panel_title(
        ax, "B", "Three traces",
        "Each row is one molecule's trace, six cycles long.", t,
    )

    n_mol, n_cyc = OBSERVATIONS.shape
    x0, w, gap = 20.0, 11.0, 2.5
    top, h, vgap = 68.0, 13.0, 3.0

    for c in range(n_cyc):
        x = x0 + c * (w + gap)
        ax.text(x + w / 2, top + 4.0, f"P{CYCLE_TO_PROBE[c]}", ha="center",
                va="bottom", fontsize=10, color=t["text_primary"])

    for i in range(n_mol):
        y = top - (i + 1) * h - i * vgap
        ax.text(x0 - 4, y + h / 2, f"Molecule {i + 1}", ha="right", va="center",
                fontsize=10.5, color=t["text_primary"])
        for c in range(n_cyc):
            call_cell(ax, x0 + c * (w + gap), y, w, h, OBSERVATIONS[i, c], t)

    ax.text(x0, 8.0,
            "Molecule 1 has a missing call at cycle 3.\n"
            "That cycle is skipped. It is not counted as a negative.",
            ha="left", va="bottom", fontsize=9.5, color=t["text_secondary"])


def panel_c(ax, t):
    blank(ax, t)
    panel_title(
        ax, "C", "The emission matrix Q",
        "Measured in advance from known proteoforms.\n"
        "Held fixed during the fit.", t,
    )

    cmap = LinearSegmentedColormap.from_list("seq", t["seq"])
    n_origin, n_probe = Q.shape
    x0, w, gap = 26.0, 17.0, 2.5
    top, h, vgap = 60.0, 15.0, 2.5

    for j in range(n_probe):
        ax.text(x0 + j * (w + gap) + w / 2, top + 4.0, f"P{j}", ha="center",
                va="bottom", fontsize=10, color=t["text_primary"])

    for k in range(n_origin):
        y = top - (k + 1) * h - k * vgap
        # swatch drawn narrower than tall so it reads square on a wide axes
        ax.add_patch(Rectangle((x0 - 5.6, y + h / 2 - 1.6), 2.3, 3.2,
                               facecolor=t["series"][k], edgecolor="none"))
        ax.text(x0 - 7.6, y + h / 2, PROTEOFORM_NAMES[k], ha="right", va="center",
                fontsize=10.5, color=t["text_primary"])
        for j in range(n_probe):
            v = Q[k, j]
            x = x0 + j * (w + gap)
            fill = cmap(v)
            ax.add_patch(Rectangle((x, y), w, h, facecolor=fill,
                                   edgecolor=t["surface"], lw=1.5))
            ax.text(x + w / 2, y + h / 2, f"{v:.2f}", ha="center", va="center",
                    fontsize=11, color=on_color(fill))

    ax.text(2.0, 20.0,
            "Each cell is the probability that a probe gives a\n"
            "positive call for that proteoform. The two proteoforms\n"
            "behave alike, so one trace cannot tell them apart.",
            ha="left", va="top", fontsize=9.5, color=t["text_secondary"])


def panel_d(ax, resp, expected_counts, weights, t):
    blank(ax, t)
    panel_title(
        ax, "D", "From traces to abundance",
        "Each trace counts as one molecule. That count is shared between the proteoforms.", t,
    )

    n_mol, n_origin = resp.shape

    # ---- split bars -------------------------------------------------------
    bx0, bw = 14.0, 34.0
    top, h, vgap = 68.0, 10.0, 4.0
    seg_gap = 0.35  # surface-coloured gap between stacked segments

    def column_header(x, title, gloss):
        ax.text(x, top + 11.0, title, ha="center", va="bottom", fontsize=10.5,
                color=t["text_primary"])
        ax.text(x, top + 6.0, gloss, ha="center", va="bottom", fontsize=9,
                color=t["text_secondary"], style="italic")

    column_header(bx0 + bw / 2, "Posterior responsibility",
                  "probability this trace is each proteoform")

    bar_y = []
    for i in range(n_mol):
        y = top - (i + 1) * h - i * vgap
        bar_y.append(y)
        ax.text(bx0 - 3, y + h / 2, f"Molecule {i + 1}", ha="right", va="center",
                fontsize=10.5, color=t["text_primary"])
        x = bx0
        for k in range(n_origin):
            seg = resp[i, k] * bw
            draw = max(seg - (seg_gap if k == 0 else 0.0), 0.1)
            ax.add_patch(Rectangle((x, y), draw, h, facecolor=t["series"][k],
                                   edgecolor="none"))
            ax.text(x + seg / 2, y + h / 2, f"{resp[i, k]:.3f}", ha="center",
                    va="center", fontsize=10, color="#ffffff")
            x += seg

    # origin legend, directly labelled
    lx = bx0
    for k in range(n_origin):
        # narrow-but-tall so the swatch reads square on this wide, short axes
        ax.add_patch(Rectangle((lx, 19.5), 1.3, 5.0, facecolor=t["series"][k],
                               edgecolor="none"))
        ax.text(lx + 2.6, 22.0, PROTEOFORM_NAMES[k], ha="left", va="center",
                fontsize=10, color=t["text_primary"])
        lx += 14.0

    # ---- accumulation into expected counts --------------------------------
    cx = [58.0, 69.0]
    cw = 8.0
    base = 30.0
    scale = 26.0 / expected_counts.max()

    column_header(cx[0] + cw + 1.5, "Expected counts",
                  "how many molecules each proteoform gets")

    for k in range(n_origin):
        y = base
        for i in range(n_mol):
            seg = resp[i, k] * scale
            ax.add_patch(Rectangle((cx[k], y), cw, max(seg - seg_gap, 0.1),
                                   facecolor=t["series"][k], edgecolor="none"))
            y += seg
        ax.text(cx[k] + cw / 2, base - 3.0, PROTEOFORM_NAMES[k].split()[-1],
                ha="center", va="top", fontsize=10, color=t["text_primary"])
        ax.text(cx[k] + cw / 2, y + 2.0, f"{expected_counts[k]:.3f}", ha="center",
                va="bottom", fontsize=11, color=t["text_primary"])

    ax.add_patch(
        FancyArrowPatch(
            (bx0 + bw + 1.5, bar_y[1] + h / 2),
            (cx[0] - 2.0, base + 13.0),
            arrowstyle="-|>", mutation_scale=13, lw=1.2,
            color=t["text_secondary"], shrinkA=0, shrinkB=0,
        )
    )
    ax.text((bx0 + bw + cx[0]) / 2, bar_y[1] + h / 2 + 2.0, "add up\neach column",
            ha="center", va="bottom", fontsize=9, color=t["text_secondary"])

    # ---- normalization to mixture weights ---------------------------------
    wx0, ww = 86.0, 12.0
    wy, wh = base, 26.0

    column_header(wx0 + ww / 2, "Mixture weights", "relative abundance")

    y = wy
    for k in range(n_origin):
        seg = weights[k] * wh
        ax.add_patch(Rectangle((wx0, y), ww, max(seg - seg_gap, 0.1),
                               facecolor=t["series"][k], edgecolor="none"))
        ax.text(wx0 + ww / 2, y + seg / 2, f"{weights[k]:.3f}", ha="center",
                va="center", fontsize=11, color="#ffffff")
        y += seg
    ax.text(wx0 + ww / 2, wy - 3.0, "adds to 1", ha="center", va="top",
            fontsize=10, color=t["text_primary"])

    ax.add_patch(
        FancyArrowPatch(
            (cx[1] + cw + 1.5, base + 13.0), (wx0 - 2.0, base + 13.0),
            arrowstyle="-|>", mutation_scale=13, lw=1.2,
            color=t["text_secondary"], shrinkA=0, shrinkB=0,
        )
    )
    ax.text((cx[1] + cw + wx0) / 2, base + 15.0, "divide by\n3 molecules",
            ha="center", va="bottom", fontsize=9, color=t["text_secondary"])

    ax.text(bx0, 15.0,
            "Each bar is one trace. Its two parts add up to 1. No trace is given to one proteoform "
            "outright.\n"
            "Adding up a column gives the number of molecules for that proteoform.\n"
            "Dividing by 3 gives the relative abundance of each proteoform among these traces, which "
            "is not its abundance in the original sample.",
            ha="left", va="top", fontsize=9.5, color=t["text_secondary"])


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def render(mode, resp, expected_counts, weights, outdir):
    t = THEMES[mode]
    rcparams()

    fig = plt.figure(figsize=(16, 8.2), facecolor=t["surface"])
    gs = fig.add_gridspec(
        2, 3, height_ratios=[0.92, 1.08],
        left=0.012, right=0.988, top=0.965, bottom=0.02, wspace=0.06, hspace=0.02,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, :])

    panel_a(ax_a, t)
    panel_b(ax_b, t)
    panel_c(ax_c, t)
    panel_d(ax_d, resp, expected_counts, weights, t)

    return save(fig, outdir, f"method-{mode}", t)


def render_measurement(mode, outdir):
    """Panels A-C only: the trace, the call matrix and the emission matrix.

    The split in panel D shows a fitted result, so a piece of writing that has
    not walked through the fit yet should use this version instead of the full
    figure. Same panels, same code, no orphaned numbers.
    """
    t = THEMES[mode]
    rcparams()

    fig = plt.figure(figsize=(16, 4.6), facecolor=t["surface"])
    gs = fig.add_gridspec(1, 3, left=0.012, right=0.988, top=0.94, bottom=0.04,
                          wspace=0.06)
    panel_a(fig.add_subplot(gs[0, 0]), t)
    panel_b(fig.add_subplot(gs[0, 1]), t)
    panel_c(fig.add_subplot(gs[0, 2]), t)

    return save(fig, outdir, f"measurement-{mode}", t)


def main():
    outdir = Path(__file__).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)

    fit, resp, expected_counts, weights = compute()

    print(f"converged in {fit.n_iter} iterations")
    print("responsibilities (molecule x origin):")
    for i, row in enumerate(resp):
        print(f"  molecule {i + 1}: " + "  ".join(f"{v:.4f}" for v in row)
              + f"   (row sum {row.sum():.6f})")
    print("expected counts: " + "  ".join(f"{v:.4f}" for v in expected_counts))
    print("mixture weights: " + "  ".join(f"{v:.4f}" for v in weights))

    for mode in ("light", "dark"):
        stem = render(mode, resp, expected_counts, weights, outdir)
        print(f"wrote {stem}.svg and {stem}@2x.png")
        stem = render_measurement(mode, outdir)
        print(f"wrote {stem}.svg and {stem}@2x.png")


if __name__ == "__main__":
    main()
