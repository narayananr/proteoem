#!/usr/bin/env python3
"""Render the abundance-recovery (parity) figure for Results 3.1.

One correctly specified fit (N = 20,000 accepted traces, seed 7) compares
weighted EM against two baselines that discard graded likelihood information:

* top-likelihood counting - assign each molecule to its single best origin;
* binary-profile EM - threshold the calibrated likelihoods to 0/1 compatibility.

Left panel: estimated vs true abundance for the proteoforms that are truly
present. Right panel: the total abundance each fit places on the proteoforms
that are truly absent. Weighted EM tracks the truth and keeps absent mass near
zero; the reductions both misestimate present proteoforms and spread mass onto
absent ones.

All data are regenerated deterministically from the tau-like benchmark
(``run_tau_like_benchmark``); nothing is read from disk. The three total-variation
errors reproduce the Table 2 / Table 3 values (0.013, 0.181, 0.222).

Usage:
    python scripts/figures/render_abundance_parity.py
Requires the ``figures`` extra:  python -m pip install -e ".[figures]"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from proteoem import run_tau_like_benchmark  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "figures" / "figure_abundance_parity"
N_MOLECULES = 20000
N_ACTIVE = 32
MISSING_RATE = 0.02


def build_data(seed: int):
    r = run_tau_like_benchmark(
        n_molecules=N_MOLECULES, n_active=N_ACTIVE,
        missing_rate=MISSING_RATE, seed=seed, block_size=4096,
    )
    truth = np.asarray(r.simulation.weights)
    est = {
        "weighted": np.asarray(r.weighted_fit.weights),
        "hard": np.asarray(r.hard_weights),
        "binary": np.asarray(r.incidence_fit.weights),
    }
    return truth, est


def render(seed: int, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from _style import DARK, GREEN, GREY, ORANGE, PURPLE, apply_style

    apply_style()
    truth, est = build_data(seed)
    pres = truth > 0
    absent = ~pres
    n_pres, n_abs = int(pres.sum()), int(absent.sum())

    # name, legend label, colour, marker, size, filled?
    methods = [
        ("weighted", "weighted EM", GREEN, "o", 40, True),
        ("hard", "top-likelihood counting", ORANGE, "^", 30, False),
        ("binary", "binary-profile EM", PURPLE, "s", 24, False),
    ]

    def median_pct_err(e):
        return float(np.median(np.abs(e[pres] - truth[pres]) / truth[pres]) * 100)

    def false_mass(e):
        return float(e[absent].sum())

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(7.4, 3.5), gridspec_kw=dict(width_ratios=[1.4, 1.0]),
    )

    # ---- Panel A: present-candidate parity (log-log) ------------------------
    # Clamp to the meaningful abundance range. One present candidate has true
    # abundance ~7e-8 (a Dirichlet fluke) and clips harmlessly to the corner.
    lo, hi = 1e-3, 4e-1
    axA.plot((lo, hi), (lo, hi), ls=(0, (4, 2)), color=GREY, lw=1.0, zorder=1)
    for key, label, color, mk, ms, filled in methods:
        e = est[key]
        x = np.clip(truth[pres], lo, hi)
        y = np.clip(e[pres], lo, hi)
        lbl = f"{label}  ({median_pct_err(e):.0f}% median err)"
        if filled:
            axA.scatter(x, y, s=ms, marker=mk, facecolors=color, edgecolors="white",
                        linewidths=0.4, alpha=0.95, zorder=5, label=lbl)
        else:
            axA.scatter(x, y, s=ms, marker=mk, facecolors="none", edgecolors=color,
                        linewidths=1.1, alpha=0.85, zorder=3, label=lbl)
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlim(lo, hi)
    axA.set_ylim(lo, hi)
    axA.set_aspect("equal")
    axA.set_xlabel("true abundance")
    axA.set_ylabel("estimated abundance")
    axA.set_title(f"Present proteoforms  (n = {n_pres})", loc="left", pad=8)
    axA.legend(loc="upper left", fontsize=6.5, handletextpad=0.25, borderpad=0.3)
    axA.annotate(f"N = {N_MOLECULES:,} traces", (0.97, 0.03), xycoords="axes fraction",
                 ha="right", va="bottom", fontsize=6.6, color="0.5")

    # ---- Panel B: false mass on absent candidates ---------------------------
    xs = np.arange(len(methods))
    vals = [false_mass(est[k]) for k, *_ in methods]
    colors = [c for _, _, c, _, _, _ in methods]
    axB.bar(xs, vals, color=colors, width=0.6, zorder=2)
    for x, v in zip(xs, vals):
        axB.annotate(f"{v:.3f}", (x, v), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=8, fontweight="bold", color=DARK)
    axB.set_xticks(xs)
    axB.set_xticklabels(["weighted\nEM", "top-likelihood", "binary\nprofile"], fontsize=7)
    axB.set_ylabel(f"abundance placed on the\n{n_abs} absent proteoforms")
    axB.set_title(f"Absent proteoforms  (n = {n_abs})", loc="left", pad=8)
    axB.set_ylim(0, max(vals) * 1.20)
    axB.grid(axis="y", color="0.92", lw=0.6)
    axB.set_axisbelow(True)

    fig.tight_layout()

    for ax, lett in ((axA, "A"), (axB, "B")):
        ax.text(0.02, 0.97, lett, transform=ax.transAxes, ha="left", va="top",
                fontsize=11, fontweight="bold", color=DARK, zorder=10,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.75))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    for key, *_ in methods:
        e = est[key]
        print(f"{key}: median%err(present)={median_pct_err(e):.1f}  "
              f"false_mass(absent)={false_mass(e):.4f}  TV={0.5*np.sum(np.abs(e-truth)):.4f}")
    print(f"matplotlib {matplotlib.__version__}, numpy {np.__version__}, seed {seed}")
    print(f"wrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT,
                    help="output path stem (writes .png and .pdf)")
    args = ap.parse_args(argv)
    render(args.seed, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
