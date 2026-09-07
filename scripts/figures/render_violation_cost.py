#!/usr/bin/env python3
"""Render the model-violation cost figure for Results 3.2.

Mean total-variation abundance error under four departures from the fitting
assumptions, against the ignorable-missingness baseline. The point is not the
individual numbers but the split: some violations leave the error at baseline
(a systematic calibration offset cancels; emission-equivalent proteoforms are
aggregated), while others bias it and need an explicit model (informative
missingness, an origin absent from the reference set).

Unlike the other result figures this one visualises an already-computed
benchmark rather than regenerating it: the model-violation grid (MNAR gating,
Q perturbation, held-out origins) is produced by the benchmark harness, not a
single fit. It reads the fixed-Q EM rows of ``summary.tsv`` from the
model-violation archive; ``provenance.json`` in that archive records the exact
command that produced it. Group-level total variation is used throughout so the
emission-equivalent scenario is scored at the resolution it supports.

Usage:
    python scripts/figures/render_violation_cost.py
Requires the ``figures`` extra:  python -m pip install -e ".[figures]"
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_OUT = REPO_ROOT / "figures" / "figure_violation_cost"
DEFAULT_SUMMARY = REPO_ROOT / "outputs" / "model-violation-slim" / "summary.tsv"
METHOD = "em_fixed_q"
METRIC = "group_total_variation"

# scenario_id -> (plain label, robust?) ; robust None = the baseline itself
SCENARIOS = [
    ("I2_anti4R_contrast000", "Emission-equivalent proteoforms\n(reported at group resolution)", True),
    ("B1_mcar02", "Missing-at-random\n(ignorable baseline)", None),
    ("Q4_delta_m050_sigma025", "Miscalibrated $Q$\n(−0.5 logit + noise)", True),
    ("B5_mnar_call_dependent", "Informative missingness\n(positive calls dropped)", False),
    ("O2_heldout20", "True origin absent from reference\n(20% of mass held out)", False),
]


def load(summary: Path):
    rows = list(csv.DictReader(open(summary), delimiter="\t"))
    out = {}
    for sid, label, robust in SCENARIOS:
        m = [r for r in rows if r["scenario_id"] == sid and r["method_id"] == METHOD
             and r["metric"] == METRIC]
        if len(m) != 1:
            raise SystemExit(f"expected one {METRIC} row for {sid}/{METHOD}, found {len(m)}")
        out[sid] = (float(m[0]["mean"]), float(m[0]["sample_SD"]), int(m[0]["n_valid"]))
    return out


def render(summary: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator
    from _style import DARK, GREEN, GREY, ORANGE, apply_style

    apply_style()
    data = load(summary)
    base = data["B1_mcar02"][0]
    n_seeds = data["B1_mcar02"][2]

    ys = np.arange(len(SCENARIOS))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.axvline(base, ls=(0, (4, 2)), color=GREY, lw=1.0, zorder=1)

    for y, (sid, label, robust) in zip(ys, SCENARIOS):
        mean, sd, _ = data[sid]
        color = GREY if robust is None else (GREEN if robust else ORANGE)
        ax.errorbar(mean, y, xerr=sd, fmt="o", ms=7, color=color, ecolor=color,
                    elinewidth=1.4, capsize=3, zorder=4)
        fold = mean / base
        tag = "" if robust is None else ("  ≈ baseline" if robust else f"  {fold:.1f}× baseline")
        ax.annotate(f"{mean:.3f}{tag}", (mean, y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=6.6, color=DARK)

    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab, _ in SCENARIOS], fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(0.015, 0.32)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks([0.02, 0.03, 0.05, 0.1, 0.2, 0.3])
    ax.set_xticklabels(["0.02", "0.03", "0.05", "0.1", "0.2", "0.3"])
    ax.set_xlabel(f"abundance error  (total variation, mean ± SD across {n_seeds} seeds)")
    ax.set_ylim(-0.7, ys.max() + 0.9)
    ax.grid(axis="x", color="0.93", lw=0.6)
    ax.set_axisbelow(True)

    legend = [
        Line2D([], [], marker="o", ls="none", color=GREEN, ms=7, label="robust: error ≈ baseline"),
        Line2D([], [], marker="o", ls="none", color=ORANGE, ms=7, label="biased: needs an explicit model"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=6.6, handletextpad=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    for sid, *_ in SCENARIOS:
        m, sd, n = data[sid]
        print(f"  {sid:26s} {m:.4f} ± {sd:.4f}  (n={n})  fold={m/base:.2f}")
    print(f"matplotlib {matplotlib.__version__}, numpy {np.__version__}")
    print(f"wrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY,
                    help="model-violation summary.tsv (fixed-Q EM rows)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT,
                    help="output path stem (writes .png and .pdf)")
    args = ap.parse_args(argv)
    render(args.summary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
