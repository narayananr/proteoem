#!/usr/bin/env python3
"""Render the per-molecule evidence-accumulation figure.

One simulated molecule, shown two ways:

* Left  - with the full 12-probe panel, its posterior concentrates on the true
  proteoform as calls accumulate over the 36 cycles (three probe-repeat rounds).
* Right - if no probe distinguished 3R from 4R, the true origin and its 3R/4R
  twin become emission-equivalent: the pair (observable group) is identified but
  the split between its members is not.

The three cycles where the 3R/4R probe (``anti_4R``) fires are marked on the
x-axis: active in the left panel, removed in the right. The three rounds re-apply
the same 12 probes, so the staircase reflects noise-averaging by repetition, not
new features (final assignment is invariant to probe order).

The posterior uses a flat prior over the 768 candidates, so it is the
normalised per-molecule trace likelihood (single-molecule evidence, not the
mixture responsibility). All data are regenerated deterministically from the
tau-like benchmark at ``--seed`` (default 7); nothing is read from disk.

Usage:
    python scripts/figures/render_evidence_accumulation.py
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

from proteoem.benchmark import (  # noqa: E402
    default_tau_probe_rates,
    make_tau_like_panel,
    sparse_tau_weights,
)
from proteoem.simulate import simulate_traces  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "figures" / "figure_evidence_accumulation"
DISABLE_PROBE = "anti_4R"          # the sole 3R/4R discriminator
N_MOLECULES = 3000
N_ACTIVE = 32
MISSING_RATE = 0.02


# --------------------------------------------------------------------------- #
# data (regenerated deterministically from the seed)
# --------------------------------------------------------------------------- #
def build_data(seed: int):
    panel = make_tau_like_panel(repeats=3)
    alpha, beta = default_tau_probe_rates(panel)
    truth = sparse_tau_weights(panel, n_active=N_ACTIVE, seed=seed)
    sim = simulate_traces(
        panel.profiles, N_MOLECULES, weights=truth,
        alpha=alpha, beta=beta, missing_rate=MISSING_RATE, seed=seed + 1,
    )
    Y = np.asarray(sim.observations)
    Q = np.asarray(sim.Q)                       # (K, C) positive-call probability per cycle
    true = np.asarray(sim.identities)
    c2p = np.asarray(panel.cycle_to_probe)      # (C,) logical probe per cycle
    Lp = np.asarray(panel.logical_profiles)     # (K, J) feature carried by candidate
    probe_ids = list(panel.logical_probe_ids)
    return panel, Y, Q, true, c2p, Lp, probe_ids


def posteriors(Y_i, Q):
    """Cumulative flat-prior posterior over candidates, per cycle. Returns (K, C)."""
    logp = np.log(np.clip(Q, 1e-12, 1 - 1e-12))
    log1mp = np.log(np.clip(1 - Q, 1e-12, 1 - 1e-12))
    K, C = Q.shape
    con = np.zeros((K, C))
    con[:, Y_i == 1] = logp[:, Y_i == 1]
    con[:, Y_i == 0] = log1mp[:, Y_i == 0]
    cum = np.cumsum(con, axis=1)
    cum -= cum.max(axis=0, keepdims=True)
    p = np.exp(cum)
    p /= p.sum(axis=0, keepdims=True)
    return p


def final_posteriors(Y, Q):
    """Vectorised final-cycle flat-prior posterior for every molecule. Returns (N, K)."""
    logp = np.log(np.clip(Q, 1e-12, 1 - 1e-12))
    log1mp = np.log(np.clip(1 - Q, 1e-12, 1 - 1e-12))
    pos = (Y == 1).astype(float)
    neg = (Y == 0).astype(float)
    logL = pos @ logp.T + neg @ log1mp.T        # (N, K)
    logL -= logL.max(axis=1, keepdims=True)
    p = np.exp(logL)
    p /= p.sum(axis=1, keepdims=True)
    return p


def select_example(Y, Q, true, Lp, disable_idx, n_logical):
    """Deterministically pick a molecule that (a) resolves cleanly under the
    full panel and (b) forms a concentrated, evenly-split pair once the 3R/4R
    probe is disabled. Returns (molecule_index, true_origin, twin_origin, cols)."""
    K, J = Lp.shape
    keep = [j for j in range(J) if j != disable_idx]
    groups: dict[bytes, list[int]] = {}
    for k in range(K):
        groups.setdefault(Lp[k, keep].tobytes(), []).append(k)
    twin_of = {k: [t for t in groups[Lp[k, keep].tobytes()] if t != k] for k in range(K)}

    cols = np.where(np.tile(np.arange(n_logical), Q.shape[1] // n_logical) == disable_idx)[0]
    Qd = Q.copy()
    Qd[:, cols] = 0.5

    pf_full = final_posteriors(Y, Q)
    pf_dis = final_posteriors(Y, Qd)
    best = None
    for i in range(len(true)):
        tk = int(true[i])
        tw = twin_of[tk]
        if len(tw) != 1:
            continue
        tw = tw[0]
        a = pf_full[i, tk]
        pair = pf_dis[i, tk] + pf_dis[i, tw]
        if pair <= 0:
            continue
        split = pf_dis[i, tk] / pair
        if a >= 0.90 and pair >= 0.90 and abs(split - 0.5) <= 0.05:
            score = a * pair * (1 - 2 * abs(split - 0.5))
            if best is None or score > best[0]:
                best = (score, i, tk, tw)
    if best is None:
        raise SystemExit("no molecule met the selection criteria for this seed")
    _, i, tk, tw = best
    return i, tk, tw, cols


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render(seed: int, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator
    from _style import DARK, GREEN, GREY, ORANGE, apply_style

    apply_style()
    panel, Y, Q, true, c2p, Lp, probe_ids = build_data(seed)
    disable_idx = probe_ids.index(DISABLE_PROBE)
    n_logical = len(probe_ids)
    mol, TRUE, TWIN, dis_cols = select_example(Y, Q, true, Lp, disable_idx, n_logical)
    K, C = Q.shape
    cycles = np.arange(1, C + 1)
    anti4r = np.where(c2p == disable_idx)[0] + 1        # 1-based cycles of the 3R/4R probe

    def id_short(k):                                    # "2N4R" from "2N4R|pT181+..."
        return panel.candidate_ids[k].split("|")[0]

    Qd = Q.copy()
    Qd[:, dis_cols] = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)

    def draw(ax, Qm, disabled, title):
        p = posteriors(Y[mol], Qm)
        field = [k for k in range(K) if p[k].max() > 0.01 and k not in (TRUE, TWIN)]
        for b in (12, 24):
            ax.axvline(b, color="0.88", lw=0.8, zorder=0)
        for k in field:
            ax.plot(cycles, p[k], color=GREY, lw=0.6, alpha=0.30, zorder=1)
        if disabled:
            ax.plot(cycles, p[TRUE] + p[TWIN], color=DARK, lw=1.3, ls=(0, (4, 2)),
                    zorder=2, label=f"group total  ({id_short(TRUE)} + {id_short(TWIN)})")
        ax.plot(cycles, p[TWIN], color=ORANGE, lw=2.0, zorder=3,
                label=f"{id_short(TWIN)}  (3R/4R twin)")
        ax.plot(cycles, p[TRUE], color=GREEN, lw=2.6, zorder=4,
                label=f"{id_short(TRUE)}  (true origin)")
        # mark the 3R/4R probe cycles just under the axis
        mcol = GREY if disabled else ORANGE
        for cyc in anti4r:
            ax.plot([cyc], [-0.055], marker="^", ms=5.5, color=mcol,
                    mfc=("none" if disabled else mcol), mew=1.1, clip_on=False, zorder=5)
        ax.set_title(title, pad=13, loc="left")
        ax.set_xlabel("cycle")
        ax.set_xlim(0.5, C + 0.5)
        ax.set_ylim(-0.09, 1.03)
        ax.xaxis.set_major_locator(MultipleLocator(12))
        ax.xaxis.set_minor_locator(MultipleLocator(6))
        ax.grid(axis="y", color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        for r, xc in zip(("round 1", "round 2", "round 3"), (6, 18, 30)):
            ax.text(xc, 1.045, r, ha="center", va="bottom", fontsize=6.6, color="0.5")
        return p

    pA = draw(axes[0], Q, False, "Full panel resolves the proteoform")
    pB = draw(axes[1], Qd, True, "Drop the 3R / 4R probe: the pair is unresolvable")
    axes[0].annotate(f"{pA[TRUE, -1]:.2f}", (C, pA[TRUE, -1]), xytext=(-2, 4),
                     textcoords="offset points", ha="right", color=GREEN,
                     fontsize=8, fontweight="bold")
    half = pB[TRUE, -1]
    axes[1].annotate(f"{half:.2f} each", (C - 2.5, half), xytext=(0, 14),
                     textcoords="offset points", ha="right", color=DARK, fontsize=7.6,
                     arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7))
    axes[1].annotate(f"{pB[TRUE, -1] + pB[TWIN, -1]:.2f}", (C, pB[TRUE, -1] + pB[TWIN, -1]),
                     xytext=(-2, 4), textcoords="offset points", ha="right",
                     color=DARK, fontsize=8, fontweight="bold")
    axes[0].set_ylabel("posterior probability of origin\ngiven calls through cycle $c$")
    proxy_a = Line2D([], [], marker="^", color=ORANGE, mfc=ORANGE, ls="none", ms=6,
                     label="3R/4R probe (anti_4R)")
    ha, la = axes[0].get_legend_handles_labels()
    axes[0].legend(ha + [proxy_a], la + [proxy_a.get_label()],
                   loc="upper left", bbox_to_anchor=(0.03, 0.99))
    proxy_b = Line2D([], [], marker="^", color=GREY, mfc="none", mew=1.1, ls="none", ms=6,
                     label="3R/4R probe (removed)")
    hb, lb = axes[1].get_legend_handles_labels()
    axes[1].legend(hb + [proxy_b], lb + [proxy_b.get_label()],
                   loc="center left", bbox_to_anchor=(0.05, 0.66))

    fig.tight_layout()

    for ax, lett in ((axes[0], "A"), (axes[1], "B")):
        ax.text(0.02, 0.97, lett, transform=ax.transAxes, ha="left", va="top",
                fontsize=11, fontweight="bold", color=DARK, zorder=10,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.75))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"molecule {mol}: true={TRUE} ({panel.candidate_ids[TRUE]}), "
          f"twin={TWIN} ({panel.candidate_ids[TWIN]})")
    print(f"full-panel P(true)={pA[TRUE, -1]:.3f} | disabled: "
          f"P(true)={pB[TRUE, -1]:.3f} P(twin)={pB[TWIN, -1]:.3f} "
          f"group={pB[TRUE, -1] + pB[TWIN, -1]:.3f}")
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
