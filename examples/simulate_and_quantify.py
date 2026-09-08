#!/usr/bin/env python3
"""
simulate_and_quantify.py -- the step-by-step tutorial as one runnable script.

    python examples/simulate_and_quantify.py           # prints all 7 steps
    python examples/simulate_and_quantify.py --plots    # also writes the two figures
                                                        # (needs the [figures] extra)

This is the un-packaged version of examples/try_from_file.py: instead of hiding
the simulation inside a helper, it builds the whole pipeline from primitives, one
stage at a time, so you can see exactly what the simulator knows and what the
estimator is allowed to use. It is the script behind SIMULATION_TUTORIAL.md.

The seven stages:
  1. panel        -- the 768 candidate proteoforms and the 12-antibody schedule
  2. rates        -- the calibrated per-antibody positive-call rates (alpha, beta)
  3. truth        -- the hidden ground-truth composition (sparse, heavy-tailed)
  4. simulate     -- read N molecules into 1/0/-1 traces (the only data)
  5. freeze Q     -- the emission matrix the estimator may use
  6. quantify     -- one fit_em call
  7. score        -- estimate vs the hidden truth (impossible on real data)

The clean line: the SIMULATOR owns `truth` and `identities` (stages 3-4); the
ESTIMATOR sees only `Q` and `observations` (stages 5-6). On real data, stages
1-2 and 5-6 are identical, stages 3-4 become your assay export, and stage 7 does
not exist because there is no truth to score against.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import proteoem as P

# Fixed seeds make every number here reproducible. Seed 7 draws the composition;
# its top state sits at fraction 0.1546, the anchor used across the benchmarks.
MIXTURE_SEED = 7
TRACE_SEED = 8
N_MOLECULES = 3000
N_ACTIVE = 32
CONCENTRATION = 0.4
MISSING_RATE = 0.02

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "docs" / "figures"

# Palette shared with docs/figures/figstyle.py so the tutorial figures read as
# part of the same set.
INK = "#0b0b0b"
MUTED = "#52514e"
RULE = "#c8c7c2"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"


def build_everything():
    """Run stages 1-6 and return every object the tutorial refers to."""
    # --- 1. panel --------------------------------------------------------
    panel = P.make_tau_like_panel(repeats=3)
    cand = list(panel.candidate_ids)

    # --- 2. calibrated rates --------------------------------------------
    logical_alpha, logical_beta = P.default_tau_logical_probe_rates(panel)  # per probe (12)
    cycle_alpha, cycle_beta = P.default_tau_probe_rates(panel)              # per cycle (36)

    # --- 3. hidden ground truth -----------------------------------------
    truth = P.sparse_tau_weights(
        panel, n_active=N_ACTIVE, concentration=CONCENTRATION, seed=MIXTURE_SEED
    )

    # --- 4. simulate molecule traces ------------------------------------
    sim = P.simulate_traces(
        panel.profiles,
        N_MOLECULES,
        weights=truth,
        alpha=cycle_alpha,
        beta=cycle_beta,
        missing_rate=MISSING_RATE,
        seed=TRACE_SEED,
    )
    obs = np.asarray(sim.observations)
    ident = np.asarray(sim.identities)  # true origin per molecule (simulation-only)

    # --- 5. freeze the emission matrix ----------------------------------
    Q = P.build_emission_matrix(
        panel.logical_profiles, alpha=logical_alpha, beta=logical_beta
    )

    # --- 6. quantify -----------------------------------------------------
    fit = P.fit_em(obs, Q=Q, cycle_to_probe=panel.cycle_to_probe)

    return dict(
        panel=panel, cand=cand, truth=truth, sim=sim, obs=obs, ident=ident,
        Q=Q, fit=fit, est=fit.weights,
        logical_alpha=logical_alpha, logical_beta=logical_beta,
    )


def report(ctx) -> None:
    """Print the seven-stage walkthrough."""
    panel, cand = ctx["panel"], ctx["cand"]
    truth, obs, ident = ctx["truth"], ctx["obs"], ctx["ident"]
    fit, est = ctx["fit"], ctx["est"]

    def rule(title):
        print(f"\n{'-' * 70}\n{title}\n{'-' * 70}")

    rule("1. panel")
    print(f"  {len(cand)} candidates = {len(set(panel.isoforms))} isoforms x 2^7 phospho-combinations")
    print(f"  {panel.n_logical_probes} antibodies over {np.asarray(panel.profiles).shape[1]} cycles "
          f"(12 probes x {panel.repeats} repeats)")

    rule("2. calibrated antibody rates")
    la, lb = ctx["logical_alpha"], ctx["logical_beta"]
    print(f"  alpha (on-target +): {la.min():.2f}-{la.max():.2f}   beta (off-target +): "
          f"{lb.min():.2f}-{lb.max():.2f}")

    rule("3. hidden ground-truth composition")
    order = np.argsort(truth)[::-1]
    print(f"  {int(np.count_nonzero(truth))} of {truth.size} candidates present, sums to {truth.sum():.4f}")
    print(f"  top state: {cand[order[0]]}  ({truth[order[0]]:.4f} = {int(round(truth[order[0]] * 1e6))} cpm)")

    rule("4. simulated molecule traces")
    vals, counts = np.unique(obs, return_counts=True)
    mix = {int(v): c / obs.size for v, c in zip(vals, counts)}
    print(f"  observations: {obs.shape[0]} molecules x {obs.shape[1]} cycles")
    print(f"  call mix: positive={mix.get(1, 0):.3f}  negative={mix.get(0, 0):.3f}  NA={mix.get(-1, 0):.3f}")
    sampled = np.unique(ident)
    print(f"  present states that drew >=1 molecule: {len(set(np.where(truth > 0)[0]) & set(sampled))} of "
          f"{int(np.count_nonzero(truth))}")

    rule("5. frozen emission matrix Q")
    print(f"  Q: {ctx['Q'].shape} = candidates x logical probes (features x calibrated rates; no truth, no obs)")

    rule("6. quantify")
    print(f"  converged={fit.converged} in {fit.n_iter} iters; "
          f"{obs.shape[0]} molecules -> {fit.responsibilities.shape[0]} trace classes")

    rule("7. score against the hidden truth")
    tv = 0.5 * float(np.sum(np.abs(est - truth)))
    print(f"  total-variation error = {tv:.4f}")
    print(f"  {'candidate':44s} {'est cpm':>9s} {'true cpm':>9s}")
    for k in np.argsort(est)[::-1][:8]:
        print(f"  {cand[k]:44s} {int(round(est[k] * 1e6)):>9d} {int(round(truth[k] * 1e6)):>9d}")
    present = set(np.where(truth > 0)[0])
    missed = present - set(sampled)
    absent_max = int(round(float(est[truth == 0].max()) * 1e6)) if np.any(truth == 0) else 0
    print(f"  present-but-unsampled: {len(missed)} -> est {[int(round(est[k] * 1e6)) for k in missed]} cpm "
          f"(correctly ~0)")
    print(f"  largest estimate on any absent candidate: {absent_max} cpm (no mass invented)")


def make_figures(ctx) -> list[Path]:
    """Write the two tutorial figures; returns the paths written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.size": 11,
        "axes.edgecolor": RULE, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    })

    truth, est, cand = ctx["truth"], ctx["est"], ctx["cand"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    # --- Figure 1: the composition (what is in the sample) --------------
    present = np.where(truth > 0)[0]
    present = present[np.argsort(truth[present])[::-1]]
    cpm = truth[present] * 1e6
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(np.arange(1, len(present) + 1), np.maximum(cpm, 1.0), color=BLUE, width=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("present proteoform, ranked by abundance")
    ax.set_ylabel("true abundance (cpm, log scale)")
    ax.set_title("The simulated sample: 32 of 768 proteoforms present",
                 color=INK, fontweight="bold")
    ax.text(0.97, 0.93, "heavy-tailed: the top state holds 15% of the sample;\n"
                        "the 32 present abundances span more than five orders\n"
                        "of magnitude down to the rarest few states",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p1 = FIG_DIR / "tutorial-composition.png"
    fig.savefig(p1, dpi=150, metadata={"Software": ""})
    plt.close(fig)
    written.append(p1)

    # --- Figure 2: recovery (estimated vs true) -- the hero -------------
    tv = 0.5 * float(np.sum(np.abs(est - truth)))
    absent_max = float(est[truth == 0].max()) * 1e6 if np.any(truth == 0) else 0.0
    tflo, eflo = truth * 1e6, np.maximum(est * 1e6, 1.0)  # floor est at 1 cpm for log axis
    pres = truth > 0
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    lim = [1, max(tflo.max(), eflo.max()) * 1.6]
    ax.plot(lim, lim, "--", color=RULE, lw=1.2, zorder=1, label="perfect recovery (y = x)")
    ax.scatter(tflo[pres], eflo[pres], s=42, color=BLUE, edgecolor="white",
               linewidth=0.5, zorder=3, label=f"{int(pres.sum())} present proteoforms")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("true abundance (cpm)")
    ax.set_ylabel("estimated abundance (cpm)")
    ax.set_title("Recovery: estimated vs true abundance", color=INK, fontweight="bold")
    ax.text(0.04, 0.96, f"total-variation error = {tv:.3f}\n"
                        f"largest estimate on an absent\ncandidate: {absent_max:.0f} cpm",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color=MUTED)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p2 = FIG_DIR / "tutorial-recovery.png"
    fig.savefig(p2, dpi=150, metadata={"Software": ""})
    plt.close(fig)
    written.append(p2)

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plots", action="store_true",
                    help="also write the two tutorial figures under docs/figures/")
    args = ap.parse_args()

    ctx = build_everything()
    report(ctx)

    if args.plots:
        try:
            paths = make_figures(ctx)
        except ImportError:
            print("\n[--plots needs matplotlib: pip install -e \".[figures]\"]")
            return
        print("\nwrote figures:")
        for p in paths:
            print(f"  {p.relative_to(REPO_ROOT)}")
    else:
        print("\n(pass --plots to also write the two figures under docs/figures/)")


if __name__ == "__main__":
    main()
