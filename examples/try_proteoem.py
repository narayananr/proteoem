#!/usr/bin/env python3
"""
try_proteoem.py — a runnable walk-through of TUTORIAL.md ("Using ProteoEM on your
own data"). Run it and read the output alongside the comments here.

    python examples/try_proteoem.py           # from a repo checkout, with proteoem installed

Every number below is a tiny illustrative value chosen so the arithmetic is easy
to follow. A real fit is one fit_em() call; the work is preparing its inputs.
"""
import numpy as np
from proteoem import (
    fit_em,                          # the estimator
    build_emission_matrix,          # build Q from a feature matrix + rates
    fit_likelihood_em,              # fit from a precomputed likelihood matrix
    effective_observation_yield,    # e_k = r_k * v_k
    analyzed_to_source_composition, # accepted composition -> source composition
    positive_gate_visibility,       # v_k under an any-positive keep-rule
)


def section(n, title):
    print("\n" + "=" * 70 + f"\nSTEP {n}. {title}\n" + "=" * 70)


# ----------------------------------------------------------------------------
section(1, "A minimal run  (tutorial section 2)")
# The three inputs ProteoEM needs:
#   observations Y : one row per molecule, one column per cycle.
#                    1 = positive call, 0 = negative call, -1 = NA (unavailable).
#   cycle_to_probe : which logical probe each cycle used (repeats share an index).
#   Q              : calibrated positive-call rate of each probe on each origin.
observations = np.array([[1, 0, -1, 1, 0, 0],
                         [0, 1,  0, 0, 1, 1],
                         [1, 0,  1, 1, 0, 1]], dtype=np.int8)
cycle_to_probe = np.array([0, 1, 2, 0, 1, 2])       # 3 probes, each applied twice
Q = np.array([[0.90, 0.10, 0.80],                   # origin 0: probe0 yes, probe1 no
              [0.15, 0.85, 0.80]])                  # origin 1: the reverse

fit = fit_em(observations, Q=Q, cycle_to_probe=cycle_to_probe,
             return_responsibilities=True)
print("inputs :  Y", observations.shape, " cycle_to_probe", cycle_to_probe.shape, " Q", Q.shape)
print("weights:", np.round(fit.weights, 4), " (composition of the accepted molecules)")
print("converged:", fit.converged, " in", fit.n_iter, "iterations   <- always check this")


# ----------------------------------------------------------------------------
section(2, "Build Q from features instead of typing it  (tutorial section 3)")
# If you have a panel design (which origin carries which epitope) rather than a
# measured Q, give the binary feature matrix E and per-probe on/off rates.
E = np.array([[1, 0, 1],           # origin 0 carries features of probes 0 and 2
              [0, 1, 1]], dtype=np.int8)
Q_from_E = build_emission_matrix(E, alpha=0.9, beta=0.1)   # q = E*alpha + (1-E)*beta
print("Q built from E:\n", Q_from_E)
fitE = fit_em(observations, E, alpha=0.9, beta=0.1, cycle_to_probe=cycle_to_probe)
print("weights (fit straight from E):", np.round(fitE.weights, 4))


# ----------------------------------------------------------------------------
section(3, "Read the result object  (tutorial section 4)")
print("weights          :", np.round(fit.weights, 4))
print("converged        :", fit.converged)
# responsibilities: posterior over origins, ONE ROW PER TRACE CLASS (not per molecule).
# ProteoEM groups identical/proportional traces, so T <= N rows.
print("responsibilities :", fit.responsibilities.shape, "= (trace classes T, origins K)")
print(np.round(fit.responsibilities, 3))
# observable_groups: origins the panel cannot tell apart (see Step 4). Here all distinct.
print("observable_groups:", fit.observable_groups)
# a few health diagnostics
for k in ("monotonic", "terminal_em_residual", "expected_count_total_error"):
    print(f"diagnostics[{k}] =", fit.diagnostics.get(k))


# ----------------------------------------------------------------------------
section(4, "When two origins are indistinguishable  (tutorial section 4 / 10)")
# Give origins 0 and 1 IDENTICAL emission rows: no probe can separate them.
Q_equiv = np.array([[0.90, 0.10, 0.80],
                    [0.90, 0.10, 0.80],   # identical to row 0
                    [0.15, 0.85, 0.80]])
fit_eq = fit_em(observations, Q=Q_equiv, cycle_to_probe=cycle_to_probe)
print("per-candidate weights:", np.round(fit_eq.weights, 4), " (the 0-vs-1 split is arbitrary)")
print("observable_groups    :", fit_eq.observable_groups)
for g in fit_eq.observable_groups:
    print(f"  group {g}: identifiable total = {fit_eq.weights[list(g)].sum():.4f}")
print(">> report the GROUP TOTAL, never the split inside a group.")


# ----------------------------------------------------------------------------
section(5, "Bring your own likelihood  (tutorial section 6)")
# If you compute evidence yourself (intensities, correlated cycles, censoring),
# skip Q and pass an N x K log-likelihood matrix directly.
log_L = np.log(np.clip(np.array([[0.7, 0.02, 0.02],
                                 [0.02, 0.7, 0.02],
                                 [0.5, 0.3, 0.01]]), 1e-9, None))
fitL = fit_likelihood_em(log_L, return_responsibilities=True)
print("weights from a supplied log-likelihood matrix:", np.round(fitL.weights, 4))


# ----------------------------------------------------------------------------
section(6, "Accepted composition vs source composition  (tutorial section 7)")
# fit.weights is the composition among molecules that PASSED your keep-rule.
# To get back to the tube, divide by the effective yield e_k = r_k * v_k.
visibility = positive_gate_visibility(Q, cycle_to_probe=cycle_to_probe)
print("visibility v_k under an any-positive gate:", np.round(visibility, 3))
# Illustrative: one origin far less visible than the other skews what you observe.
recovery   = np.array([1.0, 1.0])
vis        = np.array([0.93, 0.49])
e          = effective_observation_yield(recovery, vis)
accepted   = np.array([0.66, 0.34])                       # what you measure
source     = analyzed_to_source_composition(accepted, e)  # theta ~ pi / e
print("effective yield e_k     :", np.round(e, 3))
print("accepted (measured)     :", accepted)
print("source (yield-corrected):", np.round(source, 3), " <- recovers the ~50/50 tube")


# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Done. A real fit is just Step 1 with your own (observations, cycle_to_probe, Q).")
print("For a full synthetic dataset end-to-end, try the CLI:")
print("    proteoem benchmark-tau --output /tmp/try --molecules 5000 --seed 7")
print("Full guide: TUTORIAL.md")
print("=" * 70)
