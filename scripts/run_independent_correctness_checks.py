#!/usr/bin/env python3
"""Run the independent optimizer, binary-limit, and threshold checks.

The benchmark is intentionally small.  Its purpose is implementation
validation and a controlled demonstration of likelihood binarization, not an
estimate of iterative-affinity assay performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from typing import Any

import numpy as np

from archive_provenance import (
    benchmark_source_paths,
    capture_run_provenance,
    run_archive_main,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from proteoem import (  # noqa: E402
    build_emission_matrix,
    candidate_log_likelihoods,
    fit_likelihood_em,
    simulate_traces,
)
from proteoem.validation_oracles import (  # noqa: E402
    deterministic_profile_compatibility,
    fit_binary_compatibility_em,
    fixed_mixture_log_likelihood,
    likelihood_ratio_compatibility,
    maximize_fixed_mixture_on_simplex,
)


SCHEMA_VERSION = 1
EM_MAX_ITER = 20_000
EM_TOL = 1e-11
ORACLE_MAX_ITER = 50_000
ORACLE_TOL = 2e-8
BINARY_MAX_ITER = 100_000
BINARY_TOL = 1e-11

SMALL_PROFILES = np.array(
    [[0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0]],
    dtype=np.int8,
)
SMALL_TRUTH = np.array([0.22, 0.47, 0.31])

MODERATE_PROFILES = np.array(
    [
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
    ],
    dtype=np.int8,
)
MODERATE_TRUTH = np.array([0.07, 0.11, 0.16, 0.09, 0.13, 0.18, 0.14, 0.12])

THRESHOLD_PROFILES = np.array(
    [
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
    ],
    dtype=np.int8,
)
THRESHOLD_TRUTH = np.array([0.36, 0.24, 0.16, 0.11, 0.08, 0.05])
THRESHOLD_ALPHA = np.array([0.72, 0.76, 0.80, 0.68, 0.74, 0.82, 0.70, 0.78])
THRESHOLD_BETA = np.array([0.12, 0.18, 0.08, 0.22, 0.15, 0.10, 0.20, 0.14])
THRESHOLDS = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0]
THRESHOLD_SEEDS = [7, 17, 27, 37, 47]
THRESHOLD_MOLECULES = 10_000


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _total_variation(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(estimate - truth)))


def _component_distribution_rank(profiles: np.ndarray) -> int:
    """Rank of the complete-outcome component distributions for a case."""

    q = build_emission_matrix(profiles, alpha=0.84, beta=0.07)
    outcomes = (
        np.arange(2 ** profiles.shape[1])[:, None]
        >> np.arange(profiles.shape[1])[None, :]
    ) & 1
    probabilities = np.prod(
        np.where(outcomes[:, None, :] == 1, q[None, :, :], 1.0 - q[None, :, :]),
        axis=2,
    )
    return int(np.linalg.matrix_rank(probabilities))


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _sd(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _optimizer_case(
    name: str,
    profiles: np.ndarray,
    truth: np.ndarray,
    n_molecules: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    simulation = simulate_traces(
        profiles,
        n_molecules,
        weights=truth,
        alpha=0.84,
        beta=0.07,
        seed=seed,
    )
    logs = candidate_log_likelihoods(simulation.observations, simulation.Q)
    em = fit_likelihood_em(
        logs,
        max_iter=EM_MAX_ITER,
        tol=EM_TOL,
        return_responsibilities=False,
    )
    oracle = maximize_fixed_mixture_on_simplex(
        logs,
        max_iter=ORACLE_MAX_ITER,
        tol=ORACLE_TOL,
    )
    independently_evaluated_em_objective = fixed_mixture_log_likelihood(
        logs, em.weights
    )
    objective_difference = float(
        abs(oracle.log_likelihood - independently_evaluated_em_objective)
    )
    max_weight_difference = float(np.max(np.abs(oracle.weights - em.weights)))
    component_rank = _component_distribution_rank(profiles)
    passed = bool(
        em.converged
        and oracle.converged
        and component_rank == profiles.shape[0]
        and max_weight_difference <= 2e-7
        and objective_difference <= 2e-7
    )
    row = {
        "case": name,
        "n_molecules": n_molecules,
        "n_candidates": profiles.shape[0],
        "n_probes": profiles.shape[1],
        "seed": seed,
        "complete_outcome_component_rank": component_rank,
        "full_component_rank": component_rank == profiles.shape[0],
        "em_converged": em.converged,
        "em_iterations": em.n_iter,
        "oracle_converged": oracle.converged,
        "oracle_iterations": oracle.n_iter,
        "oracle_projected_gradient_residual": oracle.projected_gradient_residual,
        "oracle_line_search_failures": oracle.line_search_failures,
        "em_reported_log_likelihood": em.log_likelihood,
        "em_independently_evaluated_log_likelihood": independently_evaluated_em_objective,
        "oracle_log_likelihood": oracle.log_likelihood,
        "absolute_objective_difference": objective_difference,
        "max_absolute_weight_difference": max_weight_difference,
        "check_passed": passed,
    }
    weights = [
        {
            "case": name,
            "candidate": index,
            "generating_weight": truth[index],
            "em_weight": em.weights[index],
            "oracle_weight": oracle.weights[index],
            "absolute_difference": abs(em.weights[index] - oracle.weights[index]),
        }
        for index in range(profiles.shape[0])
    ]
    return row, weights


def _deterministic_binary_check() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profiles = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
    observations = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1], [0, -1], [1, -1]],
        dtype=np.int8,
    )
    counts = np.array([35, 15, 20, 10, 10, 10], dtype=np.int64)
    closed_form = np.array([0.42, 0.18, 0.8 / 3.0, 0.4 / 3.0])
    compatibility = deterministic_profile_compatibility(observations, profiles)
    deterministic_logs = candidate_log_likelihoods(observations, profiles.astype(float))
    support_equal = bool(
        np.array_equal(np.isfinite(deterministic_logs), compatibility)
        and np.all(deterministic_logs[compatibility] == 0)
    )
    reference = fit_binary_compatibility_em(
        compatibility,
        counts=counts,
        max_iter=BINARY_MAX_ITER,
        tol=1e-13,
    )
    em = fit_likelihood_em(
        deterministic_logs,
        counts=counts,
        max_iter=EM_MAX_ITER,
        tol=1e-12,
        return_responsibilities=False,
    )
    em_reference_difference = float(np.max(np.abs(em.weights - reference.weights)))
    reference_closed_form_difference = float(
        np.max(np.abs(reference.weights - closed_form))
    )
    objective_difference = float(abs(em.log_likelihood - reference.log_likelihood))
    passed = bool(
        support_equal
        and em.converged
        and reference.converged
        and em_reference_difference <= 2e-10
        and reference_closed_form_difference <= 2e-10
        and objective_difference <= 2e-9
    )
    row = {
        "n_trace_classes": observations.shape[0],
        "n_observations": int(counts.sum()),
        "n_candidates": profiles.shape[0],
        "support_matrices_exactly_equal": support_equal,
        "em_converged": em.converged,
        "em_iterations": em.n_iter,
        "independent_binary_converged": reference.converged,
        "independent_binary_iterations": reference.n_iter,
        "independent_binary_fixed_point_residual": reference.fixed_point_residual,
        "max_weight_difference_em_vs_independent_binary": em_reference_difference,
        "max_weight_difference_independent_binary_vs_closed_form": reference_closed_form_difference,
        "absolute_objective_difference": objective_difference,
        "check_passed": passed,
    }
    weights = [
        {
            "candidate": index,
            "closed_form_weight": closed_form[index],
            "em_weight": em.weights[index],
            "independent_binary_weight": reference.weights[index],
        }
        for index in range(profiles.shape[0])
    ]
    return row, weights


def _threshold_sweep() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    q = build_emission_matrix(
        THRESHOLD_PROFILES,
        alpha=THRESHOLD_ALPHA,
        beta=THRESHOLD_BETA,
    )
    rows: list[dict[str, Any]] = []
    for seed in THRESHOLD_SEEDS:
        simulation = simulate_traces(
            THRESHOLD_PROFILES,
            THRESHOLD_MOLECULES,
            weights=THRESHOLD_TRUTH,
            Q=q,
            seed=seed,
        )
        logs = candidate_log_likelihoods(simulation.observations, q)
        graded = fit_likelihood_em(
            logs,
            max_iter=EM_MAX_ITER,
            tol=1e-10,
            return_responsibilities=False,
        )
        if not graded.converged:
            raise RuntimeError(f"graded fit did not converge for seed {seed}")
        graded_objective = fixed_mixture_log_likelihood(logs, graded.weights)
        graded_tv = _total_variation(graded.weights, THRESHOLD_TRUTH)
        for threshold in THRESHOLDS:
            compatibility = likelihood_ratio_compatibility(logs, threshold)
            binary = fit_binary_compatibility_em(
                compatibility,
                max_iter=BINARY_MAX_ITER,
                tol=BINARY_TOL,
            )
            if not binary.converged:
                raise RuntimeError(
                    f"binary fit did not converge for seed {seed}, threshold {threshold}"
                )
            binary_tv = _total_variation(binary.weights, THRESHOLD_TRUTH)
            graded_objective_at_binary = fixed_mixture_log_likelihood(
                logs, binary.weights
            )
            support_sizes = compatibility.sum(axis=1)
            rows.append(
                {
                    "seed": seed,
                    "n_molecules": THRESHOLD_MOLECULES,
                    "n_candidates": THRESHOLD_PROFILES.shape[0],
                    "n_probes": THRESHOLD_PROFILES.shape[1],
                    "minimum_relative_likelihood": threshold,
                    "graded_converged": graded.converged,
                    "graded_iterations": graded.n_iter,
                    "binary_converged": binary.converged,
                    "binary_iterations": binary.n_iter,
                    "graded_total_variation": graded_tv,
                    "binary_total_variation": binary_tv,
                    "binary_minus_graded_total_variation": binary_tv - graded_tv,
                    "graded_optimum_log_likelihood": graded_objective,
                    "graded_log_likelihood_at_binary_weights": graded_objective_at_binary,
                    "graded_log_likelihood_loss_per_molecule": (
                        graded_objective - graded_objective_at_binary
                    )
                    / THRESHOLD_MOLECULES,
                    "mean_compatible_candidates": float(np.mean(support_sizes)),
                    "unique_compatibility_fraction": float(np.mean(support_sizes == 1)),
                    "compatibility_density": float(np.mean(compatibility)),
                    "true_origin_retained_fraction": float(
                        np.mean(
                            compatibility[
                                np.arange(THRESHOLD_MOLECULES),
                                simulation.identities,
                            ]
                        )
                    ),
                }
            )

    summaries: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        selected = [
            row
            for row in rows
            if row["minimum_relative_likelihood"] == threshold
        ]
        summary: dict[str, Any] = {
            "minimum_relative_likelihood": threshold,
            "n_seeds": len(selected),
        }
        for field in (
            "graded_total_variation",
            "binary_total_variation",
            "binary_minus_graded_total_variation",
            "graded_log_likelihood_loss_per_molecule",
            "mean_compatible_candidates",
            "unique_compatibility_fraction",
            "compatibility_density",
            "true_origin_retained_fraction",
        ):
            values = [float(row[field]) for row in selected]
            summary[f"{field}_mean"] = _mean(values)
            summary[f"{field}_sd"] = _sd(values)
        summaries.append(summary)
    return rows, summaries


def _configuration() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": (
            "Independent numerical correctness checks and a constructed "
            "imperfect-probe likelihood-binarization demonstration; not an assay "
            "performance benchmark."
        ),
        "optimizer": {
            "algorithm": "projected_gradient_ascent_with_armijo_backtracking",
            "uses_proteoem_em_internals": False,
            "max_iter": ORACLE_MAX_ITER,
            "tolerance": ORACLE_TOL,
            "comparison_em_max_iter": EM_MAX_ITER,
            "comparison_em_tolerance": EM_TOL,
            "cases": [
                {
                    "name": "small",
                    "n_molecules": 800,
                    "n_candidates": 3,
                    "n_probes": 6,
                    "seed": 107,
                    "alpha": 0.84,
                    "beta": 0.07,
                    "truth": SMALL_TRUTH.tolist(),
                    "profiles": SMALL_PROFILES.tolist(),
                },
                {
                    "name": "moderate",
                    "n_molecules": 6000,
                    "n_candidates": 8,
                    "n_probes": 8,
                    "seed": 211,
                    "alpha": 0.84,
                    "beta": 0.07,
                    "truth": MODERATE_TRUTH.tolist(),
                    "profiles": MODERATE_PROFILES.tolist(),
                },
            ],
        },
        "deterministic_binary": {
            "description": (
                "Four deterministic two-bit references, four unique classes, and "
                "two wildcard multimapping classes with an analytic MLE."
            ),
            "counts": [35, 15, 20, 10, 10, 10],
            "closed_form_weights": [0.42, 0.18, 0.8 / 3.0, 0.4 / 3.0],
            "independent_em_max_iter": BINARY_MAX_ITER,
            "independent_em_tolerance": 1e-13,
        },
        "threshold_sweep": {
            "compatibility_rule": (
                "retain k when L_ik / max_h(L_ih) >= threshold; at threshold 0, "
                "retain all candidates with positive likelihood"
            ),
            "thresholds": THRESHOLDS,
            "seeds": THRESHOLD_SEEDS,
            "n_molecules": THRESHOLD_MOLECULES,
            "truth": THRESHOLD_TRUTH.tolist(),
            "profiles": THRESHOLD_PROFILES.tolist(),
            "alpha_by_probe": THRESHOLD_ALPHA.tolist(),
            "beta_by_probe": THRESHOLD_BETA.tolist(),
            "binary_em_max_iter": BINARY_MAX_ITER,
            "binary_em_tolerance": BINARY_TOL,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "independent-correctness-checks",
    )
    return parser


def _run_benchmark(args, provenance: dict[str, Any]) -> None:
    optimizer_rows: list[dict[str, Any]] = []
    optimizer_weight_rows: list[dict[str, Any]] = []
    for case in (
        ("small", SMALL_PROFILES, SMALL_TRUTH, 800, 107),
        ("moderate", MODERATE_PROFILES, MODERATE_TRUTH, 6_000, 211),
    ):
        row, weights = _optimizer_case(*case)
        optimizer_rows.append(row)
        optimizer_weight_rows.extend(weights)
    if not all(bool(row["check_passed"]) for row in optimizer_rows):
        raise RuntimeError("at least one direct-optimizer comparison failed")

    binary_row, binary_weight_rows = _deterministic_binary_check()
    if not bool(binary_row["check_passed"]):
        raise RuntimeError("deterministic binary compatibility check failed")

    threshold_rows, threshold_summaries = _threshold_sweep()
    _write_tsv(args.output / "optimizer_checks.tsv", optimizer_rows)
    _write_tsv(args.output / "optimizer_weights.tsv", optimizer_weight_rows)
    _write_tsv(args.output / "deterministic_binary_check.tsv", [binary_row])
    _write_tsv(args.output / "deterministic_binary_weights.tsv", binary_weight_rows)
    _write_tsv(args.output / "threshold_sweep_runs.tsv", threshold_rows)
    _write_tsv(args.output / "threshold_sweep_summary.tsv", threshold_summaries)

    configuration = _configuration()
    (args.output / "configuration.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    environment = {
        **provenance,
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
    }
    (args.output / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    best_binary = min(
        threshold_summaries,
        key=lambda row: float(row["binary_total_variation_mean"]),
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "optimizer_all_checks_passed": True,
        "optimizer_max_absolute_weight_difference": max(
            float(row["max_absolute_weight_difference"]) for row in optimizer_rows
        ),
        "optimizer_max_absolute_objective_difference": max(
            float(row["absolute_objective_difference"]) for row in optimizer_rows
        ),
        "deterministic_binary_check_passed": True,
        "deterministic_binary_support_exact": bool(
            binary_row["support_matrices_exactly_equal"]
        ),
        "deterministic_binary_max_weight_difference": float(
            binary_row["max_weight_difference_em_vs_independent_binary"]
        ),
        "deterministic_binary_max_closed_form_difference": float(
            binary_row["max_weight_difference_independent_binary_vs_closed_form"]
        ),
        "threshold_sweep_n_seeds": len(THRESHOLD_SEEDS),
        "threshold_sweep_n_molecules_per_seed": THRESHOLD_MOLECULES,
        "graded_total_variation_mean": float(
            threshold_summaries[0]["graded_total_variation_mean"]
        ),
        "graded_total_variation_sd": float(
            threshold_summaries[0]["graded_total_variation_sd"]
        ),
        "best_binary_threshold_by_mean_total_variation": float(
            best_binary["minimum_relative_likelihood"]
        ),
        "best_binary_total_variation_mean": float(
            best_binary["binary_total_variation_mean"]
        ),
        "best_binary_total_variation_sd": float(
            best_binary["binary_total_variation_sd"]
        ),
        "best_binary_graded_objective_loss_per_molecule_mean": float(
            best_binary["graded_log_likelihood_loss_per_molecule_mean"]
        ),
        "claim_boundary": (
            "Results describe these fixed synthetic cases only. The sweep is not "
            "evidence that any threshold or calibrated model is universally best."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = build_parser().parse_args()
    runner_path = Path(__file__).resolve()
    provenance = capture_run_provenance(
        repo_root=REPO_ROOT,
        command=sys.argv,
        source_paths=benchmark_source_paths(REPO_ROOT, runner_path),
    )
    return run_archive_main(
        output_dir=args.output,
        provenance=provenance,
        function=lambda: _run_benchmark(args, provenance),
    )


if __name__ == "__main__":
    raise SystemExit(main())
