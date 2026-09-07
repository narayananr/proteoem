#!/usr/bin/env python3
"""Benchmark accepted-trace and source-composition inference under selection.

The benchmark deliberately separates two targets:

* ``pi``: the origin composition among accepted traces; and
* ``theta``: the composition in the source population before recovery/gating.

Three scenarios use fractional fixed-emission EM.  A fourth applies a
uniform-prior posterior threshold followed by hard decoding, as a deliberately
biased comparator.  Every run records both population truth and the realized
hidden-origin fractions so that Monte Carlo variation is not mistaken for
estimation error.
"""

from __future__ import annotations

import argparse
import csv
import itertools
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
    analyzed_to_source_composition,
    candidate_log_likelihoods,
    condition_on_deterministic_gate,
    fit_likelihood_em,
    simulate_selected_traces,
    source_to_analyzed_composition,
)


THETA = np.array([0.5, 0.5], dtype=float)
PROTEIN_LENGTHS = np.array([300.0, 600.0], dtype=float)
CONFIDENCE_THRESHOLD = 0.90
CORRECTIONS = (
    "oracle_yield",
    "equal_yield",
    "raw_probe_count",
    "raw_protein_length",
    "misspecified_yield",
)


def _scenarios() -> dict[str, dict[str, Any]]:
    opportunity_q = np.array(
        [
            [0.2] * 12,
            [0.2] * 3 + [0.0] * 9,
        ],
        dtype=float,
    )
    tau_q = np.array(
        [
            [0.65] * 6 + [0.75] * 3 + [0.05] * 3,
            [0.65, 0.0, 0.0, 0.65, 0.0, 0.0]
            + [0.05] * 3
            + [0.75] * 3,
        ],
        dtype=float,
    )
    confidence_q = np.array(
        [
            [0.50] * 4 + [0.02] * 8,
            [0.02] * 4 + [0.25] * 2 + [0.02] * 6,
        ],
        dtype=float,
    )
    return {
        "all_traces": {
            "label": "All registered traces retained",
            "estimator": "fractional_em",
            "Q": opportunity_q,
            "recovery": np.ones(2),
            "retention_rule": "all",
            "groups": None,
            "probe_opportunities": np.array([12.0, 3.0]),
            "description": (
                "One registered trace per recovered molecule; all-negative and "
                "ambiguous traces are retained. Twelve versus three binding "
                "opportunities alter the trace likelihood, not observation yield."
            ),
        },
        "any_positive": {
            "label": "Any-positive trace gate",
            "estimator": "fractional_em",
            "Q": opportunity_q,
            "recovery": np.ones(2),
            "retention_rule": "any_positive",
            "groups": None,
            "probe_opportunities": np.array([12.0, 3.0]),
            "description": (
                "The same 12-versus-3 panel, but a trace is retained only when "
                "at least one positive call is recorded."
            ),
        },
        "tau_anchor": {
            "label": "Tau-like N- and C-anchor gate",
            "estimator": "fractional_em",
            "Q": tau_q,
            "recovery": np.array([0.85, 0.95]),
            "retention_rule": "required_positive_groups",
            "groups": ((0, 1, 2), (3, 4, 5)),
            "probe_opportunities": np.array([9.0, 5.0]),
            "description": (
                "Retention requires at least one N-anchor and one C-anchor "
                "positive. Origin A has three opportunities per anchor group; "
                "origin B has one, and physical recovery also differs."
            ),
        },
        "confidence_hard_decode": {
            "label": "Confidence-filtered hard decode",
            "estimator": "hard_decode",
            "Q": confidence_q,
            "recovery": np.ones(2),
            "retention_rule": "all",
            "downstream_filter": "uniform_prior_posterior_confidence",
            "groups": None,
            "probe_opportunities": np.array([4.0, 2.0]),
            "description": (
                "A trace is retained only when its largest uniform-prior "
                "posterior assignment probability is at least 0.90, then is "
                "assigned wholly to the top origin. Origin B has deliberately "
                "weaker evidence and is less likely to pass."
            ),
        },
    }


def _normalize(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array / array.sum()


def _total_variation(estimate: Any, truth: Any) -> float:
    return float(0.5 * np.abs(np.asarray(estimate) - np.asarray(truth)).sum())


def _posterior_uniform(log_likelihoods: np.ndarray) -> np.ndarray:
    centered = log_likelihoods - np.max(log_likelihoods, axis=1, keepdims=True)
    probabilities = np.exp(centered)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def _confidence_gate_truth(Q: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    """Enumerate the trace space to obtain exact hard-gate probabilities."""

    traces = np.asarray(
        tuple(itertools.product((0, 1), repeat=Q.shape[1])), dtype=np.int8
    )
    logs = candidate_log_likelihoods(traces, Q)
    posterior = _posterior_uniform(logs)
    keep = posterior.max(axis=1) >= threshold
    decoded = posterior.argmax(axis=1)
    trace_probability = np.exp(logs)
    visibility = np.sum(trace_probability * keep[:, None], axis=0)
    confusion = np.zeros((Q.shape[0], Q.shape[0]), dtype=float)
    for origin in range(Q.shape[0]):
        for label in range(Q.shape[0]):
            confusion[origin, label] = np.sum(
                trace_probability[:, origin] * (keep & (decoded == label))
            )
    return {
        "visibility": visibility,
        "confusion": confusion,
        "n_trace_patterns": np.array([traces.shape[0]], dtype=int),
    }


def _assumed_yields(
    true_yield: np.ndarray, probe_opportunities: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the five predeclared correction vectors.

    Only ratios matter when mapping ``pi`` back to ``theta``. Raw metadata are
    therefore scaled to a maximum of one. The misspecified vector shrinks 25%
    of the true between-origin yield contrast toward their common mean.
    """

    mean_yield = np.full(true_yield.shape, float(np.mean(true_yield)))
    return {
        "oracle_yield": np.asarray(true_yield, dtype=float),
        "equal_yield": np.ones(true_yield.shape, dtype=float),
        "raw_probe_count": probe_opportunities / probe_opportunities.max(),
        "raw_protein_length": PROTEIN_LENGTHS / PROTEIN_LENGTHS.max(),
        "misspecified_yield": 0.75 * true_yield + 0.25 * mean_yield,
    }


def _empirical_yield(source_ids: np.ndarray, accepted_ids: np.ndarray) -> np.ndarray:
    source_counts = np.bincount(source_ids, minlength=THETA.size).astype(float)
    accepted_counts = np.bincount(accepted_ids, minlength=THETA.size).astype(float)
    return accepted_counts / source_counts


def _base_run_fields(
    *,
    scenario_name: str,
    scenario: dict[str, Any],
    seed: int,
    n_source: int,
    n_accepted: int,
    source_realized: np.ndarray,
    accepted_realized: np.ndarray,
    pi_population: np.ndarray,
    pi_hat: np.ndarray,
    effective_yield: np.ndarray,
    empirical_yield: np.ndarray,
    visibility: np.ndarray,
    converged: bool,
    iterations: int,
    terminal_em_residual: float | None,
    misclassification_rate: float | None,
) -> dict[str, Any]:
    return {
        "run_id": f"{scenario_name}_n{n_source}_seed{seed}",
        "scenario": scenario_name,
        "scenario_label": scenario["label"],
        "estimator": scenario["estimator"],
        "seed": seed,
        "n_source_molecules": n_source,
        "n_accepted_traces": n_accepted,
        "accepted_fraction_realized": n_accepted / n_source,
        "accepted_fraction_population": float(np.dot(THETA, effective_yield)),
        "converged": converged,
        "iterations": iterations,
        "terminal_em_residual": terminal_em_residual,
        "source_A_population": THETA[0],
        "source_B_population": THETA[1],
        "source_A_realized": source_realized[0],
        "source_B_realized": source_realized[1],
        "accepted_A_population": pi_population[0],
        "accepted_B_population": pi_population[1],
        "accepted_A_realized": accepted_realized[0],
        "accepted_B_realized": accepted_realized[1],
        "accepted_A_estimated": pi_hat[0],
        "accepted_B_estimated": pi_hat[1],
        "accepted_tv_population": _total_variation(pi_hat, pi_population),
        "accepted_tv_realized": _total_variation(pi_hat, accepted_realized),
        "recovery_A": scenario["recovery"][0],
        "recovery_B": scenario["recovery"][1],
        "visibility_A": visibility[0],
        "visibility_B": visibility[1],
        "effective_yield_A": effective_yield[0],
        "effective_yield_B": effective_yield[1],
        "empirical_yield_A": empirical_yield[0],
        "empirical_yield_B": empirical_yield[1],
        "misclassification_rate_accepted": misclassification_rate,
    }


def _fractional_em_run(
    scenario_name: str,
    scenario: dict[str, Any],
    *,
    n_source: int,
    seed: int,
    max_iter: int,
    tol: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    simulation = simulate_selected_traces(
        None,
        n_source,
        source_composition=THETA,
        recovery_probability=scenario["recovery"],
        Q=scenario["Q"],
        retention_rule=scenario["retention_rule"],
        required_positive_groups=scenario["groups"],
        seed=seed,
    )
    aggregated = simulation.aggregated
    pre_gate_logs = candidate_log_likelihoods(aggregated, simulation.cycle_Q)
    logs = pre_gate_logs
    if scenario["retention_rule"] != "all":
        logs = condition_on_deterministic_gate(logs, simulation.panel_visibility)
    fit = fit_likelihood_em(
        logs,
        counts=aggregated.counts,
        max_iter=max_iter,
        tol=tol,
        return_responsibilities=False,
    )
    if scenario["retention_rule"] == "all":
        naive_fit = fit
    else:
        naive_fit = fit_likelihood_em(
            pre_gate_logs,
            counts=aggregated.counts,
            max_iter=max_iter,
            tol=tol,
            return_responsibilities=False,
        )
    base = _base_run_fields(
        scenario_name=scenario_name,
        scenario=scenario,
        seed=seed,
        n_source=n_source,
        n_accepted=simulation.n_accepted,
        source_realized=simulation.empirical_source_composition,
        accepted_realized=simulation.empirical_analyzed_composition,
        pi_population=simulation.analyzed_composition,
        pi_hat=fit.weights,
        effective_yield=simulation.effective_yield,
        empirical_yield=simulation.empirical_effective_yield,
        visibility=simulation.panel_visibility,
        converged=fit.converged,
        iterations=fit.n_iter,
        terminal_em_residual=float(fit.diagnostics["terminal_em_residual"]),
        misclassification_rate=None,
    )
    base.update(
        {
            "accepted_A_naive_estimated": naive_fit.weights[0],
            "accepted_B_naive_estimated": naive_fit.weights[1],
            "accepted_tv_naive_population": _total_variation(
                naive_fit.weights, simulation.analyzed_composition
            ),
            "accepted_tv_naive_realized": _total_variation(
                naive_fit.weights, simulation.empirical_analyzed_composition
            ),
            "naive_converged": naive_fit.converged,
            "naive_iterations": naive_fit.n_iter,
            "naive_terminal_em_residual": float(
                naive_fit.diagnostics["terminal_em_residual"]
            ),
        }
    )
    arrays = {
        "pi_hat": fit.weights,
        "effective_yield": simulation.effective_yield,
        "pi_population": simulation.analyzed_composition,
    }
    return base, arrays


def _confidence_run(
    scenario_name: str,
    scenario: dict[str, Any],
    *,
    n_source: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    simulation = simulate_selected_traces(
        None,
        n_source,
        source_composition=THETA,
        recovery_probability=scenario["recovery"],
        Q=scenario["Q"],
        retention_rule="all",
        seed=seed,
    )
    logs = candidate_log_likelihoods(simulation.observations, scenario["Q"])
    posterior = _posterior_uniform(logs)
    gate_mask = posterior.max(axis=1) >= CONFIDENCE_THRESHOLD
    decoded = posterior[gate_mask].argmax(axis=1)
    accepted_ids = simulation.identities[gate_mask]
    n_accepted = int(gate_mask.sum())
    if n_accepted == 0:
        raise RuntimeError("confidence gate accepted no traces")

    exact = _confidence_gate_truth(scenario["Q"], CONFIDENCE_THRESHOLD)
    effective_yield = scenario["recovery"] * exact["visibility"]
    pi_population = source_to_analyzed_composition(THETA, effective_yield)
    pi_hat = _normalize(np.bincount(decoded, minlength=THETA.size))
    accepted_realized = _normalize(
        np.bincount(accepted_ids, minlength=THETA.size)
    )
    misclassification = float(np.mean(decoded != accepted_ids))
    empirical_yield = _empirical_yield(
        simulation.source_identities, accepted_ids
    )
    base = _base_run_fields(
        scenario_name=scenario_name,
        scenario=scenario,
        seed=seed,
        n_source=n_source,
        n_accepted=n_accepted,
        source_realized=simulation.empirical_source_composition,
        accepted_realized=accepted_realized,
        pi_population=pi_population,
        pi_hat=pi_hat,
        effective_yield=effective_yield,
        empirical_yield=empirical_yield,
        visibility=exact["visibility"],
        converged=True,
        iterations=0,
        terminal_em_residual=None,
        misclassification_rate=misclassification,
    )
    base.update(
        {
            "decoded_A_count": int(np.sum(decoded == 0)),
            "decoded_B_count": int(np.sum(decoded == 1)),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "accepted_A_naive_estimated": None,
            "accepted_B_naive_estimated": None,
            "accepted_tv_naive_population": None,
            "accepted_tv_naive_realized": None,
            "naive_converged": None,
            "naive_iterations": None,
            "naive_terminal_em_residual": None,
        }
    )
    arrays = {
        "pi_hat": pi_hat,
        "effective_yield": effective_yield,
        "pi_population": pi_population,
    }
    exact_fields = {
        "confidence_trace_patterns_enumerated": int(exact["n_trace_patterns"][0]),
        "confidence_confusion_probability": exact["confusion"].tolist(),
    }
    return base, arrays, exact_fields


def _correction_rows(
    base: dict[str, Any], arrays: dict[str, np.ndarray], scenario: dict[str, Any]
) -> list[dict[str, Any]]:
    assumed = _assumed_yields(
        arrays["effective_yield"], scenario["probe_opportunities"]
    )
    rows: list[dict[str, Any]] = []
    for correction in CORRECTIONS:
        yield_vector = assumed[correction]
        theta_hat = analyzed_to_source_composition(arrays["pi_hat"], yield_vector)
        row = dict(base)
        row.update(
            {
                "correction": correction,
                "assumed_yield_A": yield_vector[0],
                "assumed_yield_B": yield_vector[1],
                "source_A_estimated": theta_hat[0],
                "source_B_estimated": theta_hat[1],
                "source_tv_population": _total_variation(theta_hat, THETA),
            }
        )
        rows.append(row)
    return rows


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    group_keys = sorted({(row["scenario"], row["correction"]) for row in rows})
    for scenario, correction in group_keys:
        selected = [
            row
            for row in rows
            if row["scenario"] == scenario and row["correction"] == correction
        ]
        summary: dict[str, Any] = {
            "scenario": scenario,
            "scenario_label": selected[0]["scenario_label"],
            "estimator": selected[0]["estimator"],
            "correction": correction,
            "n_runs": len(selected),
            "all_converged": all(bool(row["converged"]) for row in selected),
        }
        for key in (
            "n_accepted_traces",
            "accepted_fraction_realized",
            "accepted_fraction_population",
            "accepted_A_population",
            "accepted_A_realized",
            "accepted_A_estimated",
            "accepted_tv_population",
            "accepted_tv_realized",
            "source_A_estimated",
            "source_tv_population",
            "effective_yield_A",
            "effective_yield_B",
            "empirical_yield_A",
            "empirical_yield_B",
        ):
            values = [float(row[key]) for row in selected]
            summary[f"{key}_mean"] = _mean(values)
            summary[f"{key}_sd"] = _sd(values)
        for key in (
            "accepted_A_naive_estimated",
            "accepted_tv_naive_population",
            "accepted_tv_naive_realized",
            "terminal_em_residual",
            "naive_terminal_em_residual",
        ):
            values = [
                float(row[key])
                for row in selected
                if row.get(key) not in (None, "")
            ]
            summary[f"{key}_mean"] = _mean(values) if values else None
            summary[f"{key}_sd"] = _sd(values) if values else None
        classification = [
            float(row["misclassification_rate_accepted"])
            for row in selected
            if row["misclassification_rate_accepted"] not in (None, "")
        ]
        summary["misclassification_rate_accepted_mean"] = (
            _mean(classification) if classification else None
        )
        summaries.append(summary)
    return summaries


def _figure_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in summaries:
        by_scenario.setdefault(str(row["scenario"]), []).append(row)
    for scenario, selected in by_scenario.items():
        accepted = next(row for row in selected if row["correction"] == "oracle_yield")
        rows.append(
            {
                "panel": "accepted_composition",
                "scenario": scenario,
                "scenario_label": accepted["scenario_label"],
                "estimator": accepted["estimator"],
                "correction": "not_applicable",
                "truth_A": accepted["accepted_A_population_mean"],
                "estimate_A_mean": accepted["accepted_A_estimated_mean"],
                "estimate_A_sd": accepted["accepted_A_estimated_sd"],
                "total_variation_mean": accepted["accepted_tv_population_mean"],
                "total_variation_sd": accepted["accepted_tv_population_sd"],
            }
        )
        for summary in selected:
            # The hard-decode scenario is shown only as an accepted-stage
            # comparator. Its archived inverse-yield calculations should not
            # be interpreted as repairing classification/filtering error.
            if summary["estimator"] == "hard_decode":
                continue
            rows.append(
                {
                    "panel": "source_composition",
                    "scenario": scenario,
                    "scenario_label": summary["scenario_label"],
                    "estimator": summary["estimator"],
                    "correction": summary["correction"],
                    "truth_A": THETA[0],
                    "estimate_A_mean": summary["source_A_estimated_mean"],
                    "estimate_A_sd": summary["source_A_estimated_sd"],
                    "total_variation_mean": summary["source_tv_population_mean"],
                    "total_variation_sd": summary["source_tv_population_sd"],
                }
            )
    return rows


def _truth_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeated runs/corrections to one exact-truth row per scenario."""

    result: list[dict[str, Any]] = []
    for scenario in sorted({str(row["scenario"]) for row in rows}):
        row = next(item for item in rows if item["scenario"] == scenario)
        result.append(
            {
                "scenario": scenario,
                "scenario_label": row["scenario_label"],
                "estimator": row["estimator"],
                "source_A_population": row["source_A_population"],
                "source_B_population": row["source_B_population"],
                "recovery_A": row["recovery_A"],
                "recovery_B": row["recovery_B"],
                "visibility_A": row["visibility_A"],
                "visibility_B": row["visibility_B"],
                "effective_yield_A": row["effective_yield_A"],
                "effective_yield_B": row["effective_yield_B"],
                "accepted_fraction_population": row[
                    "accepted_fraction_population"
                ],
                "accepted_A_population": row["accepted_A_population"],
                "accepted_B_population": row["accepted_B_population"],
            }
        )
    return result


def _jsonable_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in scenario.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, tuple):
            result[key] = [list(item) if isinstance(item, tuple) else item for item in value]
        else:
            result[key] = value
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "observation-yield-benchmark",
    )
    parser.add_argument("--molecules", type=int, default=50_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-9)
    return parser


def _run_benchmark(args, provenance: dict[str, Any]) -> None:
    if args.molecules < 1:
        raise ValueError("--molecules must be positive")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed")

    scenarios = _scenarios()
    rows: list[dict[str, Any]] = []
    exact_metadata: dict[str, Any] = {}
    for scenario_name, scenario in scenarios.items():
        for seed in args.seeds:
            if scenario["estimator"] == "fractional_em":
                base, arrays = _fractional_em_run(
                    scenario_name,
                    scenario,
                    n_source=args.molecules,
                    seed=seed,
                    max_iter=args.max_iter,
                    tol=args.tol,
                )
            else:
                base, arrays, metadata = _confidence_run(
                    scenario_name,
                    scenario,
                    n_source=args.molecules,
                    seed=seed,
                )
                exact_metadata[scenario_name] = metadata
            rows.extend(_correction_rows(base, arrays, scenario))

    summaries = _summaries(rows)
    _write_tsv(args.output / "runs.tsv", rows)
    _write_tsv(args.output / "summary.tsv", summaries)
    _write_tsv(args.output / "scenario_truth.tsv", _truth_rows(rows))
    _write_tsv(args.output / "figure_data.tsv", _figure_rows(summaries))

    configuration = {
        "benchmark_version": 1,
        "source_composition": THETA.tolist(),
        "protein_lengths_aa": PROTEIN_LENGTHS.tolist(),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "corrections": {
            "oracle_yield": "true r_k times true gate visibility v_k",
            "equal_yield": "no source-level correction",
            "raw_probe_count": "accessible probe-opportunity count, scaled by its maximum",
            "raw_protein_length": "protein length in amino acids, scaled by its maximum",
            "misspecified_yield": "75% true yield plus 25% across-origin mean yield",
        },
        "scenarios": {
            name: _jsonable_scenario(scenario)
            for name, scenario in scenarios.items()
        },
        "exact_metadata": exact_metadata,
        "run": {
            "n_source_molecules": args.molecules,
            "seeds": args.seeds,
            "max_iter": args.max_iter,
            "tol": args.tol,
        },
    }
    (args.output / "configuration.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
