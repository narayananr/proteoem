#!/usr/bin/env python3
"""Execute the frozen ProteoEM systematic model-violation benchmark.

The runner consumes the frozen manifest directly.  It never invents a smaller
replacement for a declared debug or primary run.  ``--smoke`` is an explicitly
nonconformant, non-headline pipeline check whose overrides are recorded in
provenance; ``--preflight`` may schedule an exact subset of a tier without
changing any scientific setting.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import resource
import signal
import statistics
import subprocess
import sys
import threading
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from proteoem import (  # noqa: E402
    TraceLikelihoodClasses,
    alignment_profile_log_likelihoods,
    build_trace_likelihood_classes,
    default_tau_logical_probe_rates,
    find_observable_groups,
    fit_likelihood_em,
    hard_assignment_counts,
    logsumexp,
    make_tau_like_panel,
    probability_assignment_metrics,
)
from proteoem.validation_oracles import (  # noqa: E402
    maximize_fixed_mixture_on_simplex,
)


DEFAULT_MANIFEST = REPO_ROOT / "configs" / "systematic_benchmark_manifest_v4.json"
RUNNER_VERSION = 3
V4_RUNNER_VERSION = 4
V1_SCHEMA_VERSION = "1.0.0"
V2_SCHEMA_VERSION = "2.0.0"
V3_SCHEMA_VERSION = "3.0.0"
V4_SCHEMA_VERSION = "4.0.0"
V1_MANIFEST_ID = "proteoem-tau768-systematic-v1"
V2_MANIFEST_ID = "proteoem-tau768-systematic-v2"
V3_MANIFEST_ID = "proteoem-tau768-systematic-v3"
V4_MANIFEST_ID = "proteoem-tau768-systematic-v4"
MODERN_SCHEMA_VERSIONS = frozenset(
    {V2_SCHEMA_VERSION, V3_SCHEMA_VERSION, V4_SCHEMA_VERSION}
)
DIRECT_SVD_SCHEMA_VERSIONS = frozenset({V3_SCHEMA_VERSION, V4_SCHEMA_VERSION})
V4_METRIC_SCHEMA_VERSIONS = frozenset({V4_SCHEMA_VERSION})
EXACT_UNRESOLVED_SCENARIOS = frozenset(
    {"B4_anti4R_drop", "I2_anti4R_contrast000"}
)
IDENTIFIABILITY_SCENARIOS = frozenset(
    {
        "B4_anti4R_drop",
        "I1_anti4R_contrast025",
        "I2_anti4R_contrast000",
    }
)
CALIBRATED_PROBABILITY_METHODS = frozenset(
    {"em_fixed_q", "soft_uniform_one_step", "oracle_repeat_aware_likelihood_em"}
)
TOP_LABEL_METHODS = frozenset(
    {"top_likelihood_equal_ties", "NA_as_zero_top_likelihood_antipattern"}
)
FROZEN_MANIFEST_SHA256 = {
    V1_MANIFEST_ID: "e86ba9a6bb75f223fb012670fe8887e12fdd838ccd65b0eb40c616bc3c7cef49",
    V2_MANIFEST_ID: "629d88d9c49037ebc395f1bec3cbceb5b54555fab1d1b2ba734467824557b696",
    V3_MANIFEST_ID: "36b02cf856d43ffbe19d5a34ae55edbcb0b5b031f36ae1666400dae5aaad9a18",
    V4_MANIFEST_ID: "c55a0ac2d0e8c08edae84c8d9b4053375fa093872912e50012ae395eaf098e60",
}
FROZEN_MANIFEST_CANONICAL_SHA256 = {
    V1_MANIFEST_ID: "ca35d1cfe4e8584e775337c4977ddcb4779059aef892e2a4505d08838b3f09fe",
    V2_MANIFEST_ID: "aa523f4b2f5e32d20c515a8ae834f87e58721b35bfb9484c8ac47e36693c9895",
    V3_MANIFEST_ID: "c0a0bd549372cacbc2c35560874a13b859eca09b5b43a3dbf0a100b4ff79fb35",
    V4_MANIFEST_ID: "fbee02a0fa7b7fd7a0ade62d1a0b19a1a7c3a07bf2c9b2e3aee5e81bf4355b2f",
}
SMOKE_MOLECULES = 192
SMOKE_BOOTSTRAPS = 2
SMOKE_CANDIDATES_PER_ISOFORM = 32
INTERVAL_SCOPE = "fixed_fitted_Q|fixed_candidate_dictionary|realized_cycle_or_probe_availability"
WORKER_GRACE_SECONDS = 60.0
RESOURCE_TERMINATION_CODES = frozenset(
    {
        "point_estimate_wall_seconds",
        "identifiability_wall_seconds",
        "bootstrap_wall_seconds",
        "bootstrap_wall_limit",
        "worker_wall_timeout",
        "maximum_rss_bytes_per_worker",
        "maximum_dense_likelihood_entries",
        "maximum_archive_bytes",
    }
)

BOOTSTRAP_FIELDS = (
    "manifest_id",
    "run_id",
    "tier",
    "scenario_id",
    "master_seed",
    "method_id",
    "estimand_type",
    "estimand_id",
    "members",
    "truth",
    "point_estimate",
    "lower",
    "upper",
    "width",
    "covered",
    "n_successful",
    "n_attempted",
    "success_fraction",
    "interval_valid",
    "confidence_level",
    "interval_type",
    "conditional_on",
)

BOOTSTRAP_DIAGNOSTIC_FIELDS = (
    "manifest_id",
    "run_id",
    "tier",
    "scenario_id",
    "method_id",
    "master_seed",
    "bootstrap_index_0_based",
    "status",
    "retry_used",
    "initial_status",
    "initial_iterations",
    "initial_terminal_em_residual",
    "final_iterations",
    "final_terminal_em_residual",
    "failure_code",
    "failure_reason",
)

V3_BOOTSTRAP_DIAGNOSTIC_FIELDS = (
    "manifest_id",
    "run_id",
    "tier",
    "scenario_id",
    "method_id",
    "master_seed",
    "bootstrap_index_0_based",
    "status",
    "retry_used",
    "initial_status",
    "initial_iterations",
    "initial_terminal_em_residual",
    "initial_last_log_likelihood_gain",
    "initial_max_weight_change",
    "final_iterations",
    "final_terminal_em_residual",
    "final_last_log_likelihood_gain",
    "final_max_weight_change",
    "failure_code",
    "failure_reason",
)

COVERAGE_FIELDS = (
    "manifest_id",
    "tier",
    "scenario_id",
    "method_id",
    "estimand_type",
    "n_intervals",
    "n_valid_intervals",
    "covered",
    "coverage",
    "coverage_wilson_lower",
    "coverage_wilson_upper",
    "mean_width",
    "median_width",
    "headline_eligible",
)

V2_COVERAGE_FIELDS = (
    "manifest_id",
    "tier",
    "scenario_id",
    "method_id",
    "estimand_type",
    "estimand_id",
    "n_intervals",
    "n_valid_intervals",
    "covered",
    "coverage",
    "coverage_wilson_lower",
    "coverage_wilson_upper",
    "mean_width",
    "median_width",
    "headline_eligible",
)

FAILURE_FIELDS = (
    "manifest_id",
    "run_id",
    "tier",
    "scenario_id",
    "method_id",
    "master_seed",
    "stage",
    "bootstrap_index_0_based",
    "failure_code",
    "failure_reason",
)

PAIRED_FIELDS = (
    "manifest_id",
    "tier",
    "scenario_id",
    "paired_reference",
    "method_id",
    "master_seed",
    "metric",
    "scenario_value",
    "reference_value",
    "paired_difference",
    "pair_valid",
)

V2_PAIRED_FIELDS = (
    "manifest_id",
    "tier",
    "comparison_type",
    "scenario_id",
    "paired_reference",
    "method_id",
    "reference_method_id",
    "master_seed",
    "metric",
    "scenario_value",
    "reference_value",
    "paired_difference",
    "pair_valid",
)

V2_SUMMARY_FIELDS = (
    "manifest_id",
    "tier",
    "scenario_id",
    "paired_reference",
    "comparison_type",
    "reference_scenario_id",
    "reference_method_id",
    "method_id",
    "metric",
    "n_planned",
    "n_valid",
    "n_failed",
    "mean",
    "sample_SD",
    "Monte_Carlo_SE",
    "median",
    "q25",
    "q75",
    "paired_mean_difference",
    "paired_difference_sample_SD",
    "paired_difference_Monte_Carlo_SE",
    "headline_eligible",
)

RELIABILITY_FIELDS = (
    "manifest_id",
    "run_id",
    "tier",
    "scenario_id",
    "method_id",
    "master_seed",
    "resolution",
    "reportable",
    "bin_index",
    "bin_lower",
    "bin_upper",
    "count",
    "mean_confidence",
    "accuracy",
    "absolute_gap",
)

OOD_FIELDS = (
    "manifest_id",
    "run_id",
    "tier",
    "scenario_id",
    "method_id",
    "master_seed",
    "n_in_dictionary",
    "n_ood",
    "AUROC_maximum_posterior",
    "AUPRC_maximum_posterior",
    "AUROC_negative_maximum_log_likelihood_per_observed_call",
    "AUPRC_negative_maximum_log_likelihood_per_observed_call",
    "false_accept_fraction_at_maximum_posterior_at_least_0_80",
    "mean_maximum_posterior_in_dictionary",
    "mean_maximum_posterior_ood",
)

IDENTIFIABILITY_FIELDS = (
    "manifest_id",
    "run_id",
    "tier",
    "scenario_id",
    "method_id",
    "master_seed",
    "n_observable_groups",
    "n_unresolved_groups",
    "maximum_group_size",
    "centered_likelihood_rank",
    "centered_likelihood_singular_max",
    "centered_likelihood_singular_min_retained",
    "centered_likelihood_condition_number",
    "relative_singular_value_floor",
    "plus_0_05_Q_weight_TV",
    "minus_0_05_Q_weight_TV",
    "maximum_plus_or_minus_0_05_Q_weight_TV",
)

V3_IDENTIFIABILITY_FIELDS = (
    "manifest_id",
    "run_id",
    "tier",
    "scenario_id",
    "method_id",
    "master_seed",
    "n_observable_groups",
    "n_unresolved_groups",
    "maximum_group_size",
    "centered_likelihood_rank",
    "centered_likelihood_singular_max",
    "centered_likelihood_singular_min_retained",
    "centered_likelihood_condition_number",
    "relative_singular_value_floor",
    "plus_0_05_Q_status",
    "plus_0_05_Q_retry_used",
    "plus_0_05_Q_iterations",
    "plus_0_05_Q_terminal_em_residual",
    "plus_0_05_Q_last_log_likelihood_gain",
    "plus_0_05_Q_max_weight_change",
    "plus_0_05_Q_weight_TV",
    "minus_0_05_Q_status",
    "minus_0_05_Q_retry_used",
    "minus_0_05_Q_iterations",
    "minus_0_05_Q_terminal_em_residual",
    "minus_0_05_Q_last_log_likelihood_gain",
    "minus_0_05_Q_max_weight_change",
    "minus_0_05_Q_weight_TV",
    "maximum_plus_or_minus_0_05_Q_weight_TV",
)

FEATURE_FIELDS = (
    "manifest_id",
    "run_id",
    "tier",
    "scenario_id",
    "method_id",
    "master_seed",
    "isoform_total_variation",
    "PTM_marginal_MAE",
)

V4_FEATURE_FIELDS = (
    *FEATURE_FIELDS,
    "PTM_pair_MAE",
)

IDENTIFIABILITY_SINGULAR_VALUE_FIELDS = (
    "manifest_id",
    "run_id",
    "tier",
    "scenario_id",
    "method_id",
    "master_seed",
    "singular_value_index_0_based",
    "singular_value",
    "retained_above_relative_floor",
)

SCENARIO_INDEX_FIELDS = (
    "manifest_id",
    "tier",
    "scenario_id",
    "scenario_family",
    "paired_reference",
    "method_id",
    "bootstrap_eligible",
    "headline_eligible",
    "definition_json",
)

FIGURE_FIELDS = (
    "manifest_id",
    "tier",
    "row_type",
    "scenario_id",
    "scenario_family",
    "paired_reference",
    "method_id",
    "metric",
    "resolution",
    "estimate",
    "uncertainty",
    "bin_index",
    "bin_lower",
    "bin_upper",
    "count",
    "mean_confidence",
    "accuracy",
    "reportable",
)

V2_FIGURE_FIELDS = (
    "manifest_id",
    "tier",
    "row_type",
    "value_kind",
    "scenario_id",
    "scenario_family",
    "paired_reference",
    "comparison_type",
    "reference_scenario_id",
    "reference_method_id",
    "method_id",
    "metric",
    "resolution",
    "estimate",
    "uncertainty",
    "bin_index",
    "bin_lower",
    "bin_upper",
    "count",
    "mean_confidence",
    "accuracy",
    "reportable",
)


class ResourceLimit(RuntimeError):
    """A declared manifest resource limit prevented one affected run."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@contextmanager
def _deadline(seconds: float, code: str):
    """Raise a typed resource failure when a worker exceeds wall time."""

    if seconds <= 0:
        raise ValueError("deadline seconds must be positive")
    previous_handler = signal.getsignal(signal.SIGALRM)

    def handle_timeout(_signum: int, _frame: Any) -> None:
        raise ResourceLimit(code, f"manifest wall limit of {seconds} seconds reached")

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


@dataclass(frozen=True)
class ExecutionPlan:
    tier: str
    scientific_tier: str
    scenario_ids: tuple[str, ...]
    master_seeds: tuple[int, ...]
    n_molecules: int
    bootstrap_resamples: int
    methods_filter: tuple[str, ...] | None
    preflight: bool
    skip_bootstrap: bool
    manifest_conformant: bool
    headline_eligible: bool
    overrides: dict[str, Any]


@dataclass(frozen=True)
class Design:
    profiles: np.ndarray
    candidate_ids: tuple[str, ...]
    isoforms: tuple[str, ...]
    ptm_states: np.ndarray
    probe_ids: tuple[str, ...]
    cycle_to_probe: np.ndarray
    base_Q: np.ndarray
    original_indices: np.ndarray

    @property
    def n_candidates(self) -> int:
        return int(self.profiles.shape[0])

    @property
    def n_probes(self) -> int:
        return int(self.profiles.shape[1])

    @property
    def n_cycles(self) -> int:
        return int(self.cycle_to_probe.size)


@dataclass(frozen=True)
class SeedState:
    master_seed: int
    support: np.ndarray
    base_weights: np.ndarray
    heldout_support: np.ndarray
    base_identities: np.ndarray
    emission_uniforms: np.ndarray
    missingness_uniforms: np.ndarray
    q_error_normals: np.ndarray
    shared_effects: np.ndarray
    ood_kind_uniform: np.ndarray
    ood_choice_uniform: np.ndarray
    ood_second_identity_uniform: np.ndarray
    ood_truncation_h: np.ndarray


@dataclass(frozen=True)
class Dataset:
    observations: np.ndarray
    identities: np.ndarray
    truth_weights: np.ndarray
    ood_mask: np.ndarray
    ood_subtype: np.ndarray
    true_Q: np.ndarray
    n_active_truth: int
    data_sha256: str


@dataclass(frozen=True)
class MethodFit:
    weights: np.ndarray
    classes: TraceLikelihoodClasses
    posterior_weights: np.ndarray | None
    probability_method: bool
    status: str
    failure_code: str | None
    failure_reason: str | None
    converged: bool | None
    retry_used: bool
    iterations: int | None
    terminal_residual: float | None
    minimum_gain: float | None
    last_gain: float | None
    max_weight_change: float | None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(*values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical_json(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _schema_names(entries: Sequence[str]) -> tuple[str, ...]:
    return tuple(entry.split(":", maxsplit=1)[0] for entry in entries)


def _manifest_canonical_sha256(manifest: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(manifest).encode("utf-8"))


def _load_frozen_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    manifest_id = manifest.get("manifest_id")
    expected = FROZEN_MANIFEST_SHA256.get(manifest_id)
    observed = _sha256_bytes(raw)
    if expected is None or observed != expected:
        raise ValueError(
            "supplied frozen manifest bytes do not match the runner-pinned SHA-256 "
            f"for {manifest_id!r}: expected {expected}, got {observed}"
        )
    _validate_manifest(manifest)
    return manifest, observed


def _require_manifest_value(actual: Any, expected: Any, path: str) -> None:
    if actual != expected:
        raise ValueError(
            f"frozen manifest invariant {path} must be {expected!r}, got {actual!r}"
        )


def _require_manifest_integer(actual: Any, expected: int, path: str) -> None:
    if type(actual) is not int or actual != expected:
        raise ValueError(
            f"frozen manifest invariant {path} must be integer {expected}, "
            f"got {actual!r}"
        )


def _validate_modern_manifest(manifest: dict[str, Any]) -> None:
    """Validate operational invariants shared by the frozen v2/v3 designs."""

    schema_version = str(manifest["schema_version"])
    if schema_version not in MODERN_SCHEMA_VERSIONS:
        raise ValueError(
            "modern manifest validation requires schema 2.0.0, 3.0.0, or 4.0.0"
        )

    primary = manifest["tiers"]["primary"]
    debug = manifest["tiers"]["debug"]
    primary_scenarios = tuple(primary["scenario_ids"])
    expected_scenarios = (
        "B0_complete",
        "B1_mcar02",
        "B2_mcar10",
        "B3_cycle18_drop",
        "B4_anti4R_drop",
        "B5_mnar_call_dependent",
        "Q1_delta_m050_sigma000",
        "Q2_delta_p050_sigma000",
        "Q3_delta_000_sigma025",
        "Q4_delta_m050_sigma025",
        "Q5_delta_p050_sigma025",
        "D1_shared035",
        "D2_shared065",
        "I1_anti4R_contrast025",
        "I2_anti4R_contrast000",
        "O1_heldout05",
        "O2_heldout20",
        "O3_artifact_mix05",
    )
    expected_seeds = tuple(range(7, 198, 10))
    _require_manifest_value(primary_scenarios, expected_scenarios, "tiers.primary.scenario_ids")
    _require_manifest_value(
        tuple(debug["scenario_ids"]), expected_scenarios, "tiers.debug.scenario_ids"
    )
    _require_manifest_value(
        tuple(primary["master_seeds"]), expected_seeds, "tiers.primary.master_seeds"
    )
    _require_manifest_value(
        tuple(debug["master_seeds"]), (7, 17, 27), "tiers.debug.master_seeds"
    )
    for tier_name, molecules, resamples, headline in (
        ("primary", 10_000, 50, True),
        ("debug", 5_000, 10, False),
    ):
        tier = manifest["tiers"][tier_name]
        _require_manifest_integer(
            tier["molecules_per_dataset"], molecules,
            f"tiers.{tier_name}.molecules_per_dataset",
        )
        _require_manifest_integer(
            tier["bootstrap_resamples"], resamples,
            f"tiers.{tier_name}.bootstrap_resamples",
        )
        _require_manifest_value(
            tier["headline_eligible"], headline,
            f"tiers.{tier_name}.headline_eligible",
        )

    expected_targeted = {
        "B1_mcar02": [
            "soft_uniform_one_step",
            "top_likelihood_equal_ties",
            "binary_best_hamming_em",
        ],
        "B2_mcar10": [
            "top_likelihood_equal_ties",
            "NA_as_zero_top_likelihood_antipattern",
        ],
        "D1_shared035": ["oracle_repeat_aware_likelihood_em"],
        "D2_shared065": ["oracle_repeat_aware_likelihood_em"],
    }
    _require_manifest_value(
        manifest["methods"]["run_on_all_scenarios"],
        ["em_fixed_q"],
        "methods.run_on_all_scenarios",
    )
    _require_manifest_value(
        manifest["methods"]["targeted_methods"],
        expected_targeted,
        "methods.targeted_methods",
    )
    _require_manifest_value(
        manifest["methods"]["headline_scenario_contrasts"],
        {
            "em_fixed_q": "all_nonself_declared_paired_references",
            "top_likelihood_equal_ties": ["B2_mcar10"],
        },
        "methods.headline_scenario_contrasts",
    )
    expected_contrasts = [
        ("B1_mcar02", "soft_uniform_one_step", "em_fixed_q"),
        ("B1_mcar02", "top_likelihood_equal_ties", "em_fixed_q"),
        ("B1_mcar02", "binary_best_hamming_em", "em_fixed_q"),
        (
            "B2_mcar10",
            "NA_as_zero_top_likelihood_antipattern",
            "top_likelihood_equal_ties",
        ),
        ("D1_shared035", "oracle_repeat_aware_likelihood_em", "em_fixed_q"),
        ("D2_shared065", "oracle_repeat_aware_likelihood_em", "em_fixed_q"),
    ]
    actual_contrasts = [
        (
            entry["scenario_id"],
            entry["method_id"],
            entry["reference_method_id"],
        )
        for entry in manifest["methods"]["headline_method_contrasts"]
    ]
    _require_manifest_value(
        actual_contrasts, expected_contrasts, "methods.headline_method_contrasts"
    )
    for scenario, method, reference_method in actual_contrasts:
        declared = {"em_fixed_q", *expected_targeted.get(scenario, [])}
        if method not in declared or reference_method not in declared:
            raise ValueError(
                "headline method contrast references a method not declared for "
                f"{scenario}"
            )
    scenario_rules = manifest["methods"]["headline_scenario_contrasts"]
    for scenario in expected_scenarios:
        reference = manifest["scenario_definitions"][scenario]["paired_reference"]
        for method in ("em_fixed_q", *expected_targeted.get(scenario, [])):
            rule = scenario_rules.get(method)
            scenario_declared = (
                rule == "all_nonself_declared_paired_references"
                and scenario != reference
            ) or (isinstance(rule, list) and scenario in rule)
            method_declared = sum(
                candidate_scenario == scenario and candidate_method == method
                for candidate_scenario, candidate_method, _ in actual_contrasts
            )
            if int(scenario_declared) + method_declared > 1:
                raise ValueError(
                    "v2 summary row model permits at most one declared contrast "
                    f"for {scenario}:{method}"
                )

    bootstrap = manifest["bootstrap"]
    _require_manifest_value(
        bootstrap["eligible_primary_scenarios"],
        ["B0_complete", "B1_mcar02", "B2_mcar10"],
        "bootstrap.eligible_primary_scenarios",
    )
    _require_manifest_integer(
        bootstrap["primary_resamples"], 50, "bootstrap.primary_resamples"
    )
    _require_manifest_integer(
        bootstrap["debug_resamples"], 10, "bootstrap.debug_resamples"
    )
    _require_manifest_value(
        bootstrap["coverage_grouping"],
        ["scenario_id", "method_id", "estimand_type", "estimand_id"],
        "bootstrap.coverage_grouping",
    )
    _require_manifest_integer(
        bootstrap["coverage_minimum_valid_seed_intervals"],
        18,
        "bootstrap.coverage_minimum_valid_seed_intervals",
    )
    _require_manifest_value(
        bootstrap["diagnostics_required"], True, "bootstrap.diagnostics_required"
    )
    _require_manifest_value(
        bootstrap["minimum_success_fraction"],
        0.98,
        "bootstrap.minimum_success_fraction",
    )
    em_settings = manifest["stopping_and_failures"]["em"]
    _require_manifest_value(
        em_settings["maximum_weight_change_tolerance"],
        em_settings["relative_objective_tolerance"],
        "stopping_and_failures.em.maximum_weight_change_tolerance",
    )
    _require_manifest_integer(
        em_settings["consecutive_qualifying_iterations"],
        1,
        "stopping_and_failures.em.consecutive_qualifying_iterations",
    )
    _require_manifest_integer(
        em_settings["retry_once_if_nonconverged"]["max_iterations"],
        1_000 if schema_version == V2_SCHEMA_VERSION else 10_000,
        "stopping_and_failures.em.retry_once_if_nonconverged.max_iterations",
    )
    _require_manifest_value(
        manifest["change_control"]["v1_manifest_sha256"],
        "e86ba9a6bb75f223fb012670fe8887e12fdd838ccd65b0eb40c616bc3c7cef49",
        "change_control.v1_manifest_sha256",
    )
    _require_manifest_value(
        manifest["change_control"]["v1_execution_status"],
        "preflighted_but_full_primary_grid_unexecuted",
        "change_control.v1_execution_status",
    )
    _require_manifest_value(
        manifest["change_control"]["headline_eligible"],
        False,
        "change_control.headline_eligible",
    )
    if schema_version in DIRECT_SVD_SCHEMA_VERSIONS:
        _require_manifest_value(
            manifest["supersedes_manifest_id"],
            (
                V2_MANIFEST_ID
                if schema_version == V3_SCHEMA_VERSION
                else V3_MANIFEST_ID
            ),
            "supersedes_manifest_id",
        )
        _require_manifest_value(
            manifest["change_control"]["v2_execution_status"],
            "full_debug_failed_no_primary_run",
            "change_control.v2_execution_status",
        )
        _require_manifest_value(
            manifest["change_control"]["v2_manifest_sha256"],
            FROZEN_MANIFEST_SHA256[V2_MANIFEST_ID],
            "change_control.v2_manifest_sha256",
        )
        _require_manifest_value(
            manifest["change_control"]["v2_failed_debug_evidence"][
                "archive_output_checksums_sha256"
            ],
            "45136e437f5063ed7eb35091447ae9bcfa0281015638cd57e49365fcaca35aee",
            "change_control.v2_failed_debug_evidence.archive_output_checksums_sha256",
        )
        _require_manifest_value(
            manifest["change_control"]["v2_failed_debug_evidence"][
                "retained_archive_relative_path"
            ],
            "outputs/model-violation-benchmark-v2-debug-failed",
            "change_control.v2_failed_debug_evidence.retained_archive_relative_path",
        )
        _require_manifest_value(
            manifest["stopping_and_failures"]["failure_precedence"],
            [
                "nonfinite_fit",
                "likelihood_nonmonotonic",
                "terminal_residual_exceeded",
                "em_nonconvergence",
            ],
            "stopping_and_failures.failure_precedence",
        )
        _require_manifest_value(
            manifest["stopping_and_failures"]["identifiability_sensitivity_em"],
            {
                "initialization": "converged_base_fit_weights",
                "initial_max_iterations": 500,
                "retry_once_if_nonconverged": {
                    "max_iterations": 10_000,
                    "initialization": (
                        "continue_from_initial_sensitivity_fit_weights"
                    ),
                },
                "relative_objective_tolerance": 1e-8,
                "maximum_weight_change_tolerance": 1e-8,
                "terminal_fixed_point_residual_maximum": 1e-7,
                "require_status_ok_for_full_tier": True,
            },
            "stopping_and_failures.identifiability_sensitivity_em",
        )
        convergence_preflight = manifest["change_control"][
            "primary_i1_convergence_preflight"
        ]
        _require_manifest_value(
            convergence_preflight["audited_master_seeds"],
            list(expected_seeds),
            "change_control.primary_i1_convergence_preflight.audited_master_seeds",
        )
        _require_manifest_integer(
            convergence_preflight["maximum_observed_retry_iterations"],
            5_780,
            "change_control.primary_i1_convergence_preflight.maximum_observed_retry_iterations",
        )
        _require_manifest_value(
            (
                convergence_preflight["all_converged_at_retry_cap_10000"],
                convergence_preflight[
                    "all_monotonic_and_terminal_residual_at_most_1e-7"
                ],
            ),
            (True, True),
            "change_control.primary_i1_convergence_preflight pass flags",
        )
        identifiability_preflight = manifest["change_control"][
            "debug_i1_identifiability_preflight"
        ]
        _require_manifest_value(
            identifiability_preflight["audited_master_seeds"],
            [7, 17, 27],
            "change_control.debug_i1_identifiability_preflight.audited_master_seeds",
        )
        _require_manifest_value(
            (
                identifiability_preflight[
                    "all_point_and_sensitivity_fits_status_ok"
                ],
                identifiability_preflight[
                    "all_direct_svd_ranks_and_conditions_finite"
                ],
                identifiability_preflight[
                    "all_within_declared_wall_and_RSS_limits"
                ],
            ),
            (True, True, True),
            "change_control.debug_i1_identifiability_preflight pass flags",
        )
        if schema_version in V4_METRIC_SCHEMA_VERSIONS:
            _require_manifest_value(
                manifest["change_control"]["v3_manifest_sha256"],
                FROZEN_MANIFEST_SHA256[V3_MANIFEST_ID],
                "change_control.v3_manifest_sha256",
            )
            _require_manifest_value(
                manifest["change_control"]["v3_execution_status"],
                (
                    "two_full_debug_archives_passed_primary_interrupted_before_"
                    "archive_after_reporting_contract_audit"
                ),
                "change_control.v3_execution_status",
            )
    _require_manifest_value(
        [
            (entry["role"], entry["git_commit"])
            for entry in manifest["change_control"]["preflight_evidence"]
        ],
        [
            (
                "exact_v1_primary_B1_seed7_all_point_methods_without_bootstrap",
                "6a7038f741060a1f4d1bcbf43533c50267ae3e08",
            ),
            (
                "exact_v1_debug_B1_seed7_full_25_resample_bootstrap_after_retry_fix",
                "28c26f11521b6e70272ef70ef78c735608307c60",
            ),
        ],
        "change_control.preflight_evidence",
    )
    _require_manifest_value(
        manifest["change_control"]["projected_v1_full_primary_wall_days"],
        {"lower": 10.6, "upper": 11.1},
        "change_control.projected_v1_full_primary_wall_days",
    )
    operational_target = manifest["change_control"][
        "v2_operational_completion_target_wall_hours"
    ]
    _require_manifest_value(
        (operational_target["lower"], operational_target["upper"]),
        (2.0, 4.0),
        "change_control.v2_operational_completion_target_wall_hours",
    )
    if schema_version in DIRECT_SVD_SCHEMA_VERSIONS:
        v3_operational_target = manifest["change_control"][
            "v3_operational_completion_target_wall_hours"
        ]
        _require_manifest_value(
            (v3_operational_target["lower"], v3_operational_target["upper"]),
            (2.0, 6.0),
            "change_control.v3_operational_completion_target_wall_hours",
        )

    resources = manifest["resources"]
    expected_resources = {
        "maximum_concurrent_primary_workers": 2,
        "maximum_rss_bytes_per_worker": 4 * 1024**3,
        "point_estimate_wall_seconds_per_scenario_seed": (
            300 if schema_version == V2_SCHEMA_VERSION else 600
        ),
        "bootstrap_wall_seconds_per_scenario_seed": 600,
        "maximum_dense_likelihood_entries": 10_000_000,
        "maximum_archive_bytes": 1024**3,
        "full_fit_molecule_limit": 10_000,
        "worker_blas_threads": 1,
    }
    if schema_version in DIRECT_SVD_SCHEMA_VERSIONS:
        expected_resources[
            "identifiability_wall_seconds_per_scenario_seed"
        ] = 300
    for key, expected in expected_resources.items():
        _require_manifest_integer(resources[key], expected, f"resources.{key}")
    _require_manifest_value(
        resources["archive_size_definition"],
        (
            "Sum of the byte sizes of all regular top-level archive files except "
            "output_checksums.sha256; the checksum inventory is excluded to avoid "
            "a self-referential size calculation."
        ),
        "resources.archive_size_definition",
    )
    expected_thread_variables = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ]
    _require_manifest_value(
        resources["worker_thread_environment_variables"],
        expected_thread_variables,
        "resources.worker_thread_environment_variables",
    )
    _require_manifest_integer(
        manifest["stopping_and_failures"]["primary_summary_minimum_valid_seed_pairs"],
        18,
        "stopping_and_failures.primary_summary_minimum_valid_seed_pairs",
    )
    _require_manifest_integer(
        manifest["stopping_and_failures"][
            "primary_summary_minimum_valid_absolute_runs"
        ],
        18,
        "stopping_and_failures.primary_summary_minimum_valid_absolute_runs",
    )
    _require_manifest_integer(
        manifest["stopping_and_failures"]["coverage_headline_minimum_valid_seed_intervals"],
        18,
        "stopping_and_failures.coverage_headline_minimum_valid_seed_intervals",
    )
    systematic_policy = manifest["stopping_and_failures"][
        "systematic_failure_stop_policy"
    ]
    _require_manifest_integer(
        systematic_policy["threshold_failed_scenario_seed_attempts"],
        3,
        "stopping_and_failures.systematic_failure_stop_policy.threshold_failed_scenario_seed_attempts",
    )
    _require_manifest_value(
        systematic_policy["fatal_failure_rules"],
        [
            {
                "stage": "point_estimate",
                "failure_codes": "all_except_systematic_failure_stop",
            },
            {
                "stage": "worker",
                "failure_codes": "all_except_systematic_failure_stop",
            },
            {
                "stage": "bootstrap",
                "failure_codes": [
                    "bootstrap_wall_seconds",
                    "bootstrap_wall_limit",
                    "maximum_rss_bytes_per_worker",
                ],
            },
            {
                "stage": "bootstrap_summary",
                "failure_codes": [
                    "bootstrap_success_fraction_below_minimum"
                ],
            },
        ],
        "stopping_and_failures.systematic_failure_stop_policy.fatal_failure_rules",
    )
    _require_manifest_value(
        systematic_policy["nonfatal_bootstrap_replicate_rule"],
        {
            "stage": "bootstrap",
            "bootstrap_index_0_based": "integer",
            "failure_codes": "all_except_declared_fatal_bootstrap_codes",
            "action": (
                "Retain each replicate failure and diagnostic row, but do not "
                "increment the systematic-stop counter. Apply "
                "bootstrap.minimum_success_fraction to the complete "
                "scenario-seed bootstrap result."
            ),
        },
        "stopping_and_failures.systematic_failure_stop_policy.nonfatal_bootstrap_replicate_rule",
    )
    _require_manifest_value(
        systematic_policy["seed_replacement"],
        False,
        "stopping_and_failures.systematic_failure_stop_policy.seed_replacement",
    )

    outputs = manifest["outputs"]
    _require_manifest_value(
        outputs["root"],
        (
            "outputs/model-violation-benchmark-v2"
            if schema_version == V2_SCHEMA_VERSION
            else (
                "outputs/model-violation-benchmark-v3"
                if schema_version == V3_SCHEMA_VERSION
                else "outputs/model-violation-benchmark-v4"
            )
        ),
        "outputs.root",
    )
    _require_manifest_value(
        _schema_names(outputs["paired_contrast_schema"]),
        V2_PAIRED_FIELDS,
        "outputs.paired_contrast_schema",
    )
    _require_manifest_value(
        _schema_names(outputs["summary_schema"]),
        V2_SUMMARY_FIELDS,
        "outputs.summary_schema",
    )
    _require_manifest_value(
        _schema_names(outputs["bootstrap_diagnostic_schema"]),
        (
            BOOTSTRAP_DIAGNOSTIC_FIELDS
            if schema_version == V2_SCHEMA_VERSION
            else V3_BOOTSTRAP_DIAGNOSTIC_FIELDS
        ),
        "outputs.bootstrap_diagnostic_schema",
    )
    if schema_version in DIRECT_SVD_SCHEMA_VERSIONS:
        _require_manifest_value(
            _schema_names(outputs["identifiability_diagnostic_schema"]),
            V3_IDENTIFIABILITY_FIELDS,
            "outputs.identifiability_diagnostic_schema",
        )
    if schema_version in V4_METRIC_SCHEMA_VERSIONS:
        _require_manifest_value(
            _schema_names(outputs["feature_metric_schema"]),
            V4_FEATURE_FIELDS,
            "outputs.feature_metric_schema",
        )
        _require_manifest_value(
            _schema_names(outputs["identifiability_singular_value_schema"]),
            IDENTIFIABILITY_SINGULAR_VALUE_FIELDS,
            "outputs.identifiability_singular_value_schema",
        )
    _require_manifest_value(
        _schema_names(outputs["coverage_summary_schema"]),
        V2_COVERAGE_FIELDS,
        "outputs.coverage_summary_schema",
    )
    _require_manifest_value(
        _schema_names(outputs["figure_data_schema"]),
        V2_FIGURE_FIELDS,
        "outputs.figure_data_schema",
    )
    _require_manifest_value(
        outputs["execution_status_schema"],
        {
            "schema_version": "string",
            "manifest_id": "string",
            "tier": "string",
            "execution_scope": "string",
            "status": "passed|failed|not_applicable",
            "reasons": "array[{code:string,message:string}]",
            "counts": "object",
        },
        "outputs.execution_status_schema",
    )

    gates = manifest["execution_gates"]
    _require_manifest_value(
        (
            gates["full_tier_requires_clean_git"],
            gates["failed_full_tier_exit_code_nonzero"],
            gates["smoke_and_preflight_status"],
        ),
        (True, True, "not_applicable"),
        "execution_gates top-level controls",
    )
    debug_gate = gates["full_primary_debug_gate"]
    _require_manifest_integer(
        debug_gate["required_archive_count"],
        2,
        "execution_gates.full_primary_debug_gate.required_archive_count",
    )
    _require_manifest_value(
        (
            debug_gate["required_tier"],
            debug_gate["required_execution_scope"],
            debug_gate["required_execution_status"],
            debug_gate["required_git_dirty"],
        ),
        ("debug", "full_tier", "passed", False),
        "execution_gates.full_primary_debug_gate provenance/status requirements",
    )
    comparison = debug_gate["deterministic_pair_comparison"]
    _require_manifest_value(
        comparison["normalized_run_excluded_fields"],
        ["wall_seconds", "peak_RSS_bytes"],
        "execution_gates.full_primary_debug_gate.deterministic_pair_comparison.normalized_run_excluded_fields",
    )


def _validate_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "manifest_id",
        "frozen",
        "design",
        "replicate_semantics",
        "randomness",
        "tiers",
        "scenario_definitions",
        "methods",
        "bootstrap",
        "stopping_and_failures",
        "resources",
        "outputs",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"manifest lacks required fields: {sorted(missing)}")
    schema_version = manifest["schema_version"]
    expected_ids = {
        V1_SCHEMA_VERSION: V1_MANIFEST_ID,
        V2_SCHEMA_VERSION: V2_MANIFEST_ID,
        V3_SCHEMA_VERSION: V3_MANIFEST_ID,
        V4_SCHEMA_VERSION: V4_MANIFEST_ID,
    }
    if schema_version not in expected_ids or manifest["frozen"] is not True:
        raise ValueError(
            "runner requires a supported frozen schema_version 1.0.0, 2.0.0, 3.0.0, or 4.0.0 manifest"
        )
    _require_manifest_value(
        manifest["manifest_id"], expected_ids[schema_version], "manifest_id"
    )
    semantic_expected = FROZEN_MANIFEST_CANONICAL_SHA256[manifest["manifest_id"]]
    semantic_observed = _manifest_canonical_sha256(manifest)
    if semantic_observed != semantic_expected:
        raise ValueError(
            "manifest scientific/configuration content differs from the frozen "
            f"{manifest['manifest_id']} object"
        )
    if manifest["randomness"]["bit_generator"] != "PCG64":
        raise ValueError("manifest bit_generator must be PCG64")
    if manifest["randomness"]["seed_sequence"] != "numpy.random.SeedSequence":
        raise ValueError("manifest seed_sequence must be numpy.random.SeedSequence")
    expected_spawn = [
        "domain_code",
        "biological_replicate_index_0_based",
        "technical_run_index_0_based",
        "bootstrap_index_0_based",
    ]
    if manifest["randomness"]["spawn_key_order"] != expected_spawn:
        raise ValueError("unsupported randomness.spawn_key_order")
    resources = manifest["resources"]
    if resources["worker_process_isolation"] is not True:
        raise ValueError("frozen manifests require worker_process_isolation=true")
    expected_concurrency = 1 if schema_version == V1_SCHEMA_VERSION else 2
    if int(resources["maximum_concurrent_primary_workers"]) != expected_concurrency:
        raise ValueError(
            "frozen manifest invariant resources.maximum_concurrent_primary_workers "
            f"must be {expected_concurrency}"
        )
    for key in (
        "maximum_rss_bytes_per_worker",
        "point_estimate_wall_seconds_per_scenario_seed",
        "bootstrap_wall_seconds_per_scenario_seed",
        "maximum_dense_likelihood_entries",
        "maximum_archive_bytes",
    ):
        if float(resources[key]) <= 0:
            raise ValueError(f"resources.{key} must be positive")
    scenarios = set(manifest["scenario_definitions"])
    for tier_name, tier in manifest["tiers"].items():
        unknown = set(tier["scenario_ids"]) - scenarios
        if unknown:
            raise ValueError(f"tier {tier_name} references unknown scenarios: {sorted(unknown)}")
        declared = manifest["randomness"][f"{tier_name}_master_seeds"]
        if tier["master_seeds"] != declared:
            raise ValueError(f"tier {tier_name} seeds disagree with randomness")
    required_outputs = set(manifest["outputs"]["required_files"])
    expected_outputs = {
        "manifest.json",
        "provenance.json",
        "candidate_dictionary.tsv",
        "probe_panel.tsv",
        "scenario_index.tsv",
        "runs.tsv",
        "paired_contrasts.tsv",
        "summary.tsv",
        "bootstrap_intervals.tsv",
        "coverage_summary.tsv",
        "failures.tsv",
        "output_checksums.sha256",
    }
    if schema_version in MODERN_SCHEMA_VERSIONS:
        expected_outputs.update(
            {"bootstrap_diagnostics.tsv", "execution_status.json"}
        )
    if schema_version in DIRECT_SVD_SCHEMA_VERSIONS:
        expected_outputs.update(
            {
                "reliability.tsv",
                "ood_metrics.tsv",
                "identifiability_diagnostics.tsv",
                "feature_metrics.tsv",
                "figure_data.tsv",
            }
        )
    if schema_version in V4_METRIC_SCHEMA_VERSIONS:
        expected_outputs.add("identifiability_singular_values.tsv")
    if required_outputs != expected_outputs:
        raise ValueError(
            f"manifest required_files are not the supported frozen {schema_version} set"
        )
    run_names = _schema_names(manifest["outputs"]["run_schema"])
    summary_names = _schema_names(manifest["outputs"]["summary_schema"])
    if len(run_names) != len(set(run_names)) or len(summary_names) != len(set(summary_names)):
        raise ValueError("manifest output schemas contain duplicate field names")
    if schema_version in DIRECT_SVD_SCHEMA_VERSIONS:
        diagnostics_index = run_names.index("minimum_log_likelihood_gain")
        _require_manifest_value(
            run_names[diagnostics_index : diagnostics_index + 3],
            (
                "minimum_log_likelihood_gain",
                "last_log_likelihood_gain",
                "max_weight_change",
            ),
            "outputs.run_schema convergence diagnostics",
        )
    if schema_version in V4_METRIC_SCHEMA_VERSIONS:
        metrics_valid_index = run_names.index("metrics_valid")
        _require_manifest_value(
            run_names[metrics_valid_index : metrics_valid_index + 4],
            (
                "metrics_valid",
                "abundance_resolution",
                "truth_out_of_dictionary_mass",
                "total_variation",
            ),
            "outputs.run_schema v4 abundance audit fields",
        )
        candidate_rmse_index = run_names.index("candidate_RMSE")
        _require_manifest_value(
            run_names[candidate_rmse_index : candidate_rmse_index + 9],
            (
                "candidate_RMSE",
                "maximum_absolute_error",
                "detection_sensitivity",
                "detection_false_discovery_rate",
                "n_truth_detectable",
                "n_called_detected",
                "n_true_positive_detected",
                "n_false_positive_detected",
                "group_total_variation",
            ),
            "outputs.run_schema v4 abundance metrics",
        )
        top5_index = run_names.index("top5_recall")
        _require_manifest_value(
            run_names[top5_index : top5_index + 4],
            (
                "top5_recall",
                "posterior_entropy",
                "posterior_entropy_resolution",
                "posterior_entropy_evaluated_n",
            ),
            "outputs.run_schema v4 posterior entropy",
        )
    if schema_version in MODERN_SCHEMA_VERSIONS:
        _validate_modern_manifest(manifest)


def _declared_worker_thread_environment(
    manifest: dict[str, Any],
) -> dict[str, str]:
    resources = manifest["resources"]
    threads = resources.get("worker_blas_threads")
    variables = resources.get("worker_thread_environment_variables", [])
    if threads is None:
        return {}
    return {str(variable): str(int(threads)) for variable in variables}


def _configure_worker_thread_environment(
    manifest: dict[str, Any],
) -> dict[str, str]:
    """Install the frozen BLAS-thread settings before spawning fresh workers."""

    declared = _declared_worker_thread_environment(manifest)
    for variable, value in declared.items():
        os.environ[variable] = value
    return {variable: os.environ[variable] for variable in declared}


def _verify_worker_thread_environment(manifest: dict[str, Any]) -> None:
    declared = _declared_worker_thread_environment(manifest)
    mismatches = {
        variable: os.environ.get(variable)
        for variable, expected in declared.items()
        if os.environ.get(variable) != expected
    }
    if mismatches:
        raise RuntimeError(
            "worker BLAS-thread environment differs from the frozen manifest: "
            f"{mismatches}"
        )


def _rng(
    manifest: dict[str, Any],
    master_seed: int,
    domain: str,
    *,
    bootstrap_index: int = 0,
) -> np.random.Generator:
    domain_code = int(manifest["randomness"]["domain_codes"][domain])
    sequence = np.random.SeedSequence(
        master_seed,
        spawn_key=(domain_code, 0, 0, int(bootstrap_index)),
    )
    return np.random.Generator(np.random.PCG64(sequence))


def _expit(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _logit(values: Any) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-12, 1.0 - 1e-12)
    return np.log(values) - np.log1p(-values)


def _sample_from_uniform(uniform: np.ndarray, weights: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, uniform, side="right").astype(np.int64)


def _build_plan(args: argparse.Namespace, manifest: dict[str, Any]) -> ExecutionPlan:
    if args.smoke:
        scientific_tier = "debug"
        tier = "smoke"
        source = manifest["tiers"][scientific_tier]
        scenario_ids = tuple(source["scenario_ids"])
        master_seeds = (int(source["master_seeds"][0]),)
        return ExecutionPlan(
            tier=tier,
            scientific_tier=scientific_tier,
            scenario_ids=scenario_ids,
            master_seeds=master_seeds,
            n_molecules=SMOKE_MOLECULES,
            bootstrap_resamples=SMOKE_BOOTSTRAPS,
            methods_filter=None,
            preflight=True,
            skip_bootstrap=False,
            manifest_conformant=False,
            headline_eligible=False,
            overrides={
                "purpose": "fast pipeline smoke; never a manifest debug or primary result",
                "molecules_per_dataset": SMOKE_MOLECULES,
                "bootstrap_resamples": SMOKE_BOOTSTRAPS,
                "master_seeds": list(master_seeds),
                "candidate_dictionary": f"stratified_{6 * SMOKE_CANDIDATES_PER_ISOFORM}_state_subset",
            },
        )

    scientific_tier = str(args.tier)
    source = manifest["tiers"][scientific_tier]
    scenarios = tuple(args.scenario_id or source["scenario_ids"])
    unknown = set(scenarios) - set(source["scenario_ids"])
    if unknown:
        raise ValueError(f"scenarios are not declared for {scientific_tier}: {sorted(unknown)}")
    seeds = tuple(int(seed) for seed in (args.master_seed or source["master_seeds"]))
    unknown_seeds = set(seeds) - set(source["master_seeds"])
    if unknown_seeds:
        raise ValueError(f"master seeds are not declared for {scientific_tier}: {sorted(unknown_seeds)}")
    partial = scenarios != tuple(source["scenario_ids"]) or seeds != tuple(source["master_seeds"])
    if (partial or args.method_id or args.skip_bootstrap) and not args.preflight:
        raise ValueError("partial scheduling and --skip-bootstrap require --preflight")
    if args.skip_bootstrap and not args.preflight:
        raise ValueError("--skip-bootstrap is allowed only for an explicit preflight")
    return ExecutionPlan(
        tier=scientific_tier,
        scientific_tier=scientific_tier,
        scenario_ids=scenarios,
        master_seeds=seeds,
        n_molecules=int(source["molecules_per_dataset"]),
        bootstrap_resamples=int(source["bootstrap_resamples"]),
        methods_filter=tuple(args.method_id) if args.method_id else None,
        preflight=bool(args.preflight),
        skip_bootstrap=bool(args.skip_bootstrap),
        manifest_conformant=True,
        headline_eligible=bool(source["headline_eligible"] and not args.preflight and not partial),
        overrides={},
    )


def _prepare_design(manifest: dict[str, Any], plan: ExecutionPlan) -> Design:
    repeats = int(manifest["design"]["repeats_per_logical_probe"])
    panel = make_tau_like_panel(repeats=repeats)
    if plan.tier == "smoke":
        indices = np.concatenate(
            [
                np.arange(
                    isoform * 128,
                    isoform * 128 + SMOKE_CANDIDATES_PER_ISOFORM,
                    dtype=np.int64,
                )
                for isoform in range(6)
            ]
        )
    else:
        indices = np.arange(panel.n_candidates, dtype=np.int64)
    logical_profiles = panel.logical_profiles[indices]
    alpha, beta = default_tau_logical_probe_rates(panel)
    base_q = np.where(logical_profiles == 1, alpha[None, :], beta[None, :])
    cycle_to_probe = np.asarray(manifest["design"]["cycle_to_probe_1_based"], dtype=np.int64) - 1
    if not np.array_equal(cycle_to_probe, panel.cycle_to_probe):
        raise ValueError("manifest cycle schedule disagrees with the tau-like panel")
    if plan.tier != "smoke":
        if panel.n_candidates != int(manifest["design"]["candidate_count"]):
            raise ValueError("manifest candidate count disagrees with generated dictionary")
        if tuple(panel.logical_probe_ids) != tuple(manifest["design"]["logical_probe_ids"]):
            raise ValueError("manifest logical probe IDs disagree with generated panel")
    return Design(
        profiles=np.asarray(logical_profiles, dtype=np.int8),
        candidate_ids=tuple(panel.candidate_ids[int(index)] for index in indices),
        isoforms=tuple(panel.isoforms[int(index)] for index in indices),
        ptm_states=np.asarray(logical_profiles[:, 5:], dtype=np.int8),
        probe_ids=tuple(panel.logical_probe_ids),
        cycle_to_probe=cycle_to_probe,
        base_Q=np.asarray(base_q, dtype=float),
        original_indices=indices,
    )


def _support_and_weights(
    manifest: dict[str, Any], design: Design, master_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = _rng(manifest, master_seed, "support_and_weights")
    isoforms = np.asarray(design.isoforms)
    mandatory = [
        int(rng.choice(np.flatnonzero(isoforms == isoform)))
        for isoform in manifest["design"]["isoforms"]
    ]
    remaining = np.setdiff1d(np.arange(design.n_candidates), np.asarray(mandatory))
    active_count = int(manifest["design"]["mixture"]["active_candidate_count"])
    extra = rng.choice(remaining, size=active_count - len(mandatory), replace=False)
    support = np.concatenate((np.asarray(mandatory), np.asarray(extra, dtype=np.int64)))
    draws = rng.dirichlet(np.full(active_count, 0.4))
    weights = np.zeros(design.n_candidates, dtype=float)
    weights[support] = draws
    return support, weights


def _heldout_support(
    manifest: dict[str, Any], design: Design, support: np.ndarray, master_seed: int
) -> np.ndarray:
    rng = _rng(manifest, master_seed, "ood_support")
    available = np.setdiff1d(np.arange(design.n_candidates), support)
    isoforms = np.asarray(design.isoforms)
    selected: list[int] = []
    for isoform in manifest["design"]["isoforms"]:
        pool = np.intersect1d(available, np.flatnonzero(isoforms == isoform))
        choice = int(rng.choice(pool))
        selected.append(choice)
        available = available[available != choice]
    selected.extend(map(int, rng.choice(available, size=2, replace=False)))
    return np.asarray(selected, dtype=np.int64)


def _prepare_seed_state(
    manifest: dict[str, Any], design: Design, plan: ExecutionPlan, master_seed: int
) -> SeedState:
    support, weights = _support_and_weights(manifest, design, master_seed)
    heldout = _heldout_support(manifest, design, support, master_seed)
    n = plan.n_molecules
    identity_uniform = _rng(manifest, master_seed, "molecule_identities").random(n)
    base_identities = _sample_from_uniform(identity_uniform, weights)
    emission_uniforms = _rng(manifest, master_seed, "emission_uniforms").random(
        (n, design.n_cycles)
    )
    missingness_uniforms = _rng(manifest, master_seed, "missingness_uniforms").random(
        (n, design.n_cycles)
    )
    q_error_normals = _rng(manifest, master_seed, "q_error_standard_normals").standard_normal(
        design.base_Q.shape
    )
    shared_effects = _rng(manifest, master_seed, "shared_repeat_effects").standard_normal(
        (n, design.n_probes)
    )
    ood_rng = _rng(manifest, master_seed, "ood_generation")
    return SeedState(
        master_seed=master_seed,
        support=support,
        base_weights=weights,
        heldout_support=heldout,
        base_identities=base_identities,
        emission_uniforms=emission_uniforms,
        missingness_uniforms=missingness_uniforms,
        q_error_normals=q_error_normals,
        shared_effects=shared_effects,
        ood_kind_uniform=ood_rng.random(n),
        ood_choice_uniform=ood_rng.random(n),
        ood_second_identity_uniform=ood_rng.random(n),
        ood_truncation_h=ood_rng.integers(1, 10, size=n),
    )


def _marginal_intercepts(q: np.ndarray, sigma: float) -> np.ndarray:
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    normal_nodes = np.sqrt(2.0) * nodes
    normal_weights = weights / np.sqrt(np.pi)
    result = np.empty_like(q)
    for value in np.unique(q):
        low, high = -40.0, 40.0
        for _ in range(200):
            midpoint = 0.5 * (low + high)
            expectation = float(
                np.dot(normal_weights, _expit(midpoint + sigma * normal_nodes))
            )
            if expectation < value:
                low = midpoint
            else:
                high = midpoint
            if high - low <= 1e-12:
                break
        result[q == value] = 0.5 * (low + high)
    return result


def _identifiability_q(spec: dict[str, Any], design: Design) -> np.ndarray:
    q = design.base_Q.copy()
    probe = design.probe_ids.index(str(spec["logical_probe"]))
    q[:, probe] = np.where(
        design.profiles[:, probe] == 1,
        float(spec["true_and_fitted_alpha"]),
        float(spec["true_and_fitted_beta"]),
    )
    return q


def _artifact_profiles(
    state: SeedState, design: Design, artifact_kind: np.ndarray
) -> np.ndarray:
    profiles = design.profiles[state.base_identities].copy()
    collision = artifact_kind == 0
    truncation = artifact_kind == 1
    background = artifact_kind == 2
    if np.any(collision):
        second = _sample_from_uniform(
            state.ood_second_identity_uniform[collision], state.base_weights
        )
        profiles[collision] = np.maximum(profiles[collision], design.profiles[second])
    if np.any(truncation):
        profiles[truncation, 1] = 0
        indices = np.flatnonzero(truncation)
        for molecule in indices:
            h = int(state.ood_truncation_h[molecule])
            profiles[molecule, 2 + h : 2 + 10] = 0
    profiles[background] = 0
    return profiles


def _dataset_hash(dataset_parts: Sequence[Any]) -> str:
    return _array_sha256(*dataset_parts)


def _build_dataset(
    scenario_id: str,
    manifest: dict[str, Any],
    design: Design,
    state: SeedState,
    cache: dict[str, Dataset],
) -> Dataset:
    if scenario_id in cache:
        return cache[scenario_id]
    spec = manifest["scenario_definitions"][scenario_id]
    if "reuse_observations_from" in spec:
        reused = _build_dataset(
            str(spec["reuse_observations_from"]), manifest, design, state, cache
        )
        cache[scenario_id] = reused
        return reused

    true_q = design.base_Q.copy()
    identities = state.base_identities.copy()
    truth_weights = state.base_weights.copy()
    ood_mask = np.zeros(identities.size, dtype=bool)
    ood_subtype = np.full(identities.size, -1, dtype=np.int8)
    probabilities: np.ndarray

    if spec["family"] in {"near_identifiability", "exact_identifiability"}:
        true_q = _identifiability_q(spec, design)

    if spec["family"] == "candidate_omission":
        heldout_fraction = float(spec["heldout_fraction"])
        heldout = state.ood_kind_uniform < heldout_fraction
        chosen = np.minimum(
            (state.ood_choice_uniform[heldout] * state.heldout_support.size).astype(int),
            state.heldout_support.size - 1,
        )
        identities[heldout] = state.heldout_support[chosen]
        truth_weights *= 1.0 - heldout_fraction
        truth_weights[state.heldout_support] += heldout_fraction / state.heldout_support.size
        ood_mask = heldout
        ood_subtype[heldout] = 3
        probabilities = true_q[identities][:, design.cycle_to_probe]
    elif spec["family"] == "out_of_dictionary_artifacts":
        fractions = spec["subtype_fractions"]
        collision_cut = float(fractions["two_molecule_collision"])
        truncation_cut = collision_cut + float(fractions["truncation"])
        background_cut = truncation_cut + float(fractions["non_target_background"])
        ood_subtype[state.ood_kind_uniform < collision_cut] = 0
        ood_subtype[
            (state.ood_kind_uniform >= collision_cut)
            & (state.ood_kind_uniform < truncation_cut)
        ] = 1
        ood_subtype[
            (state.ood_kind_uniform >= truncation_cut)
            & (state.ood_kind_uniform < background_cut)
        ] = 2
        ood_mask = ood_subtype >= 0
        profiles = _artifact_profiles(state, design, ood_subtype)
        identities[ood_mask] = -1
        alpha = np.asarray([0.97, 0.97] + [0.91] * 3 + [0.84] * 7)
        beta = np.asarray([0.002, 0.002] + [0.010] * 3 + [0.015] * 7)
        molecule_q = np.where(profiles == 1, alpha[None, :], beta[None, :])
        probabilities = molecule_q[:, design.cycle_to_probe]
        truth_weights *= float(spec["base_in_dictionary_fraction"])
    elif spec["family"] == "conditional_dependence":
        sigma = float(spec["shared_logit_standard_deviation"])
        intercepts = _marginal_intercepts(true_q, sigma)
        probabilities = _expit(
            intercepts[identities][:, design.cycle_to_probe]
            + sigma * state.shared_effects[:, design.cycle_to_probe]
        )
    else:
        probabilities = true_q[identities][:, design.cycle_to_probe]

    calls = (state.emission_uniforms < probabilities).astype(np.int8)
    missing = np.zeros_like(calls, dtype=bool)
    missing_spec = spec.get("missingness", {"kind": "complete", "probability": 0.0})
    kind = missing_spec["kind"]
    if kind in {
        "independent_mcar",
        "independent_mcar_plus_forced_cycle",
        "independent_mcar_plus_forced_logical_probe",
    }:
        missing |= state.missingness_uniforms < float(missing_spec["probability"])
    if kind == "independent_mcar_plus_forced_cycle":
        missing[:, int(missing_spec["forced_missing_cycle_id_1_based"]) - 1] = True
    if kind == "independent_mcar_plus_forced_logical_probe":
        probe = design.probe_ids.index(str(missing_spec["forced_missing_logical_probe"]))
        missing[:, design.cycle_to_probe == probe] = True
    if kind == "latent_call_dependent":
        missing_probability = np.where(
            calls == 1,
            float(missing_spec["probability_if_latent_call_1"]),
            float(missing_spec["probability_if_latent_call_0"]),
        )
        additional = float(missing_spec.get("additional_mcar_probability", 0.0))
        missing_probability = 1.0 - (1.0 - missing_probability) * (1.0 - additional)
        missing |= state.missingness_uniforms < missing_probability

    observations = calls.copy()
    observations[missing] = -1
    n_active = int(np.count_nonzero(truth_weights))
    if spec["family"] == "out_of_dictionary_artifacts":
        n_active += 3
    data_hash = _dataset_hash(
        (observations, identities, truth_weights, ood_mask, ood_subtype, true_q)
    )
    result = Dataset(
        observations=observations,
        identities=identities,
        truth_weights=truth_weights,
        ood_mask=ood_mask,
        ood_subtype=ood_subtype,
        true_Q=true_q,
        n_active_truth=n_active,
        data_sha256=data_hash,
    )
    cache[scenario_id] = result
    return result


def _fitted_q_and_indices(
    scenario_id: str,
    manifest: dict[str, Any],
    design: Design,
    state: SeedState,
    dataset: Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    spec = manifest["scenario_definitions"][scenario_id]
    fitted_q = dataset.true_Q.copy()
    if spec["family"] == "Q_misspecification":
        fitted_q = _expit(
            _logit(design.base_Q)
            + float(spec["delta"])
            + float(spec["sigma"]) * state.q_error_normals
        )
        fitted_q = np.clip(fitted_q, *map(float, spec["clip"]))
    indices = np.arange(design.n_candidates, dtype=np.int64)
    if spec["family"] == "candidate_omission":
        indices = np.setdiff1d(indices, state.heldout_support)
        fitted_q = fitted_q[indices]
    return fitted_q, indices


def _compress_base_logs(
    absolute_logs: np.ndarray,
    count_class_counts: np.ndarray,
    molecule_to_count_class: np.ndarray,
    missing_fraction: float,
) -> TraceLikelihoodClasses:
    offsets = np.max(absolute_logs, axis=1)
    if np.any(~np.isfinite(offsets)):
        raise ValueError("at least one observed class has no supported candidate")
    relative = absolute_logs - offsets[:, None]
    logs, count_to_likelihood = np.unique(relative, axis=0, return_inverse=True)
    counts = np.bincount(
        count_to_likelihood,
        weights=count_class_counts,
        minlength=logs.shape[0],
    ).astype(np.int64)
    membership = count_to_likelihood[molecule_to_count_class]
    molecule_offsets = offsets[molecule_to_count_class]
    return TraceLikelihoodClasses(
        log_likelihoods=logs,
        counts=counts,
        molecule_to_class=membership,
        molecule_log_offsets=molecule_offsets,
        log_likelihood_offset=float(np.dot(count_class_counts, offsets)),
        missing_fraction=missing_fraction,
    )


def _binary_classes(
    observations: np.ndarray, expanded_profiles: np.ndarray
) -> TraceLikelihoodClasses:
    unique, membership, counts = np.unique(
        observations, axis=0, return_inverse=True, return_counts=True
    )
    logs = alignment_profile_log_likelihoods(unique, expanded_profiles, mode="best")
    return _compress_base_logs(logs, counts, membership, float(np.mean(observations == -1)))


def _repeat_aware_classes(
    observations: np.ndarray,
    q: np.ndarray,
    cycle_to_probe: np.ndarray,
    sigma: float,
) -> TraceLikelihoodClasses:
    n, n_cycles = observations.shape
    n_probes = q.shape[1]
    positive = np.zeros((n, n_probes), dtype=np.int8)
    observed = np.zeros_like(positive)
    for probe in range(n_probes):
        selected = cycle_to_probe == probe
        calls = observations[:, selected]
        positive[:, probe] = np.sum(calls == 1, axis=1)
        observed[:, probe] = np.sum(calls != -1, axis=1)
    keys = np.concatenate((observed, positive), axis=1)
    unique, membership, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    unique_observed = unique[:, :n_probes]
    unique_positive = unique[:, n_probes:]
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    z = np.sqrt(2.0) * nodes
    gh_weights = weights / np.sqrt(np.pi)
    intercepts = _marginal_intercepts(q, sigma)
    table = np.full((q.shape[0], n_probes, 4, 4), np.nan)
    for probe in range(n_probes):
        probability = _expit(intercepts[:, probe, None] + sigma * z[None, :])
        for observed_count in range(4):
            for positive_count in range(observed_count + 1):
                integrand = probability**positive_count * (1.0 - probability) ** (
                    observed_count - positive_count
                )
                table[:, probe, observed_count, positive_count] = np.log(
                    integrand @ gh_weights
                )
    logs = np.zeros((unique.shape[0], q.shape[0]), dtype=float)
    for probe in range(n_probes):
        logs += table[
            :,
            probe,
            unique_observed[:, probe],
            unique_positive[:, probe],
        ].T
    return _compress_base_logs(logs, counts, membership, float(np.mean(observations == -1)))


def _posterior(logs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        joint = logs + np.log(weights)[None, :]
    normalizer = np.asarray(logsumexp(joint, axis=1))
    result = np.exp(joint - normalizer[:, None])
    result /= result.sum(axis=1, keepdims=True)
    return result


def _top_tie_initial(logs: np.ndarray, counts: np.ndarray) -> np.ndarray:
    values = hard_assignment_counts(logs, counts=counts, split_ties=True) + 0.5
    return values / values.sum()


def _em_fit_status(
    fit: Any,
    settings: dict[str, Any],
    schema_version: str = V1_SCHEMA_VERSION,
) -> tuple[str, str | None, str | None]:
    terminal = float(fit.diagnostics["terminal_em_residual"])
    monotonic = bool(fit.diagnostics["monotonic"])
    finite = bool(np.all(np.isfinite(fit.weights)) and np.isfinite(fit.log_likelihood))
    residual_ok = terminal <= float(settings["terminal_fixed_point_residual_maximum"])
    if schema_version in DIRECT_SVD_SCHEMA_VERSIONS:
        # V3 reports the most diagnostic numerical-safety failure even when
        # the optimizer also exhausted its iteration ceiling.  Legacy order is
        # retained below so v1/v2 archives remain reproducible.
        if not finite or not np.isfinite(terminal):
            return (
                "numerical_failure",
                "nonfinite_fit",
                "EM returned a nonfinite weight, objective, or terminal residual",
            )
        if not monotonic:
            return (
                "numerical_failure",
                "likelihood_nonmonotonic",
                "EM likelihood monotonicity invariant failed",
            )
        if not residual_ok:
            return (
                "numerical_failure",
                "terminal_residual_exceeded",
                f"terminal EM residual {terminal} exceeds the manifest maximum",
            )
        if not fit.converged:
            return (
                "nonconverged",
                "em_nonconvergence",
                "EM did not satisfy stopping rules",
            )
        return "ok", None, None
    if not fit.converged:
        return (
            "nonconverged",
            "em_nonconvergence",
            "EM did not satisfy stopping rules",
        )
    if not finite:
        return (
            "numerical_failure",
            "nonfinite_fit",
            "EM returned a nonfinite weight or objective",
        )
    if not monotonic:
        return (
            "numerical_failure",
            "likelihood_nonmonotonic",
            "EM likelihood monotonicity invariant failed",
        )
    if not residual_ok:
        return (
            "numerical_failure",
            "terminal_residual_exceeded",
            f"terminal EM residual {terminal} exceeds the manifest maximum",
        )
    return "ok", None, None


def _em_fit(
    classes: TraceLikelihoodClasses,
    manifest: dict[str, Any],
    *,
    initial_weights: np.ndarray | None = None,
    continue_warm_start_on_retry: bool = False,
) -> MethodFit:
    settings = manifest["stopping_and_failures"]["em"]
    fit = fit_likelihood_em(
        classes.log_likelihoods,
        counts=classes.counts,
        initial_weights=initial_weights,
        max_iter=int(settings["max_iterations"]),
        tol=float(settings["relative_objective_tolerance"]),
        log_likelihood_offset=classes.log_likelihood_offset,
        block_size=int(settings["block_size"]),
        return_responsibilities=False,
    )
    retry_used = False
    retry_allowed = initial_weights is None or (
        manifest["schema_version"] in DIRECT_SVD_SCHEMA_VERSIONS
        and continue_warm_start_on_retry
    )
    if not fit.converged and retry_allowed:
        retry = settings["retry_once_if_nonconverged"]
        retry_initial = (
            _top_tie_initial(classes.log_likelihoods, classes.counts)
            if initial_weights is None
            else np.asarray(fit.weights)
        )
        fit = fit_likelihood_em(
            classes.log_likelihoods,
            counts=classes.counts,
            initial_weights=retry_initial,
            max_iter=int(retry["max_iterations"]),
            tol=float(settings["relative_objective_tolerance"]),
            log_likelihood_offset=classes.log_likelihood_offset,
            block_size=int(settings["block_size"]),
            return_responsibilities=False,
        )
        retry_used = True
    terminal = float(fit.diagnostics["terminal_em_residual"])
    status, code, reason = _em_fit_status(
        fit, settings, str(manifest["schema_version"])
    )
    if retry_used and reason is not None:
        reason = f"{reason} after the declared retry"
    return MethodFit(
        weights=np.asarray(fit.weights),
        classes=classes,
        posterior_weights=np.asarray(fit.weights),
        probability_method=True,
        status=status,
        failure_code=code,
        failure_reason=reason,
        converged=bool(fit.converged),
        retry_used=retry_used,
        iterations=int(fit.n_iter),
        terminal_residual=terminal,
        minimum_gain=float(fit.diagnostics["minimum_log_likelihood_gain"]),
        last_gain=float(fit.diagnostics["last_log_likelihood_gain"]),
        max_weight_change=float(fit.diagnostics["max_weight_change"]),
    )


def _run_method(
    method_id: str,
    observations: np.ndarray,
    fitted_q: np.ndarray,
    fit_indices: np.ndarray,
    design: Design,
    scenario_spec: dict[str, Any],
    manifest: dict[str, Any],
) -> MethodFit:
    modeled = observations
    if method_id == "NA_as_zero_top_likelihood_antipattern":
        modeled = observations.copy()
        modeled[modeled == -1] = 0
    if method_id == "binary_best_hamming_em":
        profiles = design.profiles[fit_indices][:, design.cycle_to_probe]
        classes = _binary_classes(modeled, profiles)
    elif method_id == "oracle_repeat_aware_likelihood_em":
        classes = _repeat_aware_classes(
            modeled,
            fitted_q,
            design.cycle_to_probe,
            float(scenario_spec["shared_logit_standard_deviation"]),
        )
    else:
        classes = build_trace_likelihood_classes(
            modeled, fitted_q, cycle_to_probe=design.cycle_to_probe
        )

    dense_entries = classes.n_classes * classes.n_candidates
    maximum_entries = int(manifest["resources"]["maximum_dense_likelihood_entries"])
    if dense_entries > maximum_entries:
        raise ResourceLimit(
            "maximum_dense_likelihood_entries",
            f"dense likelihood entries {dense_entries} exceed manifest maximum {maximum_entries}"
        )
    if method_id in {"em_fixed_q", "binary_best_hamming_em", "oracle_repeat_aware_likelihood_em"}:
        result = _em_fit(classes, manifest)
        if (
            manifest["schema_version"] in V4_METRIC_SCHEMA_VERSIONS
            and method_id == "binary_best_hamming_em"
        ):
            # The Hamming-support likelihood is an intentionally uncalibrated
            # binary compatibility model.  Its mixture weights remain valid as
            # an abundance comparator, but its class posteriors must not be
            # exposed as calibrated assignment probabilities.
            return MethodFit(
                weights=result.weights,
                classes=result.classes,
                posterior_weights=None,
                probability_method=False,
                status=result.status,
                failure_code=result.failure_code,
                failure_reason=result.failure_reason,
                converged=result.converged,
                retry_used=result.retry_used,
                iterations=result.iterations,
                terminal_residual=result.terminal_residual,
                minimum_gain=result.minimum_gain,
                last_gain=result.last_gain,
                max_weight_change=result.max_weight_change,
            )
        return result
    if method_id == "soft_uniform_one_step":
        uniform = np.full(classes.n_candidates, 1.0 / classes.n_candidates)
        responsibilities = _posterior(classes.log_likelihoods, uniform)
        weights = np.sum(classes.counts[:, None] * responsibilities, axis=0)
        weights /= weights.sum()
        return MethodFit(
            weights, classes, uniform, True, "ok", None, None, None,
            False, 1, None, None, None, None,
        )
    if method_id in {"top_likelihood_equal_ties", "NA_as_zero_top_likelihood_antipattern"}:
        weights = hard_assignment_counts(
            classes.log_likelihoods, counts=classes.counts, split_ties=True
        )
        weights /= weights.sum()
        return MethodFit(
            weights, classes, None, False, "ok", None, None, None,
            False, 0, None, None, None, None,
        )
    if method_id == "direct_constrained_mixture_optimizer":
        # The independent optimizer is a certification path, not a production
        # estimator.  Starting it at the production optimum makes the debug
        # check practical at K=768 while its projected-gradient residual and
        # objective are still evaluated by independent code.
        em_start = _em_fit(classes, manifest)
        direct = maximize_fixed_mixture_on_simplex(
            classes.log_likelihoods,
            counts=classes.counts,
            initial_weights=em_start.weights,
            max_iter=50_000,
            tol=1e-7,
        )
        status = "ok" if direct.converged else "nonconverged"
        return MethodFit(
            np.asarray(direct.weights),
            classes,
            np.asarray(direct.weights),
            True,
            status,
            None if direct.converged else "optimizer_nonconvergence",
            None if direct.converged else "direct constrained optimizer did not converge",
            bool(direct.converged),
            False,
            int(direct.n_iter),
            float(direct.projected_gradient_residual),
            None,
            None,
            None,
        )
    raise ValueError(f"unsupported frozen-manifest method: {method_id}")


def _observable_groups(
    fitted_q: np.ndarray, observations: np.ndarray, cycle_to_probe: np.ndarray
) -> tuple[tuple[int, ...], ...]:
    observed = np.asarray(
        [
            np.any(observations[:, cycle_to_probe == probe] != -1)
            for probe in range(fitted_q.shape[1])
        ],
        dtype=bool,
    )
    if not np.any(observed):
        return (tuple(range(fitted_q.shape[0])),)
    return find_observable_groups(fitted_q[:, observed]).groups


def _tv(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(estimate - truth)))


def _js_base2(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if estimate.ndim != 1 or estimate.shape != truth.shape or estimate.size == 0:
        raise ValueError("Jensen-Shannon inputs must be non-empty vectors of equal length")
    if (
        not np.all(np.isfinite(estimate))
        or not np.all(np.isfinite(truth))
        or np.any(estimate < 0)
        or np.any(truth < 0)
    ):
        raise ValueError("Jensen-Shannon inputs must be finite and non-negative")
    if not np.isclose(estimate.sum(), 1.0, rtol=0.0, atol=1e-12) or not np.isclose(
        truth.sum(), 1.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Jensen-Shannon inputs must each sum to one")

    def contribution(left: np.ndarray, right: np.ndarray) -> float:
        selected = left > 0
        positive = left[selected]
        # log2(left / ((left + right) / 2)) is evaluated in log space.
        # Forming the midpoint first can underflow to zero when ``left`` is
        # the smallest positive subnormal, turning the valid 0*log term into
        # an infinity that later violates strict JSON serialization.
        log_ratio = (
            1.0
            + np.log2(positive)
            - np.log2(positive + right[selected])
        )
        return float(np.dot(positive, log_ratio))

    divergence = 0.5 * (
        contribution(estimate, truth) + contribution(truth, estimate)
    )
    tolerance = 64.0 * np.finfo(float).eps
    if -tolerance <= divergence < 0.0:
        divergence = 0.0
    elif 1.0 < divergence <= 1.0 + tolerance:
        divergence = 1.0
    if not np.isfinite(divergence) or divergence < 0.0 or divergence > 1.0:
        raise FloatingPointError(
            f"Jensen-Shannon divergence fell outside [0, 1]: {divergence}"
        )
    return float(divergence)


def _truth_mapping(design: Design, fit_indices: np.ndarray) -> np.ndarray:
    mapping = np.full(design.n_candidates, -1, dtype=np.int64)
    mapping[fit_indices] = np.arange(fit_indices.size, dtype=np.int64)
    return mapping


def _topk_recall(
    classes: TraceLikelihoodClasses,
    posterior_weights: np.ndarray,
    identities: np.ndarray,
    mapping: np.ndarray,
    k: int,
) -> float | None:
    truth = np.where(identities >= 0, mapping[np.maximum(identities, 0)], -1)
    valid = truth >= 0
    if not np.any(valid):
        return None
    posterior = _posterior(classes.log_likelihoods, posterior_weights)
    width = min(k, posterior.shape[1])
    top = np.argpartition(posterior, -width, axis=1)[:, -width:]
    molecule_top = top[classes.molecule_to_class[valid]]
    return float(np.mean(np.any(molecule_top == truth[valid, None], axis=1)))


def _hard_top1(
    classes: TraceLikelihoodClasses, identities: np.ndarray, mapping: np.ndarray
) -> float | None:
    truth = np.where(identities >= 0, mapping[np.maximum(identities, 0)], -1)
    valid = truth >= 0
    if not np.any(valid):
        return None
    labels = np.argmax(classes.log_likelihoods, axis=1)
    return float(np.mean(labels[classes.molecule_to_class[valid]] == truth[valid]))


def _feature_metrics(
    estimate: np.ndarray,
    fit_indices: np.ndarray,
    truth: np.ndarray,
    design: Design,
    schema_version: str = V1_SCHEMA_VERSION,
) -> dict[str, float | None]:
    isoforms = tuple(dict.fromkeys(design.isoforms))
    isoform_array = np.asarray(design.isoforms)
    estimated_isoform = np.asarray(
        [estimate[isoform_array[fit_indices] == label].sum() for label in isoforms]
    )
    true_isoform = np.asarray([truth[isoform_array == label].sum() for label in isoforms])
    estimated_ptm = estimate @ design.ptm_states[fit_indices]
    true_ptm = truth @ design.ptm_states
    result = {
        "isoform_total_variation": _tv(estimated_isoform, true_isoform),
        "PTM_marginal_MAE": float(np.mean(np.abs(estimated_ptm - true_ptm))),
    }
    if schema_version in V4_METRIC_SCHEMA_VERSIONS:
        left, right = np.triu_indices(design.ptm_states.shape[1], k=1)
        fitted_pairs = (
            design.ptm_states[fit_indices][:, left]
            * design.ptm_states[fit_indices][:, right]
        )
        truth_pairs = design.ptm_states[:, left] * design.ptm_states[:, right]
        estimated_pair_abundance = estimate @ fitted_pairs
        true_pair_abundance = truth @ truth_pairs
        result["PTM_pair_MAE"] = float(
            np.mean(np.abs(estimated_pair_abundance - true_pair_abundance))
        )
    return result


def _posterior_entropy(
    classes: TraceLikelihoodClasses,
    weights: np.ndarray,
    groups: Sequence[Sequence[int]],
    true_identities: np.ndarray,
    true_to_decoder: np.ndarray,
    *,
    group_resolution: bool,
    block_size: int,
) -> tuple[float | None, int]:
    """Return mean entropy over molecules whose truth is in the fitted set."""

    identities = np.asarray(true_identities, dtype=np.int64)
    mapping = np.asarray(true_to_decoder, dtype=np.int64)
    valid = mapping[identities] >= 0
    valid_class_counts = np.bincount(
        classes.molecule_to_class[valid], minlength=classes.n_classes
    ).astype(np.int64)
    valid_total = int(valid_class_counts.sum())
    if not valid_total:
        return None, 0
    total = 0.0
    for start in range(0, classes.n_classes, block_size):
        stop = min(start + block_size, classes.n_classes)
        posterior = _posterior(classes.log_likelihoods[start:stop], weights)
        if group_resolution:
            posterior = np.column_stack(
                [posterior[:, list(group)].sum(axis=1) for group in groups]
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(posterior > 0.0, posterior * np.log(posterior), 0.0)
        entropy = -np.sum(terms, axis=1)
        total += float(np.dot(valid_class_counts[start:stop], entropy))
    return total / valid_total, valid_total


def _detection_metrics(
    estimate: np.ndarray,
    fit_indices: np.ndarray,
    truth: np.ndarray,
    n_molecules: int,
) -> tuple[float | None, float, int, int, int, int]:
    """Score candidate detection at the frozen expected-count threshold."""

    estimated_full = np.zeros_like(truth, dtype=float)
    estimated_full[fit_indices] = estimate
    truth_detected = n_molecules * truth >= 20.0
    estimate_detected = n_molecules * estimated_full >= 20.0
    true_positive = int(np.sum(truth_detected & estimate_detected))
    false_positive = int(np.sum(~truth_detected & estimate_detected))
    n_truth = int(np.sum(truth_detected))
    n_called = true_positive + false_positive
    sensitivity = true_positive / n_truth if n_truth else None
    false_discovery_rate = false_positive / n_called if n_called else 0.0
    return (
        float(sensitivity) if sensitivity is not None else None,
        float(false_discovery_rate),
        n_truth,
        n_called,
        true_positive,
        false_positive,
    )


def _maximum_absolute_error_with_ood(
    estimate: np.ndarray,
    truth_fit: np.ndarray,
    truth_out_of_dictionary_mass: float,
) -> float:
    """Return max abundance error including the aggregate OOD coordinate."""

    difference = np.append(
        np.asarray(estimate, dtype=float) - np.asarray(truth_fit, dtype=float),
        -float(truth_out_of_dictionary_mass),
    )
    return float(np.max(np.abs(difference)))


def _rank_auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=float)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_precision(scores: np.ndarray, positive: np.ndarray) -> float | None:
    positive = np.asarray(positive, dtype=bool)
    if not np.any(positive) or np.all(positive):
        return None
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    labels = positive[order]
    precision = np.cumsum(labels) / np.arange(1, labels.size + 1)
    return float(np.sum(precision * labels) / labels.sum())


def _ood_metrics(
    manifest_id: str,
    run_id: str,
    tier: str,
    scenario_id: str,
    method_id: str,
    master_seed: int,
    fit: MethodFit,
    dataset: Dataset,
    observations: np.ndarray,
) -> dict[str, Any] | None:
    if fit.posterior_weights is None or not np.any(dataset.ood_mask):
        return None
    posterior = _posterior(fit.classes.log_likelihoods, fit.posterior_weights)
    confidence = np.max(posterior, axis=1)[fit.classes.molecule_to_class]
    observed_calls = np.maximum(np.sum(observations != -1, axis=1), 1)
    maximum_log_likelihood = fit.classes.molecule_log_offsets
    negative_max_per_call = -maximum_log_likelihood / observed_calls
    ood_score_posterior = -confidence
    ood = dataset.ood_mask
    return {
        "manifest_id": manifest_id,
        "run_id": run_id,
        "tier": tier,
        "scenario_id": scenario_id,
        "method_id": method_id,
        "master_seed": master_seed,
        "n_in_dictionary": int((~ood).sum()),
        "n_ood": int(ood.sum()),
        "AUROC_maximum_posterior": _rank_auc(ood_score_posterior, ood),
        "AUPRC_maximum_posterior": _average_precision(ood_score_posterior, ood),
        "AUROC_negative_maximum_log_likelihood_per_observed_call": _rank_auc(negative_max_per_call, ood),
        "AUPRC_negative_maximum_log_likelihood_per_observed_call": _average_precision(negative_max_per_call, ood),
        "false_accept_fraction_at_maximum_posterior_at_least_0_80": float(np.mean(confidence[ood] >= 0.80)),
        "mean_maximum_posterior_in_dictionary": float(np.mean(confidence[~ood])),
        "mean_maximum_posterior_ood": float(np.mean(confidence[ood])),
    }


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _blank_run(
    manifest: dict[str, Any],
    plan: ExecutionPlan,
    scenario_id: str,
    method_id: str,
    master_seed: int,
    seed_index: int,
) -> dict[str, Any]:
    spec = manifest["scenario_definitions"][scenario_id]
    run_id = manifest["outputs"]["run_id_template"].format(
        tier=plan.tier,
        scenario_id=scenario_id,
        seed_index_1_based=seed_index,
        master_seed=master_seed,
        biological_index_1_based=1,
        technical_index_1_based=1,
        method_id=method_id,
    )
    return {
        "manifest_id": manifest["manifest_id"],
        "run_id": run_id,
        "tier": plan.tier,
        "scenario_id": scenario_id,
        "scenario_family": spec["family"],
        "paired_reference": spec["paired_reference"],
        "method_id": method_id,
        "master_seed": master_seed,
        "seed_index_1_based": seed_index,
        "biological_replicate_id": "bio01",
        "technical_run_id": "tech01",
        "n_source": plan.n_molecules,
        "n_accepted": plan.n_molecules,
        "n_logical_probes": int(manifest["design"]["logical_probe_ids"].__len__()),
        "n_cycles": int(manifest["design"]["cycle_count"]),
        "status": "exception",
        "retry_used": False,
        "wall_seconds": 0.0,
        "peak_RSS_bytes": _peak_rss_bytes(),
        "metrics_valid": False,
    }


def _result_hash(row: dict[str, Any]) -> str:
    excluded = {"wall_seconds", "peak_RSS_bytes", "result_sha256"}
    stable = {key: value for key, value in row.items() if key not in excluded}
    return _sha256_bytes(_canonical_json(stable).encode())


def _run_one_method(
    manifest: dict[str, Any],
    plan: ExecutionPlan,
    design: Design,
    state: SeedState,
    dataset: Dataset,
    scenario_id: str,
    method_id: str,
    master_seed: int,
    seed_index: int,
    *,
    point_wall_seconds: float | None = None,
) -> tuple[
    dict[str, Any],
    MethodFit | None,
    np.ndarray | None,
    tuple[tuple[int, ...], ...] | None,
    list[dict[str, Any]],
    dict[str, Any] | None,
    dict[str, float | None] | None,
]:
    row = _blank_run(manifest, plan, scenario_id, method_id, master_seed, seed_index)
    reliability: list[dict[str, Any]] = []
    ood_row: dict[str, Any] | None = None
    feature_row: dict[str, float | None] | None = None
    deadline_context: Any | None = None
    start = time.perf_counter()
    try:
        point_limit = (
            float(manifest["resources"]["point_estimate_wall_seconds_per_scenario_seed"])
            if point_wall_seconds is None
            else float(point_wall_seconds)
        )
        if point_limit <= 0:
            raise ResourceLimit(
                "point_estimate_wall_seconds",
                f"shared scenario-seed point-estimate wall budget was exhausted before {method_id}",
            )
        deadline_context = _deadline(point_limit, "point_estimate_wall_seconds")
        deadline_context.__enter__()
        fitted_q, fit_indices = _fitted_q_and_indices(
            scenario_id, manifest, design, state, dataset
        )
        row["n_candidates_fit"] = int(fit_indices.size)
        row["n_active_truth"] = dataset.n_active_truth
        row["missing_fraction"] = float(np.mean(dataset.observations == -1))
        fit = _run_method(
            method_id,
            dataset.observations,
            fitted_q,
            fit_indices,
            design,
            manifest["scenario_definitions"][scenario_id],
            manifest,
        )
        peak_after_fit = _peak_rss_bytes()
        maximum_rss = int(manifest["resources"]["maximum_rss_bytes_per_worker"])
        if peak_after_fit > maximum_rss:
            raise ResourceLimit(
                "maximum_rss_bytes_per_worker",
                f"worker peak RSS {peak_after_fit} exceeds manifest maximum {maximum_rss}",
            )
        row.update(
            {
                "trace_likelihood_classes": fit.classes.n_classes,
                "dense_entries_per_iteration": fit.classes.n_classes * fit.classes.n_candidates,
                "status": fit.status,
                "failure_code": fit.failure_code,
                "failure_reason": fit.failure_reason,
                "converged": fit.converged,
                "retry_used": fit.retry_used,
                "iterations": fit.iterations,
                "terminal_em_residual": fit.terminal_residual,
                "minimum_log_likelihood_gain": fit.minimum_gain,
                "data_sha256": dataset.data_sha256,
            }
        )
        if manifest["schema_version"] in DIRECT_SVD_SCHEMA_VERSIONS:
            row.update(
                {
                    "last_log_likelihood_gain": fit.last_gain,
                    "max_weight_change": fit.max_weight_change,
                }
            )
        groups = _observable_groups(fitted_q, dataset.observations, design.cycle_to_probe)
        unresolved = any(len(group) > 1 for group in groups)
        metrics_valid = fit.status == "ok"
        row["metrics_valid"] = metrics_valid
        if metrics_valid:
            truth_fit = dataset.truth_weights[fit_indices]
            ood_mass = max(0.0, 1.0 - float(truth_fit.sum()))
            if manifest["schema_version"] in V4_METRIC_SCHEMA_VERSIONS:
                row["abundance_resolution"] = (
                    "group" if unresolved else "candidate"
                )
                row["truth_out_of_dictionary_mass"] = ood_mass
            truth_augmented = np.append(truth_fit, ood_mass)
            estimate_augmented = np.append(fit.weights, 0.0)
            group_estimate = np.asarray(
                [fit.weights[list(group)].sum() for group in groups]
            )
            group_truth = np.asarray(
                [truth_fit[list(group)].sum() for group in groups]
            )
            group_tv = _tv(np.append(group_estimate, 0.0), np.append(group_truth, ood_mass))
            if not unresolved:
                candidate_difference = estimate_augmented - truth_augmented
                candidate_metrics = {
                    "total_variation": _tv(estimate_augmented, truth_augmented),
                    "jensen_shannon_base2": _js_base2(
                        estimate_augmented, truth_augmented
                    ),
                    "candidate_MAE": float(np.mean(np.abs(candidate_difference))),
                    "candidate_RMSE": float(
                        np.sqrt(np.mean(candidate_difference**2))
                    ),
                }
                if manifest["schema_version"] in V4_METRIC_SCHEMA_VERSIONS:
                    (
                        sensitivity,
                        false_discovery_rate,
                        n_truth_detectable,
                        n_called_detected,
                        n_true_positive_detected,
                        n_false_positive_detected,
                    ) = _detection_metrics(
                        fit.weights,
                        fit_indices,
                        dataset.truth_weights,
                        plan.n_molecules,
                    )
                    candidate_metrics.update(
                        {
                            "maximum_absolute_error": (
                                _maximum_absolute_error_with_ood(
                                    fit.weights,
                                    truth_fit,
                                    ood_mass,
                                )
                            ),
                            "detection_sensitivity": sensitivity,
                            "detection_false_discovery_rate": (
                                false_discovery_rate
                            ),
                            "n_truth_detectable": n_truth_detectable,
                            "n_called_detected": n_called_detected,
                            "n_true_positive_detected": n_true_positive_detected,
                            "n_false_positive_detected": (
                                n_false_positive_detected
                            ),
                        }
                    )
                row.update(candidate_metrics)
            row["group_total_variation"] = group_tv
            feature_row = _feature_metrics(
                fit.weights,
                fit_indices,
                dataset.truth_weights,
                design,
                str(manifest["schema_version"]),
            )
            if manifest["schema_version"] in V4_METRIC_SCHEMA_VERSIONS and unresolved:
                ptm = design.ptm_states[fit_indices]
                if any(
                    not np.all(ptm[list(group)] == ptm[group[0]])
                    for group in groups
                ):
                    raise FloatingPointError(
                        "exact observable group combines different PTM vectors; "
                        "feature-level PTM metrics are not identifiable"
                    )
                feature_row["isoform_total_variation"] = None
            if (
                manifest["schema_version"] in V4_METRIC_SCHEMA_VERSIONS
                and scenario_id == "O3_artifact_mix05"
            ):
                # Artifact profiles have no preserved isoform/PTM truth in the
                # frozen dataset object, so feature-level scores are undefined.
                feature_row = None
            mapping = _truth_mapping(design, fit_indices)
            if fit.probability_method and fit.posterior_weights is not None:
                assignment_identities = np.where(
                    dataset.identities >= 0,
                    dataset.identities,
                    design.n_candidates,
                )
                assignment_mapping = np.append(mapping, -1)
                assignment, bins = probability_assignment_metrics(
                    fit.classes,
                    fit.posterior_weights,
                    assignment_identities,
                    true_to_decoder=assignment_mapping,
                    observable_groups=groups,
                    n_bins=10,
                    binning="equal_frequency",
                    block_size=int(manifest["stopping_and_failures"]["em"]["block_size"]),
                )
                resolution = "group" if unresolved else "candidate"
                row.update(
                    {
                        "log_score": assignment[f"{resolution}_assignment_log_score"],
                        "Brier_score": assignment[f"{resolution}_assignment_brier"],
                        "ECE_10_equal_frequency": assignment[f"{resolution}_assignment_ece"],
                        "top1_accuracy": assignment[f"{resolution}_assignment_accuracy"],
                        "top5_recall": _topk_recall(
                            fit.classes,
                            fit.posterior_weights,
                            dataset.identities,
                            mapping,
                            5,
                        ),
                    }
                )
                if manifest["schema_version"] in V4_METRIC_SCHEMA_VERSIONS:
                    (
                        row["posterior_entropy"],
                        row["posterior_entropy_evaluated_n"],
                    ) = _posterior_entropy(
                        fit.classes,
                        fit.posterior_weights,
                        groups,
                        assignment_identities,
                        assignment_mapping,
                        group_resolution=unresolved,
                        block_size=int(
                            manifest["stopping_and_failures"]["em"]["block_size"]
                        ),
                    )
                    row["posterior_entropy_resolution"] = (
                        "group" if unresolved else "candidate"
                    )
                for item in bins:
                    level = str(item["level"])
                    reliability.append(
                        {
                            "manifest_id": manifest["manifest_id"],
                            "run_id": row["run_id"],
                            "tier": plan.tier,
                            "scenario_id": scenario_id,
                            "method_id": method_id,
                            "master_seed": master_seed,
                            "resolution": level,
                            "reportable": level == "group" or not unresolved,
                            **item,
                        }
                    )
                ood_row = _ood_metrics(
                    manifest["manifest_id"],
                    row["run_id"],
                    plan.tier,
                    scenario_id,
                    method_id,
                    master_seed,
                    fit,
                    dataset,
                    dataset.observations,
                )
            else:
                row["top1_accuracy"] = _hard_top1(
                    fit.classes, dataset.identities, mapping
                )
        row["result_sha256"] = _result_hash(row)
        return row, fit, fit_indices, groups, reliability, ood_row, feature_row
    except ResourceLimit as exc:
        reason = str(exc)
        if exc.code == "point_estimate_wall_seconds" and point_wall_seconds is not None:
            reason = (
                "shared scenario-seed point-estimate wall budget was exhausted "
                f"while scheduling or running {method_id}; remaining allowance at "
                f"method start was {max(float(point_wall_seconds), 0.0):.6g} seconds"
            )
        row.update(
            {
                "status": "resource_skipped",
                "failure_code": exc.code,
                "failure_reason": reason,
                "data_sha256": dataset.data_sha256,
                "metrics_valid": False,
            }
        )
    except MemoryError as exc:
        row.update(
            {
                "status": "resource_skipped",
                "failure_code": "maximum_rss_bytes_per_worker",
                "failure_reason": str(exc) or "worker allocation failed under the RSS/address-space limit",
                "data_sha256": dataset.data_sha256,
                "metrics_valid": False,
            }
        )
    except FloatingPointError as exc:
        row.update(
            {
                "status": "numerical_failure",
                "failure_code": type(exc).__name__,
                "failure_reason": str(exc),
                "data_sha256": dataset.data_sha256,
                "metrics_valid": False,
            }
        )
    except Exception as exc:
        row.update(
            {
                "status": "exception",
                "failure_code": type(exc).__name__,
                "failure_reason": str(exc),
                "data_sha256": dataset.data_sha256,
                "metrics_valid": False,
            }
        )
    finally:
        if deadline_context is not None:
            deadline_context.__exit__(None, None, None)
        row["wall_seconds"] = time.perf_counter() - start
        row["peak_RSS_bytes"] = _peak_rss_bytes()
    row["result_sha256"] = _result_hash(row)
    return row, None, None, None, reliability, ood_row, feature_row


def _method_ids(
    manifest: dict[str, Any], plan: ExecutionPlan, scenario_id: str
) -> tuple[str, ...]:
    unique = _declared_method_ids(manifest, plan.scientific_tier, scenario_id)
    if plan.methods_filter is None:
        return unique
    unknown = set(plan.methods_filter) - set(unique)
    if unknown:
        raise ValueError(
            f"methods are not declared for {plan.scientific_tier}:{scenario_id}: {sorted(unknown)}"
        )
    return tuple(method for method in unique if method in plan.methods_filter)


def _declared_method_ids(
    manifest: dict[str, Any], scientific_tier: str, scenario_id: str
) -> tuple[str, ...]:
    methods = list(manifest["methods"]["run_on_all_scenarios"])
    targeted = manifest["methods"]["targeted_methods"]
    methods.extend(targeted.get(scenario_id, []))
    methods.extend(targeted.get(f"{scientific_tier}:{scenario_id}", []))
    return tuple(dict.fromkeys(methods))


def _bootstrap_estimands(
    weights: np.ndarray,
    truth: np.ndarray,
    fit_indices: np.ndarray,
    groups: Sequence[Sequence[int]],
    design: Design,
    n_molecules: int,
) -> list[tuple[str, str, np.ndarray, float]]:
    result: list[tuple[str, str, np.ndarray, float]] = []
    isoform_array = np.asarray(design.isoforms)
    for isoform in dict.fromkeys(design.isoforms):
        members = np.flatnonzero(isoform_array[fit_indices] == isoform)
        result.append(("isoform_total", str(isoform), members, float(truth[fit_indices[members]].sum())))
    for feature_index, feature in enumerate(
        ("pT181", "pS202_pT205", "pT205", "pS214", "pT217", "pT231", "pS396")
    ):
        members = np.flatnonzero(design.ptm_states[fit_indices, feature_index] == 1)
        result.append(("PTM_marginal", feature, members, float(truth[fit_indices[members]].sum())))
    candidate_to_group = np.empty(fit_indices.size, dtype=np.int64)
    for group_index, group in enumerate(groups):
        candidate_to_group[list(group)] = group_index
    for decoder_index, original_index in enumerate(fit_indices):
        if (
            truth[original_index] > 0
            and n_molecules * truth[original_index] >= 20
            and len(groups[int(candidate_to_group[decoder_index])]) == 1
        ):
            result.append(
                (
                    "active_singleton_candidate",
                    design.candidate_ids[original_index],
                    np.asarray([decoder_index]),
                    float(truth[original_index]),
                )
            )
    for group_index, group in enumerate(groups):
        original = fit_indices[list(group)]
        if len(group) > 1 and np.any(truth[original] > 0):
            result.append(
                (
                    "active_exact_observable_group",
                    f"group_{group_index}",
                    np.asarray(group, dtype=np.int64),
                    float(truth[original].sum()),
                )
            )
    return result


def _fit_bootstrap_em(
    log_likelihoods: np.ndarray,
    sampled_counts: np.ndarray,
    original_weights: np.ndarray,
    settings: dict[str, Any],
    schema_version: str = V1_SCHEMA_VERSION,
) -> tuple[Any, dict[str, Any]]:
    """Fit one fixed-Q bootstrap replicate with the manifest EM retry."""

    initial = fit_likelihood_em(
        log_likelihoods,
        counts=sampled_counts,
        initial_weights=original_weights,
        max_iter=int(settings["max_iterations"]),
        tol=float(settings["relative_objective_tolerance"]),
        block_size=int(settings["block_size"]),
        return_responsibilities=False,
    )
    initial_status, _, _ = _em_fit_status(initial, settings, schema_version)
    final = initial
    retry_used = False
    if not initial.converged:
        retry = settings["retry_once_if_nonconverged"]
        final = fit_likelihood_em(
            log_likelihoods,
            counts=sampled_counts,
            initial_weights=_top_tie_initial(log_likelihoods, sampled_counts),
            max_iter=int(retry["max_iterations"]),
            tol=float(settings["relative_objective_tolerance"]),
            block_size=int(settings["block_size"]),
            return_responsibilities=False,
        )
        retry_used = True
    status, failure_code, failure_reason = _em_fit_status(
        final, settings, schema_version
    )
    if retry_used and failure_reason is not None:
        failure_reason = f"{failure_reason} after the declared bootstrap retry"
    return final, {
        "status": status,
        "retry_used": retry_used,
        "initial_status": initial_status,
        "initial_iterations": int(initial.n_iter),
        "initial_terminal_em_residual": float(
            initial.diagnostics["terminal_em_residual"]
        ),
        "initial_last_log_likelihood_gain": float(
            initial.diagnostics["last_log_likelihood_gain"]
        ),
        "initial_max_weight_change": float(
            initial.diagnostics["max_weight_change"]
        ),
        "final_iterations": int(final.n_iter),
        "final_terminal_em_residual": float(
            final.diagnostics["terminal_em_residual"]
        ),
        "final_last_log_likelihood_gain": float(
            final.diagnostics["last_log_likelihood_gain"]
        ),
        "final_max_weight_change": float(
            final.diagnostics["max_weight_change"]
        ),
        "failure_code": failure_code,
        "failure_reason": failure_reason,
    }


def _bootstrap_run_unbounded(
    manifest: dict[str, Any],
    plan: ExecutionPlan,
    row: dict[str, Any],
    fit: MethodFit,
    fit_indices: np.ndarray,
    groups: Sequence[Sequence[int]],
    design: Design,
    dataset: Dataset,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    attempts = plan.bootstrap_resamples
    replicas = np.full((attempts, fit.weights.size), np.nan)
    failures: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    settings = manifest["stopping_and_failures"]["em"]
    start = time.perf_counter()
    wall_limit = float(manifest["resources"]["bootstrap_wall_seconds_per_scenario_seed"])
    probabilities = fit.classes.counts / fit.classes.counts.sum()
    for index in range(attempts):
        if time.perf_counter() - start > wall_limit:
            failures.append(
                {
                    "manifest_id": manifest["manifest_id"],
                    "run_id": row["run_id"],
                    "tier": plan.tier,
                    "scenario_id": row["scenario_id"],
                    "method_id": row["method_id"],
                    "master_seed": row["master_seed"],
                    "stage": "bootstrap",
                    "bootstrap_index_0_based": index,
                    "failure_code": "bootstrap_wall_limit",
                    "failure_reason": f"manifest wall limit {wall_limit} seconds reached",
                }
            )
            break
        rng = _rng(
            manifest,
            int(row["master_seed"]),
            "bootstrap_multinomial",
            bootstrap_index=index,
        )
        sampled = rng.multinomial(fit.classes.n_molecules, probabilities)
        diagnostic_base = {
            "manifest_id": manifest["manifest_id"],
            "run_id": row["run_id"],
            "tier": plan.tier,
            "scenario_id": row["scenario_id"],
            "method_id": row["method_id"],
            "master_seed": row["master_seed"],
            "bootstrap_index_0_based": index,
        }
        try:
            replicate, diagnostic = _fit_bootstrap_em(
                fit.classes.log_likelihoods,
                sampled,
                fit.weights,
                settings,
                str(manifest["schema_version"]),
            )
            diagnostics.append({**diagnostic_base, **diagnostic})
            if diagnostic["status"] == "ok":
                replicas[index] = replicate.weights
            else:
                failures.append(
                    {
                        **diagnostic_base,
                        "stage": "bootstrap",
                        "failure_code": diagnostic["failure_code"],
                        "failure_reason": diagnostic["failure_reason"],
                    }
                )
        except ResourceLimit:
            # The scenario-seed bootstrap deadline is fatal and must reach the
            # outer handler rather than being mislabeled as a tolerated
            # individual-resample exception.
            raise
        except MemoryError as exc:
            diagnostics.append(
                {
                    **diagnostic_base,
                    "status": "resource_skipped",
                    "retry_used": None,
                    "initial_status": None,
                    "initial_iterations": None,
                    "initial_terminal_em_residual": None,
                    "initial_last_log_likelihood_gain": None,
                    "initial_max_weight_change": None,
                    "final_iterations": None,
                    "final_terminal_em_residual": None,
                    "final_last_log_likelihood_gain": None,
                    "final_max_weight_change": None,
                    "failure_code": "maximum_rss_bytes_per_worker",
                    "failure_reason": str(exc) or "bootstrap allocation failed",
                }
            )
            failures.append(
                {
                    **diagnostic_base,
                    "stage": "bootstrap",
                    "failure_code": "maximum_rss_bytes_per_worker",
                    "failure_reason": str(exc) or "bootstrap allocation failed",
                }
            )
            break
        except Exception as exc:
            diagnostics.append(
                {
                    **diagnostic_base,
                    "status": "exception",
                    "retry_used": None,
                    "initial_status": None,
                    "initial_iterations": None,
                    "initial_terminal_em_residual": None,
                    "initial_last_log_likelihood_gain": None,
                    "initial_max_weight_change": None,
                    "final_iterations": None,
                    "final_terminal_em_residual": None,
                    "final_last_log_likelihood_gain": None,
                    "final_max_weight_change": None,
                    "failure_code": type(exc).__name__,
                    "failure_reason": str(exc),
                }
            )
            failures.append(
                {
                    **diagnostic_base,
                    "stage": "bootstrap",
                    "failure_code": type(exc).__name__,
                    "failure_reason": str(exc),
                }
            )
    successful = np.all(np.isfinite(replicas), axis=1)
    n_success = int(successful.sum())
    success_fraction = n_success / attempts
    interval_valid = success_fraction >= float(manifest["bootstrap"]["minimum_success_fraction"])
    if manifest["schema_version"] in MODERN_SCHEMA_VERSIONS and not interval_valid:
        fatal_bootstrap_codes = {
            code
            for rule in manifest["stopping_and_failures"][
                "systematic_failure_stop_policy"
            ]["fatal_failure_rules"]
            if rule["stage"] == "bootstrap"
            for code in rule["failure_codes"]
        }
        fatal_resource_failure = any(
            failure.get("stage") == "bootstrap"
            and failure.get("failure_code") in fatal_bootstrap_codes
            for failure in failures
        )
        if not fatal_resource_failure:
            failures.append(
                {
                    "manifest_id": manifest["manifest_id"],
                    "run_id": row["run_id"],
                    "tier": plan.tier,
                    "scenario_id": row["scenario_id"],
                    "method_id": row["method_id"],
                    "master_seed": row["master_seed"],
                    "stage": "bootstrap_summary",
                    "bootstrap_index_0_based": None,
                    "failure_code": "bootstrap_success_fraction_below_minimum",
                    "failure_reason": (
                        f"{n_success}/{attempts} successful bootstrap refits gives "
                        f"{success_fraction:.6g}, below the declared minimum "
                        f"{manifest['bootstrap']['minimum_success_fraction']}"
                    ),
                }
            )
    confidence = float(manifest["bootstrap"]["confidence_level"])
    tail = (1.0 - confidence) * 50.0
    rows: list[dict[str, Any]] = []
    for estimand_type, estimand_id, members, truth_value in _bootstrap_estimands(
        fit.weights,
        dataset.truth_weights,
        fit_indices,
        groups,
        design,
        plan.n_molecules,
    ):
        point = float(fit.weights[members].sum())
        values = replicas[successful][:, members].sum(axis=1) if n_success else np.asarray([])
        if n_success:
            lower, upper = np.percentile(values, [tail, 100.0 - tail], method="linear")
            lower, upper = float(lower), float(upper)
        else:
            lower = upper = None
        rows.append(
            {
                "manifest_id": manifest["manifest_id"],
                "run_id": row["run_id"],
                "tier": plan.tier,
                "scenario_id": row["scenario_id"],
                "master_seed": row["master_seed"],
                "method_id": row["method_id"],
                "estimand_type": estimand_type,
                "estimand_id": estimand_id,
                "members": "|".join(map(str, fit_indices[members].tolist())),
                "truth": truth_value,
                "point_estimate": point,
                "lower": lower,
                "upper": upper,
                "width": upper - lower if lower is not None else None,
                "covered": bool(lower <= truth_value <= upper) if lower is not None else None,
                "n_successful": n_success,
                "n_attempted": attempts,
                "success_fraction": success_fraction,
                "interval_valid": interval_valid,
                "confidence_level": confidence,
                "interval_type": manifest["bootstrap"]["interval_type"],
                "conditional_on": INTERVAL_SCOPE,
            }
        )
    return rows, failures, diagnostics


def _bootstrap_run(
    manifest: dict[str, Any],
    plan: ExecutionPlan,
    row: dict[str, Any],
    fit: MethodFit,
    fit_indices: np.ndarray,
    groups: Sequence[Sequence[int]],
    design: Design,
    dataset: Dataset,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    wall_limit = float(
        manifest["resources"]["bootstrap_wall_seconds_per_scenario_seed"]
    )
    try:
        with _deadline(wall_limit, "bootstrap_wall_seconds"):
            return _bootstrap_run_unbounded(
                manifest,
                plan,
                row,
                fit,
                fit_indices,
                groups,
                design,
                dataset,
            )
    except ResourceLimit as exc:
        return (
            [],
            [
                {
                    "manifest_id": manifest["manifest_id"],
                    "run_id": row["run_id"],
                    "tier": plan.tier,
                    "scenario_id": row["scenario_id"],
                    "method_id": row["method_id"],
                    "master_seed": row["master_seed"],
                    "stage": "bootstrap",
                    "bootstrap_index_0_based": None,
                    "failure_code": exc.code,
                    "failure_reason": str(exc),
                }
            ],
            [],
        )


def _likelihood_conditioning(
    classes: TraceLikelihoodClasses,
    schema_version: str = V3_SCHEMA_VERSION,
    *,
    return_singular_values: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], np.ndarray]:
    centered = classes.log_likelihoods - np.mean(classes.log_likelihoods, axis=1, keepdims=True)
    weighted = centered * np.sqrt(classes.counts[:, None])
    floor = 1e-12
    if schema_version in {V1_SCHEMA_VERSION, V2_SCHEMA_VERSION}:
        # Preserve the frozen legacy calculation exactly.  V3 moves to direct
        # singular values because forming the Gram matrix squares the condition
        # number and can promote roundoff in null directions to false rank.
        gram = weighted.T @ weighted
        singular = np.sqrt(
            np.maximum(np.linalg.eigvalsh(0.5 * (gram + gram.T)), 0.0)
        )
        maximum = float(singular[-1]) if singular.size else 0.0
        retained = singular > floor * maximum
        rank = int(retained.sum())
        minimum = float(singular[retained][0]) if rank else 0.0
        condition = (
            maximum / minimum if rank == singular.size and minimum else math.inf
        )
        result = {
            "centered_likelihood_rank": rank,
            "centered_likelihood_singular_max": maximum,
            "centered_likelihood_singular_min_retained": minimum,
            "centered_likelihood_condition_number": condition,
            "relative_singular_value_floor": floor,
        }
        return (result, singular) if return_singular_values else result

    singular = np.linalg.svd(weighted, compute_uv=False)
    maximum = float(singular[0]) if singular.size else 0.0
    retained = singular > floor * maximum
    rank = int(retained.sum())
    minimum = float(singular[retained][-1]) if rank else 0.0
    # With no retained likelihood direction there is no nonzero subspace to
    # condition.  Use a finite zero sentinel alongside rank=0 and min=0 rather
    # than reporting an infinite condition number for absent information.
    condition = maximum / minimum if rank else 0.0
    result = {
        "centered_likelihood_rank": rank,
        "centered_likelihood_singular_max": maximum,
        "centered_likelihood_singular_min_retained": minimum,
        "centered_likelihood_condition_number": condition,
        "relative_singular_value_floor": floor,
    }
    return (result, singular) if return_singular_values else result


def _identifiability_row(
    manifest: dict[str, Any],
    plan: ExecutionPlan,
    row: dict[str, Any],
    fit: MethodFit,
    fit_indices: np.ndarray,
    groups: Sequence[Sequence[int]],
    design: Design,
    dataset: Dataset,
    fitted_q: np.ndarray,
) -> dict[str, Any]:
    if manifest["schema_version"] in V4_METRIC_SCHEMA_VERSIONS:
        conditioning, singular_values = _likelihood_conditioning(
            fit.classes,
            str(manifest["schema_version"]),
            return_singular_values=True,
        )
    else:
        conditioning = _likelihood_conditioning(
            fit.classes, str(manifest["schema_version"])
        )
        singular_values = None
    result: dict[str, Any] = {
        "manifest_id": manifest["manifest_id"],
        "run_id": row["run_id"],
        "tier": plan.tier,
        "scenario_id": row["scenario_id"],
        "method_id": row["method_id"],
        "master_seed": row["master_seed"],
        "n_observable_groups": len(groups),
        "n_unresolved_groups": sum(len(group) > 1 for group in groups),
        "maximum_group_size": max(map(len, groups)),
        **conditioning,
    }
    if singular_values is not None:
        result["_singular_values"] = singular_values
    if row["scenario_id"] in {"I1_anti4R_contrast025", "I2_anti4R_contrast000"}:
        probe = design.probe_ids.index("anti_4R")
        changes: list[float | None] = []
        for sign, prefix in ((1.0, "plus_0_05_Q"), (-1.0, "minus_0_05_Q")):
            perturbed = fitted_q.copy()
            perturbed[:, probe] = _expit(_logit(perturbed[:, probe]) + sign * 0.05)
            classes = build_trace_likelihood_classes(
                dataset.observations,
                perturbed,
                cycle_to_probe=design.cycle_to_probe,
            )
            sensitivity = _em_fit(
                classes,
                manifest,
                initial_weights=fit.weights,
                continue_warm_start_on_retry=(
                    manifest["schema_version"] in DIRECT_SVD_SCHEMA_VERSIONS
                ),
            )
            if manifest["schema_version"] in DIRECT_SVD_SCHEMA_VERSIONS:
                result.update(
                    {
                        f"{prefix}_status": sensitivity.status,
                        f"{prefix}_retry_used": sensitivity.retry_used,
                        f"{prefix}_iterations": sensitivity.iterations,
                        f"{prefix}_terminal_em_residual": (
                            sensitivity.terminal_residual
                        ),
                        f"{prefix}_last_log_likelihood_gain": sensitivity.last_gain,
                        f"{prefix}_max_weight_change": sensitivity.max_weight_change,
                    }
                )
                if sensitivity.status != "ok":
                    raise FloatingPointError(
                        f"{prefix} identifiability sensitivity fit failed: "
                        f"{sensitivity.failure_code}: {sensitivity.failure_reason}"
                    )
            value = _tv(sensitivity.weights, fit.weights) if sensitivity.status == "ok" else None
            result[f"{prefix}_weight_TV"] = value
            changes.append(value)
        finite = [value for value in changes if value is not None]
        result["maximum_plus_or_minus_0_05_Q_weight_TV"] = max(finite) if finite else None
    return result


def _wilson(covered: int, total: int) -> tuple[float | None, float | None]:
    if not total:
        return None, None
    z = 1.959963984540054
    proportion = covered / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def _coverage_summary(
    manifest: dict[str, Any], plan: ExecutionPlan, rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    named_estimands = manifest["schema_version"] in MODERN_SCHEMA_VERSIONS
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["scenario_id"]),
            str(row["method_id"]),
            str(row["estimand_type"]),
        )
        if named_estimands:
            key += (str(row["estimand_id"]),)
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for key, selected in sorted(grouped.items()):
        scenario, method, estimand = key[:3]
        estimand_id = key[3] if named_estimands else None
        valid = [row for row in selected if row["interval_valid"] and row["covered"] is not None]
        covered = sum(bool(row["covered"]) for row in valid)
        lower, upper = _wilson(covered, len(valid))
        widths = [float(row["width"]) for row in valid]
        minimum = int(
            manifest["bootstrap"].get(
                "coverage_minimum_valid_seed_intervals",
                manifest["stopping_and_failures"]["primary_summary_minimum_valid_seed_pairs"],
            )
        )
        result.append(
            {
                "manifest_id": manifest["manifest_id"],
                "tier": plan.tier,
                "scenario_id": scenario,
                "method_id": method,
                "estimand_type": estimand,
                "estimand_id": estimand_id,
                "n_intervals": len(selected),
                "n_valid_intervals": len(valid),
                "covered": covered,
                "coverage": covered / len(valid) if valid else None,
                "coverage_wilson_lower": lower,
                "coverage_wilson_upper": upper,
                "mean_width": statistics.fmean(widths) if widths else None,
                "median_width": statistics.median(widths) if widths else None,
                "headline_eligible": bool(
                    plan.headline_eligible
                    and len(valid) >= minimum
                ),
            }
        )
    return result


SUMMARY_METRICS = (
    "total_variation",
    "jensen_shannon_base2",
    "candidate_MAE",
    "candidate_RMSE",
    "group_total_variation",
    "log_score",
    "Brier_score",
    "ECE_10_equal_frequency",
    "top1_accuracy",
    "top5_recall",
)

V4_SUMMARY_METRICS = (
    "total_variation",
    "jensen_shannon_base2",
    "candidate_MAE",
    "candidate_RMSE",
    "maximum_absolute_error",
    "detection_sensitivity",
    "detection_false_discovery_rate",
    "group_total_variation",
    "log_score",
    "Brier_score",
    "ECE_10_equal_frequency",
    "top1_accuracy",
    "top5_recall",
    "posterior_entropy",
)


def _summary_metrics(manifest: dict[str, Any]) -> tuple[str, ...]:
    return (
        V4_SUMMARY_METRICS
        if manifest["schema_version"] in V4_METRIC_SCHEMA_VERSIONS
        else SUMMARY_METRICS
    )


def _declared_contrast_metadata(
    manifest: dict[str, Any], scenario_id: str, method_id: str
) -> tuple[str | None, str | None, str | None]:
    reference_id = str(
        manifest["scenario_definitions"][scenario_id]["paired_reference"]
    )
    if manifest["schema_version"] == V1_SCHEMA_VERSION:
        return "scenario", reference_id, method_id
    declared: list[tuple[str, str, str]] = []
    rule = manifest["methods"]["headline_scenario_contrasts"].get(method_id)
    scenario_declared = (
        rule == "all_nonself_declared_paired_references"
        and scenario_id != reference_id
    ) or (isinstance(rule, list) and scenario_id in rule)
    if scenario_declared:
        declared.append(("scenario", reference_id, method_id))
    declared.extend(
        ("method", scenario_id, str(contrast["reference_method_id"]))
        for contrast in manifest["methods"].get("headline_method_contrasts", [])
        if contrast["scenario_id"] == scenario_id
        and contrast["method_id"] == method_id
    )
    if len(declared) > 1:
        raise ValueError(
            "summary row model permits at most one declared contrast for "
            f"{scenario_id}:{method_id}"
        )
    return declared[0] if declared else (None, None, None)


def _paired_rows(
    manifest: dict[str, Any], plan: ExecutionPlan, runs: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    lookup = {
        (row["scenario_id"], row["method_id"], row["master_seed"]): row for row in runs
    }
    result: list[dict[str, Any]] = []

    def append_comparison(
        row: dict[str, Any],
        reference: dict[str, Any] | None,
        *,
        comparison_type: str,
        paired_reference: str,
        reference_method_id: str,
    ) -> None:
        for metric in _summary_metrics(manifest):
            scenario_value = row.get(metric)
            reference_value = reference.get(metric) if reference else None
            valid = (
                row.get("metrics_valid") is True
                and reference is not None
                and reference.get("metrics_valid") is True
                and scenario_value not in (None, "")
                and reference_value not in (None, "")
            )
            result.append(
                {
                    "manifest_id": manifest["manifest_id"],
                    "tier": plan.tier,
                    "comparison_type": comparison_type,
                    "scenario_id": row["scenario_id"],
                    "paired_reference": paired_reference,
                    "method_id": row["method_id"],
                    "reference_method_id": reference_method_id,
                    "master_seed": row["master_seed"],
                    "metric": metric,
                    "scenario_value": scenario_value,
                    "reference_value": reference_value,
                    "paired_difference": (
                        float(scenario_value) - float(reference_value)
                        if valid
                        else None
                    ),
                    "pair_valid": valid,
                }
            )

    if manifest["schema_version"] == V1_SCHEMA_VERSION:
        for row in runs:
            reference_id = str(row["paired_reference"])
            reference = lookup.get(
                (reference_id, row["method_id"], row["master_seed"])
            )
            append_comparison(
                row,
                reference,
                comparison_type="scenario",
                paired_reference=reference_id,
                reference_method_id=str(row["method_id"]),
            )
        return result

    for row in runs:
        method_id = str(row["method_id"])
        scenario_id = str(row["scenario_id"])
        reference_id = str(row["paired_reference"])
        comparison_type, _, _ = _declared_contrast_metadata(
            manifest, scenario_id, method_id
        )
        if comparison_type != "scenario":
            continue
        reference = lookup.get((reference_id, method_id, row["master_seed"]))
        append_comparison(
            row,
            reference,
            comparison_type="scenario",
            paired_reference=reference_id,
            reference_method_id=method_id,
        )

    for contrast in manifest["methods"]["headline_method_contrasts"]:
        scenario_id = str(contrast["scenario_id"])
        method_id = str(contrast["method_id"])
        reference_method_id = str(contrast["reference_method_id"])
        for master_seed in plan.master_seeds:
            row = lookup.get((scenario_id, method_id, master_seed))
            if row is None:
                continue
            reference = lookup.get((scenario_id, reference_method_id, master_seed))
            append_comparison(
                row,
                reference,
                comparison_type="method",
                paired_reference=scenario_id,
                reference_method_id=reference_method_id,
            )
    return result


def _summary_rows(
    manifest: dict[str, Any],
    plan: ExecutionPlan,
    runs: Sequence[dict[str, Any]],
    paired: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    fields = _schema_names(manifest["outputs"]["summary_schema"])
    result: list[dict[str, Any]] = []
    planned_seeds = len(manifest["tiers"][plan.scientific_tier]["master_seeds"])
    combinations = sorted({(row["scenario_id"], row["method_id"]) for row in runs})
    for scenario, method in combinations:
        selected = [row for row in runs if row["scenario_id"] == scenario and row["method_id"] == method]
        comparison_type, reference_scenario, reference_method = (
            _declared_contrast_metadata(manifest, scenario, method)
        )
        legacy_reference = manifest["scenario_definitions"][scenario][
            "paired_reference"
        ]
        summary_reference = (
            legacy_reference
            if manifest["schema_version"] == V1_SCHEMA_VERSION
            else reference_scenario
        )
        for metric in _summary_metrics(manifest):
            values = [
                float(row[metric])
                for row in selected
                if row.get("metrics_valid") and row.get(metric) not in (None, "")
            ]
            declared_contrasts = [
                row
                for row in paired
                if row["scenario_id"] == scenario
                and row["method_id"] == method
                and row["metric"] == metric
            ]
            differences = [
                float(row["paired_difference"])
                for row in declared_contrasts
                if row["pair_valid"]
            ]
            n_valid = len(values)
            pair_minimum = int(
                manifest["stopping_and_failures"][
                    "primary_summary_minimum_valid_seed_pairs"
                ]
            )
            absolute_minimum = int(
                manifest["stopping_and_failures"].get(
                    "primary_summary_minimum_valid_absolute_runs", pair_minimum
                )
            )
            absolute_valid = n_valid >= absolute_minimum
            contrast_valid = (
                len(differences) >= pair_minimum
                if comparison_type is not None
                else True
            )
            output = {
                "manifest_id": manifest["manifest_id"],
                "tier": plan.tier,
                "scenario_id": scenario,
                "paired_reference": summary_reference,
                "comparison_type": comparison_type,
                "reference_scenario_id": reference_scenario,
                "reference_method_id": reference_method,
                "method_id": method,
                "metric": metric,
                "n_planned": planned_seeds,
                "n_valid": n_valid,
                "n_failed": sum(row.get("status") != "ok" for row in selected),
                "mean": statistics.fmean(values) if values else None,
                "sample_SD": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
                "Monte_Carlo_SE": statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else (0.0 if values else None),
                "median": statistics.median(values) if values else None,
                "q25": float(np.quantile(values, 0.25)) if values else None,
                "q75": float(np.quantile(values, 0.75)) if values else None,
                "paired_mean_difference": statistics.fmean(differences) if differences else None,
                "paired_difference_sample_SD": statistics.stdev(differences) if len(differences) > 1 else (0.0 if differences else None),
                "paired_difference_Monte_Carlo_SE": statistics.stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else (0.0 if differences else None),
                "headline_eligible": bool(
                    plan.headline_eligible and absolute_valid and contrast_valid
                ),
            }
            result.append({field: output.get(field) for field in fields})
    return result


def _scenario_index(
    manifest: dict[str, Any], plan: ExecutionPlan
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eligible = set(manifest["bootstrap"]["eligible_primary_scenarios"])
    for scenario in plan.scenario_ids:
        spec = manifest["scenario_definitions"][scenario]
        for method in _method_ids(manifest, plan, scenario):
            rows.append(
                {
                    "manifest_id": manifest["manifest_id"],
                    "tier": plan.tier,
                    "scenario_id": scenario,
                    "scenario_family": spec["family"],
                    "paired_reference": spec["paired_reference"],
                    "method_id": method,
                    "bootstrap_eligible": method == manifest["bootstrap"]["method"] and scenario in eligible,
                    "headline_eligible": plan.headline_eligible,
                    "definition_json": _canonical_json(spec),
                }
            )
    return rows


def _write_tsv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _candidate_rows(manifest: dict[str, Any], design: Design) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local, original in enumerate(design.original_indices):
        rows.append(
            {
                "manifest_id": manifest["manifest_id"],
                "candidate_index_0_based": local,
                "tau768_index_0_based": int(original),
                "candidate_id": design.candidate_ids[local],
                "isoform": design.isoforms[local],
                "PTM_count": int(design.ptm_states[local].sum()),
                "logical_profile": "".join(map(str, design.profiles[local].tolist())),
            }
        )
    return rows


def _probe_rows(manifest: dict[str, Any], design: Design) -> list[dict[str, Any]]:
    alpha = np.asarray([0.97, 0.97] + [0.91] * 3 + [0.84] * 7)
    beta = np.asarray([0.002, 0.002] + [0.010] * 3 + [0.015] * 7)
    return [
        {
            "manifest_id": manifest["manifest_id"],
            "logical_probe_index_0_based": index,
            "logical_probe_id": probe,
            "alpha": alpha[index],
            "beta": beta[index],
            "cycle_ids_1_based": "|".join(
                map(str, (np.flatnonzero(design.cycle_to_probe == index) + 1).tolist())
            ),
        }
        for index, probe in enumerate(design.probe_ids)
    ]


def _figure_rows(
    manifest: dict[str, Any],
    summaries: Sequence[dict[str, Any]],
    reliability: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    is_modern = manifest["schema_version"] in MODERN_SCHEMA_VERSIONS
    for summary in summaries:
        if summary["mean"] in (None, ""):
            continue
        common = {
            "manifest_id": manifest["manifest_id"],
            "tier": summary["tier"],
            "row_type": "metric_summary",
            "scenario_id": summary["scenario_id"],
            "scenario_family": manifest["scenario_definitions"][summary["scenario_id"]]["family"],
            "method_id": summary["method_id"],
            "metric": summary["metric"],
            "resolution": "declared",
            "reportable": summary["headline_eligible"],
        }
        if not is_modern:
            rows.append(
                {
                    **common,
                    "paired_reference": summary["paired_reference"],
                    "estimate": summary["mean"],
                    "uncertainty": summary["sample_SD"],
                }
            )
            continue
        rows.append(
            {
                **common,
                "value_kind": "absolute_mean",
                "paired_reference": None,
                "comparison_type": None,
                "reference_scenario_id": None,
                "reference_method_id": None,
                "estimate": summary["mean"],
                "uncertainty": summary["sample_SD"],
            }
        )
        if summary.get("comparison_type") is not None:
            rows.append(
                {
                    **common,
                    "value_kind": "paired_mean_difference",
                    "paired_reference": summary.get("paired_reference"),
                    "comparison_type": summary.get("comparison_type"),
                    "reference_scenario_id": summary.get("reference_scenario_id"),
                    "reference_method_id": summary.get("reference_method_id"),
                    "estimate": summary.get("paired_mean_difference"),
                    "uncertainty": summary.get("paired_difference_sample_SD"),
                    "reportable": bool(
                        summary["headline_eligible"]
                        and summary.get("paired_mean_difference") not in (None, "")
                    ),
                }
            )
    for item in reliability:
        rows.append(
            {
                "manifest_id": manifest["manifest_id"],
                "tier": item["tier"],
                "row_type": "reliability",
                "value_kind": "reliability_bin" if is_modern else None,
                "scenario_id": item["scenario_id"],
                "scenario_family": manifest["scenario_definitions"][item["scenario_id"]]["family"],
                "paired_reference": (
                    None
                    if is_modern
                    else manifest["scenario_definitions"][item["scenario_id"]][
                        "paired_reference"
                    ]
                ),
                "comparison_type": None,
                "reference_scenario_id": None,
                "reference_method_id": None,
                "method_id": item["method_id"],
                "metric": "ECE_10_equal_frequency",
                "resolution": item["resolution"],
                "bin_index": item["bin_index"],
                "bin_lower": item["bin_lower"],
                "bin_upper": item["bin_upper"],
                "count": item["count"],
                "mean_confidence": item["mean_confidence"],
                "accuracy": item["accuracy"],
                "reportable": item["reportable"],
            }
        )
    return rows


def _git_metadata() -> tuple[str | None, bool | None]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, capture_output=True
    )
    if commit.returncode != 0:
        return None, None
    return commit.stdout.strip(), bool(status.stdout.strip()) if status.returncode == 0 else None


def _source_sha256_map() -> dict[str, str]:
    paths = {Path(__file__).resolve()}
    paths.update(
        path.resolve()
        for path in (SRC_ROOT / "proteoem").glob("*.py")
        if path.is_file()
    )
    return {
        str(path.relative_to(REPO_ROOT)): _file_sha256(path)
        for path in sorted(paths, key=lambda item: str(item.relative_to(REPO_ROOT)))
    }


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _execution_scope(plan: ExecutionPlan) -> str:
    if plan.tier == "smoke":
        return "smoke"
    return "preflight" if plan.preflight else "full_tier"


def _full_tier_plan(manifest: dict[str, Any], tier: str) -> ExecutionPlan:
    source = manifest["tiers"][tier]
    return ExecutionPlan(
        tier=tier,
        scientific_tier=tier,
        scenario_ids=tuple(source["scenario_ids"]),
        master_seeds=tuple(map(int, source["master_seeds"])),
        n_molecules=int(source["molecules_per_dataset"]),
        bootstrap_resamples=int(source["bootstrap_resamples"]),
        methods_filter=None,
        preflight=False,
        skip_bootstrap=False,
        manifest_conformant=True,
        headline_eligible=bool(source["headline_eligible"]),
        overrides={},
    )


def _missing_value(value: Any) -> bool:
    return value is None or value == ""


def _finite_value(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if _missing_value(value):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _row_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row["scenario_id"]),
        int(row["master_seed"]),
        str(row["method_id"]),
    )


def _v4_reporting_contract_counts(
    expected_points: set[tuple[str, int, str]],
    runs: Sequence[dict[str, Any]],
    reliability: Sequence[dict[str, Any]],
    ood: Sequence[dict[str, Any]],
    identifiability: Sequence[dict[str, Any]],
    singular_values: Sequence[dict[str, Any]],
    features: Sequence[dict[str, Any]],
) -> dict[str, int]:
    """Validate that every v4 reporting product is present and applicable."""

    run_lookup = {_row_key(row): row for row in runs if _row_key(row) in expected_points}
    reliability_count_totals: dict[tuple[str, int, str, str], int] = {}
    for reliability_row in reliability:
        try:
            total_key = (*_row_key(reliability_row), str(reliability_row["resolution"]))
            reliability_count_totals[total_key] = (
                reliability_count_totals.get(total_key, 0)
                + int(reliability_row["count"])
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    invalid_run_metrics = 0
    candidate_abundance_fields = (
        "total_variation",
        "jensen_shannon_base2",
        "candidate_MAE",
        "candidate_RMSE",
        "maximum_absolute_error",
    )
    candidate_fields = (
        *candidate_abundance_fields,
        "detection_sensitivity",
        "detection_false_discovery_rate",
    )
    detection_count_fields = (
        "n_truth_detectable",
        "n_called_detected",
        "n_true_positive_detected",
        "n_false_positive_detected",
    )
    probability_fields = (
        "log_score",
        "Brier_score",
        "ECE_10_equal_frequency",
        "top1_accuracy",
        "top5_recall",
        "posterior_entropy",
    )
    for key, row in run_lookup.items():
        scenario, _seed, method = key
        unresolved = scenario in EXACT_UNRESOLVED_SCENARIOS
        try:
            expected_abundance_resolution = "group" if unresolved else "candidate"
            if row.get("abundance_resolution") != expected_abundance_resolution:
                raise ValueError("wrong abundance resolution")
            out_of_dictionary = _finite_value(
                row, "truth_out_of_dictionary_mass"
            )
            if out_of_dictionary is None or not 0.0 <= out_of_dictionary <= 1.0:
                raise ValueError("invalid out-of-dictionary mass")
            group_tv = _finite_value(row, "group_total_variation")
            if group_tv is None or not 0.0 <= group_tv <= 1.0:
                raise ValueError("invalid group TV")
            if unresolved:
                if any(not _missing_value(row.get(field)) for field in candidate_fields):
                    raise ValueError("candidate metric at unresolved resolution")
                if any(
                    not _missing_value(row.get(field))
                    for field in detection_count_fields
                ):
                    raise ValueError("detection count at unresolved resolution")
            else:
                for field in candidate_abundance_fields:
                    value = _finite_value(row, field)
                    if value is None or not 0.0 <= value <= 1.0:
                        raise ValueError(f"invalid {field}")
                if float(row["maximum_absolute_error"]) + 1e-12 < out_of_dictionary:
                    raise ValueError(
                        "maximum absolute error omits out-of-dictionary mass"
                    )
                n_truth, n_called, n_true_positive, n_false_positive = (
                    int(row[field]) for field in detection_count_fields
                )
                if (
                    min(
                        n_truth,
                        n_called,
                        n_true_positive,
                        n_false_positive,
                    )
                    < 0
                    or n_true_positive > n_truth
                    or n_true_positive > n_called
                    or n_true_positive + n_false_positive != n_called
                    or max(n_truth, n_called) > 768
                    or n_false_positive > 768 - n_truth
                ):
                    raise ValueError("invalid detection counts")
                sensitivity = _finite_value(row, "detection_sensitivity")
                if n_truth:
                    if sensitivity is None or not math.isclose(
                        sensitivity,
                        n_true_positive / n_truth,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise ValueError("inconsistent detection sensitivity")
                elif not _missing_value(row.get("detection_sensitivity")):
                    raise ValueError("sensitivity must be null with no truth calls")
                fdr = _finite_value(row, "detection_false_discovery_rate")
                expected_fdr = (
                    (n_called - n_true_positive) / n_called if n_called else 0.0
                )
                if fdr is None or not math.isclose(
                    fdr, expected_fdr, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError("inconsistent detection FDR")

            if method in CALIBRATED_PROBABILITY_METHODS:
                for field in probability_fields:
                    if _finite_value(row, field) is None:
                        raise ValueError(f"missing probability metric {field}")
                for field in (
                    "Brier_score",
                    "ECE_10_equal_frequency",
                    "top1_accuracy",
                    "top5_recall",
                ):
                    value = float(row[field])
                    if not 0.0 <= value <= 1.0:
                        raise ValueError(f"invalid probability metric {field}")
                entropy = float(row["posterior_entropy"])
                n_candidates = int(row["n_candidates_fit"])
                maximum_entropy = math.log(n_candidates)
                if entropy < -1e-12 or entropy > maximum_entropy + 1e-12:
                    raise ValueError("posterior entropy outside cardinality bounds")
                evaluated_n = int(row["posterior_entropy_evaluated_n"])
                n_accepted = int(row["n_accepted"])
                if evaluated_n <= 0 or evaluated_n > n_accepted:
                    raise ValueError("invalid posterior entropy denominator")
                if any(
                    reliability_count_totals.get((*key, resolution), -1)
                    != evaluated_n
                    for resolution in ("candidate", "group")
                ):
                    raise ValueError(
                        "posterior entropy denominator disagrees with reliability counts"
                    )
                expected_resolution = "group" if unresolved else "candidate"
                if row.get("posterior_entropy_resolution") != expected_resolution:
                    raise ValueError("wrong posterior entropy resolution")
            else:
                forbidden = (
                    "log_score",
                    "Brier_score",
                    "ECE_10_equal_frequency",
                    "top5_recall",
                    "posterior_entropy",
                    "posterior_entropy_resolution",
                    "posterior_entropy_evaluated_n",
                )
                if any(not _missing_value(row.get(field)) for field in forbidden):
                    raise ValueError("inapplicable probability metric populated")
                top1 = _finite_value(row, "top1_accuracy")
                if top1 is None or not 0.0 <= top1 <= 1.0:
                    raise ValueError("missing top-label accuracy")
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid_run_metrics += 1

    feature_keys = [_row_key(row) for row in features]
    feature_set = set(feature_keys)
    expected_features = {
        key for key in expected_points if key[0] != "O3_artifact_mix05"
    }
    invalid_feature_rows = 0
    for row in features:
        key = _row_key(row)
        try:
            if key not in expected_features:
                continue
            scenario = key[0]
            isoform = _finite_value(row, "isoform_total_variation")
            ptm = _finite_value(row, "PTM_marginal_MAE")
            pair = _finite_value(row, "PTM_pair_MAE")
            if scenario in EXACT_UNRESOLVED_SCENARIOS:
                if not _missing_value(row.get("isoform_total_variation")):
                    raise ValueError("unresolved isoform score must be null")
            elif isoform is None or not 0.0 <= isoform <= 1.0:
                raise ValueError("invalid isoform score")
            if ptm is None or pair is None or not 0.0 <= ptm <= 1.0 or not 0.0 <= pair <= 1.0:
                raise ValueError("invalid PTM score")
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid_feature_rows += 1

    calibrated_points = {
        key for key in expected_points if key[2] in CALIBRATED_PROBABILITY_METHODS
    }
    expected_reliability = {
        (*key, resolution, bin_index)
        for key in calibrated_points
        for resolution in ("candidate", "group")
        for bin_index in range(10)
    }
    observed_reliability = [
        (*_row_key(row), str(row["resolution"]), int(row["bin_index"]))
        for row in reliability
    ]
    invalid_reliability_rows = 0
    for row in reliability:
        try:
            key = _row_key(row)
            resolution = str(row["resolution"])
            count = int(row["count"])
            confidence = _finite_value(row, "mean_confidence")
            accuracy = _finite_value(row, "accuracy")
            gap = _finite_value(row, "absolute_gap")
            if (
                key not in calibrated_points
                or resolution not in {"candidate", "group"}
                or count <= 0
                or confidence is None
                or accuracy is None
                or gap is None
                or not 0.0 <= confidence <= 1.0
                or not 0.0 <= accuracy <= 1.0
                or not math.isclose(
                    gap, abs(confidence - accuracy), rel_tol=0.0, abs_tol=1e-12
                )
            ):
                raise ValueError("invalid reliability row")
            unresolved = key[0] in EXACT_UNRESOLVED_SCENARIOS
            expected_reportable = resolution == "group" or not unresolved
            if _bool_value(row.get("reportable")) != expected_reportable:
                raise ValueError("wrong reliability resolution")
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid_reliability_rows += 1

    ood_scenarios = {
        "O1_heldout05",
        "O2_heldout20",
        "O3_artifact_mix05",
    }
    expected_ood = {
        key for key in calibrated_points if key[0] in ood_scenarios
    }
    ood_keys = [_row_key(row) for row in ood]
    invalid_ood_rows = 0
    for row in ood:
        try:
            if _row_key(row) not in expected_ood:
                raise ValueError("unexpected OOD row")
            for field in (
                "AUROC_maximum_posterior",
                "AUPRC_maximum_posterior",
                "AUROC_negative_maximum_log_likelihood_per_observed_call",
                "AUPRC_negative_maximum_log_likelihood_per_observed_call",
                "false_accept_fraction_at_maximum_posterior_at_least_0_80",
                "mean_maximum_posterior_in_dictionary",
                "mean_maximum_posterior_ood",
            ):
                value = _finite_value(row, field)
                if value is None or not 0.0 <= value <= 1.0:
                    raise ValueError(f"invalid OOD metric {field}")
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid_ood_rows += 1

    expected_identifiability = {
        key
        for key in expected_points
        if key[0] in IDENTIFIABILITY_SCENARIOS and key[2] == "em_fixed_q"
    }
    ident_keys = [_row_key(row) for row in identifiability]
    ident_lookup = {_row_key(row): row for row in identifiability}
    singular_grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in singular_values:
        singular_grouped.setdefault(_row_key(row), []).append(row)
    invalid_singular_spectra = 0
    for key in expected_identifiability:
        try:
            diagnostic = ident_lookup[key]
            selected = sorted(
                singular_grouped[key],
                key=lambda row: int(row["singular_value_index_0_based"]),
            )
            run = run_lookup[key]
            expected_count = min(
                int(run["trace_likelihood_classes"]), int(run["n_candidates_fit"])
            )
            indices = [int(row["singular_value_index_0_based"]) for row in selected]
            values = [float(row["singular_value"]) for row in selected]
            if indices != list(range(expected_count)):
                raise ValueError("incomplete singular spectrum")
            if any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ValueError("invalid singular value")
            if any(left < right for left, right in zip(values, values[1:])):
                raise ValueError("singular values not descending")
            floor = float(diagnostic["relative_singular_value_floor"])
            maximum = values[0] if values else 0.0
            retained = [value > floor * maximum for value in values]
            if [
                _bool_value(row["retained_above_relative_floor"])
                for row in selected
            ] != retained:
                raise ValueError("singular retained flags disagree")
            retained_values = [
                value for value, keep in zip(values, retained, strict=True) if keep
            ]
            rank = len(retained_values)
            minimum = retained_values[-1] if retained_values else 0.0
            condition = maximum / minimum if rank else 0.0
            for field, expected in (
                ("centered_likelihood_rank", rank),
                ("centered_likelihood_singular_max", maximum),
                ("centered_likelihood_singular_min_retained", minimum),
                ("centered_likelihood_condition_number", condition),
            ):
                observed = float(diagnostic[field])
                if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(f"singular summary mismatch: {field}")
        except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
            invalid_singular_spectra += 1

    singular_keys = set(singular_grouped)
    return {
        "invalid_v4_run_metric_rows": invalid_run_metrics,
        "missing_v4_feature_rows": len(expected_features - feature_set),
        "unexpected_v4_feature_rows": len(feature_set - expected_features),
        "duplicate_v4_feature_rows": len(feature_keys) - len(feature_set),
        "invalid_v4_feature_rows": invalid_feature_rows,
        "missing_v4_reliability_rows": len(
            expected_reliability - set(observed_reliability)
        ),
        "unexpected_v4_reliability_rows": len(
            set(observed_reliability) - expected_reliability
        ),
        "duplicate_v4_reliability_rows": len(observed_reliability)
        - len(set(observed_reliability)),
        "invalid_v4_reliability_rows": invalid_reliability_rows,
        "missing_v4_ood_rows": len(expected_ood - set(ood_keys)),
        "unexpected_v4_ood_rows": len(set(ood_keys) - expected_ood),
        "duplicate_v4_ood_rows": len(ood_keys) - len(set(ood_keys)),
        "invalid_v4_ood_rows": invalid_ood_rows,
        "missing_v4_identifiability_rows": len(
            expected_identifiability - set(ident_keys)
        ),
        "unexpected_v4_identifiability_rows": len(
            set(ident_keys) - expected_identifiability
        ),
        "duplicate_v4_identifiability_rows": len(ident_keys) - len(set(ident_keys)),
        "missing_v4_singular_spectra": len(
            expected_identifiability - singular_keys
        ),
        "unexpected_v4_singular_spectra": len(
            singular_keys - expected_identifiability
        ),
        "invalid_v4_singular_spectra": invalid_singular_spectra,
    }


def _evaluate_execution_status(
    manifest: dict[str, Any],
    plan: ExecutionPlan,
    runs: Sequence[dict[str, Any]],
    intervals: Sequence[dict[str, Any]],
    diagnostics: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
    reliability: Sequence[dict[str, Any]] = (),
    ood: Sequence[dict[str, Any]] = (),
    identifiability: Sequence[dict[str, Any]] = (),
    singular_values: Sequence[dict[str, Any]] = (),
    features: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    expected_points = {
        (scenario, int(seed), method)
        for seed in plan.master_seeds
        for scenario in plan.scenario_ids
        for method in _declared_method_ids(
            manifest, plan.scientific_tier, scenario
        )
        if plan.methods_filter is None or method in plan.methods_filter
    }
    observed_point_keys = [
        (str(row["scenario_id"]), int(row["master_seed"]), str(row["method_id"]))
        for row in runs
    ]
    observed_point_set = set(observed_point_keys)
    duplicate_point_runs = len(observed_point_keys) - len(observed_point_set)
    missing_point_runs = expected_points - observed_point_set
    unexpected_point_runs = observed_point_set - expected_points
    invalid_point_runs = sum(
        key in expected_points
        and (
            str(row.get("status")) != "ok"
            or not _bool_value(row.get("metrics_valid"))
        )
        for key, row in zip(observed_point_keys, runs, strict=True)
    )

    bootstrap_method = str(manifest["bootstrap"]["method"])
    eligible = set(manifest["bootstrap"]["eligible_primary_scenarios"])
    expected_bootstrap_runs = {
        (scenario, int(seed), bootstrap_method)
        for seed in plan.master_seeds
        for scenario in plan.scenario_ids
        if not plan.skip_bootstrap
        and scenario in eligible
        and bootstrap_method
        in _declared_method_ids(manifest, plan.scientific_tier, scenario)
        and (
            plan.methods_filter is None
            or bootstrap_method in plan.methods_filter
        )
    }
    expected_diagnostics = {
        (*key, index)
        for key in expected_bootstrap_runs
        for index in range(plan.bootstrap_resamples)
    }
    observed_diagnostic_keys = [
        (
            str(row["scenario_id"]),
            int(row["master_seed"]),
            str(row["method_id"]),
            int(row["bootstrap_index_0_based"]),
        )
        for row in diagnostics
    ]
    observed_diagnostic_set = set(observed_diagnostic_keys)
    duplicate_diagnostics = len(observed_diagnostic_keys) - len(
        observed_diagnostic_set
    )
    missing_diagnostics = expected_diagnostics - observed_diagnostic_set
    unexpected_diagnostics = observed_diagnostic_set - expected_diagnostics

    intervals_by_run: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in intervals:
        key = (
            str(row["scenario_id"]),
            int(row["master_seed"]),
            str(row["method_id"]),
        )
        intervals_by_run.setdefault(key, []).append(row)
    invalid_bootstrap_runs = 0
    minimum_success = float(manifest["bootstrap"]["minimum_success_fraction"])
    for key in expected_bootstrap_runs:
        selected = intervals_by_run.get(key, [])
        successful_diagnostics = sum(
            str(row.get("status")) == "ok"
            for diagnostic_key, row in zip(
                observed_diagnostic_keys, diagnostics, strict=True
            )
            if diagnostic_key[:3] == key
        )
        derived_success_fraction = (
            successful_diagnostics / plan.bootstrap_resamples
        )
        derived_interval_valid = derived_success_fraction >= minimum_success
        valid = bool(selected) and derived_interval_valid and all(
            _bool_value(row.get("interval_valid")) == derived_interval_valid
            and int(row.get("n_attempted", -1)) == plan.bootstrap_resamples
            and int(row.get("n_successful", -1)) == successful_diagnostics
            and math.isclose(
                float(row.get("success_fraction", -1.0)),
                derived_success_fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for row in selected
        )
        invalid_bootstrap_runs += int(not valid)
    unexpected_interval_runs = set(intervals_by_run) - expected_bootstrap_runs

    fatal_events = {
        (
            str(row.get("scenario_id")),
            int(row.get("master_seed")),
            str(row.get("failure_code")),
        )
        for row in failures
        if _failure_is_fatal(manifest, row)
    }
    counts = {
        "planned_point_runs": len(expected_points),
        "observed_point_runs": len(runs),
        "missing_point_runs": len(missing_point_runs),
        "unexpected_point_runs": len(unexpected_point_runs),
        "duplicate_point_runs": duplicate_point_runs,
        "invalid_point_runs": invalid_point_runs,
        "fatal_systematic_failure_events": len(fatal_events),
        "planned_bootstrap_runs": len(expected_bootstrap_runs),
        "planned_bootstrap_resamples": len(expected_diagnostics),
        "observed_bootstrap_diagnostics": len(diagnostics),
        "missing_bootstrap_diagnostics": len(missing_diagnostics),
        "unexpected_bootstrap_diagnostics": len(unexpected_diagnostics),
        "duplicate_bootstrap_diagnostics": duplicate_diagnostics,
        "invalid_bootstrap_runs": invalid_bootstrap_runs,
        "unexpected_interval_runs": len(unexpected_interval_runs),
    }
    v4_reporting_violations = 0
    if manifest["schema_version"] in V4_METRIC_SCHEMA_VERSIONS:
        v4_counts = _v4_reporting_contract_counts(
            expected_points,
            runs,
            reliability,
            ood,
            identifiability,
            singular_values,
            features,
        )
        counts.update(v4_counts)
        v4_reporting_violations = sum(v4_counts.values())
    scope = _execution_scope(plan)
    applicable = (
        manifest["schema_version"] in MODERN_SCHEMA_VERSIONS
        and scope == "full_tier"
        and plan.tier in {"debug", "primary"}
    )
    reasons: list[dict[str, str]] = []
    if not applicable:
        reasons.append(
            {
                "code": "gate_not_applicable",
                "message": (
                    "execution gates apply only to full v2 debug or primary tiers"
                    if manifest["schema_version"] == V2_SCHEMA_VERSION
                    else (
                        "execution gates apply only to full v3/v4 debug or primary "
                        "tiers"
                    )
                ),
            }
        )
        status = "not_applicable"
    else:
        checks = [
            (
                "point_run_key_mismatch",
                counts["missing_point_runs"]
                + counts["unexpected_point_runs"]
                + counts["duplicate_point_runs"],
                "planned point-run keys were incomplete, duplicated, or unexpected",
            ),
            (
                "invalid_point_runs",
                counts["invalid_point_runs"],
                "one or more planned point runs were not status=ok with valid metrics",
            ),
            (
                "fatal_systematic_failures",
                counts["fatal_systematic_failure_events"],
                "one or more manifest-declared fatal failure events occurred",
            ),
            (
                "bootstrap_diagnostic_key_mismatch",
                counts["missing_bootstrap_diagnostics"]
                + counts["unexpected_bootstrap_diagnostics"]
                + counts["duplicate_bootstrap_diagnostics"],
                "bootstrap diagnostics were incomplete, duplicated, or unexpected",
            ),
            (
                "invalid_bootstrap_runs",
                counts["invalid_bootstrap_runs"]
                + counts["unexpected_interval_runs"],
                "bootstrap intervals were missing, invalid, or unexpected",
            ),
        ]
        if manifest["schema_version"] in V4_METRIC_SCHEMA_VERSIONS:
            checks.append(
                (
                    "reporting_contract_violation",
                    v4_reporting_violations,
                    (
                        "v4 metric applicability, reliability, OOD, feature, or "
                        "identifiability-spectrum products were incomplete or invalid"
                    ),
                )
            )
        reasons.extend(
            {"code": code, "message": message}
            for code, count, message in checks
            if count
        )
        status = "failed" if reasons else "passed"
    return {
        "schema_version": "1.0.0",
        "manifest_id": manifest["manifest_id"],
        "tier": plan.tier,
        "execution_scope": scope,
        "status": status,
        "reasons": reasons,
        "counts": counts,
    }


def _read_tsv_rows(path: Path, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != tuple(expected_fields):
            raise ValueError(
                f"archive schema mismatch for {path.name}: {reader.fieldnames}"
            )
        return list(reader)


def _verify_archive_checksums(archive: Path) -> str:
    checksum_path = archive / "output_checksums.sha256"
    if not checksum_path.is_file():
        raise ValueError(f"debug gate archive lacks {checksum_path.name}: {archive}")
    unsafe_entries = sorted(
        path.name
        for path in archive.iterdir()
        if path.is_symlink() or not path.is_file()
    )
    if unsafe_entries:
        raise ValueError(
            "debug gate archive must contain only regular top-level files: "
            f"{unsafe_entries}"
        )
    declared: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        try:
            digest, filename = line.split("  ", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"malformed checksum line in {checksum_path}") from exc
        if Path(filename).name != filename or filename in declared:
            raise ValueError(f"unsafe or duplicate checksum filename {filename!r}")
        declared[filename] = digest
    actual_files = {
        path.name
        for path in archive.iterdir()
        if path.is_file() and path.name != checksum_path.name
    }
    if set(declared) != actual_files:
        raise ValueError(
            f"checksum inventory mismatch in {archive}: declared={sorted(declared)}, "
            f"actual={sorted(actual_files)}"
        )
    for filename, expected in declared.items():
        observed = _file_sha256(archive / filename)
        if observed != expected:
            raise ValueError(
                f"checksum mismatch for {archive / filename}: expected {expected}, "
                f"got {observed}"
            )
    return _file_sha256(checksum_path)


def _validate_debug_gate_archive(
    archive: Path,
    manifest: dict[str, Any],
    current_source_sha256: dict[str, str],
) -> dict[str, Any]:
    archive = archive.resolve()
    if not archive.is_dir():
        raise ValueError(f"debug gate archive is not a directory: {archive}")
    archive_bytes = _archive_payload_bytes(archive)
    maximum_archive_bytes = int(manifest["resources"]["maximum_archive_bytes"])
    if archive_bytes > maximum_archive_bytes:
        raise ValueError(
            f"debug gate archive size {archive_bytes} exceeds manifest maximum "
            f"{maximum_archive_bytes} bytes (output_checksums.sha256 excluded): "
            f"{archive}"
        )
    checksum_manifest_sha256 = _verify_archive_checksums(archive)
    archived_manifest, archived_manifest_sha256 = _load_frozen_manifest(
        archive / "manifest.json"
    )
    if archived_manifest != manifest:
        raise ValueError(f"debug gate archive uses a different manifest: {archive}")
    provenance = json.loads((archive / "provenance.json").read_text(encoding="utf-8"))
    plan = _full_tier_plan(manifest, "debug")
    required_provenance = {
        "manifest_sha256": archived_manifest_sha256,
        "manifest_conformant": True,
        "execution_scope": "full_tier",
        "scientific_tier": "debug",
        "git_dirty": False,
        "scheduled_scenarios": list(plan.scenario_ids),
        "scheduled_master_seeds": list(plan.master_seeds),
        "source_sha256": current_source_sha256,
    }
    for key, expected in required_provenance.items():
        if provenance.get(key) != expected:
            raise ValueError(
                f"debug gate provenance {key} mismatch in {archive}: "
                f"expected {expected!r}, got {provenance.get(key)!r}"
            )
    if not provenance.get("git_commit"):
        raise ValueError(f"debug gate archive has no clean Git commit: {archive}")

    runs = _read_tsv_rows(
        archive / "runs.tsv", _schema_names(manifest["outputs"]["run_schema"])
    )
    intervals = _read_tsv_rows(
        archive / "bootstrap_intervals.tsv", BOOTSTRAP_FIELDS
    )
    diagnostics = _read_tsv_rows(
        archive / "bootstrap_diagnostics.tsv",
        _schema_names(manifest["outputs"]["bootstrap_diagnostic_schema"]),
    )
    failures = _read_tsv_rows(archive / "failures.tsv", FAILURE_FIELDS)
    if manifest["schema_version"] in V4_METRIC_SCHEMA_VERSIONS:
        reliability = _read_tsv_rows(
            archive / "reliability.tsv", RELIABILITY_FIELDS
        )
        ood = _read_tsv_rows(archive / "ood_metrics.tsv", OOD_FIELDS)
        identifiability = _read_tsv_rows(
            archive / "identifiability_diagnostics.tsv",
            V3_IDENTIFIABILITY_FIELDS,
        )
        singular_values = _read_tsv_rows(
            archive / "identifiability_singular_values.tsv",
            IDENTIFIABILITY_SINGULAR_VALUE_FIELDS,
        )
        features = _read_tsv_rows(
            archive / "feature_metrics.tsv", V4_FEATURE_FIELDS
        )
    else:
        reliability = ood = identifiability = singular_values = features = []
    evaluated = _evaluate_execution_status(
        manifest,
        plan,
        runs,
        intervals,
        diagnostics,
        failures,
        reliability,
        ood,
        identifiability,
        singular_values,
        features,
    )
    recorded = json.loads(
        (archive / "execution_status.json").read_text(encoding="utf-8")
    )
    if evaluated != recorded or evaluated["status"] != "passed":
        raise ValueError(
            f"debug gate execution status is not independently valid: {archive}"
        )
    return {
        "path": str(archive),
        "output_checksums_sha256": checksum_manifest_sha256,
        "git_commit": provenance["git_commit"],
        "manifest_sha256": archived_manifest_sha256,
    }


def _normalized_scientific_file(archive: Path, filename: str) -> bytes:
    if filename != "runs.tsv":
        return (archive / filename).read_bytes()
    with (archive / filename).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    normalized = [
        {
            key: value
            for key, value in row.items()
            if key not in {"wall_seconds", "peak_RSS_bytes"}
        }
        for row in rows
    ]
    return _canonical_json(normalized).encode("utf-8")


def _validate_primary_debug_gates(
    archives: Sequence[Path],
    manifest: dict[str, Any],
    current_source_sha256: dict[str, str],
) -> list[dict[str, Any]]:
    gate = manifest["execution_gates"]["full_primary_debug_gate"]
    required_count = int(gate["required_archive_count"])
    resolved = [path.resolve() for path in archives]
    if len(resolved) != required_count or len(set(resolved)) != required_count:
        raise ValueError(
            "full primary execution requires "
            f"{required_count} distinct debug gate archives"
        )
    metadata = [
        _validate_debug_gate_archive(path, manifest, current_source_sha256)
        for path in resolved
    ]
    if len({item["output_checksums_sha256"] for item in metadata}) != required_count:
        raise ValueError("debug gate archives must be distinct complete reruns")
    scientific_files = gate["deterministic_pair_comparison"]["scientific_files"]
    first, second = resolved
    mismatches = [
        filename
        for filename in scientific_files
        if _normalized_scientific_file(first, filename)
        != _normalized_scientific_file(second, filename)
    ]
    if mismatches:
        raise ValueError(
            "debug gate scientific products are not deterministic: "
            f"{mismatches}"
        )
    return sorted(metadata, key=lambda item: item["output_checksums_sha256"])


def _archive_payload_bytes(output: Path) -> int:
    """Return archive payload bytes, excluding the self-describing checksum file."""

    return sum(
        path.stat().st_size
        for path in output.iterdir()
        if path.is_file() and path.name != "output_checksums.sha256"
    )


def _finalize_modern_archive_size_failure(
    output: Path,
    manifest: dict[str, Any],
    plan: ExecutionPlan,
    failures: Sequence[dict[str, Any]],
    execution_status: dict[str, Any],
    observed_bytes: int,
) -> int:
    """Finalize a coherent failed modern archive and return its payload size."""

    maximum = int(manifest["resources"]["maximum_archive_bytes"])
    base_failures = [
        row
        for row in failures
        if not (
            row.get("stage") == "archive"
            and row.get("failure_code") == "maximum_archive_bytes"
        )
    ]
    reasons = [
        reason
        for reason in execution_status["reasons"]
        if reason.get("code") != "archive_size_limit"
    ]
    current = observed_bytes
    for _ in range(10):
        message = (
            f"archive payload size {current} exceeds manifest maximum {maximum} "
            "bytes; output_checksums.sha256 is excluded from both values"
        )
        execution_status["status"] = "failed"
        execution_status["reasons"] = [
            *reasons,
            {"code": "archive_size_limit", "message": message},
        ]
        execution_status["counts"].update(
            {
                "archive_bytes_excluding_output_checksums": current,
                "maximum_archive_bytes": maximum,
                "archive_size_limit_exceeded": 1,
            }
        )
        archive_failure = {
            "manifest_id": manifest["manifest_id"],
            "run_id": "archive",
            "tier": plan.tier,
            "scenario_id": None,
            "method_id": None,
            "master_seed": None,
            "stage": "archive",
            "bootstrap_index_0_based": None,
            "failure_code": "maximum_archive_bytes",
            "failure_reason": message,
        }
        _write_tsv(
            output / "failures.tsv",
            FAILURE_FIELDS,
            [*base_failures, archive_failure],
        )
        _write_json(output / "execution_status.json", execution_status)
        finalized = _archive_payload_bytes(output)
        if finalized == current:
            return finalized
        current = finalized
    raise RuntimeError("archive-size failure metadata did not reach a stable byte count")


def _archive(
    output: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    plan: ExecutionPlan,
    design: Design,
    runs: Sequence[dict[str, Any]],
    paired: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    intervals: Sequence[dict[str, Any]],
    bootstrap_diagnostics: Sequence[dict[str, Any]],
    coverage: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
    reliability: Sequence[dict[str, Any]],
    ood: Sequence[dict[str, Any]],
    identifiability: Sequence[dict[str, Any]],
    identifiability_singular_values: Sequence[dict[str, Any]],
    feature_rows: Sequence[dict[str, Any]],
    git_snapshot: tuple[str | None, bool | None],
    creation_time_utc: str,
    worker_thread_environment: dict[str, str],
    source_sha256: dict[str, str],
    execution_status: dict[str, Any] | None,
    debug_gate_archives: Sequence[dict[str, Any]],
) -> None:
    (output / "manifest.json").write_bytes(manifest_path.read_bytes())
    _write_tsv(
        output / "candidate_dictionary.tsv",
        (
            "manifest_id",
            "candidate_index_0_based",
            "tau768_index_0_based",
            "candidate_id",
            "isoform",
            "PTM_count",
            "logical_profile",
        ),
        _candidate_rows(manifest, design),
    )
    _write_tsv(
        output / "probe_panel.tsv",
        ("manifest_id", "logical_probe_index_0_based", "logical_probe_id", "alpha", "beta", "cycle_ids_1_based"),
        _probe_rows(manifest, design),
    )
    _write_tsv(output / "scenario_index.tsv", SCENARIO_INDEX_FIELDS, _scenario_index(manifest, plan))
    _write_tsv(output / "runs.tsv", _schema_names(manifest["outputs"]["run_schema"]), runs)
    paired_fields = (
        _schema_names(manifest["outputs"]["paired_contrast_schema"])
        if manifest["schema_version"] in MODERN_SCHEMA_VERSIONS
        else PAIRED_FIELDS
    )
    coverage_fields = (
        _schema_names(manifest["outputs"]["coverage_summary_schema"])
        if manifest["schema_version"] in MODERN_SCHEMA_VERSIONS
        else COVERAGE_FIELDS
    )
    diagnostic_fields = (
        _schema_names(manifest["outputs"]["bootstrap_diagnostic_schema"])
        if manifest["schema_version"] in MODERN_SCHEMA_VERSIONS
        else BOOTSTRAP_DIAGNOSTIC_FIELDS
    )
    _write_tsv(output / "paired_contrasts.tsv", paired_fields, paired)
    _write_tsv(output / "summary.tsv", _schema_names(manifest["outputs"]["summary_schema"]), summaries)
    _write_tsv(output / "bootstrap_intervals.tsv", BOOTSTRAP_FIELDS, intervals)
    _write_tsv(
        output / "bootstrap_diagnostics.tsv",
        diagnostic_fields,
        bootstrap_diagnostics,
    )
    _write_tsv(output / "coverage_summary.tsv", coverage_fields, coverage)
    _write_tsv(output / "failures.tsv", FAILURE_FIELDS, failures)
    _write_tsv(output / "reliability.tsv", RELIABILITY_FIELDS, reliability)
    _write_tsv(output / "ood_metrics.tsv", OOD_FIELDS, ood)
    identifiability_fields = (
        V3_IDENTIFIABILITY_FIELDS
        if manifest["schema_version"] in DIRECT_SVD_SCHEMA_VERSIONS
        else IDENTIFIABILITY_FIELDS
    )
    _write_tsv(
        output / "identifiability_diagnostics.tsv",
        identifiability_fields,
        identifiability,
    )
    if manifest["schema_version"] in V4_METRIC_SCHEMA_VERSIONS:
        _write_tsv(
            output / "identifiability_singular_values.tsv",
            IDENTIFIABILITY_SINGULAR_VALUE_FIELDS,
            identifiability_singular_values,
        )
    _write_tsv(
        output / "feature_metrics.tsv",
        (
            V4_FEATURE_FIELDS
            if manifest["schema_version"] in V4_METRIC_SCHEMA_VERSIONS
            else FEATURE_FIELDS
        ),
        feature_rows,
    )
    figure_fields = (
        _schema_names(manifest["outputs"]["figure_data_schema"])
        if manifest["schema_version"] in MODERN_SCHEMA_VERSIONS
        else FIGURE_FIELDS
    )
    _write_tsv(
        output / "figure_data.tsv",
        figure_fields,
        _figure_rows(manifest, summaries, reliability),
    )

    commit, dirty = git_snapshot
    if manifest["schema_version"] in MODERN_SCHEMA_VERSIONS:
        assert execution_status is not None
        _write_json(output / "execution_status.json", execution_status)
    provenance = {
        "runner_version": (
            V4_RUNNER_VERSION
            if manifest["schema_version"] in V4_METRIC_SCHEMA_VERSIONS
            else RUNNER_VERSION
        ),
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": _file_sha256(manifest_path),
        "manifest_conformant": plan.manifest_conformant,
        "execution_scope": "smoke" if plan.tier == "smoke" else ("preflight" if plan.preflight else "full_tier"),
        "scientific_tier": plan.scientific_tier,
        "scheduled_scenarios": list(plan.scenario_ids),
        "scheduled_master_seeds": list(plan.master_seeds),
        "overrides": plan.overrides,
        "headline_eligible": plan.headline_eligible,
        "git_commit": commit,
        "git_dirty": dirty,
        "command": sys.argv,
        "creation_time_utc": creation_time_utc,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "dependency_lock_sha256": _file_sha256(REPO_ROOT / "uv.lock"),
        "source_sha256": source_sha256,
        "hardware_without_device_identifiers": {
            "platform": platform.system(),
            "machine_architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        },
        "contains_actual_nautilus_data": False,
        "fitted_to_actual_nautilus_raw_data": False,
        "worker_process_isolation": bool(
            manifest["resources"]["worker_process_isolation"]
        ),
        "maximum_concurrent_workers": int(
            manifest["resources"]["maximum_concurrent_primary_workers"]
        ),
        "worker_blas_threads": manifest["resources"].get("worker_blas_threads"),
        "worker_thread_environment": worker_thread_environment,
        "maximum_rss_bytes_per_worker": int(
            manifest["resources"]["maximum_rss_bytes_per_worker"]
        ),
        "deterministic_archive_order": "seed_major_then_scenario_manifest_order",
        "debug_gate_archives": list(debug_gate_archives),
    }
    _write_json(output / "provenance.json", provenance)
    archive_bytes = _archive_payload_bytes(output)
    maximum_archive_bytes = int(manifest["resources"]["maximum_archive_bytes"])
    full_modern = (
        manifest["schema_version"] in MODERN_SCHEMA_VERSIONS
        and execution_status is not None
        and execution_status["execution_scope"] == "full_tier"
    )
    if archive_bytes > maximum_archive_bytes and full_modern:
        archive_bytes = _finalize_modern_archive_size_failure(
            output,
            manifest,
            plan,
            failures,
            execution_status,
            archive_bytes,
        )
    files = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "output_checksums.sha256"
    )
    checksum_text = "".join(f"{_file_sha256(path)}  {path.name}\n" for path in files)
    (output / "output_checksums.sha256").write_text(checksum_text, encoding="utf-8")
    enforced_archive_bytes = archive_bytes
    if manifest["schema_version"] == V1_SCHEMA_VERSION:
        # Preserve the frozen v1 definition, which included the checksum file.
        enforced_archive_bytes = sum(
            path.stat().st_size for path in output.iterdir() if path.is_file()
        )
    if enforced_archive_bytes > maximum_archive_bytes and not full_modern:
        detail = (
            f"archive size {enforced_archive_bytes} exceeds manifest maximum "
            f"{maximum_archive_bytes}"
            if manifest["schema_version"] == V1_SCHEMA_VERSION
            else (
                f"archive payload size {enforced_archive_bytes} exceeds manifest "
                f"maximum {maximum_archive_bytes} bytes; "
                "output_checksums.sha256 is excluded"
            )
        )
        raise ResourceLimit(
            "maximum_archive_bytes",
            detail,
        )


def _systematic_stop_code(
    failure_counts: dict[tuple[str, str], int],
    scenario_id: str,
    threshold: int = 3,
) -> str | None:
    candidates = sorted(
        code
        for (scenario, code), count in failure_counts.items()
        if scenario == scenario_id and count >= threshold
    )
    return candidates[0] if candidates else None


def _failure_is_fatal(
    manifest: dict[str, Any], failure: dict[str, Any]
) -> bool:
    stage = failure.get("stage")
    code = failure.get("failure_code")
    if code in {None, "", "systematic_failure_stop"}:
        return False
    policy = manifest["stopping_and_failures"]["systematic_failure_stop_policy"]
    for rule in policy["fatal_failure_rules"]:
        if rule["stage"] != stage:
            continue
        declared_codes = rule["failure_codes"]
        if declared_codes == "all_except_systematic_failure_stop":
            return True
        return code in declared_codes
    return False


def _execute_scenario_seed(
    manifest: dict[str, Any],
    plan: ExecutionPlan,
    design: Design,
    scenario_id: str,
    master_seed: int,
    seed_index: int,
    systematic_stop_code: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    bundle: dict[str, list[dict[str, Any]]] = {
        "runs": [],
        "intervals": [],
        "bootstrap_diagnostics": [],
        "failures": [],
        "reliability": [],
        "ood": [],
        "identifiability": [],
        "identifiability_singular_values": [],
        "features": [],
    }
    state = _prepare_seed_state(manifest, design, plan, master_seed)
    dataset = _build_dataset(scenario_id, manifest, design, state, {})
    eligible_bootstrap = set(manifest["bootstrap"]["eligible_primary_scenarios"])
    point_limit = float(
        manifest["resources"]["point_estimate_wall_seconds_per_scenario_seed"]
    )
    point_started = time.monotonic()
    bootstrap_jobs: list[
        tuple[
            dict[str, Any],
            MethodFit,
            np.ndarray,
            tuple[tuple[int, ...], ...],
        ]
    ] = []
    for method_id in _method_ids(manifest, plan, scenario_id):
        if systematic_stop_code is not None:
            row = _blank_run(
                manifest, plan, scenario_id, method_id, master_seed, seed_index
            )
            fitted_q, fit_indices = _fitted_q_and_indices(
                scenario_id, manifest, design, state, dataset
            )
            row.update(
                {
                    "n_candidates_fit": int(fit_indices.size),
                    "n_active_truth": dataset.n_active_truth,
                    "missing_fraction": float(np.mean(dataset.observations == -1)),
                    "status": "resource_skipped",
                    "failure_code": "systematic_failure_stop",
                    "failure_reason": (
                        "scenario scheduling stopped after three failures with code "
                        f"{systematic_stop_code}"
                    ),
                    "data_sha256": dataset.data_sha256,
                }
            )
            row["result_sha256"] = _result_hash(row)
            fit = fit_indices = groups = None
            run_reliability: list[dict[str, Any]] = []
            ood_row = feature = None
        else:
            point_remaining = max(
                0.0, point_limit - (time.monotonic() - point_started)
            )
            (
                row,
                fit,
                fit_indices,
                groups,
                run_reliability,
                ood_row,
                feature,
            ) = _run_one_method(
                manifest,
                plan,
                design,
                state,
                dataset,
                scenario_id,
                method_id,
                master_seed,
                seed_index,
                point_wall_seconds=point_remaining,
            )
        bundle["runs"].append(row)
        bundle["reliability"].extend(run_reliability)
        if ood_row is not None:
            bundle["ood"].append(ood_row)
        if feature is not None:
            bundle["features"].append(
                {
                    "manifest_id": manifest["manifest_id"],
                    "run_id": row["run_id"],
                    "tier": plan.tier,
                    "scenario_id": scenario_id,
                    "method_id": method_id,
                    "master_seed": master_seed,
                    **feature,
                }
            )
        if row["status"] != "ok":
            bundle["failures"].append(
                {
                    "manifest_id": manifest["manifest_id"],
                    "run_id": row["run_id"],
                    "tier": plan.tier,
                    "scenario_id": scenario_id,
                    "method_id": method_id,
                    "master_seed": master_seed,
                    "stage": "point_estimate",
                    "bootstrap_index_0_based": None,
                    "failure_code": row.get("failure_code"),
                    "failure_reason": row.get("failure_reason"),
                }
            )
        if (
            fit is not None
            and fit_indices is not None
            and groups is not None
            and row["metrics_valid"]
            and method_id == manifest["bootstrap"]["method"]
            and scenario_id in eligible_bootstrap
            and not plan.skip_bootstrap
        ):
            bootstrap_jobs.append((row, fit, fit_indices, groups))
        if (
            fit is not None
            and fit_indices is not None
            and groups is not None
            and row["metrics_valid"]
            and method_id == "em_fixed_q"
            and manifest["scenario_definitions"][scenario_id]["family"]
            in {
                "near_identifiability",
                "exact_identifiability",
                "missingness_and_exact_identifiability",
            }
        ):
            fitted_q, _ = _fitted_q_and_indices(
                scenario_id, manifest, design, state, dataset
            )
            if manifest["schema_version"] in DIRECT_SVD_SCHEMA_VERSIONS:
                with _deadline(
                    float(
                        manifest["resources"][
                            "identifiability_wall_seconds_per_scenario_seed"
                        ]
                    ),
                    "identifiability_wall_seconds",
                ):
                    identifiability_row = _identifiability_row(
                        manifest,
                        plan,
                        row,
                        fit,
                        fit_indices,
                        groups,
                        design,
                        dataset,
                        fitted_q,
                    )
            else:
                identifiability_row = _identifiability_row(
                    manifest,
                    plan,
                    row,
                    fit,
                    fit_indices,
                    groups,
                    design,
                    dataset,
                    fitted_q,
                )
            singular_values = identifiability_row.pop("_singular_values", None)
            bundle["identifiability"].append(identifiability_row)
            if singular_values is not None:
                largest = float(singular_values[0]) if singular_values.size else 0.0
                threshold = (
                    float(identifiability_row["relative_singular_value_floor"])
                    * largest
                )
                bundle["identifiability_singular_values"].extend(
                    {
                        "manifest_id": manifest["manifest_id"],
                        "run_id": row["run_id"],
                        "tier": plan.tier,
                        "scenario_id": scenario_id,
                        "method_id": method_id,
                        "master_seed": master_seed,
                        "singular_value_index_0_based": index,
                        "singular_value": float(value),
                        "retained_above_relative_floor": bool(value > threshold),
                    }
                    for index, value in enumerate(singular_values)
                )
    # Bootstrap has its own scenario-seed budget and therefore starts only
    # after every point method has either completed or been explicitly skipped
    # under the single shared point-estimation budget.
    for row, fit, fit_indices, groups in bootstrap_jobs:
        run_intervals, run_failures, run_diagnostics = _bootstrap_run(
            manifest, plan, row, fit, fit_indices, groups, design, dataset
        )
        bundle["intervals"].extend(run_intervals)
        bundle["bootstrap_diagnostics"].extend(run_diagnostics)
        bundle["failures"].extend(run_failures)
    return bundle


def _worker_entry(
    send_connection: Any,
    manifest: dict[str, Any],
    plan: ExecutionPlan,
    design: Design,
    scenario_id: str,
    master_seed: int,
    seed_index: int,
    systematic_stop_code: str | None,
) -> None:
    stop_monitor = threading.Event()
    maximum = int(manifest["resources"]["maximum_rss_bytes_per_worker"])

    def monitor_rss() -> None:
        while not stop_monitor.wait(0.05):
            if _peak_rss_bytes() > maximum:
                os._exit(88)

    monitor = threading.Thread(target=monitor_rss, daemon=True)
    monitor.start()
    try:
        _verify_worker_thread_environment(manifest)
        if hasattr(resource, "RLIMIT_AS"):
            _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            effective = maximum if hard == resource.RLIM_INFINITY else min(maximum, hard)
            # The hard limit can safely be lowered in this disposable worker;
            # doing so is portable across macOS/Python combinations that
            # reject a finite soft limit paired with RLIM_INFINITY.
            try:
                resource.setrlimit(resource.RLIMIT_AS, (effective, effective))
            except (OSError, ValueError):
                # Some macOS builds expose RLIMIT_AS but reject any finite
                # value.  The child background monitor independently
                # terminates on observed RSS, so inability to install this
                # additional address-space guard does not disable the limit.
                pass
        payload: dict[str, Any] = {
            "bundle": _execute_scenario_seed(
                manifest,
                plan,
                design,
                scenario_id,
                master_seed,
                seed_index,
                systematic_stop_code,
            )
        }
    except BaseException as exc:
        payload = {
            "worker_error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "failure_code": (
                    exc.code if isinstance(exc, ResourceLimit) else "worker_exception"
                ),
            }
        }
    if _peak_rss_bytes() > maximum:
        os._exit(88)
    stop_monitor.set()
    monitor.join(timeout=0.2)
    try:
        send_connection.send(payload)
    finally:
        send_connection.close()


def _worker_failure_bundle(
    manifest: dict[str, Any],
    plan: ExecutionPlan,
    scenario_id: str,
    master_seed: int,
    seed_index: int,
    code: str,
    reason: str,
) -> dict[str, list[dict[str, Any]]]:
    status = "resource_skipped" if code in RESOURCE_TERMINATION_CODES else "exception"
    bundle = {
        "runs": [],
        "intervals": [],
        "bootstrap_diagnostics": [],
        "failures": [],
        "reliability": [],
        "ood": [],
        "identifiability": [],
        "identifiability_singular_values": [],
        "features": [],
    }
    for method_id in _method_ids(manifest, plan, scenario_id):
        row = _blank_run(
            manifest, plan, scenario_id, method_id, master_seed, seed_index
        )
        row.update(
            {
                "status": status,
                "failure_code": code,
                "failure_reason": reason,
                "data_sha256": "unavailable_due_worker_termination",
            }
        )
        row["result_sha256"] = _result_hash(row)
        bundle["runs"].append(row)
        bundle["failures"].append(
            {
                "manifest_id": manifest["manifest_id"],
                "run_id": row["run_id"],
                "tier": plan.tier,
                "scenario_id": scenario_id,
                "method_id": method_id,
                "master_seed": master_seed,
                "stage": "worker",
                "bootstrap_index_0_based": None,
                "failure_code": code,
                "failure_reason": reason,
            }
        )
    return bundle


def _systematic_stop_bundle(
    manifest: dict[str, Any],
    plan: ExecutionPlan,
    scenario_id: str,
    master_seed: int,
    seed_index: int,
    repeated_code: str,
) -> dict[str, list[dict[str, Any]]]:
    bundle: dict[str, list[dict[str, Any]]] = {
        "runs": [],
        "intervals": [],
        "bootstrap_diagnostics": [],
        "failures": [],
        "reliability": [],
        "ood": [],
        "identifiability": [],
        "identifiability_singular_values": [],
        "features": [],
    }
    reason = (
        "scenario was not scheduled after three prior paired-seed failures "
        f"with code {repeated_code}"
    )
    for method_id in _method_ids(manifest, plan, scenario_id):
        row = _blank_run(
            manifest, plan, scenario_id, method_id, master_seed, seed_index
        )
        row.update(
            {
                "status": "resource_skipped",
                "failure_code": "systematic_failure_stop",
                "failure_reason": reason,
                "data_sha256": "not_generated_due_systematic_failure_stop",
            }
        )
        row["result_sha256"] = _result_hash(row)
        bundle["runs"].append(row)
        bundle["failures"].append(
            {
                "manifest_id": manifest["manifest_id"],
                "run_id": row["run_id"],
                "tier": plan.tier,
                "scenario_id": scenario_id,
                "method_id": method_id,
                "master_seed": master_seed,
                "stage": "systematic_stop",
                "bootstrap_index_0_based": None,
                "failure_code": "systematic_failure_stop",
                "failure_reason": reason,
            }
        )
    return bundle


def _worker_timeout_seconds(
    manifest: dict[str, Any],
    plan: ExecutionPlan,
    scenario_id: str,
    systematic_stop_code: str | None,
) -> float:
    methods = _method_ids(manifest, plan, scenario_id)
    bootstrap_scheduled = (
        systematic_stop_code is None
        and not plan.skip_bootstrap
        and manifest["bootstrap"]["method"] in methods
        and scenario_id in set(manifest["bootstrap"]["eligible_primary_scenarios"])
    )
    identifiability_scheduled = (
        manifest["schema_version"] in DIRECT_SVD_SCHEMA_VERSIONS
        and systematic_stop_code is None
        and "em_fixed_q" in methods
        and manifest["scenario_definitions"][scenario_id]["family"]
        in {
            "near_identifiability",
            "exact_identifiability",
            "missingness_and_exact_identifiability",
        }
    )
    return (
        float(manifest["resources"]["point_estimate_wall_seconds_per_scenario_seed"])
        + (
            float(manifest["resources"]["bootstrap_wall_seconds_per_scenario_seed"])
            if bootstrap_scheduled
            else 0.0
        )
        + (
            float(
                manifest["resources"][
                    "identifiability_wall_seconds_per_scenario_seed"
                ]
            )
            if identifiability_scheduled
            else 0.0
        )
        + WORKER_GRACE_SECONDS
    )


def _worker_exit_failure(exitcode: int | None) -> tuple[str, str]:
    if exitcode == 88:
        return (
            "maximum_rss_bytes_per_worker",
            "isolated worker self-terminated after exceeding the manifest RSS limit",
        )
    return (
        "worker_exception",
        f"isolated worker exited with code {exitcode} without a result",
    )


def _run_isolated_scenario_seed(
    manifest: dict[str, Any],
    plan: ExecutionPlan,
    design: Design,
    scenario_id: str,
    master_seed: int,
    seed_index: int,
    systematic_stop_code: str | None,
) -> dict[str, list[dict[str, Any]]]:
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    worker = context.Process(
        target=_worker_entry,
        args=(
            send,
            manifest,
            plan,
            design,
            scenario_id,
            master_seed,
            seed_index,
            systematic_stop_code,
        ),
    )
    worker.start()
    send.close()
    timeout = _worker_timeout_seconds(
        manifest, plan, scenario_id, systematic_stop_code
    )
    started = time.monotonic()
    payload: dict[str, Any] | None = None
    termination_code: str | None = None
    termination_reason: str | None = None
    while time.monotonic() - started <= timeout:
        if receive.poll(0.1):
            try:
                payload = receive.recv()
            except EOFError:
                worker.join(timeout=1.0)
                termination_code, termination_reason = _worker_exit_failure(
                    worker.exitcode
                )
            break
        if not worker.is_alive() and payload is None:
            termination_code, termination_reason = _worker_exit_failure(
                worker.exitcode
            )
            break
    if payload is None and termination_code is None:
        termination_code = "worker_wall_timeout"
        termination_reason = f"isolated scenario-seed worker exceeded {timeout} seconds"
    if worker.is_alive():
        worker.terminate()
    worker.join(timeout=10.0)
    receive.close()
    if termination_code is not None:
        return _worker_failure_bundle(
            manifest,
            plan,
            scenario_id,
            master_seed,
            seed_index,
            termination_code,
            str(termination_reason),
        )
    assert payload is not None
    if "bundle" in payload:
        return payload["bundle"]
    error = payload["worker_error"]
    return _worker_failure_bundle(
        manifest,
        plan,
        scenario_id,
        master_seed,
        seed_index,
        str(error.get("failure_code", "worker_exception")),
        f"{error['type']}: {error['message']}",
    )


def _run_seed_batch(
    manifest: dict[str, Any],
    plan: ExecutionPlan,
    design: Design,
    master_seed: int,
    seed_index: int,
    systematic_stop_codes: dict[str, str | None],
    run_scenario: Callable[
        [dict[str, Any], ExecutionPlan, Design, str, int, int, str | None],
        dict[str, list[dict[str, Any]]],
    ],
) -> list[dict[str, list[dict[str, Any]]]]:
    """Run one paired-seed scenario batch and return manifest-ordered bundles."""

    def invoke(scenario_id: str) -> dict[str, list[dict[str, Any]]]:
        repeated_code = systematic_stop_codes[scenario_id]
        if (
            repeated_code is not None
            and manifest["schema_version"] in MODERN_SCHEMA_VERSIONS
        ):
            return _systematic_stop_bundle(
                manifest,
                plan,
                scenario_id,
                master_seed,
                seed_index,
                repeated_code,
            )
        try:
            return run_scenario(
                manifest,
                plan,
                design,
                scenario_id,
                master_seed,
                seed_index,
                repeated_code,
            )
        except Exception as exc:
            return _worker_failure_bundle(
                manifest,
                plan,
                scenario_id,
                master_seed,
                seed_index,
                "worker_exception",
                f"parent scheduler caught {type(exc).__name__}: {exc}",
            )

    maximum_workers = int(
        manifest["resources"]["maximum_concurrent_primary_workers"]
    )
    if maximum_workers == 1:
        return [invoke(scenario_id) for scenario_id in plan.scenario_ids]
    with ThreadPoolExecutor(
        max_workers=maximum_workers,
        thread_name_prefix="proteoem-scenario",
    ) as executor:
        futures = [executor.submit(invoke, scenario_id) for scenario_id in plan.scenario_ids]
        # Waiting in submission order makes archive order independent of worker
        # completion order while later futures continue running concurrently.
        return [future.result() for future in futures]


def _record_systematic_failures(
    manifest: dict[str, Any],
    repeated_failures: dict[tuple[str, str], int],
    scenario_id: str,
    bundle: dict[str, list[dict[str, Any]]],
) -> None:
    if manifest["schema_version"] == V1_SCHEMA_VERSION:
        # Preserve the frozen v1 runner semantics exactly: only point-estimate
        # rows count, and each failed method row increments the counter.
        for failure in bundle["failures"]:
            if failure.get("stage") != "point_estimate":
                continue
            code = str(failure.get("failure_code"))
            if code == "systematic_failure_stop":
                continue
            key = (scenario_id, code)
            repeated_failures[key] = repeated_failures.get(key, 0) + 1
        return

    # Modern manifests count one fatal scenario-seed event per code even when several
    # inherited method rows describe the same worker/resource termination.
    codes = {
        str(failure.get("failure_code"))
        for failure in bundle["failures"]
        if _failure_is_fatal(manifest, failure)
    }
    for code in sorted(codes):
        key = (scenario_id, code)
        repeated_failures[key] = repeated_failures.get(key, 0) + 1


def _iter_scenario_seed_bundles(
    manifest: dict[str, Any],
    plan: ExecutionPlan,
    design: Design,
    *,
    run_scenario: Callable[
        [dict[str, Any], ExecutionPlan, Design, str, int, int, str | None],
        dict[str, list[dict[str, Any]]],
    ] | None = None,
) -> Iterable[tuple[int, str, dict[str, list[dict[str, Any]]]]]:
    """Yield deterministic seed-major bundles with between-seed stop decisions."""

    scenario_runner = run_scenario or _run_isolated_scenario_seed
    repeated_failures: dict[tuple[str, str], int] = {}
    threshold = (
        int(
            manifest["stopping_and_failures"]["systematic_failure_stop_policy"][
                "threshold_failed_scenario_seed_attempts"
            ]
        )
        if manifest["schema_version"] in MODERN_SCHEMA_VERSIONS
        else 3
    )
    for seed_index, master_seed in enumerate(plan.master_seeds, start=1):
        stop_codes = {
            scenario_id: _systematic_stop_code(
                repeated_failures, scenario_id, threshold
            )
            for scenario_id in plan.scenario_ids
        }
        bundles = _run_seed_batch(
            manifest,
            plan,
            design,
            master_seed,
            seed_index,
            stop_codes,
            scenario_runner,
        )
        for scenario_id, bundle in zip(plan.scenario_ids, bundles, strict=True):
            _record_systematic_failures(
                manifest, repeated_failures, scenario_id, bundle
            )
            yield master_seed, scenario_id, bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tier", choices=("debug", "primary"))
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true", help="schedule an exact manifest subset without headline eligibility")
    parser.add_argument("--scenario-id", action="append")
    parser.add_argument("--master-seed", type=int, action="append")
    parser.add_argument("--method-id", action="append")
    parser.add_argument("--skip-bootstrap", action="store_true", help="preflight timing only; never valid for a full tier archive")
    parser.add_argument(
        "--debug-gate-archive",
        type=Path,
        action="append",
        default=[],
        help="repeat twice for a full modern-manifest primary run",
    )
    return parser


def _default_output_path(
    manifest: dict[str, Any], plan: ExecutionPlan
) -> Path:
    if manifest["schema_version"] == V1_SCHEMA_VERSION:
        legacy_name = (
            "model-violation-benchmark-smoke"
            if plan.tier == "smoke"
            else f"model-violation-benchmark-{plan.tier}"
        )
        return (REPO_ROOT / "outputs" / legacy_name).resolve()
    declared_root = (REPO_ROOT / str(manifest["outputs"]["root"])).resolve()
    if plan.tier == "primary":
        return declared_root
    return declared_root.with_name(f"{declared_root.name}-{plan.tier}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    manifest, _manifest_sha256 = _load_frozen_manifest(manifest_path)
    plan = _build_plan(args, manifest)
    git_snapshot = _git_metadata()
    full_modern = (
        manifest["schema_version"] in MODERN_SCHEMA_VERSIONS
        and not plan.preflight
        and plan.tier in {"debug", "primary"}
    )
    if args.debug_gate_archive and not (full_modern and plan.tier == "primary"):
        raise ValueError(
            "--debug-gate-archive is accepted only for a full modern-manifest "
            "primary run"
        )
    if full_modern and manifest["execution_gates"]["full_tier_requires_clean_git"]:
        if git_snapshot[1] is not False:
            raise RuntimeError(
                "full modern-manifest debug/primary execution requires a clean "
                "Git snapshot"
            )
    source_sha256 = _source_sha256_map()
    debug_gate_archives: list[dict[str, Any]] = []
    if full_modern and plan.tier == "primary":
        debug_gate_archives = _validate_primary_debug_gates(
            args.debug_gate_archive, manifest, source_sha256
        )
    worker_thread_environment = _configure_worker_thread_environment(manifest)
    creation_time_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    output = (args.output or _default_output_path(manifest, plan)).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty archive: {output}")
    output.mkdir(parents=True, exist_ok=True)
    design = _prepare_design(manifest, plan)

    runs: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    bootstrap_diagnostics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    reliability: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []
    identifiability_rows: list[dict[str, Any]] = []
    identifiability_singular_value_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for _master_seed, _scenario_id, bundle in _iter_scenario_seed_bundles(
        manifest, plan, design
    ):
        runs.extend(bundle["runs"])
        intervals.extend(bundle["intervals"])
        bootstrap_diagnostics.extend(bundle["bootstrap_diagnostics"])
        failures.extend(bundle["failures"])
        reliability.extend(bundle["reliability"])
        ood_rows.extend(bundle["ood"])
        identifiability_rows.extend(bundle["identifiability"])
        identifiability_singular_value_rows.extend(
            bundle["identifiability_singular_values"]
        )
        feature_rows.extend(bundle["features"])

    paired = _paired_rows(manifest, plan, runs)
    summaries = _summary_rows(manifest, plan, runs, paired)
    coverage = _coverage_summary(manifest, plan, intervals)
    execution_status = (
        _evaluate_execution_status(
            manifest,
            plan,
            runs,
            intervals,
            bootstrap_diagnostics,
            failures,
            reliability,
            ood_rows,
            identifiability_rows,
            identifiability_singular_value_rows,
            feature_rows,
        )
        if manifest["schema_version"] in MODERN_SCHEMA_VERSIONS
        else None
    )
    _archive(
        output,
        manifest_path,
        manifest,
        plan,
        design,
        runs,
        paired,
        summaries,
        intervals,
        bootstrap_diagnostics,
        coverage,
        failures,
        reliability,
        ood_rows,
        identifiability_rows,
        identifiability_singular_value_rows,
        feature_rows,
        git_snapshot,
        creation_time_utc,
        worker_thread_environment,
        source_sha256,
        execution_status,
        debug_gate_archives,
    )
    return int(execution_status is not None and execution_status["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
