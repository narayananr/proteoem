#!/usr/bin/env python3
"""Render Figure 5 from the archived observation-yield benchmark tables."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "outputs" / "observation-yield-benchmark"
DEFAULT_OUTPUT = REPO_ROOT / "figures" / "figure5_observation_yield_benchmark.svg"

SCENARIOS = ("all_traces", "any_positive", "tau_anchor")
DISPLAY_SCENARIOS = (*SCENARIOS, "confidence_hard_decode")
SCENARIO_LABELS = {
    "all_traces": "All traces",
    "any_positive": "Any-positive gate",
    "tau_anchor": "Synthetic N+C gate",
    "confidence_hard_decode": "Confidence-filtered decode",
}
SCENARIO_COLORS = {
    "all_traces": "#1B9E77",
    "any_positive": "#4C78A8",
    "tau_anchor": "#D95F02",
    "confidence_hard_decode": "#7570B3",
}
SCENARIO_SHAPES = {
    "all_traces": "circle",
    "any_positive": "square",
    "tau_anchor": "triangle",
    "confidence_hard_decode": "diamond",
}
CORRECTIONS = (
    "oracle_yield",
    "misspecified_yield",
    "equal_yield",
    "raw_probe_count",
    "raw_protein_length",
)
CORRECTION_LABELS = {
    "oracle_yield": "Oracle e = r × v",
    "misspecified_yield": "Misspecified e",
    "equal_yield": "Equal yield",
    "raw_probe_count": "Probe-count proxy",
    "raw_protein_length": "Protein-length proxy",
}


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _text(
    x: float,
    y: float,
    value: str,
    css: str,
    *,
    anchor: str | None = None,
    fill: str | None = None,
) -> str:
    attrs = [f'x="{x:.1f}"', f'y="{y:.1f}"', f'class="t {css}"']
    if anchor:
        attrs.append(f'text-anchor="{anchor}"')
    if fill:
        attrs.append(f'style="fill:{fill}"')
    return f"<text {' '.join(attrs)}>{escape(value)}</text>"


def _stacked_bar(x: float, y: float, width: float, share_a: float) -> list[str]:
    if not 0.0 <= share_a <= 1.0:
        raise ValueError(f"Composition share outside [0, 1]: {share_a}")
    a_width = width * share_a
    return [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{a_width:.1f}" height="44" rx="7" fill="#4C78A8"/>',
        f'<path d="M{x + a_width:.1f},{y:.1f} H{x + width - 7:.1f} '
        f'Q{x + width:.1f},{y:.1f} {x + width:.1f},{y + 7:.1f} '
        f'V{y + 37:.1f} Q{x + width:.1f},{y + 44:.1f} {x + width - 7:.1f},{y + 44:.1f} '
        f'H{x + a_width:.1f} Z" fill="#F28E2B"/>',
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="44" rx="7" fill="none" stroke="#AEBBC6" stroke-width="2"/>',
        _text(x + a_width / 2, y + 31, f"{100 * share_a:.1f}", "bar", anchor="middle", fill="#FFFFFF"),
        _text(
            x + a_width + (width - a_width) / 2,
            y + 31,
            f"{100 * (1 - share_a):.1f}",
            "bar",
            anchor="middle",
            fill="#17324D",
        ),
    ]


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty value list")
    return sum(values) / len(values)


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _x(value: float, left: float, width: float, maximum: float) -> float:
    return left + width * value / maximum


def _marker(
    x: float,
    y: float,
    *,
    color: str,
    shape: str,
    size: float,
    opacity: float = 1.0,
    stroke: str | None = None,
) -> str:
    stroke_color = stroke or color
    common = f'fill="{color}" stroke="{stroke_color}" stroke-width="2" opacity="{opacity:.2f}"'
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size:.1f}" {common}/>'
    if shape == "square":
        side = 2 * size
        return f'<rect x="{x - size:.1f}" y="{y - size:.1f}" width="{side:.1f}" height="{side:.1f}" rx="2" {common}/>'
    if shape == "triangle":
        return (
            f'<path d="M{x:.1f},{y - size:.1f} L{x + size:.1f},{y + size:.1f} '
            f'L{x - size:.1f},{y + size:.1f} Z" {common}/>'
        )
    if shape == "diamond":
        return (
            f'<path d="M{x:.1f},{y - size:.1f} L{x + size:.1f},{y:.1f} '
            f'L{x:.1f},{y + size:.1f} L{x - size:.1f},{y:.1f} Z" {common}/>'
        )
    raise ValueError(f"Unknown marker shape: {shape}")


def _error_mark(
    values: list[float],
    y: float,
    *,
    left: float,
    width: float,
    maximum: float,
    color: str,
    shape: str,
    label: str | None = None,
) -> list[str]:
    if any(value < 0.0 or value > maximum for value in values):
        raise ValueError(f"Values fall outside plotting range 0..{maximum}: {values}")
    mean = _mean(values)
    sd = _sample_sd(values)
    mean_x = _x(mean, left, width, maximum)
    low_x = _x(max(0.0, mean - sd), left, width, maximum)
    high_x = _x(min(maximum, mean + sd), left, width, maximum)
    result = [
        f'<line x1="{low_x:.1f}" y1="{y:.1f}" x2="{high_x:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="5"/>',
        f'<line x1="{low_x:.1f}" y1="{y - 9:.1f}" x2="{low_x:.1f}" y2="{y + 9:.1f}" stroke="{color}" stroke-width="4"/>',
        f'<line x1="{high_x:.1f}" y1="{y - 9:.1f}" x2="{high_x:.1f}" y2="{y + 9:.1f}" stroke="{color}" stroke-width="4"/>',
    ]
    count = len(values)
    offsets = [0.0] if count == 1 else [(-12.0 + 24.0 * index / (count - 1)) for index in range(count)]
    for value, offset in zip(values, offsets, strict=True):
        result.append(
            _marker(
                _x(value, left, width, maximum),
                y + offset,
                color=color,
                shape=shape,
                size=5.5,
                opacity=0.42,
            )
        )
    result.append(
        _marker(mean_x, y, color=color, shape=shape, size=9.5, stroke="#FFFFFF")
    )
    if label:
        result.append(_text(mean_x + 17, y - 13, label, "value", fill=color))
    return result


def _validate_inputs(
    truth_rows: list[dict[str, str]], runs: list[dict[str, str]]
) -> dict[str, dict[str, str]]:
    truth: dict[str, dict[str, str]] = {}
    for row in truth_rows:
        scenario = row["scenario"]
        if scenario in truth:
            raise ValueError(f"Duplicate scenario truth row: {scenario}")
        truth[scenario] = row
    missing_truth = set(DISPLAY_SCENARIOS) - set(truth)
    if missing_truth:
        raise ValueError(f"Missing scenario truth rows: {sorted(missing_truth)}")

    for scenario in SCENARIOS:
        for correction in CORRECTIONS:
            selected = [
                row
                for row in runs
                if row["scenario"] == scenario and row["correction"] == correction
            ]
            if not selected:
                raise ValueError(f"No runs for {scenario}/{correction}")
            seeds = [row["seed"] for row in selected]
            if len(seeds) != len(set(seeds)):
                raise ValueError(f"Duplicate seeds for {scenario}/{correction}")

    hard_rows = [
        row
        for row in runs
        if row["scenario"] == "confidence_hard_decode"
        and row["correction"] == "oracle_yield"
    ]
    if not hard_rows:
        raise ValueError("No confidence-filtered hard-decode runs")
    return truth


def render(input_dir: Path, output: Path) -> None:
    truth_rows = _read_tsv(input_dir / "scenario_truth.tsv")
    runs = _read_tsv(input_dir / "runs.tsv")
    truth = _validate_inputs(truth_rows, runs)
    oracle_runs = [row for row in runs if row["correction"] == "oracle_yield"]

    source_shares = [_float(truth[scenario], "source_A_population") for scenario in DISPLAY_SCENARIOS]
    if max(source_shares) - min(source_shares) > 1e-12:
        raise ValueError("Panel A requires a common source composition across scenarios")
    source_share = source_shares[0]

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1550" viewBox="0 0 1800 1550" role="img" aria-labelledby="title desc">',
        '<title id="title">Observation yield separates accepted-trace composition from source composition</title>',
        '<desc id="desc">Panel A shows one common source composition transformed by four recovery and retention rules into different accepted or retained populations. Panel B evaluates valid accepted-composition fits on a magnified low-error axis and isolates the unconditioned any-positive likelihood as a negative control. Panel C applies alternative effective-yield assumptions to the same accepted-composition fit and compares source-composition error using color and marker shape for each scenario.</desc>',
        '<style>.t{font-family:Arial,Helvetica,sans-serif;fill:#17324D}.panel{font-size:54px;font-weight:700}.head{font-size:39px;font-weight:700}.body{font-size:32px}.small{font-size:29px;fill:#536475}.compact{font-size:25px;fill:#536475}.label{font-size:33px;font-weight:700}.axis{font-size:29px;fill:#536475}.bar{font-size:29px;font-weight:700}.value{font-size:29px;font-weight:700}.card{fill:#F7F9FB;stroke:#CAD5DF;stroke-width:3}</style>',
        '<rect width="1800" height="1550" fill="#FFFFFF"/>',
    ]

    # Panel A: one shared source composition feeds the three fractional-EM
    # scenarios; the confidence-filtered card remains a visually separate
    # comparator because it uses a different emission design.
    lines.extend(
        [
            _text(35, 70, "A", "panel"),
            _text(95, 70, "How recovery and retention shape the analyzed population", "head"),
            _text(95, 112, "Exact population truth; bars show PF-A / PF-B (%)", "small"),
            '<circle cx="1430" cy="104" r="11" fill="#4C78A8"/>',
            _text(1450, 114, "PF-A", "small"),
            '<circle cx="1575" cy="104" r="11" fill="#F28E2B"/>',
            _text(1595, 114, "PF-B", "small"),
            '<rect class="card" x="35" y="165" width="260" height="315" rx="20"/>',
            _text(165, 220, "One source", "label", anchor="middle"),
            _text(165, 266, "source composition", "small", anchor="middle"),
            _text(1040, 150, "scenario-specific effective yield e = r × v and retention rule", "small", anchor="middle"),
        ]
    )
    lines.extend(_stacked_bar(55, 290, 220, source_share))
    lines.extend(
        [
            _text(165, 382, "source = 50:50", "label", anchor="middle"),
            _text(165, 415, "shared by fractional-EM", "compact", anchor="middle"),
            _text(165, 447, "scenarios", "compact", anchor="middle"),
        ]
    )

    card_x = {
        "all_traces": 350,
        "any_positive": 700,
        "tau_anchor": 1050,
        "confidence_hard_decode": 1400,
    }
    card_notes = {
        "all_traces": ("all registered molecules", "retained"),
        "any_positive": ("trace-dependent gate", "use conditioned f/v"),
        "tau_anchor": ("unequal recovery", "+ N/C visibility (synthetic)"),
        "confidence_hard_decode": ("separate comparator", "different emissions"),
    }
    for scenario in DISPLAY_SCENARIOS:
        row = truth[scenario]
        x = card_x[scenario]
        lines.append(f'<rect class="card" x="{x}" y="165" width="330" height="315" rx="20"/>')
        if scenario == "confidence_hard_decode":
            lines.append(_text(x + 165, 208, "Confidence-filtered", "label", anchor="middle"))
            lines.append(_text(x + 165, 246, "hard decode", "label", anchor="middle"))
        else:
            lines.append(_text(x + 165, 224, SCENARIO_LABELS[scenario], "label", anchor="middle"))
        lower_label = "retained-set truth" if scenario == "confidence_hard_decode" else "accepted truth"
        lines.append(_text(x + 165, 277, lower_label, "axis", anchor="middle"))
        lines.extend(
            _stacked_bar(
                x + 55,
                292,
                220,
                _float(row, "accepted_A_population"),
            )
        )
        retained = 100 * _float(row, "accepted_fraction_population")
        lines.append(_text(x + 165, 382, f"{retained:.1f}% retained", "label", anchor="middle"))
        note_1, note_2 = card_notes[scenario]
        lines.append(_text(x + 165, 425, note_1, "small", anchor="middle"))
        lines.append(_text(x + 165, 460, note_2, "small", anchor="middle"))

    # Fan the shared source into every fractional-EM card from below so the
    # connectors do not cross card contents. The confidence-filtered card is
    # deliberately excluded from this fan.
    lines.extend(
        [
            '<path d="M165 480 V495 H1215" fill="none" stroke="#17324D" stroke-width="4"/>',
            '<line x1="515" y1="495" x2="515" y2="480" stroke="#17324D" stroke-width="4"/>',
            '<path d="M506,486 L515,472 L524,486 Z" fill="#17324D"/>',
            '<line x1="865" y1="495" x2="865" y2="480" stroke="#17324D" stroke-width="4"/>',
            '<path d="M856,486 L865,472 L874,486 Z" fill="#17324D"/>',
            '<line x1="1215" y1="495" x2="1215" y2="480" stroke="#17324D" stroke-width="4"/>',
            '<path d="M1206,486 L1215,472 L1224,486 Z" fill="#17324D"/>',
        ]
    )

    lines.extend(
        [
            '<line x1="30" y1="510" x2="1770" y2="510" stroke="#D7E0E8" stroke-width="3"/>',
            '<line x1="900" y1="540" x2="900" y2="1515" stroke="#D7E0E8" stroke-width="3"/>',
        ]
    )

    # Panel B: a magnified valid-estimate axis plus one separated negative control.
    lines.extend(
        [
            _text(35, 595, "B", "panel"),
            _text(95, 595, "Estimate accepted composition", "head"),
            _text(95, 638, "TV error against each scenario's exact accepted-population truth", "small"),
            _text(95, 678, "small marks: seeds; large mark and whisker: mean ± sample SD", "axis"),
        ]
    )
    b_rows = {
        "all_traces": 765,
        "any_positive": 885,
        "tau_anchor": 1005,
        "confidence_hard_decode": 1125,
    }
    b_values: dict[str, list[float]] = {}
    for scenario in DISPLAY_SCENARIOS:
        selected = [row for row in oracle_runs if row["scenario"] == scenario]
        b_values[scenario] = [_float(row, "accepted_tv_population") for row in selected]
    valid_max = max(value for values in b_values.values() for value in values)
    b_max = max(0.01, math.ceil(valid_max / 0.002) * 0.002)
    b_left, b_width = 390.0, 455.0
    for tick_index in range(6):
        tick = b_max * tick_index / 5
        tick_x = _x(tick, b_left, b_width, b_max)
        lines.append(f'<line x1="{tick_x:.1f}" y1="715" x2="{tick_x:.1f}" y2="1175" stroke="#E1E7EC" stroke-width="2"/>')
        lines.append(_text(tick_x, 1215, f"{tick:.3f}", "axis", anchor="middle"))
    lines.append(f'<rect x="{b_left:.1f}" y="715" width="{b_width:.1f}" height="460" fill="none" stroke="#CAD5DF" stroke-width="2.5"/>')

    b_labels = {
        "all_traces": ("All traces", "valid accepted fit"),
        "any_positive": ("Any-positive gate", "conditioned f/v"),
        "tau_anchor": ("Synthetic N+C gate", "valid accepted fit"),
        "confidence_hard_decode": ("Confidence-filtered", "hard count vs retained truth"),
    }
    for scenario, y in b_rows.items():
        top, bottom = b_labels[scenario]
        lines.append(_text(55, y - 13, top, "label"))
        lines.append(_text(75, y + 27, bottom, "axis"))
        color = "#1B9E77" if scenario != "confidence_hard_decode" else SCENARIO_COLORS[scenario]
        shape = "circle" if scenario != "confidence_hard_decode" else "diamond"
        mean = _mean(b_values[scenario])
        lines.extend(
            _error_mark(
                b_values[scenario],
                y,
                left=b_left,
                width=b_width,
                maximum=b_max,
                color=color,
                shape=shape,
                label=f"{mean:.3f}",
            )
        )
    lines.append(_text(b_left + b_width / 2, 1260, "Accepted TV error (0 = truth)", "label", anchor="middle"))

    any_naive = [
        _float(row, "accepted_tv_naive_population")
        for row in oracle_runs
        if row["scenario"] == "any_positive"
    ]
    naive_mean = _mean(any_naive)
    naive_sd = _sample_sd(any_naive)
    lines.extend(
        [
            '<rect x="55" y="1300" width="790" height="180" rx="20" fill="#FFF3F2" stroke="#C44E52" stroke-width="3"/>',
            _text(85, 1350, "Negative control: omit gate conditioning", "label", fill="#A43F44"),
            _text(85, 1397, "Any-positive unconditioned f", "body"),
            _text(805, 1397, f"TV = {naive_mean:.3f} ± {naive_sd:.3f}", "label", anchor="end", fill="#C44E52"),
            _text(85, 1445, "Recovery r remains outside the conditional trace likelihood", "small"),
        ]
    )

    # Panel C: apply alternative yield assumptions to the same accepted-composition estimate.
    lines.extend(
        [
            _text(930, 595, "C", "panel"),
            _text(995, 595, "Yield assumptions shape correction", "head"),
            _text(995, 638, "Within each scenario, the accepted-composition fit is held fixed", "small"),
        ]
    )
    legend_x = {"all_traces": 1010, "any_positive": 1260, "tau_anchor": 1530}
    for scenario in SCENARIOS:
        x = legend_x[scenario]
        lines.append(
            _marker(
                x,
                680,
                color=SCENARIO_COLORS[scenario],
                shape=SCENARIO_SHAPES[scenario],
                size=9,
            )
        )
        legend_label = "Synthetic N+C" if scenario == "tau_anchor" else SCENARIO_LABELS[scenario]
        lines.append(_text(x + 20, 690, legend_label, "axis"))
    lines.append(_text(995, 730, "Hard-decode comparator is not source-corrected", "axis"))

    all_c_values = [
        _float(row, "source_tv_population")
        for row in runs
        if row["scenario"] in SCENARIOS and row["correction"] in CORRECTIONS
    ]
    c_max = max(0.32, math.ceil(max(all_c_values) / 0.05) * 0.05)
    c_left, c_width = 1320.0, 430.0
    lines.append('<rect x="935" y="1108" width="825" height="248" rx="18" fill="#FFF9E8"/>')
    lines.append(_text(970, 1145, "uncalibrated proxies", "axis", fill="#9A6B00"))
    for tick in (0.0, 0.1, 0.2, 0.3):
        if tick > c_max:
            continue
        tick_x = _x(tick, c_left, c_width, c_max)
        lines.append(f'<line x1="{tick_x:.1f}" y1="760" x2="{tick_x:.1f}" y2="1355" stroke="#E1E7EC" stroke-width="2"/>')
        lines.append(_text(tick_x, 1395, f"{tick:.1f}", "axis", anchor="middle"))
    lines.append(f'<rect x="{c_left:.1f}" y="760" width="{c_width:.1f}" height="595" fill="none" stroke="#CAD5DF" stroke-width="2.5"/>')
    correction_y = {
        "oracle_yield": 815,
        "misspecified_yield": 935,
        "equal_yield": 1055,
        "raw_probe_count": 1195,
        "raw_protein_length": 1315,
    }
    for correction in CORRECTIONS:
        y = correction_y[correction]
        lines.append(_text(955, y + 10, CORRECTION_LABELS[correction], "label"))
        for offset, scenario in zip((-22, 0, 22), SCENARIOS, strict=True):
            selected = [
                row
                for row in runs
                if row["scenario"] == scenario and row["correction"] == correction
            ]
            values = [_float(row, "source_tv_population") for row in selected]
            lines.extend(
                _error_mark(
                    values,
                    y + offset,
                    left=c_left,
                    width=c_width,
                    maximum=c_max,
                    color=SCENARIO_COLORS[scenario],
                    shape=SCENARIO_SHAPES[scenario],
                )
            )
    lines.extend(
        [
            _text(c_left + c_width / 2, 1445, "Source TV error (0 = truth)", "label", anchor="middle"),
            _text(1340, 1493, "Oracle e recovers 50:50 within Monte Carlo error", "label", anchor="middle", fill="#1B9E77"),
            _text(1340, 1530, "Proxy results are mechanistic counterexamples, not a ranking", "axis", anchor="middle"),
        ]
    )

    lines.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
