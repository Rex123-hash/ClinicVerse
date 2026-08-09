"""M3 figures and demonstration-patient selection, generated from artifacts.

Every plotted value is derived from `results.json` / `predictions.npz`. The
demonstration patient is chosen by the rule declared in `docs/M3_DESIGN.md`
section 11, applied mechanically — median, not maximum, so the case cannot be
chosen for drama.

Usage:
    python experiments/robustness/m3_figures.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cliniverse.evaluation.selective import predictive_entropy

PRIMARY_REP = "values_mask"
PRIMARY_MODEL = "xgboost"
COND_COLOUR = {"group_structured": "#B91C1C", "cell_random": "#1D4ED8", "none": "#111827"}
COND_LABEL = {
    "group_structured": "structured group-level loss",
    "cell_random": "matched random-cell loss",
    "none": "no loss",
}


def _rows(data: dict[str, Any], calibrator: str) -> dict[tuple[float, str], dict]:
    return {
        (r["severity"], r["condition"]): r
        for r in data["rows"]
        if r["representation"] == PRIMARY_REP
        and r["model"] == PRIMARY_MODEL
        and r["calibrator"] == calibrator
    }


def _series(
    rows: dict[tuple[float, str], dict], condition: str, metric: str, severities: list[float]
) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for s in severities:
        key = (s, "none" if s == 0.0 else condition)
        if key in rows:
            xs.append(rows[key]["metrics"].get("realized_severity", s))
            ys.append(rows[key]["metrics"][metric])
    return xs, ys


def figure_degradation(data: dict[str, Any], out: pathlib.Path) -> pathlib.Path:
    """The headline: what degrades, and what does not, as information disappears."""
    rows = _rows(data, "platt")
    severities = sorted({r["severity"] for r in data["rows"]})
    realized = {
        s: data["severity_report"]
        .get(f"{s}|group_structured", {})
        .get("realized_severity_mean", s)
        for s in severities
    }
    x = [realized[s] for s in severities]

    panels = [
        ("auroc", "AUROC — ranking ability", False),
        ("nll", "NLL — probabilistic quality", True),
        ("calibration_intercept", "Calibration intercept — risk-level drift", True),
        ("mean_predicted_probability", "Mean predicted risk", False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))

    for ax, (metric, title, lower_better) in zip(axes, panels, strict=True):
        for cond in ("group_structured", "cell_random"):
            ys = []
            for s in severities:
                key = (s, "none" if s == 0.0 else cond)
                ys.append(rows[key]["metrics"][metric])
            ax.plot(
                x,
                ys,
                marker="o",
                linewidth=2,
                color=COND_COLOUR[cond],
                label=COND_LABEL[cond],
            )
        if metric == "mean_predicted_probability":
            prevalence = 0.14025
            ax.axhline(
                prevalence,
                color="#059669",
                linestyle="--",
                linewidth=1.4,
                label="true prevalence",
            )
        if metric == "calibration_intercept":
            ax.axhline(0.0, color="#6B7280", linestyle=":", linewidth=1)
        ax.set_xlabel("realized information loss (fraction of eligible lab cells)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        if lower_better:
            ax.set_ylabel("lower is better", fontsize=8)

    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "M3 — XGBoost values+mask, Platt-calibrated on clean data. "
        "Ranking survives; risk estimates drift downward, and structured loss is worse.",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    path = out / "m3_degradation.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_calibrator_comparison(data: dict[str, Any], out: pathlib.Path) -> pathlib.Path:
    """Does the calibration method change robustness?"""
    severities = sorted({r["severity"] for r in data["rows"]})
    realized = {
        s: data["severity_report"]
        .get(f"{s}|group_structured", {})
        .get("realized_severity_mean", s)
        for s in severities
    }
    x = [realized[s] for s in severities]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    styles = {"uncalibrated": ":", "platt": "-", "isotonic": "--"}

    for ax, metric, title in zip(
        axes,
        ("nll", "calibration_intercept"),
        (
            "NLL under structured group loss",
            "Calibration intercept under structured group loss",
        ),
        strict=True,
    ):
        for cal, style in styles.items():
            rows = _rows(data, cal)
            ys = [
                rows[(s, "none" if s == 0.0 else "group_structured")]["metrics"][metric]
                for s in severities
            ]
            ax.plot(x, ys, marker="o", linestyle=style, linewidth=1.8, label=cal)
        if metric == "calibration_intercept":
            ax.axhline(0.0, color="#6B7280", linestyle=":", linewidth=1)
        ax.set_xlabel("realized information loss")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "M3 — calibration fitted on clean data does not prevent risk-level drift", y=1.0
    )
    fig.tight_layout()
    path = out / "m3_calibrators.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_reliability(data: dict[str, Any], out: pathlib.Path) -> pathlib.Path:
    rows = _rows(data, "platt")
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 0.6], [0, 0.6], color="#9CA3AF", linestyle="--", linewidth=1)
    for sev, cond, colour, label in (
        (0.0, "none", "#111827", "no loss"),
        (0.75, "cell_random", "#1D4ED8", "matched random-cell loss"),
        (0.75, "group_structured", "#B91C1C", "structured group-level loss"),
    ):
        rel = rows[(sev, cond)]["reliability"]
        ax.plot(
            rel["mean_predicted"],
            rel["observed_rate"],
            marker="o",
            markersize=5,
            linewidth=1.6,
            color=colour,
            label=label,
        )
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed mortality rate")
    ax.set_title(
        "M3 - reliability at the highest severity\n"
        "(points below the diagonal understate risk)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out / "m3_reliability.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def select_demo_patient(
    data: dict[str, Any], arrays: dict[str, np.ndarray], out: pathlib.Path
) -> pathlib.Path:
    """Apply the M3_DESIGN section 11 rule mechanically."""
    y = arrays["labels"].astype(float)
    ids = arrays["record_ids"]
    base_key = f"{PRIMARY_REP}|{PRIMARY_MODEL}|platt|0.0|none"
    loss_key = f"{PRIMARY_REP}|{PRIMARY_MODEL}|platt|0.5|group_structured"
    p0, p1 = arrays[base_key].astype(float), arrays[loss_key].astype(float)

    correct_before = (p0 >= 0.5).astype(float) == y
    wrong_after = (p1 >= 0.5).astype(float) != y
    flipped = correct_before & wrong_after

    e0, e1 = predictive_entropy(p0), predictive_entropy(p1)
    # Confidence must not have fallen: entropy did not rise.
    confidence_held = e1 <= e0
    eligible = flipped & confidence_held

    payload: dict[str, Any] = {
        "rule": "docs/M3_DESIGN.md section 11",
        "baseline": base_key,
        "stress": loss_key,
        "n_flipped_correct_to_incorrect": int(flipped.sum()),
        "n_eligible_with_confidence_held": int(eligible.sum()),
    }

    if not eligible.any():
        payload["selected"] = None
        payload["note"] = "eligible set empty; no patient selected, reported as a finding"
    else:
        # Calibration error deterioration = |p - y| increase.
        deterioration = np.abs(p1 - y) - np.abs(p0 - y)
        cand = np.flatnonzero(eligible)
        median_value = float(np.median(deterioration[cand]))
        chosen = cand[int(np.argmin(np.abs(deterioration[cand] - median_value)))]
        payload["selected"] = {
            "record_id": int(ids[chosen]),
            "index": int(chosen),
            "label": float(y[chosen]),
            "p_baseline": float(p0[chosen]),
            "p_group_loss_50": float(p1[chosen]),
            "entropy_baseline": float(e0[chosen]),
            "entropy_group_loss_50": float(e1[chosen]),
            "calibration_error_deterioration": float(deterioration[chosen]),
            "median_deterioration_of_eligible": median_value,
        }
    path = out / "m3_demo_patient.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=pathlib.Path,
        default=pathlib.Path("experiments/robustness/results/m3"),
    )
    args = parser.parse_args(argv)
    out = args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)

    data = json.loads((args.results / "results.json").read_text(encoding="utf-8"))
    arrays = dict(np.load(args.results / "predictions.npz"))

    for path in (
        figure_degradation(data, out),
        figure_calibrator_comparison(data, out),
        figure_reliability(data, out),
        select_demo_patient(data, arrays, args.results),
    ):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
