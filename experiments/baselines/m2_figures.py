"""Generate M2 result figures from artifacts.

Reads `results.json` and `predictions.npz` and derives every value plotted. No
number is typed by hand, so a figure cannot drift from the run that produced it.

These are scientific result artifacts, not presentation graphics.

Usage:
    python experiments/baselines/m2_figures.py [--results DIR] [--out DIR]
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

from cliniverse.evaluation.metrics import reliability_curve

CORE = ("mask_only", "values_only", "values_mask")
COLOURS = {
    "mask_only": "#B45309",
    "values_only": "#1D4ED8",
    "values_mask": "#047857",
    "values_only_median_jitter": "#7C3AED",
    "values_only_empirical_marginal": "#9333EA",
    "prevalence": "#6B7280",
    "statics_only": "#9CA3AF",
    "values_mask_statics": "#065F46",
}


RunSummary = dict[str, Any]


def _load(results_dir: pathlib.Path) -> tuple[list[RunSummary], dict[str, np.ndarray]]:
    manifest = json.loads((results_dir / "results.json").read_text(encoding="utf-8"))
    arrays = dict(np.load(results_dir / "predictions.npz"))
    return manifest["runs"], arrays


def figure_baseline_comparison(runs: list[RunSummary], out: pathlib.Path) -> pathlib.Path:
    """AUROC with bootstrap intervals for every run."""
    ordered = sorted(runs, key=lambda r: r["metrics"]["auroc"])
    labels = [r["run_id"] for r in ordered]
    point = np.array([r["metrics"]["auroc"] for r in ordered])
    low = np.array([r["intervals"]["auroc"]["ci_low"] for r in ordered])
    high = np.array([r["intervals"]["auroc"]["ci_high"] for r in ordered])

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(ordered) + 1.6))
    ypos = np.arange(len(ordered))
    ax.errorbar(
        point,
        ypos,
        xerr=[point - low, high - point],
        fmt="o",
        color="#111827",
        ecolor="#9CA3AF",
        capsize=3,
        markersize=5,
    )
    ax.axvline(0.5, color="#DC2626", linestyle=":", linewidth=1, label="chance")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUROC (out-of-fold, 95% patient bootstrap CI)")
    ax.set_title("M2 — T1 in-hospital mortality at 24h, development cohort (sets a+b)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = out / "m2_baseline_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_representation_contrast(runs: list[RunSummary], out: pathlib.Path) -> pathlib.Path:
    """The three binding representations, grouped by model."""
    models = ["logreg", "xgboost"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    by_id = {r["run_id"]: r for r in runs}

    for ax, metric, label in zip(
        axes, ("auroc", "auprc"), ("AUROC", "Average precision (AP)"), strict=True
    ):
        width = 0.26
        xpos = np.arange(len(models))
        for k, rep in enumerate(CORE):
            values: list[float] = []
            errs: list[list[float]] = [[], []]
            for model in models:
                run = by_id.get(f"{rep}::{model}")
                if run is None:
                    values.append(np.nan)
                    errs[0].append(0.0)
                    errs[1].append(0.0)
                    continue
                v = run["metrics"][metric]
                ci = run["intervals"][metric]
                values.append(v)
                errs[0].append(v - ci["ci_low"])
                errs[1].append(ci["ci_high"] - v)
            ax.bar(
                xpos + (k - 1) * width,
                values,
                width,
                yerr=np.array(errs),
                capsize=3,
                label=rep,
                color=COLOURS[rep],
                edgecolor="white",
            )
        prevalence = by_id.get("prevalence")
        if prevalence is not None and metric == "auprc":
            ax.axhline(
                prevalence["metrics"]["prevalence"],
                color="#DC2626",
                linestyle=":",
                linewidth=1,
                label="prevalence",
            )
        if metric == "auroc":
            ax.axhline(0.5, color="#DC2626", linestyle=":", linewidth=1, label="chance")
        ax.set_xticks(xpos)
        ax.set_xticklabels(models)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("score (out-of-fold, 95% CI)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("M2 — measurement presence vs clinical values vs both", y=1.0)
    fig.tight_layout()
    path = out / "m2_representation_contrast.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_reliability(
    runs: list[RunSummary], arrays: dict[str, np.ndarray], out: pathlib.Path
) -> pathlib.Path:
    """Reliability curves recomputed from retained predictions."""
    y = arrays["labels"].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

    for ax, model in zip(axes, ("logreg", "xgboost"), strict=True):
        ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", linewidth=1)
        for rep in CORE:
            key = f"pred__{rep}::{model}"
            if key not in arrays:
                continue
            curve = reliability_curve(y, arrays[key].astype(float), n_bins=10)
            ax.plot(
                curve["mean_predicted"],
                curve["observed_rate"],
                marker="o",
                markersize=4,
                linewidth=1.4,
                color=COLOURS[rep],
                label=rep,
            )
        ax.set_xlabel("mean predicted probability")
        ax.set_title(model)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.8)

    axes[0].set_ylabel("observed mortality rate")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "M2 — descriptive OOF reliability (equal-mass bins); diagonal is reference",
        y=1.0,
    )
    fig.tight_layout()
    path = out / "m2_reliability.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=pathlib.Path,
        default=pathlib.Path("experiments/baselines/results/m2"),
    )
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)
    out = args.out or args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)

    runs, arrays = _load(args.results)
    for path in (
        figure_baseline_comparison(runs, out),
        figure_representation_contrast(runs, out),
        figure_reliability(runs, arrays, out),
    ):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
