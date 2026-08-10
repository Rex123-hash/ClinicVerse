"""M4 figures, generated from artifacts only.

Usage:
    python experiments/acquisition/m4_figures.py
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

COLOURS = {
    "no_acquisition": "#6B7280",
    "random_uniform_all": "#2563EB",
    "random_train_frequency": "#059669",
    "fixed_domain_order": "#B91C1C",
    "greedy_eig": "#7C3AED",
    "greedy_eig_per_cost": "#DB2777",
}


def figure_primary_nll(data: dict[str, Any], out: pathlib.Path) -> pathlib.Path:
    """NLL vs budget for every policy under the primary support-blind condition."""
    pr = data["primary"]
    grid = data["budget_grid"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for policy in data["ranked_policies"]:
        rows = sorted(
            (r for r in pr["rows"] if r["policy"] == policy),
            key=lambda r: r["budget_fraction"],
        )
        axes[0].plot(
            grid,
            [r["nll"] for r in rows],
            marker="o",
            linewidth=2,
            color=COLOURS[policy],
            label=policy,
        )
        axes[1].plot(
            grid,
            [r["brier"] for r in rows],
            marker="o",
            linewidth=2,
            color=COLOURS[policy],
            label=policy,
        )
    axes[0].set_ylabel("NLL (lower is better)")
    axes[0].set_title("Primary endpoint: NLL vs budget")
    axes[1].set_ylabel("Brier (lower is better)")
    axes[1].set_title("Co-primary: Brier vs budget")
    for ax in axes:
        ax.set_xlabel("budget (fraction of total catalogue cost)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    cond = pr["condition"]
    fig.suptitle(
        f"M4 primary — support_blind / {cond['cost_regime']} / mask {cond['mask_rate']} "
        f"(n={cond['n_patients']:,})",
        y=1.02,
    )
    fig.tight_layout()
    path = out / "m4_primary_nll_vs_budget.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_rank_flow(data: dict[str, Any], out: pathlib.Path) -> pathlib.Path:
    """Rank of each policy across every evaluation condition (slopegraph)."""
    results = data["grid"]["results"]
    policies = list(data["ranked_policies"])
    conditions = [g["condition"] for g in results]
    ranks = np.zeros((len(policies), len(conditions)))
    for j, g in enumerate(results):
        for i, p in enumerate(policies):
            ranks[i, j] = g["ranking"].index(p) + 1

    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(conditions))
    for i, p in enumerate(policies):
        ax.plot(x, ranks[i], marker="o", linewidth=2.2, color=COLOURS[p], label=p)
    boundary = sum(1 for c in conditions if c.startswith("support_blind")) - 0.5
    ax.axvline(boundary, color="#111827", linestyle="--", linewidth=1.5)
    ax.text(
        boundary - 0.15,
        0.4,
        "support_blind (fair)",
        ha="right",
        fontsize=9,
        color="#111827",
    )
    ax.text(
        boundary + 0.15,
        0.4,
        "support_aware (diagnostic oracle)",
        ha="left",
        fontsize=9,
        color="#111827",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace("support_blind|", "").replace("support_aware|", "") for c in conditions],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(range(1, len(policies) + 1))
    ax.invert_yaxis()
    ax.set_ylabel("rank by AUNLLC (1 = best)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="lower center")
    st = data["stability"]
    fig.suptitle(
        "M4 — policy rank across evaluation assumptions  "
        f"(mean Kendall tau-b {st['mean_kendall_tau_b']:.3f}, "
        f"{st['n_supported_reversals']}/{len(st['winner_changes'])} winner changes supported)",
        y=1.0,
    )
    fig.tight_layout()
    path = out / "m4_rank_flow.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=pathlib.Path,
        default=pathlib.Path("experiments/acquisition/results/m4"),
    )
    args = parser.parse_args(argv)
    out = args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((args.results / "results.json").read_text(encoding="utf-8"))
    for path in (figure_primary_nll(data, out), figure_rank_flow(data, out)):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
