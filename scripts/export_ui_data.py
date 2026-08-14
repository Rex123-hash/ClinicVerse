"""Export committed Cliniverse result artifacts into a single UI data bundle.

The web UI must never invent a scientific number. This script is the *only*
bridge between the committed result artifacts and `web/src/data/`: it reads the
JSON/NPZ artifacts produced by M2-M5v2, derives the chart series the UI renders,
and writes one machine-generated JSON bundle.

Run:

    python scripts/export_ui_data.py

Output: ``web/src/data/cliniverse-bundle.json``
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = ROOT / "web" / "src" / "data" / "cliniverse-bundle.json"
PUBLIC_EVIDENCE = ROOT / "web" / "public" / "evidence"

SETC = EXP / "robustness/results/m5v2_setc"
FREEZE = EXP / "robustness/results/m5v2_final_freeze"
M5V2 = EXP / "robustness/results/m5v2"
M5 = EXP / "robustness/results/m5"
M3 = EXP / "robustness/results/m3"
M4 = EXP / "acquisition/results/m4"
M2 = EXP / "baselines/results/m2"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def r(value: float, places: int = 6) -> float | None:
    """Round for transport without pretending to more precision than we have.

    Non-finite values become ``None`` rather than ``NaN``: the constant
    prevalence baseline has a degenerate logistic recalibration, and ``NaN`` is
    not valid JSON.
    """
    number = float(value)
    if not np.isfinite(number):
        return None
    return float(np.round(number, places))


# --------------------------------------------------------------------------
# derived chart series, computed from the committed prediction vectors
# --------------------------------------------------------------------------


def roc_curve(
    labels: np.ndarray, scores: np.ndarray, n_points: int = 60
) -> list[dict[str, float]]:
    """Downsampled ROC curve. Exact at the endpoints, monotone in between."""
    order = np.argsort(-scores)
    y = labels[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = tps / max(tps[-1], 1.0)
    fpr = fps / max(fps[-1], 1.0)
    idx = np.unique(np.linspace(0, len(y) - 1, n_points).astype(int))
    points = [{"fpr": 0.0, "tpr": 0.0}]
    points += [{"fpr": r(fpr[i], 5), "tpr": r(tpr[i], 5)} for i in idx]
    points += [{"fpr": 1.0, "tpr": 1.0}]
    return points


def reliability(
    labels: np.ndarray, scores: np.ndarray, bins: int = 10
) -> list[dict[str, float]]:
    """Equal-count reliability bins, matching the convention used in M2/M3 artifacts."""
    order = np.argsort(scores)
    y = labels[order]
    p = scores[order]
    chunks = np.array_split(np.arange(len(p)), bins)
    return [
        {
            "meanPredicted": r(float(p[c].mean()), 5),
            "observedRate": r(float(y[c].mean()), 5),
            "count": len(c),
        }
        for c in chunks
    ]


def histogram(
    scores: np.ndarray, lo: float, hi: float, bins: int = 28
) -> list[dict[str, float]]:
    counts, edges = np.histogram(scores, bins=bins, range=(lo, hi))
    return [
        {"risk": r(float((edges[i] + edges[i + 1]) / 2), 5), "count": int(counts[i])}
        for i in range(bins)
    ]


def setc_charts() -> dict[str, Any]:
    z = np.load(SETC / "setc_oneshot_predictions.npz")
    labels = z["labels"].astype(float)
    clean = z["p_clean"].astype(float)
    withheld = z["p_withheld"].astype(float)
    controls = np.vstack([z[f"p_control_{i}"] for i in range(5)]).astype(float)
    control_mean = controls.mean(axis=0)
    d_i = z["d_i"].astype(float)
    removed = z["removed_cells"].astype(int)

    lo, hi = 0.0, float(max(clean.max(), withheld.max()))
    return {
        "roc": {"clean": roc_curve(labels, clean), "withheld": roc_curve(labels, withheld)},
        "reliability": {
            "clean": reliability(labels, clean),
            "withheld": reliability(labels, withheld),
        },
        "riskDistribution": {
            "clean": histogram(clean, lo, hi),
            "withheld": histogram(withheld, lo, hi),
        },
        "meanRiskShift": [
            {"stage": "Clean", "meanRisk": r(float(clean.mean()))},
            {"stage": "Amount-matched control", "meanRisk": r(float(control_mean.mean()))},
            {"stage": "Withheld", "meanRisk": r(float(withheld.mean()))},
        ],
        "perPatientExcess": {
            "mean": r(float(d_i.mean())),
            "positiveFraction": r(float((d_i > 0).mean()), 4),
            "histogram": [
                {"excess": r(float((e0 + e1) / 2), 4), "count": int(c)}
                for c, e0, e1 in zip(
                    *(lambda h, e: (h, e[:-1], e[1:]))(
                        *np.histogram(d_i, bins=32, range=(-0.6, 0.6))
                    ),
                    strict=True,
                )
            ],
        },
        "removedCells": {
            "mean": r(float(removed.mean())),
            "median": int(np.median(removed)),
            "p10": int(np.percentile(removed, 10)),
            "p90": int(np.percentile(removed, 90)),
        },
    }


# --------------------------------------------------------------------------
# visual-artifact series
#
# Each of these is derived only from committed prediction vectors. None of them
# is a feature-attribution, a counterfactual, or a biological response surface:
# the repository contains no such capability, and naming them that way would
# overclaim. They describe where the *measured* withholding damage sits.
# --------------------------------------------------------------------------


def failure_concentration_surface(
    z: Any, risk_bins: int = 14, cell_bins: int = 12
) -> dict[str, Any]:
    """Mean per-patient excess NLL over a clean-risk x removed-cell grid.

    Descriptive only. The axes are a model output and a count of removed
    cells — not clinical measurements — so this shows where the stress test
    concentrated, not any relationship between analytes and outcome.
    """
    clean = z["p_clean"].astype(float)
    removed = z["removed_cells"].astype(int)
    d_i = z["d_i"].astype(float)

    risk_hi = float(np.quantile(clean, 0.99))
    cell_hi = int(np.quantile(removed, 0.99))
    risk_edges = np.linspace(0.0, risk_hi, risk_bins + 1)
    cell_edges = np.linspace(0.0, cell_hi, cell_bins + 1)

    ri = np.clip(np.digitize(clean, risk_edges) - 1, 0, risk_bins - 1)
    ci = np.clip(np.digitize(removed, cell_edges) - 1, 0, cell_bins - 1)

    cells: list[dict[str, Any]] = []
    for a in range(risk_bins):
        for b in range(cell_bins):
            mask = (ri == a) & (ci == b)
            n = int(mask.sum())
            cells.append(
                {
                    "riskBin": a,
                    "cellBin": b,
                    "risk": r(float((risk_edges[a] + risk_edges[a + 1]) / 2), 5),
                    "removed": r(float((cell_edges[b] + cell_edges[b + 1]) / 2), 3),
                    "n": n,
                    "meanExcess": r(float(d_i[mask].mean())) if n else None,
                }
            )

    populated = [c["meanExcess"] for c in cells if c["meanExcess"] is not None]
    return {
        "riskBins": risk_bins,
        "cellBins": cell_bins,
        "riskMax": r(risk_hi, 5),
        "removedMax": cell_hi,
        "zMin": r(min(populated)),
        "zMax": r(max(populated)),
        "cells": cells,
    }


def patient_rows(z: Any) -> list[dict[str, Any]]:
    """Per-patient committed predictions: the basis of the slice and explorer."""
    return [
        {
            "id": int(rid),
            "y": int(label),
            "c": r(float(pc), 5),
            "w": r(float(pw), 5),
            "d": r(float(d), 5),
            "rc": int(rc),
        }
        for rid, label, pc, pw, d, rc in zip(
            z["record_ids"],
            z["labels"],
            z["p_clean"],
            z["p_withheld"],
            z["d_i"],
            z["removed_cells"],
            strict=True,
        )
    ]


def withholding_burden(z: Any) -> dict[str, Any]:
    """Distribution of removed cells per patient under the frozen pattern."""
    removed = z["removed_cells"].astype(int)
    hi = int(removed.max())
    counts = np.bincount(removed, minlength=hi + 1)
    return {
        "mean": r(float(removed.mean())),
        "median": int(np.median(removed)),
        "p10": int(np.percentile(removed, 10)),
        "p90": int(np.percentile(removed, 90)),
        "max": hi,
        "nZero": int((removed == 0).sum()),
        "histogram": [{"removed": i, "count": int(counts[i])} for i in range(hi + 1)],
    }


def candidate_damage_landscape(top: int = 12) -> dict[str, Any]:
    """Per-candidate spread of development excess NLL across resplits x folds.

    This is a distribution of a *measured loss difference* per candidate
    pattern. It is not a feature-importance or attribution method.
    """
    tables = np.load(M5V2 / "m5v2_tables.npz", allow_pickle=True)
    names = [str(n) for n in tables["candidate_names"]]
    deltas = tables["deltas"]  # (candidates, resplits, folds)

    means = deltas.reshape(len(names), -1).mean(axis=1)
    order = np.argsort(-means)[:top]

    series = []
    for idx in order:
        flat = deltas[idx].reshape(-1)
        series.append(
            {
                "name": names[idx],
                "analytes": names[idx].split("+"),
                "mean": r(float(flat.mean())),
                "median": r(float(np.median(flat))),
                "p10": r(float(np.percentile(flat, 10))),
                "p90": r(float(np.percentile(flat, 90))),
                "positiveFraction": r(float((flat > 0).mean()), 4),
                "values": [r(float(v), 5) for v in flat],
            }
        )
    return {
        "nCandidates": len(names),
        "nObservationsPerCandidate": int(deltas.shape[1] * deltas.shape[2]),
        "series": series,
    }


def development_charts() -> dict[str, Any]:
    """M5-v2 development: per-resplit stability and the candidate ranking."""
    tables = np.load(M5V2 / "m5v2_tables.npz", allow_pickle=True)
    names = [str(n) for n in tables["candidate_names"]]
    deltas = tables["deltas"]  # (141 candidates, 20 resplits, 5 folds)
    clean_auroc = tables["clean_auroc"]  # (20,)
    cand_auroc = tables["candidate_auroc"]  # (141, 20)

    res = load_json(M5V2 / "results.json")
    oos = res["out_of_selection_components"]

    # How often each candidate was the pick on a held-out *fold* of the
    # out-of-selection procedure. There are 20 resplits x 5 folds = 100 such
    # picks. This is NOT the headline 11/20 resplit-level majority reported by
    # gate G2 — keep the two clearly separated in the UI.
    fold_picks: dict[str, int] = defaultdict(int)
    for row in oos:
        fold_picks["+".join(row["selected"])] += 1

    per_candidate = []
    for i, name in enumerate(names):
        mean_delta = float(deltas[i].mean())
        per_candidate.append(
            {
                "name": name,
                "analytes": name.split("+"),
                "meanExcessNll": r(mean_delta),
                "meanAuroc": r(float(cand_auroc[i].mean())),
                "meanAurocDrop": r(float((clean_auroc - cand_auroc[i]).mean())),
                "oosFoldPicks": fold_picks.get(name, 0),
            }
        )
    per_candidate.sort(key=lambda row: -row["meanExcessNll"])

    frozen = "+".join(load_json(M5V2 / "frozen_pattern.json")["frozen_pattern"])
    per_resplit_frozen: dict[int, int] = defaultdict(int)
    for row in oos:
        if "+".join(row["selected"]) == frozen:
            per_resplit_frozen[row["resplit"]] += 1

    return {
        "candidates": per_candidate,
        "nCandidates": len(names),
        "oosFoldPickTotal": len(oos),
        "resplitStability": [
            {
                "resplit": i,
                "cleanAuroc": r(float(clean_auroc[i])),
                # of the 5 held-out folds in this resplit, how many picked the
                # frozen pattern in the out-of-selection procedure
                "frozenPatternFolds": per_resplit_frozen.get(i, 0),
                "foldsPerResplit": 5,
            }
            for i in range(len(clean_auroc))
        ],
        "outOfSelection": [
            {
                "resplit": row["resplit"],
                "heldOutFold": row["held_out_fold"],
                "selected": row["selected"],
                "delta": r(row["delta_on_held_out_fold"]),
            }
            for row in oos
        ],
    }


def stress_response() -> dict[str, Any]:
    """M3 severity sweep: the only executed performance-vs-stress evidence."""
    res = load_json(M3 / "results.json")
    grouped: dict[tuple[str, float], list[dict[str, float]]] = defaultdict(list)
    for row in res["rows"]:
        if row["calibrator"] != "uncalibrated":
            continue
        grouped[(row["condition"], float(row["severity"]))].append(row["metrics"])

    series = []
    ordered = sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    for (condition, severity), metrics in ordered:
        series.append(
            {
                "condition": condition,
                "severity": severity,
                "nRuns": len(metrics),
                "auroc": r(float(np.mean([m["auroc"] for m in metrics]))),
                "nll": r(float(np.mean([m["nll"] for m in metrics]))),
                "brier": r(float(np.mean([m["brier"] for m in metrics]))),
                "calibrationIntercept": r(
                    float(np.mean([m["calibration_intercept"] for m in metrics]))
                ),
                "meanPredictedRisk": r(
                    float(np.mean([m["mean_predicted_probability"] for m in metrics]))
                ),
            }
        )

    severity_report = {
        key: {
            "requestedSeverity": value["requested_severity"],
            "realizedSeverityMean": r(value["realized_severity_mean"]),
            "totalRemovedCells": value["total_removed_cells"],
            "nPatientsWithEligibleCells": value["n_patients_with_eligible_cells"],
        }
        for key, value in res["severity_report"].items()
    }
    return {"series": series, "severityReport": severity_report}


def representation_comparison() -> list[dict[str, Any]]:
    """M2: the executed representation/model grid behind the frozen choice."""
    res = load_json(M2 / "results.json")
    rows = []
    for run in res["runs"]:
        metrics = run.get("metrics")
        if not metrics:
            continue
        intervals = run.get("intervals", {})
        rows.append(
            {
                "runId": run["run_id"],
                "representation": run["representation"],
                "model": run["model"],
                "nFeatures": run.get("n_features"),
                "auroc": r(metrics["auroc"]),
                "auprc": r(metrics["auprc"]),
                "brier": r(metrics["brier"]),
                "nll": r(metrics["nll"]),
                "calibrationIntercept": r(metrics["calibration_intercept"]),
                "calibrationSlope": r(metrics["calibration_slope"]),
                "aurocCi": [
                    r(intervals["auroc"]["ci_low"]),
                    r(intervals["auroc"]["ci_high"]),
                ]
                if "auroc" in intervals
                else None,
            }
        )
    return rows


def visual_artifacts() -> dict[str, Any]:
    """The four Set-C visual artifacts, plus the development landscape."""
    z = np.load(SETC / "setc_oneshot_predictions.npz")
    return {
        "concentrationSurface": failure_concentration_surface(z),
        "patients": patient_rows(z),
        "burden": withholding_burden(z),
        "damageLandscape": candidate_damage_landscape(),
    }


# Evidence files served to the browser verbatim, so the "Export data" action in
# the UI hands over the committed artifact itself rather than a re-serialisation
# of it. The expected digest is the same one the Artifacts page displays; keeping
# it here means a changed artifact fails the export loudly instead of silently
# desynchronising the hash the UI claims.
PUBLIC_EVIDENCE_FILES: tuple[tuple[Path, str, str], ...] = (
    (
        SETC / "results.json",
        "setc-confirmation.json",
        "7179a5744e5d9034a735fb6bcd1652a96e850e285fc60b8de61983a7d192a907",
    ),
)


# The public repository history was rewritten with `git filter-repo` after these
# experiments ran, so the commit SHAs recorded in the result artifacts at run
# time no longer resolve on the public remote. The mapping below is taken
# verbatim from `.git/filter-repo/commit-map`; every target below was verified
# reachable from the rewritten `main`.
#
# This translates *where to find the commit*, never what was measured. The
# artifacts themselves are frozen and are not edited: the value recorded at run
# time travels through as `git_sha_at_run` so nothing is lost.
GIT_SHA_REWRITE: dict[str, str] = {
    # set-c one-shot confirmation run
    "e2120eb3785647b361db9e29038259c0fa5de968": "a97f6905d3f8c34ac4030ede5114a02990c61578",
    # final freeze the confirmation was executed against
    "01bc036145e22c1821de8aae8233c2bc4a75b7a0": "e5edc13b3f242a5f8f94b4af6522018682e47df9",
    # M5-v2 development search
    "91262fd5c53c56665a042d11225abdc7b5c85777": "3f4b4e01dc58c62636a1b59e5d411a60eb51122d",
    # M5-v1, the search layer that failed its own bar
    "f6b6e6be671e868f988d05e2c64fa7e9275d7fcd": "abb9ffc012e4d763788bfc96eb0f94019489be58",
}


def rewritten(sha: str) -> str:
    """Current public SHA for a commit recorded before the history rewrite."""
    return GIT_SHA_REWRITE.get(sha, sha)


def remap_provenance(provenance: Any) -> Any:
    """Copy a provenance block with its git SHA pointed at the public history."""
    if not isinstance(provenance, dict) or "git_sha" not in provenance:
        return provenance
    recorded = provenance["git_sha"]
    updated = dict(provenance)
    updated["git_sha"] = rewritten(recorded)
    updated["git_sha_at_run"] = recorded
    return updated


def copy_public_evidence() -> None:
    """Copy committed evidence artifacts into the web app's static assets.

    Byte-for-byte, never re-serialised: a judge who downloads the file from the
    deployed application must be able to hash it and get the digest recorded in
    the result artifact and shown in the UI.
    """
    PUBLIC_EVIDENCE.mkdir(parents=True, exist_ok=True)
    for source, name, expected_sha256 in PUBLIC_EVIDENCE_FILES:
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected_sha256:
            raise SystemExit(
                f"{source.relative_to(ROOT)} hashes to {digest}, but the UI "
                f"publishes {expected_sha256}. Refusing to serve an artifact "
                "whose digest no longer matches the one displayed."
            )
        (PUBLIC_EVIDENCE / name).write_bytes(payload)
        print(f"wrote web/public/evidence/{name} ({len(payload)} bytes, sha256 {digest[:12]}…)")


def main() -> None:
    setc = load_json(SETC / "results.json")
    freeze = load_json(FREEZE / "final_freeze.json")
    m5v2 = load_json(M5V2 / "results.json")
    m5 = load_json(M5 / "results.json")

    bundle = {
        "_generated": {
            "by": "scripts/export_ui_data.py",
            "note": (
                "Machine-generated from committed result artifacts. "
                "Do not hand-edit; re-run the exporter instead."
            ),
        },
        "setcConfirmation": {
            "stage": setc["stage"],
            "cohort": setc["cohort"],
            "primary": {
                "deltaC": setc["primary"]["delta_c"],
                "lowerBound": setc["primary"]["one_sided_95_lower_bound"],
                "nBootstrap": setc["primary"]["n_bootstrap"],
                "nValidResamples": setc["primary"]["n_valid_resamples"],
                "bootstrapSeed": setc["primary"]["bootstrap_seed"],
                "decisionRule": setc["primary"]["decision_rule"],
                "pattern": setc["primary"]["pattern"],
                "passes": setc["primary"]["passes"],
            },
            "discriminationSilent": {
                "cleanAuroc": setc["discrimination_silent"]["clean_auroc"],
                "withheldAuroc": setc["discrimination_silent"]["withheld_auroc"],
                "aurocDrop": setc["discrimination_silent"]["auroc_drop"],
                "delta": setc["discrimination_silent"]["delta"],
                "passes": setc["discrimination_silent"]["passes"],
            },
            "confirmation": setc["confirmation"],
            "secondary": setc["secondary_descriptive"],
            "provenance": remap_provenance(setc["provenance"]),
            "recordIdsHash": setc["record_ids_hash"],
            "predictionsFile": setc["predictions_file"],
            "frozenArtifactHashes": setc["frozen_artifact_hashes"],
            "monteCarloLimitation": setc["monte_carlo_limitation"],
            "historicalDisclosure": setc["historical_disclosure"],
            "alternativeAnalysesRun": setc["alternative_analyses_run"],
            "startedAt": setc["started_at_utc"],
            "finishedAt": setc["finished_at_utc"],
            "freezeSourceGitSha": rewritten(setc["freeze_source_git_sha"]),
        },
        "freeze": {
            "stage": freeze["stage"],
            "frozenPattern": freeze["frozen_pattern"],
            "frozenPatternColumns": freeze["frozen_pattern_columns"],
            "frozenPatternProvenance": freeze["frozen_pattern_provenance"],
            "model": freeze["model"],
            "preprocessing": freeze["preprocessing"],
            "calibrator": freeze["calibrator"],
            "split": freeze["split"],
            "provenance": remap_provenance(freeze["provenance"]),
            "artifactHashes": freeze["artifact_hashes"],
            "setCAccess": freeze["set_c_access"],
            "fittingDiagnostics": freeze["fitting_diagnostics_not_evaluation"],
            "contract": freeze["set_c_evaluation_contract"],
        },
        "development": {
            "verdict": m5v2.get("frozen_pattern_region"),
            "nCandidates": m5v2["n_candidates"],
            "completeEnumeration": m5v2["complete_enumeration"],
            "predeclared": m5v2["predeclared"],
            "gates": m5v2["gates"],
            "estimates": m5v2["development_estimates"],
            "detectability": m5v2["detectability"],
            "provenance": remap_provenance(m5v2["provenance"]),
            "cleanAurocByResplit": [r(v) for v in m5v2["clean_auroc_by_resplit"]],
            "eligibilityCountByResplit": m5v2["eligibility_count_by_resplit"],
        },
        "m5v1": {
            "nConfigurations": m5.get("n_configurations"),
            "provenance": remap_provenance(m5.get("provenance")),
        },
        "visual": visual_artifacts(),
        "charts": {
            "setc": setc_charts(),
            "development": development_charts(),
            "stressResponse": stress_response(),
            "representations": representation_comparison(),
        },
    }

    # The frozen Platt calibrator must travel through to the UI exactly as the
    # freeze recorded it. A zeroed intercept would silently misstate the frozen
    # pipeline, so refuse to write a bundle that lost it.
    calibrator = bundle["freeze"]["calibrator"]
    if not calibrator.get("intercept") or not calibrator.get("slope"):
        raise SystemExit(
            "final_freeze.json did not yield a usable Platt calibrator "
            f"(intercept={calibrator.get('intercept')!r}, slope={calibrator.get('slope')!r})."
        )

    # Every pre-rewrite SHA must have been translated. The only occurrences left
    # in the bundle should be the ones deliberately preserved as `git_sha_at_run`,
    # so a provenance field added later that bypasses `remap_provenance` fails
    # here instead of shipping a commit reference that no longer resolves.
    serialised = json.dumps(bundle)
    for old in GIT_SHA_REWRITE:
        total = serialised.count(old)
        preserved = serialised.count(f'"git_sha_at_run": "{old}"')
        if total != preserved:
            raise SystemExit(
                f"{old} still appears {total - preserved} time(s) in the bundle "
                "outside git_sha_at_run. The public history was rewritten, so "
                "that commit no longer resolves; route the field through "
                "remap_provenance() or rewritten()."
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=1, ensure_ascii=False, sort_keys=False)
        handle.write("\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")

    copy_public_evidence()


if __name__ == "__main__":
    main()
