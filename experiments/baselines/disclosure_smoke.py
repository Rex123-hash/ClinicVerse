"""End-to-end M1 integration check on the real cohort.

Runs the disclosure engine over real PhysioNet patients under both protocols and
reports spend, disclosure and waste. This is a mechanics check, not a result:
no model is fitted and no predictive claim is made. Its purpose is to show that
the protocol behaves as specified on real data, in particular that support-blind
runs incur wasted spend on unavailable groups while support-aware runs cannot.

Usage:
    python experiments/baselines/disclosure_smoke.py [--n 300] [--budget 5]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import numpy as np

from cliniverse.acquisition import load_panel_catalogue
from cliniverse.data import load_cohort
from cliniverse.data.splits import development_cohort
from cliniverse.log import get_logger
from twinbench.cases import build_manifest, engine_for
from twinbench.disclosure import Protocol
from twinbench.episode import (
    FixedOrder,
    NoAcquisition,
    Policy,
    RandomSupportOracle,
    RandomTrainFrequency,
    RandomUniformAll,
    run_episode,
)

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300, help="patients to run")
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--rate", type=float, default=0.5, help="masking rate")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--cutoff", type=int, default=24)
    args = parser.parse_args(argv)

    from twinbench.masking import GroupHours

    catalogue = load_panel_catalogue()
    development = development_cohort(load_cohort(sets=("a", "b"))).truncate(args.cutoff)
    if args.n < 1 or args.n >= development.n_patients:
        parser.error(f"--n must be in [1, {development.n_patients - 1}]")
    evaluation_indices = np.arange(args.n, dtype=np.int64)
    training_indices = np.setdiff1d(
        np.arange(development.n_patients, dtype=np.int64), evaluation_indices
    )
    cohort = development.select(evaluation_indices)
    training = development.select(training_indices)
    mechanism = GroupHours(rate=args.rate, seed=args.seed)
    epoch_hours = tuple(sorted({args.cutoff, *(h for h in (12, 18, 24) if h <= args.cutoff)}))
    frequency_template = RandomTrainFrequency.fit(
        training.m, training.variable_names, catalogue, seed=args.seed
    )

    # A declared ordering by observed measurement frequency (E-001). Authored by
    # the engineering team, NOT clinician-designed or clinician-validated.
    order = ("BMP_like", "CBC_like", "ABG_like", "Lactate", "hepatic_like")

    policies: dict[Protocol, list[tuple[str, Callable[[int], Policy]]]] = {
        Protocol.SUPPORT_AWARE: [
            ("no_acquisition", lambda patient: NoAcquisition()),
            (
                "random_support_oracle",
                lambda patient: RandomSupportOracle(seed=args.seed + patient),
            ),
            ("fixed_order", lambda patient: FixedOrder(order=order)),
        ],
        Protocol.SUPPORT_BLIND: [
            ("no_acquisition", lambda patient: NoAcquisition()),
            (
                "random_uniform_all",
                lambda patient: RandomUniformAll(seed=args.seed + patient),
            ),
            (
                "random_train_frequency",
                lambda patient: RandomTrainFrequency(
                    weights=frequency_template.weights, seed=args.seed + patient
                ),
            ),
            ("fixed_order", lambda patient: FixedOrder(order=order)),
        ],
    }

    print(f"\nn={cohort.n_patients}  budget={args.budget}  mask={mechanism.mechanism_id}")
    print(
        f"cutoff={args.cutoff}h  epochs={epoch_hours}  regime={catalogue.schedule_name}\n"
        f"train-frequency fit: n={training.n_patients:,}, excludes all evaluation patients"
    )
    print("=" * 92)
    print(
        f"{'policy':<24}{'protocol':<16}{'spent':>8}{'requests':>10}"
        f"{'disclosed':>11}{'empty req':>11}{'wasted':>9}"
    )
    print("-" * 92)

    for protocol in (Protocol.SUPPORT_AWARE, Protocol.SUPPORT_BLIND):
        evaluation_sets = tuple(sorted(set(cohort.source_set.tolist())))
        manifest = build_manifest(
            cohort,
            mechanism,
            catalogue,
            sets=evaluation_sets,
            cutoff_hours=args.cutoff,
            protocol=protocol,
            cost_regime=catalogue.schedule_name,
            budget=args.budget,
            epoch_hours=epoch_hours,
        )
        for name, make in policies[protocol]:
            spent, reqs, disclosed, empty, wasted = [], [], [], [], []
            for case in manifest.cases:
                engine = engine_for(cohort, case, mechanism, catalogue)
                trace = run_episode(engine, make(case.patient_index))
                spent.append(trace.spent)
                reqs.append(trace.n_requests)
                disclosed.append(trace.n_disclosed)
                empty.append(trace.n_empty_requests)
                wasted.append(trace.wasted_spend)
            print(
                f"{name:<24}{protocol!s:<16}{np.mean(spent):>8.2f}"
                f"{np.mean(reqs):>10.2f}{np.mean(disclosed):>11.2f}"
                f"{np.mean(empty):>11.2f}{np.mean(wasted):>9.2f}"
            )

    print(
        "\nMechanics check only — no model fitted, no predictive claim. "
        "Wasted spend is expected to be zero under support_aware (unavailable\n"
        "groups are not requestable) and positive under support_blind, which is "
        "the cost of not being told what is available."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
