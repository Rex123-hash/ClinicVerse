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

import numpy as np

from cliniverse.acquisition import load_panel_catalogue
from cliniverse.data import load_cohort
from cliniverse.data.splits import development_cohort
from cliniverse.log import get_logger
from twinbench.cases import build_manifest, engine_for
from twinbench.disclosure import Protocol
from twinbench.episode import FixedOrder, NoAcquisition, RandomPolicy, run_episode

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300, help="patients to run")
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--rate", type=float, default=0.5, help="masking rate")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--cutoff", type=int, default=24)
    args = parser.parse_args(argv)

    from twinbench.masking import PanelEvents

    catalogue = load_panel_catalogue()
    cohort = development_cohort(load_cohort(sets=("a",))).truncate(args.cutoff)
    cohort = cohort.select(np.arange(min(args.n, cohort.n_patients), dtype=np.int64))
    mechanism = PanelEvents(rate=args.rate, seed=args.seed)

    # A declared ordering by observed measurement frequency (E-001). Authored by
    # the engineering team, NOT clinician-designed or clinician-validated.
    order = ("BMP_like", "CBC_like", "ABG_like", "Lactate", "hepatic_like")

    policies = [
        ("no_acquisition", lambda: NoAcquisition()),
        ("random", lambda: RandomPolicy(seed=args.seed)),
        ("fixed_order", lambda: FixedOrder(order=order)),
    ]

    print(f"\nn={cohort.n_patients}  budget={args.budget}  mask={mechanism.mechanism_id}")
    print(f"cutoff={args.cutoff}h  epochs=(12, 18, 24)  regime={catalogue.schedule_name}")
    print("=" * 92)
    print(
        f"{'policy':<16}{'protocol':<16}{'spent':>8}{'requests':>10}"
        f"{'disclosed':>11}{'empty req':>11}{'wasted':>9}"
    )
    print("-" * 92)

    for protocol in (Protocol.SUPPORT_AWARE, Protocol.SUPPORT_BLIND):
        manifest = build_manifest(
            cohort,
            mechanism,
            sets=("a",),
            cutoff_hours=args.cutoff,
            protocol=protocol,
            cost_regime=catalogue.schedule_name,
            budget=args.budget,
        )
        for name, make in policies:
            spent, reqs, disclosed, empty, wasted = [], [], [], [], []
            for case in manifest.cases:
                engine = engine_for(cohort, case, mechanism, catalogue)
                trace = run_episode(engine, make())
                spent.append(trace.spent)
                reqs.append(trace.n_requests)
                disclosed.append(trace.n_disclosed)
                empty.append(trace.n_empty_requests)
                wasted.append(trace.wasted_spend)
            print(
                f"{name:<16}{protocol!s:<16}{np.mean(spent):>8.2f}"
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
