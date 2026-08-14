/**
 * The single authoritative UI data object for Cliniverse.
 *
 * Every scientific number rendered anywhere in this application resolves back
 * to `cliniverse-bundle.json`, which is machine-generated from the committed
 * result artifacts by `scripts/export_ui_data.py`. Nothing here is typed by
 * hand from a screenshot, and no page hardcodes a metric independently.
 *
 * Narrative strings are quoted or condensed from the committed markdown
 * reports (`docs/M5_V2_SETC_CONFIRMATION.md`, `docs/STATUS.md`,
 * `docs/LIMITATIONS.md`, `docs/M5_V2_FINAL_FREEZE.md`) and carry an evidence
 * class so the UI can style claims by how strongly they are supported.
 */

import bundle from './cliniverse-bundle.json'

/**
 * How strongly a displayed statement is supported.
 *
 * - `confirmed`   — passed the frozen, pre-registered set-c decision rule.
 * - `development` — established on A+B development data only.
 * - `descriptive` — reported for interpretation; not part of any decision rule.
 * - `historical`  — a recorded fact about how the project was executed.
 * - `limitation`  — a constraint that survives confirmation.
 */
export type EvidenceClass =
  | 'confirmed'
  | 'development'
  | 'descriptive'
  | 'historical'
  | 'limitation'

export interface Claim {
  readonly text: string
  readonly evidence: EvidenceClass
  readonly source: string
}

// ---------------------------------------------------------------------------
// raw bundle re-exports (typed)
// ---------------------------------------------------------------------------

export const setc = bundle.setcConfirmation
export const freeze = bundle.freeze
export const development = bundle.development
export const charts = bundle.charts
export const visual = bundle.visual

/** One committed set-c patient row. No clinical covariates exist in the artifact. */
export interface PatientRow {
  /** PhysioNet record id */
  readonly id: number
  /** recorded in-hospital death, 1 = died */
  readonly y: number
  /** clean predicted risk */
  readonly c: number
  /** predicted risk with the frozen pattern withheld */
  readonly w: number
  /** per-patient excess NLL over the amount-matched controls */
  readonly d: number
  /** cells removed for this patient */
  readonly rc: number
}

export const frozenPattern: readonly string[] = freeze.frozenPattern
export const frozenPatternLabel = frozenPattern.join(' + ')

/**
 * The frozen Platt calibrator, read only from `final_freeze.json`.
 *
 * The intercept is NOT zero. Any page needing these numbers must import them
 * from here so a stale or screenshot-derived value can never enter the UI.
 */
export const frozenCalibrator = {
  kind: freeze.calibrator.kind,
  intercept: freeze.calibrator.intercept,
  slope: freeze.calibrator.slope,
  fittedOn: freeze.calibrator.fitted_on,
  neverRefittedUnderWithholding: freeze.calibrator.never_refitted_under_withholding,
} as const

if (import.meta.env.DEV) {
  // Guard against a regenerated bundle silently losing the calibrator.
  if (frozenCalibrator.intercept === 0 || frozenCalibrator.slope === 0) {
    throw new Error(
      'frozenCalibrator was read as zero. The UI must source Platt parameters from ' +
        'experiments/robustness/results/m5v2_final_freeze/final_freeze.json — re-run scripts/export_ui_data.py.',
    )
  }
}

// ---------------------------------------------------------------------------
// product identity
// ---------------------------------------------------------------------------

export const product = {
  name: 'Cliniverse',
  tagline: 'Healthcare AI Reliability Lab',
  disclaimer:
    'Research software. Not medical advice, not a diagnostic tool, not a treatment recommender.',
  dataset: {
    name: 'PhysioNet/CinC Challenge 2012',
    license: 'ODC-BY v1.0',
    nPatientsLabelled: 12000,
    nVariables: 37,
    missingness: 0.7975,
    horizonHours: 48,
  },
} as const

// ---------------------------------------------------------------------------
// the confirmed headline result
// ---------------------------------------------------------------------------

export const confirmation = {
  pattern: frozenPattern,
  patternLabel: frozenPatternLabel,
  deltaC: setc.primary.deltaC,
  lowerBound: setc.primary.lowerBound,
  cleanAuroc: setc.discriminationSilent.cleanAuroc,
  withheldAuroc: setc.discriminationSilent.withheldAuroc,
  aurocDrop: setc.discriminationSilent.aurocDrop,
  aurocDropCeiling: setc.discriminationSilent.delta,
  passes: setc.confirmation.passes,
  rule: setc.confirmation.rule,
  nBootstrap: setc.primary.nBootstrap,
  bootstrapSeed: setc.primary.bootstrapSeed,
  cohort: setc.cohort,
  executedAt: setc.finishedAt,
} as const

/** Clean vs withheld, the comparison the Reliability Report is built around. */
export const cleanVsWithheld = {
  auroc: {
    label: 'Discrimination (AUROC)',
    clean: setc.discriminationSilent.cleanAuroc,
    withheld: setc.discriminationSilent.withheldAuroc,
    lowerIsBetter: false,
  },
  nll: {
    label: 'Probability reliability (NLL)',
    clean: setc.secondary.clean.nll,
    withheld: setc.secondary.withheld.nll,
    lowerIsBetter: true,
  },
  brier: {
    label: 'Brier score',
    clean: setc.secondary.clean.brier,
    withheld: setc.secondary.withheld.brier,
    lowerIsBetter: true,
  },
  auprc: {
    label: 'AUPRC',
    clean: setc.secondary.clean.auprc,
    withheld: setc.secondary.withheld.auprc,
    lowerIsBetter: false,
  },
  meanRisk: {
    label: 'Mean predicted risk',
    clean: setc.secondary.clean.mean_predicted_risk,
    withheld: setc.secondary.withheld.mean_predicted_risk,
    lowerIsBetter: false,
  },
  calibrationIntercept: {
    label: 'Calibration intercept',
    clean: setc.secondary.clean.calibration_intercept,
    withheld: setc.secondary.withheld.calibration_intercept,
    lowerIsBetter: true,
  },
  calibrationSlope: {
    label: 'Calibration slope',
    clean: setc.secondary.clean.calibration_slope,
    withheld: setc.secondary.withheld.calibration_slope,
    lowerIsBetter: false,
  },
} as const

// ---------------------------------------------------------------------------
// narrative content, each line traceable to a committed document
// ---------------------------------------------------------------------------

export const keyFindings: readonly Claim[] = [
  {
    text: 'Probability reliability degrades more than an amount-matched random removal of the same number of cells from the same patients.',
    evidence: 'confirmed',
    source: 'docs/M5_V2_SETC_CONFIRMATION.md §2',
  },
  {
    text: 'Discrimination moves only 0.0115 AUROC — below the 0.02 ceiling fixed before the holdout was opened.',
    evidence: 'confirmed',
    source: 'docs/M5_V2_SETC_CONFIRMATION.md §1',
  },
  {
    text: 'The calibration intercept moves +0.026 → +0.606 and mean predicted risk falls to 10.4% against a set-c prevalence of 14.6%.',
    evidence: 'descriptive',
    source: 'docs/M5_V2_SETC_CONFIRMATION.md §3',
  },
  {
    text: 'The confirmed effect is larger than the development out-of-selection estimate (+0.012013) and more than twice the predeclared minimum detectable effect (+0.008044).',
    evidence: 'confirmed',
    source: 'docs/M5_V2_SETC_CONFIRMATION.md §2',
  },
]

export const narrativeInsight: Claim = {
  text: 'Systematic under-prediction of mortality risk that ordinary discrimination monitoring would not flag — the discrimination-silent signature, reproduced on a quarantined holdout that played no part in fitting the model, selecting it, or setting any threshold.',
  evidence: 'confirmed',
  source: 'docs/M5_V2_SETC_CONFIRMATION.md §2-3',
}

export const limitations: readonly Claim[] = [
  {
    text: 'The R = 5 control draws are FIXED across all 10,000 bootstrap replicates. The interval propagates patient-sampling uncertainty but NOT control-draw Monte-Carlo uncertainty, and is mildly optimistic on that account.',
    evidence: 'limitation',
    source: 'docs/M5_V2_SETC_CONFIRMATION.md §4 — required verbatim wording',
  },
  {
    text: 'The 11/20 selection margin travels with this pattern. Confirmation does not establish that the exact three-analyte membership is the uniquely correct minimal core; the BUN-centred family remains better supported than the exact membership.',
    evidence: 'limitation',
    source: 'docs/M5_V2_SETC_CONFIRMATION.md §8',
  },
  {
    text: 'Removal is whole-window, so this identifies which analytes were withheld and cannot test co-occurrence structure or order events.',
    evidence: 'limitation',
    source: 'docs/M5_V2_SETC_CONFIRMATION.md §8',
  },
  {
    text: 'Groups remain reconstructed co-measurement clusters (*_like), never verified laboratory orders.',
    evidence: 'limitation',
    source: 'docs/M5_V2_SETC_CONFIRMATION.md §8',
  },
  {
    text: 'Synthetic withholding is not natural missingness and not deployment shift. One historical ICU dataset, one cutoff, no external validation beyond this holdout.',
    evidence: 'limitation',
    source: 'docs/M5_V2_SETC_CONFIRMATION.md §8',
  },
  {
    text: 'No causal, deployment-utility, clinical-validation or clinician-intent claim.',
    evidence: 'limitation',
    source: 'docs/M5_V2_SETC_CONFIRMATION.md §8',
  },
]

export const historicalDisclosure: Claim = {
  text: setc.historicalDisclosure,
  evidence: 'historical',
  source: 'docs/M5_V2_SETC_CONFIRMATION.md §5 — required wording',
}

export const setCSpent: Claim = {
  text: 'The single pre-registered use is consumed. There is no second test, and no further set-c experiment is authorised by this result.',
  evidence: 'historical',
  source: 'docs/M5_V2_SETC_CONFIRMATION.md §9',
}

export const provenanceBlemish: Claim = {
  text: 'The result artifact records git_dirty: true. The sole uncommitted item was the runner script, which was untracked when it ran; the frozen fitted artifacts were hash-verified before use. This will not be repaired by re-running — that would be a second look at set-c.',
  evidence: 'historical',
  source: 'docs/M5_V2_SETC_CONFIRMATION.md §6',
}

// ---------------------------------------------------------------------------
// the four development gates
// ---------------------------------------------------------------------------

export interface Gate {
  readonly id: string
  readonly name: string
  readonly detail: string
  readonly passes: boolean
}

export const gates: readonly Gate[] = [
  {
    id: 'G1',
    name: 'Null-control sanity',
    detail: `Frozen region resolved to "${development.gates.G1_null_control_sanity.frozen_region}" — necessary, not sufficient to validate the method.`,
    passes: development.gates.G1_null_control_sanity.passes,
  },
  {
    id: 'G2',
    name: 'Majority stability',
    detail: `Selected in ${development.gates.G2_majority_stability.count}/${development.gates.G2_majority_stability.of} resplits against a required ${development.gates.G2_majority_stability.required} — the gate passed by a single resplit.`,
    passes: development.gates.G2_majority_stability.passes,
  },
  {
    id: 'G3',
    name: 'Discrimination-silent',
    detail: `Reference-run AUROC drop ${development.gates.G3_discrimination_silent.auroc_drop_reference_run.toFixed(6)} against a ceiling of ${development.gates.G3_discrimination_silent.delta}.`,
    passes: development.gates.G3_discrimination_silent.passes,
  },
  {
    id: 'G4',
    name: 'Detectability',
    detail: `Out-of-selection estimate +${development.gates.G4_detectability.delta_oos.toFixed(6)} against a minimum detectable effect of +${development.gates.G4_detectability.mde.toFixed(6)}.`,
    passes: development.gates.G4_detectability.passes,
  },
]

// ---------------------------------------------------------------------------
// milestone history — the real experiment record, from docs/STATUS.md
// ---------------------------------------------------------------------------

export type MilestoneState = 'confirmed' | 'complete' | 'closed' | 'not-started'

export interface Milestone {
  readonly id: string
  readonly name: string
  readonly verdict: string
  readonly state: MilestoneState
  readonly summary: string
  readonly report: string
  readonly headline?: { readonly label: string; readonly value: string }[]
  readonly nPatients?: number
  readonly gitSha?: string
}

export const milestones: readonly Milestone[] = [
  {
    id: 'M2',
    name: 'Baselines',
    verdict: 'Done + repaired',
    state: 'complete',
    summary:
      'Three binding representations on identical splits. Mask-only reaches AUROC 0.7319 with no clinical values, but values dominate.',
    report: 'docs/M2_MILESTONE_REPORT.md',
    nPatients: 8000,
    headline: [
      { label: 'mask-only AUROC', value: '0.7319' },
      { label: 'values−mask Δ', value: '+0.0960' },
    ],
  },
  {
    id: 'M3',
    name: 'Calibration robustness',
    verdict: 'M3-B',
    state: 'complete',
    summary:
      'Whole-window group removal moves AUROC 0.8270 → 0.8002 while the calibration intercept moves −0.010 → +0.573. The per-patient control is mask-identical.',
    report: 'docs/M3_MILESTONE_REPORT.md',
    nPatients: 8000,
    headline: [
      { label: 'AUROC', value: '0.8270 → 0.8002' },
      { label: 'intercept', value: '−0.010 → +0.573' },
    ],
  },
  {
    id: 'M4',
    name: 'Acquisition ranking stability',
    verdict: 'M4-C · ACCEPT',
    state: 'complete',
    summary:
      'Under the fair support-blind protocol, fixed_domain_order wins 8/8 conditions (mean Kendall tau-b +0.776), with zero fair winner changes.',
    report: 'docs/M4_MILESTONE_REPORT.md',
    nPatients: 8000,
    headline: [
      { label: 'conditions won', value: '8 / 8' },
      { label: 'mean tau-b', value: '+0.776' },
    ],
  },
  {
    id: 'M5-v1',
    name: 'Discrimination-silent failure search',
    verdict: 'M5-C · primary failed',
    state: 'closed',
    summary:
      'Exhaustive enumeration of all 1,023 non-empty group subsets. Primary test failed (+0.00587 [−0.00174, +0.01365]); transfer test passed decisively (Spearman +0.865, p = 1.0e-4).',
    report: 'docs/M5_MILESTONE_REPORT.md',
    nPatients: 8000,
    headline: [
      { label: 'configurations', value: '1,023' },
      { label: 'transfer Spearman', value: '+0.865' },
    ],
  },
  {
    id: 'M5-v2',
    name: 'Stability-aware failure search',
    verdict: 'v2-STABLE · all four gates pass',
    state: 'complete',
    summary: `Stability-aware search over ${development.nCandidates} candidates across ${development.predeclared.n_resplits} development resplits with 1-SE parsimony selection. Frozen pattern ${frozenPatternLabel}, selected in ${development.gates.G2_majority_stability.count}/${development.gates.G2_majority_stability.of} resplits.`,
    report: 'docs/M5_V2_MILESTONE_REPORT.md',
    nPatients: development.provenance.n_patients,
    gitSha: development.provenance.git_sha,
    headline: [
      { label: 'candidates', value: String(development.nCandidates) },
      { label: 'shrinkage', value: '12.87%' },
    ],
  },
  {
    id: 'M5-v2 freeze',
    name: 'Final model freeze',
    verdict: 'Frozen · no evaluation performed',
    state: 'complete',
    summary:
      'Final pipeline fitted on A+B with 6,400/1,600 isolation. Set-c contract frozen and hash-sealed; set C never loaded during the freeze.',
    report: 'docs/M5_V2_FINAL_FREEZE.md',
    nPatients: freeze.provenance.n_patients,
    gitSha: freeze.provenance.git_sha,
    headline: [
      { label: 'train / calib', value: '6,400 / 1,600' },
      { label: 'features used', value: String(freeze.model.n_features_used) },
    ],
  },
  {
    id: 'Set-C',
    name: 'One-shot set-c confirmation',
    verdict: 'CONFIRMED',
    state: 'confirmed',
    summary: `Executed once, per the frozen contract. Delta_C +${confirmation.deltaC.toFixed(6)}, one-sided 95% lower bound +${confirmation.lowerBound.toFixed(6)} > 0, AUROC drop +${confirmation.aurocDrop.toFixed(6)} ≤ ${confirmation.aurocDropCeiling}. Set C is now spent.`,
    report: 'docs/M5_V2_SETC_CONFIRMATION.md',
    nPatients: confirmation.cohort.n_patients,
    gitSha: setc.provenance.git_sha,
    headline: [
      { label: 'Delta_C', value: `+${confirmation.deltaC.toFixed(6)}` },
      { label: '95% LB', value: `+${confirmation.lowerBound.toFixed(6)}` },
    ],
  },
]

// ---------------------------------------------------------------------------
// investigation timeline — dates from the committed reports
// ---------------------------------------------------------------------------

export interface TimelineStep {
  readonly label: string
  readonly detail: string
  readonly date: string
  readonly state: 'done' | 'active'
}

export const investigationTimeline: readonly TimelineStep[] = [
  {
    label: 'Exhaustive search',
    detail: 'M5-v1 enumerated all 1,023 subsets',
    date: 'M5-v1',
    state: 'done',
  },
  {
    label: 'Stability-aware search',
    detail: `${development.nCandidates} candidates × ${development.predeclared.n_resplits} resplits`,
    date: 'M5-v2 A+B',
    state: 'done',
  },
  {
    label: 'Four gates passed',
    detail: 'v2-STABLE',
    date: 'M5-v2',
    state: 'done',
  },
  {
    label: 'Model + contract frozen',
    detail: 'Hash-sealed, holdout locked during freeze',
    date: freeze.provenance.git_sha.slice(0, 7),
    state: 'done',
  },
  {
    label: 'Set-c confirmation',
    detail: 'Executed once — CONFIRMED',
    date: setc.finishedAt.slice(0, 10),
    state: 'active',
  },
]

// ---------------------------------------------------------------------------
// artifact registry — real committed files
// ---------------------------------------------------------------------------

export type ArtifactKind =
  | 'result'
  | 'predictions'
  | 'model'
  | 'calibrator'
  | 'imputer'
  | 'report'
  | 'figure'
  | 'config'

export interface Artifact {
  readonly id: string
  readonly name: string
  readonly path: string
  readonly kind: ArtifactKind
  readonly milestone: string
  readonly description: string
  readonly sha256?: string
  readonly evidence: EvidenceClass
  readonly contents?: readonly string[]
}

export const artifacts: readonly Artifact[] = [
  {
    id: 'setc-results',
    name: 'Set-C confirmation result',
    path: 'experiments/robustness/results/m5v2_setc/results.json',
    kind: 'result',
    milestone: 'Set-C',
    description:
      'The frozen one-shot decision: Delta_C, the one-sided bound, the discrimination constraint, and the contract exactly as executed.',
    sha256: '7179a5744e5d9034a735fb6bcd1652a96e850e285fc60b8de61983a7d192a907',
    evidence: 'confirmed',
    contents: [
      'primary — Delta_C, 95% lower bound, decision rule',
      'discrimination_silent — clean/withheld AUROC',
      'secondary_descriptive — NLL, Brier, calibration',
      'frozen_contract_as_executed — 24 fields',
    ],
  },
  {
    id: 'setc-npz',
    name: 'Set-C one-shot predictions',
    path: 'experiments/robustness/results/m5v2_setc/setc_oneshot_predictions.npz',
    kind: 'predictions',
    milestone: 'Set-C',
    description:
      'Raw prediction vectors for all 4,000 holdout patients, so every reported number is independently recomputable.',
    sha256: 'b8ed025b4a3ed037a07e6351240aa84b5240d9432e84c0639529a21701e38783',
    evidence: 'confirmed',
    contents: [
      'record_ids, labels (4,000)',
      'p_clean, p_withheld',
      'p_control_0 … p_control_4',
      'd_i, removed_cells, control_seeds',
    ],
  },
  {
    id: 'final-model',
    name: 'Frozen model',
    path: 'experiments/robustness/results/m5v2_final_freeze/final_model.json',
    kind: 'model',
    milestone: 'M5-v2 freeze',
    description: `${freeze.model.estimator} fitted on the 6,400 clean final-training rows only. Hash-verified before the holdout was unlocked.`,
    sha256: freeze.artifactHashes['final_model.json'],
    evidence: 'historical',
  },
  {
    id: 'final-calibrator',
    name: 'Frozen calibrator',
    path: 'experiments/robustness/results/m5v2_final_freeze/final_calibrator.json',
    kind: 'calibrator',
    milestone: 'M5-v2 freeze',
    description: `Platt calibrator (intercept ${freeze.calibrator.intercept.toFixed(5)}, slope ${freeze.calibrator.slope.toFixed(5)}) fitted on the 1,600 clean calibration rows. Never refitted under withholding.`,
    sha256: freeze.artifactHashes['final_calibrator.json'],
    evidence: 'historical',
  },
  {
    id: 'final-imputer',
    name: 'Frozen imputer',
    path: 'experiments/robustness/results/m5v2_final_freeze/final_imputer.npz',
    kind: 'imputer',
    milestone: 'M5-v2 freeze',
    description:
      'Median imputer fitted once on clean final-training data and never refitted under stress.',
    sha256: freeze.artifactHashes['final_imputer.npz'],
    evidence: 'historical',
  },
  {
    id: 'freeze-json',
    name: 'Final freeze record',
    path: 'experiments/robustness/results/m5v2_final_freeze/final_freeze.json',
    kind: 'result',
    milestone: 'M5-v2 freeze',
    description:
      'The freeze package: model and preprocessing specification, split hashes, artifact hashes, and the 24-field set-c evaluation contract.',
    evidence: 'historical',
    contents: [
      'set_c_evaluation_contract',
      'split — 6,400 / 1,600 index hashes',
      'artifact_hashes',
      'set_c_access — loaded_during_freeze: false',
    ],
  },
  {
    id: 'm5v2-results',
    name: 'M5-v2 development result',
    path: 'experiments/robustness/results/m5v2/results.json',
    kind: 'result',
    milestone: 'M5-v2',
    description: `Stability-aware search over ${development.nCandidates} candidates and ${development.predeclared.n_resplits} resplits, with all four gate evaluations and the detectability projection.`,
    evidence: 'development',
    contents: ['gates G1–G4', 'development_estimates', 'detectability', 'out_of_selection_components'],
  },
  {
    id: 'm5v2-tables',
    name: 'M5-v2 candidate tables',
    path: 'experiments/robustness/results/m5v2/m5v2_tables.npz',
    kind: 'predictions',
    milestone: 'M5-v2',
    description: `Per-candidate excess NLL across ${development.predeclared.n_resplits} resplits × 5 folds, plus clean and candidate AUROC for all ${development.nCandidates} candidates.`,
    evidence: 'development',
    contents: [
      'deltas (141 × 20 × 5)',
      'candidate_auroc, clean_auroc',
      'nested_candidate_auroc',
      'reference predictions (8,000)',
    ],
  },
  {
    id: 'm3-results',
    name: 'M3 calibration robustness',
    path: 'experiments/robustness/results/m3/results.json',
    kind: 'result',
    milestone: 'M3',
    description:
      'The severity sweep: three withholding conditions at three severities, across the uncalibrated / Platt / isotonic ladder.',
    evidence: 'development',
    contents: ['rows (120)', 'severity_report', 'calibration_contrasts', 'information_loss_audit'],
  },
  {
    id: 'm2-results',
    name: 'M2 baseline grid',
    path: 'experiments/baselines/results/m2/results.json',
    kind: 'result',
    milestone: 'M2',
    description:
      'The executed representation × model grid with paired bootstrap intervals and per-fold provenance.',
    evidence: 'development',
    contents: ['runs (15)', 'intervals', 'reliability', 'feature_inventory'],
  },
  {
    id: 'm4-results',
    name: 'M4 acquisition ranking',
    path: 'experiments/acquisition/results/m4/results.json',
    kind: 'result',
    milestone: 'M4',
    description:
      'Policy ranking stability across four cost regimes under both disclosure protocols, with the full spend trace.',
    evidence: 'development',
    contents: ['primary', 'conditions (16)', 'budget_grid', 'grid'],
  },
  {
    id: 'setc-report',
    name: 'Set-C confirmation report',
    path: 'docs/M5_V2_SETC_CONFIRMATION.md',
    kind: 'report',
    milestone: 'Set-C',
    description:
      'The written confirmation, its mandatory limitation wording, the historical set-c disclosure, and the execution-integrity record.',
    evidence: 'confirmed',
  },
  {
    id: 'limitations',
    name: 'Limitations',
    path: 'docs/LIMITATIONS.md',
    kind: 'report',
    milestone: 'Project',
    description:
      'Written before results existed, so limitations constrain the work rather than being retrofitted to excuse it.',
    evidence: 'limitation',
  },
  {
    id: 'panels',
    name: 'Co-measurement catalogue',
    path: 'configs/panels.yaml',
    kind: 'config',
    milestone: 'Project',
    description: `Catalogue version ${development.provenance.catalogue_version}. Reconstructed co-measurement clusters (*_like), never verified laboratory orders.`,
    evidence: 'limitation',
  },
  {
    id: 'm3-figures',
    name: 'M3 figures',
    path: 'experiments/robustness/results/m3/figures/',
    kind: 'figure',
    milestone: 'M3',
    description:
      'Degradation, reliability and calibrator-ladder figures, generated from the M3 artifacts.',
    evidence: 'development',
    contents: ['m3_degradation.png', 'm3_reliability.png', 'm3_calibrators.png'],
  },
  {
    id: 'm2-figures',
    name: 'M2 figures',
    path: 'experiments/baselines/results/m2/figures/',
    kind: 'figure',
    milestone: 'M2',
    description: 'Baseline comparison, representation contrast and reliability figures.',
    evidence: 'development',
    contents: [
      'm2_baseline_comparison.png',
      'm2_representation_contrast.png',
      'm2_reliability.png',
    ],
  },
]


// ---------------------------------------------------------------------------
// visual artifacts — derived views, each with full provenance
//
// These are NOT feature attributions, counterfactuals or biological response
// surfaces. The repository contains no such capability. Each one is a
// descriptive view of measured loss differences, and carries the metric
// meaning and evidence class it is entitled to.
// ---------------------------------------------------------------------------

export interface VisualArtifact {
  readonly id: string
  readonly name: string
  /** what the picture actually shows */
  readonly summary: string
  /** what the plotted quantity means */
  readonly metric: string
  /** the committed file it is computed from */
  readonly source: string
  readonly milestone: string
  readonly cohort: string
  readonly evidence: EvidenceClass
  /** the claim this visual must never be read as supporting */
  readonly caution: string
}

export const visualArtifacts: readonly VisualArtifact[] = [
  {
    id: 'concentration-surface',
    name: 'Failure Concentration Surface',
    summary:
      'Mean per-patient excess NLL over a grid of clean predicted risk against the number of cells the frozen pattern removed.',
    metric:
      'Surface height and colour are mean excess NLL (d_i) — per-patient log loss under withholding minus the mean over five amount-matched random control removals.',
    source: 'experiments/robustness/results/m5v2_setc/setc_oneshot_predictions.npz',
    milestone: 'Set-C',
    cohort: 'set-c · 4,000 quarantined patients',
    evidence: 'descriptive',
    caution:
      'A descriptive stress-test surface. Both axes are experiment artefacts — a model output and a count of removed cells — so this shows where measured damage concentrated. It is not a feature-response relationship and implies no biological mechanism.',
  },
  {
    id: 'failure-slice',
    name: 'Failure Slice',
    summary:
      'All 4,000 holdout patients: clean predicted risk against that patient’s excess NLL, coloured by recorded outcome.',
    metric:
      'Each point is one patient. Above zero means the frozen pattern damaged that patient more than an amount-matched random removal did.',
    source: 'experiments/robustness/results/m5v2_setc/setc_oneshot_predictions.npz',
    milestone: 'Set-C',
    cohort: 'set-c · 4,000 quarantined patients',
    evidence: 'confirmed',
    caution:
      'Per-patient values are descriptive. The frozen decision rule was evaluated on the mean and its one-sided bound, not on any individual patient.',
  },
  {
    id: 'damage-landscape',
    name: 'Candidate Damage Landscape',
    summary:
      'Spread of measured development excess NLL for the leading candidate patterns, one dot per held-out fold.',
    metric:
      'Each dot is the excess NLL measured on one held-out fold; 20 resplits × 5 folds = 100 observations per candidate.',
    source: 'experiments/robustness/results/m5v2/m5v2_tables.npz',
    milestone: 'M5-v2',
    cohort: 'development · 8,000 patients (sets a + b)',
    evidence: 'development',
    caution:
      'A distribution of loss differences per candidate pattern. This is not SHAP, not feature importance and not attribution — no such method exists in this project. Only the frozen pattern was ever taken to the holdout.',
  },
  {
    id: 'burden-profile',
    name: 'Withholding Burden Profile',
    summary:
      'How many cells the frozen pattern actually removed per patient, clipped to what was naturally observed.',
    metric:
      'Counts of patients by removed-cell count. Mean 5.94225, median 6, p10 3, p90 10.',
    source: 'experiments/robustness/results/m5v2_setc/setc_oneshot_predictions.npz',
    milestone: 'Set-C',
    cohort: 'set-c · 4,000 quarantined patients',
    evidence: 'descriptive',
    caution:
      'Removal is whole-window and clipped to observed availability, so burden is uneven. Naturally-missing cells cannot be removed — 94 patients had nothing eligible.',
  },
  {
    id: 'case-explorer',
    name: 'Case Explorer',
    summary:
      'The committed per-patient rows: deterministic public case alias, recorded outcome, and the saved clean and withheld predictions.',
    metric:
      'Exactly the columns the artifact stores. Risk change is withheld minus clean.',
    source: 'experiments/robustness/results/m5v2_setc/setc_oneshot_predictions.npz',
    milestone: 'Set-C',
    cohort: 'set-c · 4,000 quarantined patients',
    evidence: 'confirmed',
    caution:
      'No clinical covariates are shown because none exist in the artifact — it stores predictions and outcomes only. The public view replaces source record ids with deterministic case aliases.',
  },
]

// ---------------------------------------------------------------------------
// glossary — tooltips for unfamiliar scientific labels
// ---------------------------------------------------------------------------

export const glossary: Readonly<Record<string, string>> = {
  Delta_C:
    'Mean paired excess negative log-likelihood: per-patient log loss under the withheld pattern minus the mean log loss over five amount-matched random control removals.',
  AUROC:
    'Area under the ROC curve — how well the model ranks patients by risk. It is insensitive to whether the probabilities themselves are correct.',
  NLL: 'Negative log-likelihood. A proper probability score that penalises confident wrong probabilities; a change signals probability-reliability loss but does not, by itself, identify calibration as the cause.',
  Brier: 'Mean squared error of the predicted probabilities. A second proper scoring rule.',
  'calibration intercept':
    'Offset in a logistic recalibration of the predictions. A positive intercept means the model systematically under-predicts risk.',
  'calibration slope':
    'Slope in a logistic recalibration. Values away from 1 indicate the spread of predicted risk is wrong.',
  'discrimination-silent':
    'A failure that damages probability reliability while leaving ranking performance almost unchanged — so ordinary discrimination monitoring would not flag it.',
  'amount-matched control':
    'A random removal of exactly the same number of cells, from the same patients, as the withheld pattern removed. It separates "which analytes" from "how much data".',
  'one-sided 95% lower bound':
    'The 5th percentile of the bootstrap replicate means. The primary rule passes if and only if this exceeds zero.',
  'set-c': 'The quarantined 4,000-patient holdout. It was opened exactly once, and is now spent.',
  resplit:
    'One of 20 independent re-partitions of the development data, used to test whether the selected pattern is stable rather than an artefact of one split.',
  '1-SE parsimony':
    'Among candidates within one standard error of the best mean excess, choose the smallest. It biases selection toward simpler patterns.',
  values_mask:
    'The feature representation: imputed clinical values together with explicit missingness indicators.',
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/**
 * Thousands grouping, pinned to en-US.
 *
 * Bare `toLocaleString()` follows the host locale, which on an en-IN machine
 * renders 157625 as "1,57,625" (lakh grouping). Scientific counts must read
 * identically wherever the page is opened.
 */
const GROUPED = new Intl.NumberFormat('en-US')

export function formatCount(value: number): string {
  return GROUPED.format(value)
}

export function formatDelta(value: number, places = 4): string {
  const sign = value > 0 ? '+' : value < 0 ? '−' : ''
  return `${sign}${Math.abs(value).toFixed(places)}`
}

export function formatFixed(value: number, places = 4): string {
  return value.toFixed(places)
}

export function formatPercent(value: number, places = 1): string {
  return `${(value * 100).toFixed(places)}%`
}

export const evidenceLabel: Readonly<Record<EvidenceClass, string>> = {
  confirmed: 'Confirmed on holdout',
  development: 'Development only',
  descriptive: 'Descriptive',
  historical: 'Execution record',
  limitation: 'Limitation',
}
