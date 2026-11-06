# Verification scope

The release contains saved paper results and compact source code. These have
different verification scopes.

The saved-result reproduction path now starts from 11,430 archived event
predictions rather than pre-aggregated scores. It recomputes 1,080 cell-score
rows, 180 seed summaries and 52 performance summaries, as well as the paired
effects, budget interactions, deletion sensitivity, residual diagnostics,
specified amplitude diagnostics, and Table 4. Nine reference-comparison
groups pass with a maximum absolute difference of approximately 2.9e-14
(acceptance tolerance: 1e-10). All five numerical figures are rendered from
the newly computed tables.

Eight regression/validation tests cover archived-value agreement, missing
seeds, duplicate events, changed labels, incorrect folds, incorrect primary
suffixes, nonfinite predictions and undefined zero-error log interactions.
These are saved-data and bookkeeping checks, not a dynamic label-poisoning
test of a trained predictor.

- The 15 PNG/PDF/SVG assets for paper Figures 2-6 are byte-identical to the
  corresponding final manuscript assets. Their filenames have been simplified.
- The supplied full-precision result tables preserve the saved values.
- The standard-library checker recomputes equal-cell averages from 1,080 saved
  cell-level rows, verifies 180 run summaries, and checks 52 paper summaries,
  including population seed standard deviations and event counts.
- Core feature, history, branch initialization, normalization, head-solve,
  grouping, and kernel functions were compared with the archived source at the
  syntax-tree level. Only repository imports and unrelated ledger or method
  branches were changed in the copied components, as applicable.
- Dataset paths, command-line entry points, and the orchestration of the thirteen
  estimators were consolidated for this release. B3/N01 retain the published
  branch and gate calculations while omitting unrelated experimental arms.

The raw-data path has now been executed in the recorded original environment.
All six original file hashes match. Every extracted feature, label and event
identity matches the frozen six-cell input tables exactly. The original
prefit system-clock admission is retained with separate audit and admission
receipts; it does not assert that the wall-clock check passed.

All six source jobs finished before target predictions began. They produced
270 final models for the thirteen estimators, with the prescribed nested
source-only selection for Raw-CV. The 24 selection records reproduce candidate
order, per-cell scores, macro scores and winners exactly. All 11,430 prediction
rows match their archived identities; the maximum absolute prediction
difference is 1.5987211554602254e-14 in capacity-ratio
units. See `config/reproduction_validation.json` for method-level results and
source-selection checks. This verification made no changes to the scientific
protocol and added no target cells or seeds.

The original manuscript assets and archived prediction fixture are preserved.
The verification run is distinct from those historical experimental files.
The complete saved-prediction command and eight analysis tests also passed
in a clean Python 3.13.2 virtual environment without PyTorch; see
`config/verified_analysis_environment.json` and the analysis lockfile.
The training environment matched all eleven recorded package versions, but
fresh-machine training and cross-platform equivalence were not tested.
These reproduction checks do not establish external scientific validity or
replace the manuscript's stated development-study limitations.

Release validation caught an array-layout difference in the compact B3/N01/M02
preprocessing path: converting the full source matrix before scaler fitting
changed floating-point accumulation order. Restoring the original column-major
layout and each original label-selection order reproduced all six archived
anchors exactly. The affected models were rebuilt in separate
validation directories; earlier artifacts were retained. This changes
neither the estimator definition nor any experimental setting. The remaining
180 models were reused after their hashes were checked. Eleven tests pass in
the full environment, including three rejection checks for clock admission;
the analysis-only environment runs eight and skips the three requiring SciPy.

Fresh-prediction analysis also passes all nine table-comparison groups at
1e-10 absolute tolerance; the maximum observed table difference is
6.5636385215839255e-13. This is separate from the
capacity-ratio tolerance used for individual predictions.
