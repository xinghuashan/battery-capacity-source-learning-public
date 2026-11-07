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

- The 15 PNG/PDF/SVG assets for paper Figures 2-6 are the archived figure
  exports supplied with the scientific snapshot. Their filenames have been
  simplified. Their relationship to the later submission PDFs is described
  under [Figure asset version scope](#figure-asset-version-scope) below.
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

## Figure asset version scope

Updated: 7 November 2026. The comparison concerns the five numerical figure
PDFs in manuscript version `v20261106_17_energy_reports` and the corresponding
repository PDFs at commit `8d03957beaae8827183bce2a1776fc6a7996beec`.
It does not cover Figure 1 or establish equivalence for every image format
or future manuscript revision.

All five submission PDFs have different SHA-256 hashes, page dimensions and
metadata from the repository PDFs. The earlier unqualified statement that
the assets are byte-identical to the final manuscript therefore does not
apply to this submission version.

The comparison found matching extracted text, vector drawing-object counts,
colors and path-item counts. The maximum absolute difference in drawing
bounding-box coordinates, normalized by each page's width and height, was
approximately 5.81e-6. These checks and visual inspection support consistency
of the displayed figure content; the file differences are consistent with
resizing and re-export for submission. This is not a byte-identity or
pixel-identity claim, nor a claim of independent experimental validation.

The hashes below identify the exact PDF pairs compared. Submission filenames
refer to that manuscript version's `figures/` directory; repository filenames
refer to this repository's `figures/` directory.

| Figure | Repository PDF | Repository SHA-256 | Submission PDF | Submission SHA-256 |
|---|---|---|---|---|
| 2 | `figure_02_overall_performance.pdf` | `5f3a86bd6b61bb0d5529675c825e303374565fe908598703d000d3c3894cd198` | `Figure_02.pdf` | `c818db4b18e42ec23282c8bdd3fcbea06b21193a82fd462d90ae63e0c9fbd14c` |
| 3 | `figure_03_event_predictions.pdf` | `e0f41ced40fcf7c94896f894cb508cf25dbaa8ed5ed7c35fae9ca1d38315011d` | `Figure_03.pdf` | `b4c635a25a13621c04c84f00c51a4d7f3739321e9cb798b2f4a6f100e30f07fe` |
| 4 | `figure_04_cell_effects.pdf` | `40921b5fba6c1105d72e968bebdfcbee6673d5d9cb2a949d591063c837777909` | `Figure_04.pdf` | `adab05ae9589543d69942a207986d28533c1b385fbc8e4beb0165ff97bbd733e` |
| 5 | `figure_05_robustness.pdf` | `079e7b88df361713c5a1c70dc069eea211ce8ac15d4ebf5842967bae5e435df6` | `Figure_05.pdf` | `2bd634269943894da30544fef994c2f785c20a6646b1bfb4fd141f284074aeb0` |
| 6 | `figure_06_correction_diagnostics.pdf` | `2ee8b049761392832c3021ed595ace1067707cbf4dd8a20543c6b687a12f3634` | `Figure_06.pdf` | `ade13545a4a23e1c8e9d334151cef0a36ecf965d13bac6ad9c37765c88f45487` |
