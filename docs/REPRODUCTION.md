# Reproduction instructions

## One-command paths

For the saved predictions supplied with this repository:

```bash
python -m pip install -r requirements-analysis.txt
python -m battery_capacity.reproduce --mode saved --output outputs/saved_reproduction
```

The output contains full-precision analysis tables, Tables 3–4 in Markdown,
all five numerical figures in three formats, a figure-input hash manifest,
and a verification report against the archived values. It requires no raw
data, PyTorch installation, model weights, fitting, or inference.

For a new run from the original six MAT files, explicitly select training:

```bash
python -m battery_capacity.reproduce --mode train --raw-dir data/raw --output outputs/new_run
```

This command prepares inputs, trains all thirteen methods for all six
fold/schedule combinations, predicts the prescribed target pairs, and runs
the same complete analysis and plotting path. All source training finishes
before the target prediction stage. Fresh numerical results are compared with
the archived values. A discrepancy is reported and produces a nonzero exit
status; the program does not change seeds, epochs, grids, or amplitudes to
force agreement. The component commands in this training path have been
executed end to end in the recorded environment, with 270 final models and
11,430 event predictions checked against the archived experiments. The saved
mode has also passed as a single command in a clean analysis environment.

To analyze results created by individual training/prediction commands, supply
all six prediction directories to `python -m battery_capacity.analysis
--predictions ... --output outputs/analysis`. Then run
`python -m battery_capacity.figures --analysis outputs/analysis --output outputs/figures`
and `python -m battery_capacity.verify --analysis outputs/analysis`.

## Software and data

Use a separate Python environment. `requirements.txt` records the main package
versions in the experimental environment; `config/recorded_environment.json`
contains the complete captured list. The recorded PyTorch package is
2.11.0+cu128, although all model calculations use CPU float64 and one thread.
Installing that exact wheel may require the corresponding PyTorch package
index. Training was verified in the original environment, not a freshly
installed training environment. Analysis-only dependency installation and
the full saved-result pipeline were verified in a fresh virtual environment.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Download `Battery Dataset.zip` from
<https://zenodo.org/records/10963339>. Put the six files
`RW_battery-{1,2,3,5,6,7}.mat` in `data/raw/`, with the original filenames.
Files from other charging protocols or other cells are not inputs to this study.
The program opens only these six named files; it does not search the whole archive.

```bash
python -m battery_capacity.prepare_data --raw-dir data/raw --output data/inputs
```

The extractor retains record order and original event ordinals, verifies the
charge phases, and uses the measured next-reference capacity divided by 2 Ah.
History contains the preceding five ordinary cycles and excludes the current
input cycle. The output has 127 eligible events. Generated files include six
CSV input tables and record-check files. Float64 values are written with 17
significant digits and read with round-trip parsing.

The original study admitted four documented system-clock reversals before
model fitting. Extraction follows published MAT record order and native
phase-relative time; it does not repair or reorder system timestamps.
`config/accepted_clock_scope.json` preserves the original admission identifier,
source hash, six raw-file hashes and exact exception locations. The compact
extractor checks these conditions and rejects other exceptions. Original
clock-check failures remain visible in each `_audit.json`; a separate
`_clock_admission.json` records the scoped decision. Admission does not mean
that the wall-clock audit passed. This restores the original preprocessing
contract rather than changing the feature or label definition.

The wheel command follows the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/).
For analysis alone, `requirements-analysis-lock.txt` records the complete
package set tested in a clean Windows Python 3.13.2 virtual environment,
without PyTorch. Cross-platform numerical equivalence is not guaranteed.

## Source training and target evaluation

For each fold 0, 1, 2 and stride 1, 10, train the desired methods into a new
output directory. `train` opens only the four source-cell input files for that
fold. It does not open the two target-cell tables. The examples below use all
thirteen methods; M05 alone can be selected with `--methods M05`.

```bash
python -m battery_capacity.train --inputs data/inputs --fold 0 --stride 10 --methods Ridge Ridge-CV KRR KRR-wide B3 N01 M01 M02 M03 M04 M05 M06 Raw-CV --output outputs/fold0_stride10/models
python -m battery_capacity.predict --inputs data/inputs --models outputs/fold0_stride10/models --output outputs/fold0_stride10/predictions
```

Repeat with folds 1 and 2, retaining distinct output directories. Repeat the
three-fold procedure with stride 1 for the full-label condition. Seeds 0-4 are
the default for stochastic methods. Deterministic methods are fitted once.
All source selection and final source fits finish before `predict` loads the
target pair. No target labels enter `Predictor.predict`, and the pipeline
does not fit target statistics or a target calibration head.

Raw-CV is substantially more expensive than the other estimators: each fold
and label schedule selects from eight candidates using four outer source
queries and 300-step nested feature training. Candidate selection is reused
for the five final seeds within one `train` invocation.

Output directories must not already exist. An interrupted run leaves its
artifacts intact. The compact runner does not silently resume or overwrite a
partial run. Load only model files from a source you trust; joblib model files
contain Python objects.

## Metrics

After predictions for all three folds of one schedule are available:

```bash
python -m battery_capacity.evaluate --predictions outputs/fold0_stride10/predictions outputs/fold1_stride10/predictions outputs/fold2_stride10/predictions --output outputs/stride10_metrics
```

The evaluator verifies coverage of the same six physical cells and 127 events.
It computes the primary 109-event and all-127 views separately. Errors are
measured within cells in percentage points, averaged equally across cells,
and then summarized across seeds with population standard deviation. It never
pools all events into one RMSE or ensembles predictions across seeds.

For the already supplied paper results, no model calculation is necessary:

```bash
python scripts/check_saved_results.py
python scripts/plot_overview.py --output outputs/overview
```

The overview script remains a compact replot from saved scores. The complete
`battery_capacity.figures` command regenerates all five numerical figures
from freshly computed analysis tables. Exact manuscript figure assets remain
in `figures/`; regenerated layouts need not be pixel-identical to those assets.
The package does not include raw MAT files, pretrained models,
manuscript drafts, unrelated experiment screens, or private development logs.
