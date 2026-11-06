"""Recompute paper tables and diagnostics from a complete saved prediction matrix."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import CELLS, TARGET_PAIRS

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('Ridge', 'Ridge-CV', 'KRR', 'KRR-wide', 'B3', 'N01', 'M01',
           'M02', 'M03', 'M04', 'M05', 'M06', 'Raw-CV')
DETERMINISTIC = {'Ridge', 'Ridge-CV', 'KRR', 'KRR-wide', 'M01'}
SCALE_METHODS = ('B3', 'N01', 'M03', 'M04', 'M05', 'M06')
IDENTITY = ['cell', 'event', 'reference_raw_record', 'feature_raw_record', 'cycle', 'eligible_rank']


def read_predictions(paths):
    files = []
    for path in map(Path, paths):
        files.extend(sorted(path.glob('*.csv')) if path.is_dir() else [path])
    if not files:
        raise ValueError('No prediction files found.')
    data = pd.concat([pd.read_csv(p, float_precision='round_trip') for p in files], ignore_index=True)
    needed = set(IDENTITY + ['method', 'stride', 'seed', 'fold', 'primary', 'truth', 'prediction'])
    if not needed.issubset(data):
        raise ValueError(f'Missing prediction columns: {needed - set(data)}')
    data['method'] = data.method.replace({'M05-raw-CV': 'Raw-CV'})
    for col in ('stride', 'seed', 'fold', 'event', 'reference_raw_record', 'feature_raw_record', 'eligible_rank'):
        values = pd.to_numeric(data[col], errors='raise')
        if not np.isfinite(values).all() or not (values == np.floor(values)).all():
            raise ValueError(f'Invalid integer field: {col}')
        data[col] = values.astype(int)
    if data.primary.dtype != bool:
        if not set(data.primary.astype(str).str.lower()) <= {'true', 'false'}:
            raise ValueError('primary must contain true/false flags.')
        data['primary'] = data.primary.astype(str).str.lower().eq('true')
    if not np.isfinite(data[['truth', 'prediction', 'cycle']].to_numpy()).all():
        raise ValueError('Nonfinite prediction, label, or cycle.')
    if data.duplicated(['method', 'stride', 'seed', 'cell', 'event']).any():
        raise ValueError('Duplicate prediction identities.')
    expected = {(m, b, s) for m in METHODS for b in (1, 10)
                for s in ([0] if m in DETERMINISTIC else range(5))}
    actual = set(data.groupby(['method', 'stride', 'seed']).groups)
    if actual != expected:
        raise ValueError(f'Incomplete paper matrix; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}')
    reference = None
    for key, frame in data.groupby(['method', 'stride', 'seed'], sort=True):
        frame = frame.sort_values(['cell', 'event'])
        if len(frame) != 127 or frame.primary.sum() != 109 or set(frame.cell) != {f'RW_battery-{c}.mat' for c in CELLS}:
            raise ValueError(f'Wrong event/cell coverage: {key}')
        if not np.array_equal(frame.primary.to_numpy(), (frame.eligible_rank >= 4).to_numpy()):
            raise ValueError(f'Primary-mask mismatch: {key}')
        for cell, part in frame.groupby('cell', sort=True):
            number = int(cell.split('-')[1].split('.')[0])
            fold = next(i for i, pair in enumerate(TARGET_PAIRS) if number in pair)
            if set(part.fold) != {fold} or not np.array_equal(part.eligible_rank, np.arange(1, len(part) + 1)):
                raise ValueError(f'Fold/rank mismatch: {key}, {cell}')
        if not (frame.feature_raw_record < frame.reference_raw_record).all():
            raise ValueError('Input must precede its reference record.')
        current = frame[IDENTITY + ['truth']].reset_index(drop=True)
        if reference is None:
            reference = current
        elif not current[IDENTITY].equals(reference[IDENTITY]) or not np.allclose(current.truth, reference.truth, atol=1e-14, rtol=0):
            raise ValueError(f'Event or label alignment mismatch: {key}')
    return data.sort_values(['method', 'stride', 'seed', 'cell', 'event']).reset_index(drop=True)


def cell_errors(frame):
    rows = []
    for (method, stride, seed), run in frame.groupby(['method', 'stride', 'seed'], sort=True):
        for scope in ('primary', 'all'):
            selected = run[run.primary] if scope == 'primary' else run
            for cell, part in selected.groupby('cell', sort=True):
                error = 100 * (part.prediction.to_numpy() - part.truth.to_numpy())
                rows.append(dict(method=method, stride=stride, seed=seed, scope=scope, cell=cell,
                                 n=len(part), rmse_pp=float(np.sqrt(np.mean(error ** 2))),
                                 mae_pp=float(np.mean(np.abs(error)))))
    return pd.DataFrame(rows)


def summarize(cells):
    seeds = cells.groupby(['method', 'stride', 'seed', 'scope'], as_index=False).agg(
        n=('n', 'sum'), physical_cells=('cell', 'nunique'), rmse_pp=('rmse_pp', 'mean'), mae_pp=('mae_pp', 'mean'))
    rows = []
    for (method, stride, scope), group in seeds.groupby(['method', 'stride', 'scope'], sort=True):
        row = dict(method=method, stride=stride, scope=scope, n_seeds=len(group),
                   deterministic=method in DETERMINISTIC, n_events=int(group.n.iloc[0]))
        for metric in ('rmse', 'mae'):
            values = group[metric + '_pp'].to_numpy()
            row[metric + '_mean_pp'] = float(np.mean(values))
            row[metric + '_std_ddof0_pp'] = float(np.std(values, ddof=0))
        rows.append(row)
    return seeds, pd.DataFrame(rows)


def paired_analysis(cells):
    index = cells[cells.scope == 'primary'].set_index(['method', 'stride', 'seed', 'cell'])
    names = [f'RW_battery-{c}.mat' for c in CELLS]
    def score(method, stride, seed, cell, metric='rmse_pp'):
        return float(index.loc[(method, stride, 0 if method in DETERMINISTIC else seed, cell), metric])
    paired, interactions, deletions = [], [], []
    for baseline in (m for m in METHODS if m != 'M05'):
        for stride in (1, 10):
            effects = {}
            for seed in range(5):
                for cell in names:
                    delta = score(baseline, stride, seed, cell) - score('M05', stride, seed, cell)
                    effects[seed, cell] = delta
                    paired.append(dict(baseline=baseline, stride=stride, seed=seed, cell=cell,
                                       rmse_effect_pp=delta,
                                       mae_effect_pp=score(baseline, stride, seed, cell, 'mae_pp') - score('M05', stride, seed, cell, 'mae_pp')))
            subsets = [('All six', [])] + [(f'Drop RW{c}', [f'RW_battery-{c}.mat']) for c in CELLS]
            subsets += [(f'Drop RW{a}+{b}', [f'RW_battery-{a}.mat', f'RW_battery-{b}.mat']) for a, b in TARGET_PAIRS]
            for label, removed in subsets:
                for seed in range(5):
                    deletions.append(dict(baseline=baseline, stride=stride, subset=label, seed=seed,
                                          rmse_effect_pp=float(np.mean([effects[seed, cell] for cell in names if cell not in removed]))))
        for seed in range(5):
            for cell in names:
                rf, rs = score(baseline, 1, seed, cell), score(baseline, 10, seed, cell)
                mf, ms = score('M05', 1, seed, cell), score('M05', 10, seed, cell)
                value = np.nan if min(rf, rs, mf, ms) == 0 else float(np.log(rs / ms) - np.log(rf / mf))
                interactions.append(dict(baseline=baseline, seed=seed, cell=cell, log_budget_interaction=value))
    return pd.DataFrame(paired), pd.DataFrame(interactions), pd.DataFrame(deletions)


def correction_analysis(data):
    anchor = data[data.method == 'Ridge'][['stride'] + IDENTITY + ['truth', 'prediction']].rename(columns={'prediction': 'anchor'})
    direction, amplitude = [], []
    for (method, stride, seed), frame in data[data.method.isin(SCALE_METHODS)].groupby(['method', 'stride', 'seed'], sort=True):
        merged = frame.merge(anchor.drop(columns='truth'), on=['stride'] + IDENTITY, validate='many_to_one')
        if len(merged) != len(frame):
            raise ValueError('Missing Ridge anchor identities.')
        for cell, part in merged[merged.primary].groupby('cell', sort=True):
            r = 100 * (part.truth.to_numpy() - part.anchor.to_numpy())
            delta = 100 * (part.prediction.to_numpy() - part.anchor.to_numpy())
            alignment, energy = float(np.mean(r * delta)), float(np.mean(delta * delta))
            gain = 2 * alignment - energy
            direct = float(np.mean(r * r) - np.mean((r - delta) ** 2))
            if abs(gain - direct) > 1e-10:
                raise ValueError('Residual decomposition identity failed.')
            direction.append(dict(method=method, stride=stride, seed=seed, cell=cell,
                                  inner_product_r_delta_pp2=alignment, correction_energy_pp2=energy,
                                  gain_pp2=gain, direction_nonpositive=alignment <= 0,
                                  identity_error_pp2=abs(gain - direct)))
        for scale in (0., .5, 1.):
            for scope in ('primary', 'all'):
                selected = merged[merged.primary] if scope == 'primary' else merged
                rmse, mae = [], []
                for _, part in selected.groupby('cell', sort=True):
                    prediction = part.anchor.to_numpy() + scale * (part.prediction.to_numpy() - part.anchor.to_numpy())
                    error = 100 * (prediction - part.truth.to_numpy())
                    rmse.append(np.sqrt(np.mean(error ** 2)))
                    mae.append(np.mean(np.abs(error)))
                amplitude.append(dict(method=method, stride=stride, seed=seed, scope=scope, scale=scale,
                                      rmse_pp=float(np.mean(rmse)), mae_pp=float(np.mean(mae))))
    direction = pd.DataFrame(direction)
    table4 = direction[direction.method.isin(['M05', 'M06'])].groupby(['method', 'stride'], as_index=False).agg(
        n_cell_seed=('cell', 'size'), nonpositive=('direction_nonpositive', 'sum'),
        alignment_pp2=('inner_product_r_delta_pp2', 'mean'), energy_pp2=('correction_energy_pp2', 'mean'), gain_pp2=('gain_pp2', 'mean'))
    return direction, pd.DataFrame(amplitude), table4


def analyze(data, output):
    cells = cell_errors(data)
    seeds, performance = summarize(cells)
    paired, interactions, deletions = paired_analysis(cells)
    direction, amplitude, table4 = correction_analysis(data)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tables = dict(cell_metrics=cells, seed_metrics=seeds, performance=performance,
                  paired_cell_effects=paired, budget_interactions=interactions,
                  deletion_sensitivity=deletions, direction_energy=direction,
                  amplitude_diagnostics=amplitude, table4=table4)
    for name, frame in tables.items():
        frame.to_csv(output / (name + '.csv'), index=False, float_format='%.17g')
    # Preserve every seed and event; figures never use an ensemble predictor.
    event = data[(data.stride == 10) & data.primary & data.method.isin(['M05', 'Ridge'])]
    event.to_csv(output / 'event_predictions.csv', index=False, float_format='%.17g')
    lines = ['# Table 3. Primary cell-macro errors (pp)', '', '| Method | Full RMSE | Full MAE | Stride10 RMSE | Stride10 MAE |', '|---|---:|---:|---:|---:|']
    for method in METHODS:
        values = []
        for stride in (1, 10):
            row = performance[(performance.method == method) & (performance.stride == stride) & (performance.scope == 'primary')].iloc[0]
            for metric in ('rmse', 'mae'):
                value = f'{row[metric + "_mean_pp"]:.3f}'
                if method not in DETERMINISTIC:
                    value += f' ± {row[metric + "_std_ddof0_pp"]:.3f}'
                values.append(value)
        lines.append('| ' + ' | '.join([method] + values) + ' |')
    (output / 'table3.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    lines = ['# Table 4. Correction diagnostics', '',
             '| Method/budget | Nonpositive / 30 | Alignment (pp²) | Energy (pp²) | MSE gain (pp²) |',
             '|---|---:|---:|---:|---:|']
    for row in table4.to_dict('records'):
        budget = 'full' if row['stride'] == 1 else 'stride10'
        lines.append(f"| {row['method']}/{budget} | {int(row['nonpositive'])} | {row['alignment_pp2']:.3f} | {row['energy_pp2']:.3f} | {row['gain_pp2']:.3f} |")
    (output / 'table4.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    report = dict(prediction_rows=len(data), methods=len(METHODS), complete_matrix=True,
                  primary_events_per_run=109, all_events_per_run=127, seeds=list(range(5)),
                  model_calls=0, fitting_operations=0, diagnostics_descriptive_only=True,
                  undefined_log_interactions=int(interactions.log_budget_interaction.isna().sum()))
    (output / 'analysis_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, nargs='+', default=[ROOT / 'results/paper_predictions.csv.gz'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(read_predictions(args.predictions), args.output)
    print(args.output)


if __name__ == '__main__':
    main()
