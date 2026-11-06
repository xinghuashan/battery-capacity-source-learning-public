"""Compare recomputed results with archived paper values at full precision."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .analysis import ROOT


def canonical(frame):
    frame = frame.copy()
    if 'method' in frame:
        frame['method'] = frame.method.replace({'M05-raw-CV': 'Raw-CV'})
    if 'seed' in frame:
        frame['seed'] = frame.seed.replace({'det': '0'}).astype(int)
    return frame


def verify(output, expected=ROOT / 'results', tolerance=1e-10):
    output, expected = Path(output), Path(expected)
    checks = []
    def compare(actual, reference, keys, columns, name):
        actual, reference = canonical(actual), canonical(reference)
        merged = reference.merge(actual, on=keys, suffixes=('_expected', '_actual'), validate='one_to_one')
        if len(merged) != len(reference):
            raise AssertionError(f'{name}: missing reference identities')
        largest = 0.
        for column in columns:
            a = merged[column + '_actual'].to_numpy(dtype=float)
            b = merged[column + '_expected'].to_numpy(dtype=float)
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise AssertionError(f'{name}: nonfinite values')
            error = float(np.max(np.abs(a - b)))
            largest = max(largest, error)
            if error > tolerance:
                raise AssertionError(f'{name}.{column}: max absolute difference {error} exceeds {tolerance}')
        checks.append(dict(name=name, matched_rows=len(merged), maximum_absolute_error=largest))
    def read(folder, name):
        return pd.read_csv(folder / name, float_precision='round_trip')
    compare(read(output, 'cell_metrics.csv'), read(expected, 'cell_metrics.csv'),
            ['method', 'stride', 'seed', 'scope', 'cell'], ['n', 'rmse_pp', 'mae_pp'], 'cell_metrics')
    compare(read(output, 'seed_metrics.csv'), read(expected, 'seed_metrics.csv'),
            ['method', 'stride', 'seed', 'scope'], ['n', 'rmse_pp', 'mae_pp'], 'seed_metrics')
    compare(read(output, 'performance.csv'), read(expected, 'performance.csv'),
            ['method', 'stride', 'scope'], ['n_seeds', 'rmse_mean_pp', 'mae_mean_pp', 'rmse_std_ddof0_pp', 'mae_std_ddof0_pp'], 'performance')
    compare(read(output, 'paired_cell_effects.csv'), read(expected, 'paired_cell_effects.csv'),
            ['baseline', 'stride', 'seed', 'cell'], ['rmse_effect_pp', 'mae_effect_pp'], 'paired_effects')
    compare(read(output, 'direction_energy.csv'), read(expected, 'expected_direction_energy.csv'),
            ['method', 'stride', 'seed', 'cell'], ['inner_product_r_delta_pp2', 'correction_energy_pp2', 'gain_pp2'], 'direction_energy')
    compare(read(output, 'table4.csv'), read(expected, 'expected_table4.csv'),
            ['method', 'stride'], ['n_cell_seed', 'nonpositive', 'alignment_pp2', 'energy_pp2', 'gain_pp2'], 'table4')
    compare(read(output, 'amplitude_diagnostics.csv').query("scope == 'primary'"), read(expected, 'saved_amplitude_display.csv'),
            ['method', 'stride', 'seed', 'scale'], ['rmse_pp'], 'amplitudes')
    interaction_rows = []
    for item in json.loads((expected / 'budget_interactions.json').read_text(encoding='utf-8')):
        for row in item['cell_seed_values']:
            interaction_rows.append(dict(baseline=item['baseline'], **row))
    compare(read(output, 'budget_interactions.csv'), pd.DataFrame(interaction_rows),
            ['baseline', 'seed', 'cell'], ['log_budget_interaction'], 'budget_interactions')
    deletion_rows = []
    for item in json.loads((expected / 'concentration_sensitivity.json').read_text(encoding='utf-8')):
        values = [('All six', float(np.mean(list(item['cell_mean_effects'].values()))))]
        values += [(f"Drop RW{key.split('-')[1].split('.')[0]}", val) for key, val in item['delete_single_cell'].items()]
        values += [('Drop RW' + '+'.join(part.split('-')[1].split('.')[0] for part in key.split('+')), val)
                   for key, val in item['delete_target_pair'].items()]
        deletion_rows += [dict(baseline=item['baseline'], stride=item['stride'], subset=label, rmse_effect_pp=value) for label, value in values]
    actual = read(output, 'deletion_sensitivity.csv').groupby(['baseline', 'stride', 'subset'], as_index=False).rmse_effect_pp.mean()
    compare(actual, pd.DataFrame(deletion_rows), ['baseline', 'stride', 'subset'], ['rmse_effect_pp'], 'deletions')
    report = dict(passed=True, absolute_tolerance=tolerance, checks=checks, model_calls=0,
                  maximum_absolute_error=max(row['maximum_absolute_error'] for row in checks),
                  scope='saved-prediction analysis; not a retraining-equivalence claim')
    (output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.analysis), indent=2))


if __name__ == '__main__':
    main()
