"""Aggregate saved prediction CSVs across the three prescribed folds."""
import argparse
from pathlib import Path
import pandas as pd
from .data import CELLS
from .metrics import per_cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, nargs='+', required=True,
                        help='Directories written by battery_capacity.predict for all three folds.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(path for directory in args.predictions for path in directory.glob('*.csv'))
    if not paths:
        raise ValueError('No prediction CSV files found.')
    data = pd.concat([pd.read_csv(path, float_precision='round_trip') for path in paths], ignore_index=True)
    assert not data.duplicated(['method', 'stride', 'seed', 'cell', 'event']).any()
    rows = []
    for (method, stride, seed), frame in data.groupby(['method', 'stride', 'seed'], sort=True):
        assert set(frame['fold']) == {0, 1, 2}, 'All three folds are required.'
        assert set(frame['cell']) == {f'RW_battery-{cell}.mat' for cell in CELLS}
        assert len(frame) == 127 and int(frame['primary'].sum()) == 109
        for scope in ('primary', 'all'):
            for row in per_cell(frame, primary=scope == 'primary'):
                rows.append(dict(method=method, stride=stride, seed=seed, scope=scope, **row))
    cells = pd.DataFrame(rows)
    seeds = cells.groupby(['method', 'stride', 'seed', 'scope'], as_index=False)[['rmse_pp', 'mae_pp']].mean()
    summary = []
    for (method, stride, scope), group in seeds.groupby(['method', 'stride', 'scope'], sort=True):
        row = dict(method=method, stride=stride, scope=scope, n_seeds=len(group))
        for metric in ('rmse', 'mae'):
            values = group[metric + '_pp']
            row[metric + '_mean_pp'] = values.mean()
            row[metric + '_std_ddof0_pp'] = values.std(ddof=0)
        summary.append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    cells.to_csv(args.output / 'cell_metrics.csv', index=False)
    seeds.to_csv(args.output / 'seed_metrics.csv', index=False)
    pd.DataFrame(summary).to_csv(args.output / 'performance.csv', index=False)


if __name__ == '__main__':
    main()
