"""Check saved cell -> seed -> paper summaries; no data loading or model fitting."""
from collections import defaultdict
import csv
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
CELLS = {f'RW_battery-{i}.mat' for i in (1, 2, 3, 5, 6, 7)}


def read(name):
    with (ROOT / 'results' / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def check_close(actual, expected, identity):
    assert math.isfinite(float(actual)), identity
    assert math.isfinite(float(expected)), identity
    assert math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12), (
        identity, actual, expected)


def main():
    cells = read('cell_metrics.csv')
    seeds = read('seed_metrics.csv')
    aggregate = read('performance.csv')
    by_run = defaultdict(list)
    for row in cells:
        key = tuple(row[k] for k in ('method', 'stride', 'seed', 'scope'))
        by_run[key].append(row)
    seed_index = {tuple(row[k] for k in ('method', 'stride', 'seed', 'scope')): row for row in seeds}
    assert len(seed_index) == len(seeds), 'Duplicate seed summaries'
    assert set(seed_index) == set(by_run), 'Run coverage mismatch'
    for key, rows in by_run.items():
        assert len(rows) == 6 and {r['cell'] for r in rows} == CELLS, key
        expected_n = 109 if key[-1] == 'primary' else 127
        assert sum(int(row['n']) for row in rows) == int(seed_index[key]['n']) == expected_n, key
        for metric in ('rmse_pp', 'mae_pp'):
            check_close(statistics.mean(float(row[metric]) for row in rows),
                        seed_index[key][metric], (key, metric))
    by_condition = defaultdict(list)
    for row in seeds:
        key = tuple(row[k] for k in ('method', 'stride', 'scope'))
        by_condition[key].append(row)
    aggregate_index = {tuple(row[k] for k in ('method', 'stride', 'scope')): row for row in aggregate}
    assert len(aggregate_index) == len(aggregate), 'Duplicate aggregate summaries'
    assert set(aggregate_index) == set(by_condition), 'Condition coverage mismatch'
    for key, rows in by_condition.items():
        saved = aggregate_index[key]
        assert len(rows) == int(saved['n_seeds']), key
        assert {row['seed'] for row in rows} in ({'det'}, {'0', '1', '2', '3', '4'}), key
        # Some legacy aggregate rows omit this field; counts were checked above
        # against every corresponding cell-level and seed-level row.
        if saved['n_events']:
            assert float(saved['n_events']) == (109 if key[-1] == 'primary' else 127), key
        for metric in ('rmse', 'mae'):
            values = [float(row[metric + '_pp']) for row in rows]
            check_close(statistics.mean(values), saved[metric + '_mean_pp'], (key, metric, 'mean'))
            check_close(statistics.pstdev(values), saved[metric + '_std_ddof0_pp'], (key, metric, 'std'))
    print(f'PASS: {len(cells)} cell rows, {len(seeds)} run summaries, {len(aggregate)} paper summaries.')
    print('Only saved numeric summaries were checked. No model was loaded or evaluated.')


if __name__ == '__main__':
    main()
