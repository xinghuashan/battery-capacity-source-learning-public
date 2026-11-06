"""Build the fixed development inputs from the six named original MAT files."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from .data import CELLS, FEATURES
from .features import DROP, HISTORY_INDEX, history_for
from .preprocessing import audit_cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.raw_dir / f'RW_battery-{cell}.mat' for cell in CELLS]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=False)
    audit = args.output / 'record_checks'
    audit.mkdir()
    counts = []
    for path in paths:
        cell, report = audit_cell(path, audit)
        if not report['scope_admitted']:
            raise ValueError(f'Record or phase check failed for {path.name}; inspect record_checks.')
        rows = loadmat(path, simplify_cells=True)['data']
        histories = np.array([history_for(rows, int(record), 5)[0]
                              for record in cell['feature_raw_record']])
        x = np.column_stack((cell['x'][:, DROP], cell['cycle'], histories[:, HISTORY_INDEX]))
        assert x.shape[1] == 17 and np.isfinite(x).all() and np.isfinite(cell['q']).all()
        frame = pd.DataFrame(x, columns=FEATURES)
        for key in ('event', 'reference_raw_record', 'feature_raw_record'):
            frame[key] = cell[key]
        frame['capacity_ratio'] = cell['q']
        frame.to_csv(args.output / f'{path.stem}.csv', index=False, float_format='%.17g')
        counts.append(len(frame))
    assert sum(counts) == 127, ('Unexpected eligible-event count', counts)
    print(f'Wrote six feature tables with {sum(counts)} eligible events.')


if __name__ == '__main__':
    main()
