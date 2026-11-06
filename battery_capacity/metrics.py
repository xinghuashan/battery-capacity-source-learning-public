"""Per-cell errors and equal-cell macro averages in percentage points."""
import numpy as np


def per_cell(frame, primary=True):
    selected = frame[frame['primary']] if primary else frame
    result = []
    for cell, rows in selected.groupby('cell', sort=True):
        error = 100 * (rows['prediction'].to_numpy() - rows['truth'].to_numpy())
        if not np.isfinite(error).all():
            raise ValueError(f'Nonfinite prediction or label for {cell}')
        result.append(dict(cell=cell, n=len(error), rmse_pp=float(np.sqrt(np.mean(error ** 2))),
                           mae_pp=float(np.mean(np.abs(error)))))
    return result


def macro(rows):
    return dict(rmse_pp=float(np.mean([r['rmse_pp'] for r in rows])),
                mae_pp=float(np.mean([r['mae_pp'] for r in rows])))
