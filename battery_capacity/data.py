"""The fixed 17-input, six-cell development contract."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

CELLS = (1, 2, 3, 5, 6, 7)
TARGET_PAIRS = ((1, 5), (2, 6), (3, 7))
CONFIG = Path(__file__).resolve().parents[1] / 'config/feature_schema.json'
FEATURES = [entry['name'] for entry in json.loads(CONFIG.read_text(encoding='utf-8'))['columns']]


def read_cell(directory, cell):
    if cell not in CELLS:
        raise ValueError('Only the six development cells are part of this release.')
    frame = pd.read_csv(Path(directory) / f'RW_battery-{cell}.csv', float_precision='round_trip')
    required = FEATURES + ['capacity_ratio', 'event', 'reference_raw_record', 'feature_raw_record']
    if not set(required).issubset(frame.columns):
        raise ValueError(f'Missing columns for RW{cell}: {set(required) - set(frame.columns)}')
    assert frame['reference_raw_record'].is_monotonic_increasing
    assert frame['reference_raw_record'].is_unique and frame['event'].is_unique
    assert (frame['feature_raw_record'] < frame['reference_raw_record']).all()
    assert np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all()
    return frame


def read_source(directory, fold, stride):
    if fold not in (0, 1, 2) or stride not in (1, 10):
        raise ValueError((fold, stride))
    ids = [cell for cell in CELLS if cell not in TARGET_PAIRS[fold]]
    frames = [read_cell(directory, cell) for cell in ids]
    frame = pd.concat(frames, ignore_index=True)
    cell_ids = np.concatenate([np.repeat(cell, len(part)) for cell, part in zip(ids, frames)])
    chosen = frame['event'].to_numpy(dtype=int) % stride == 0
    approved_y = frame.loc[chosen, 'capacity_ratio'].to_numpy(dtype=float)
    assert np.isfinite(approved_y).all()
    return dict(x=frame[FEATURES].to_numpy(dtype=float), ids=cell_ids, chosen=chosen,
                approved_y=approved_y, event=frame['event'].to_numpy(dtype=int),
                reference_raw_record=frame['reference_raw_record'].to_numpy(dtype=int))
