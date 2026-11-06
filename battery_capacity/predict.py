"""Evaluate frozen source models on the prescribed target pair without adaptation."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import FEATURES, TARGET_PAIRS, read_cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.models / 'training_manifest.json').read_text(encoding='utf-8'))
    assert manifest['completed']
    fold = manifest['fold']
    assert manifest['target_cells'] == list(TARGET_PAIRS[fold])
    args.output.mkdir(parents=True, exist_ok=False)
    for path in sorted(args.models.glob('*.joblib')):
        method, seed = path.stem.rsplit('_seed', 1)
        predictor = joblib.load(path)
        frames = []
        for cell in TARGET_PAIRS[fold]:
            data = read_cell(args.inputs, cell)
            prediction = predictor.predict(data[FEATURES].to_numpy(dtype=float))
            assert np.isfinite(prediction).all()
            frame = data[['event', 'reference_raw_record', 'feature_raw_record']].copy()
            frame['cycle'] = data['routine_cycle']
            frame['cell'] = f'RW_battery-{cell}.mat'
            frame['truth'] = data['capacity_ratio']
            frame['prediction'] = prediction
            frame['eligible_rank'] = np.arange(1, len(frame) + 1)
            frame['primary'] = frame['eligible_rank'] >= 4
            frame['method'], frame['seed'] = method, int(seed)
            frame['fold'], frame['stride'] = fold, manifest['stride']
            frames.append(frame)
        pd.concat(frames, ignore_index=True).to_csv(args.output / f'{path.stem}.csv', index=False)


if __name__ == '__main__':
    main()
