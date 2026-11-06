"""Fit a specified source fold without loading its target-cell tables."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from .data import CELLS, TARGET_PAIRS, read_source
from .episodes import build_bundle
from .models import DETERMINISTIC, METHODS, fit_method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--fold', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--stride', type=int, choices=(1, 10), required=True)
    parser.add_argument('--methods', nargs='+', choices=METHODS, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, choices=range(5), default=list(range(5)))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = read_source(args.inputs, args.fold, args.stride)
    expected = (88, 83, 83)[args.fold] if args.stride == 1 else 6
    assert int(source['chosen'].sum()) == expected, 'Source label count differs from the paper.'
    manifest = dict(fold=args.fold, stride=args.stride,
                    source_cells=[c for c in CELLS if c not in TARGET_PAIRS[args.fold]],
                    target_cells=list(TARGET_PAIRS[args.fold]),
                    source_inputs=len(source['x']), source_labels=int(source['chosen'].sum()),
                    source_input_sha256=hashlib.sha256(source['x'].tobytes()).hexdigest(),
                    source_label_mask_sha256=hashlib.sha256(source['chosen'].tobytes()).hexdigest(),
                    methods=list(dict.fromkeys(args.methods)), seeds=sorted(set(args.seeds)),
                    optimizer_steps=300, dtype='float64', device='CPU', threads=1,
                    target_tables_read=False, completed=False)
    manifest_path = args.output / 'training_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    bundle = build_bundle(source)
    selection_cache = {}
    for method in manifest['methods']:
        seeds = [0] if method in DETERMINISTIC else manifest['seeds']
        for seed in seeds:
            predictor = fit_method(bundle, method, seed, selection_cache)
            joblib.dump(predictor, args.output / f'{method}_seed{seed}.joblib')
            if predictor.selection is not None:
                (args.output / f'{method}_selection.json').write_text(
                    json.dumps(predictor.selection, indent=2), encoding='utf-8')
            print(f'{method}, fold={args.fold}, stride={args.stride}, seed={seed}: saved', flush=True)
    manifest['completed'] = True
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
