"""Run saved-result reproduction or explicitly requested raw-data training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .analysis import ROOT, METHODS, analyze, read_predictions
from .figures import render
from .verify import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('saved', 'train'), default='saved')
    parser.add_argument('--raw-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'train' and args.raw_dir is None:
        parser.error('--mode train requires --raw-dir')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = dict(mode=args.mode, completed=False, training_requested=args.mode == 'train',
                  completed_commands=[], checkpoint='initialized')
    status_path = output / 'run_status.json'
    def save():
        status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
    def command(module, *arguments):
        argv = [sys.executable, '-B', '-m', 'battery_capacity.' + module, *map(str, arguments)]
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUTF8='1')
        save()
        subprocess.run(argv, cwd=ROOT, env=env, check=True)
        status['completed_commands'].append([module, *map(str, arguments)])
        save()
    save()
    try:
        if args.mode == 'saved':
            files = [ROOT / 'results/paper_predictions.csv.gz']
        else:
            inputs = output / 'inputs'
            command('prepare_data', '--raw-dir', args.raw_dir.resolve(), '--output', inputs)
            files = []
            # All six source jobs finish before any target model evaluation.
            for stride in (1, 10):
                for fold in range(3):
                    models = output / f'fold{fold}_stride{stride}' / 'models'
                    command('train', '--inputs', inputs, '--fold', fold, '--stride', stride,
                            '--methods', *METHODS, '--output', models)
            for stride in (1, 10):
                for fold in range(3):
                    directory = output / f'fold{fold}_stride{stride}'
                    predictions = directory / 'predictions'
                    command('predict', '--inputs', inputs, '--models', directory / 'models', '--output', predictions)
                    files.append(predictions)
        status['checkpoint'] = 'predictions_available'; save()
        analyze(read_predictions(files), output / 'analysis')
        status['checkpoint'] = 'analysis_complete'; save()
        # This strict comparison reports deviations; it never changes fitting choices.
        try:
            verification = verify(output / 'analysis')
            status['matches_archived_results'] = verification['passed']
        except AssertionError as error:
            status['matches_archived_results'] = False
            status['numerical_difference'] = str(error)
            if args.mode == 'saved':
                raise
        render(output / 'analysis', output / 'figures')
        status['checkpoint'] = 'figures_complete'
        status['completed'] = True
        save()
        print(json.dumps(status, indent=2))
        if not status['matches_archived_results']:
            raise SystemExit('Run completed but differs from archived results; inspect run_status.json. No protocol was changed.')
    except Exception as error:
        status['error'] = str(error)
        save()
        raise


if __name__ == '__main__':
    main()
