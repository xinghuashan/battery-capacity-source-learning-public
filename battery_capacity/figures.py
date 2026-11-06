"""Render Figures 2-6 from recomputed analysis tables, without model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from .data import CELLS, TARGET_PAIRS

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#A07A00']
MARKERS = ['o', 's', '^', 'v', 'P']
METHODS = ['Ridge', 'Ridge-CV', 'KRR', 'KRR-wide', 'B3', 'N01', 'M01', 'M02', 'M03', 'M04', 'M06', 'Raw-CV', 'M05']


def render(analysis, output):
    analysis, output = Path(analysis), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.titlesize': 10, 'axes.labelsize': 9, 'svg.fonttype': 'none'})
    consumed = {}
    def read(name):
        path = analysis / (name + '.csv')
        consumed[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path, float_precision='round_trip')
    def save(fig, name):
        for extension in ('png', 'pdf', 'svg'):
            fig.savefig(output / (name + '.' + extension), dpi=300, bbox_inches='tight', pad_inches=.06)
        plt.close(fig)
    handles = [Line2D([0], [0], ls='none', marker=MARKERS[s], color=COLORS[s], label=f'Seed {s}') for s in range(5)]
    handles += [Line2D([0], [0], ls='none', marker='D', color='#182D44', label='Mean')]
    scores = read('seed_metrics').query("scope == 'primary'")
    fig, axes = plt.subplots(2, 2, figsize=(7.48, 6.5), layout='constrained')
    for row, stride in enumerate((1, 10)):
        for col, metric in enumerate(('rmse_pp', 'mae_pp')):
            ax = axes[row, col]
            ax.axhspan(11.5, 12.5, color='#EEF5FA')
            for j, method in enumerate(METHODS):
                values = scores[(scores.method == method) & (scores.stride == stride)]
                if len(values) == 1:
                    ax.scatter(values.iloc[0][metric], j, s=20, facecolor='white', edgecolor='#182D44')
                else:
                    if len(values) != 5:
                        raise ValueError('Incomplete seed coverage for Figure 2.')
                    for entry in values.to_dict('records'):
                        seed = int(entry['seed'])
                        ax.scatter(entry[metric], j + (seed - 2) * .075, s=15, color=COLORS[seed], marker=MARKERS[seed])
                    ax.scatter(values[metric].mean(), j, s=22, marker='D', color='#182D44')
            ax.set(yticks=range(13), yticklabels=METHODS, ylim=(12.6, -.6), xlabel=metric.split('_')[0].upper() + ' (pp)')
            ax.set_title(f"{'abcd'[row * 2 + col]}  {'Full' if stride == 1 else 'Stride10'}", loc='left')
            ax.grid(axis='x', alpha=.18)
    fig.legend(handles=handles + [Line2D([0], [0], ls='none', marker='o', mfc='white', mec='#182D44', label='Deterministic')],
               loc='outside upper center', ncol=7, fontsize=7.5, frameon=False)
    save(fig, 'figure_02_overall_performance')

    event = read('event_predictions')
    fig, axes = plt.subplots(3, 2, figsize=(7.48, 6.5), layout='constrained')
    for panel, (cell, ax) in enumerate(zip(CELLS, axes.flat)):
        part = event[event.cell == f'RW_battery-{cell}.mat']
        ridge = part[part.method == 'Ridge'].sort_values('cycle')
        ax.plot(ridge.cycle, ridge.truth, 'ko-', ms=2.5, lw=.8, label='Measured reference')
        ax.plot(ridge.cycle, ridge.prediction, '--', color='#0072B2', lw=1.1, label='Ridge')
        for seed in range(5):
            values = part[(part.method == 'M05') & (part.seed == seed)].sort_values('cycle')
            ax.plot(values.cycle, values.prediction, color='#D55E00', alpha=.55, lw=.8, label='M05 (five seeds)' if seed == 0 else None)
        ax.set_title(f"{'abcdef'[panel]}  RW{cell}", loc='left')
        ax.set(xlabel='Input ordinary-cycle index', ylabel='Next-reference capacity / 2 Ah')
        ax.grid(alpha=.15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside upper center', ncol=3, frameon=False, fontsize=8)
    save(fig, 'figure_03_event_predictions')

    paired = read('paired_cell_effects')
    selected = paired[paired.baseline.isin(['M03', 'M06'])]
    bound = float(np.ceil(selected.rmse_effect_pp.abs().max() * 20) / 20 + .05)
    fig, axes = plt.subplots(2, 2, figsize=(7.48, 5.4), layout='constrained', sharex=True, sharey=True)
    for row, stride in enumerate((1, 10)):
        for col, baseline in enumerate(('M03', 'M06')):
            ax = axes[row, col]
            ax.axvspan(-bound, 0, color='#FCF0ED'); ax.axvspan(0, bound, color='#EEF5FA')
            for j, cell in enumerate(CELLS):
                values = paired[(paired.baseline == baseline) & (paired.stride == stride) & (paired.cell == f'RW_battery-{cell}.mat')]
                for entry in values.to_dict('records'):
                    seed = int(entry['seed'])
                    ax.scatter(entry['rmse_effect_pp'], j + (seed - 2) * .075, s=18, color=COLORS[seed], marker=MARKERS[seed])
                ax.scatter(values.rmse_effect_pp.mean(), j, s=25, marker='D', color='#182D44')
            ax.axvline(0, color='#566574', lw=.7)
            ax.set(yticks=range(6), yticklabels=[f'RW{c}' for c in CELLS], ylim=(5.6, -.6), xlim=(-bound, bound))
            ax.set_title(f"{'abcd'[row * 2 + col]}  {'Full' if stride == 1 else 'Stride10'}: {baseline} − M05", loc='left')
            ax.grid(axis='x', alpha=.15)
            if row == 1:
                ax.set_xlabel('RMSE difference (pp); positive favors M05')
    fig.legend(handles=handles, loc='outside upper center', ncol=6, frameon=False, fontsize=8)
    save(fig, 'figure_04_cell_effects')

    interaction, deletion = read('budget_interactions'), read('deletion_sensitivity')
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 4.8), layout='constrained', gridspec_kw={'width_ratios': [.8, 1.3, 1.3]})
    for j, baseline in enumerate(('M03', 'M06')):
        values = interaction[interaction.baseline == baseline]
        if values.log_budget_interaction.isna().any():
            raise ValueError('Zero-RMSE log interaction is undefined; cannot render it as a number.')
        for seed, value in values.groupby('seed').log_budget_interaction.mean().items():
            axes[0].scatter(j + (seed - 2) * .075, value, color=COLORS[int(seed)], marker=MARKERS[int(seed)], s=25)
        axes[0].scatter(j, values.log_budget_interaction.mean(), marker='D', color='#182D44', s=30)
    axes[0].set(xticks=[0, 1], xticklabels=['M03', 'M06'], ylabel='Paired log-budget interaction')
    axes[0].set_title('a  Budget interaction', loc='left'); axes[0].axhline(0, color='gray', lw=.7)
    labels = ['All six'] + [f'Drop RW{c}' for c in CELLS] + [f'Drop RW{a}+{b}' for a, b in TARGET_PAIRS]
    comparison = deletion[deletion.baseline.isin(['M03', 'M06'])].groupby(['baseline', 'stride', 'subset']).rmse_effect_pp.mean()
    low, high = min(0., comparison.min()), max(0., comparison.max())
    pad = max(.02, .08 * (high - low))
    for col, baseline in enumerate(('M03', 'M06'), 1):
        ax = axes[col]
        for stride, color, marker in ((1, '#767F88', 'o'), (10, '#0072B2', 's')):
            values = deletion[(deletion.baseline == baseline) & (deletion.stride == stride)].groupby('subset').rmse_effect_pp.mean()
            ax.plot([values[label] for label in labels], np.arange(10) + (-.11 if stride == 1 else .11),
                    ls='none', marker=marker, color=color, ms=4, label='Full' if stride == 1 else 'Stride10')
        ax.set(yticks=range(10), yticklabels=labels, ylim=(9.6, -.6), xlabel='RMSE difference (pp)')
        ax.set_xlim(low - pad, high + pad)
        ax.set_title(f"{'bc'[col - 1]}  {baseline} − M05", loc='left'); ax.axvline(0, color='gray', lw=.7)
        ax.grid(axis='x', alpha=.15)
    fig.legend(handles=handles + axes[1].get_legend_handles_labels()[0],
               labels=[h.get_label() for h in handles] + axes[1].get_legend_handles_labels()[1],
               loc='outside upper center', ncol=4, frameon=False, fontsize=8)
    save(fig, 'figure_05_robustness')

    direction, amplitude = read('direction_energy'), read('amplitude_diagnostics').query("scope == 'primary'")
    fig, axes = plt.subplots(2, 2, figsize=(7.48, 5.6), layout='constrained')
    for col, stride in enumerate((1, 10)):
        ax = axes[0, col]
        values = direction[(direction.stride == stride) & direction.method.isin(['M05', 'M06'])]
        for method, color, marker in (('M05', '#D55E00', 'o'), ('M06', '#0072B2', '^')):
            part = values[values.method == method]
            ax.scatter(part.correction_energy_pp2, 2 * part.inner_product_r_delta_pp2, s=16, alpha=.7, color=color, marker=marker, label=method)
            lower = axes[1, col]
            scores = amplitude[(amplitude.method == method) & (amplitude.stride == stride)]
            for _, points in scores.groupby('seed'):
                points = points.sort_values('scale')
                lower.plot(points.scale, points.rmse_pp, color=color, alpha=.18, lw=.7)
            mean = scores.groupby('scale').rmse_pp.mean()
            lower.plot(mean.index, mean.values, color=color, marker=marker, label=method, ms=4)
        bound = values.correction_energy_pp2.max() * 1.08
        ax.plot([0, bound], [0, bound], '--', lw=.8, color='gray', label='Zero MSE gain')
        ax.axhline(0, color='gray', lw=.6)
        ax.set(xlabel=r'Energy $\langle\delta^2\rangle$ (pp$^2$)', ylabel=r'$2\langle r\delta\rangle$ (pp$^2$)', xlim=(-.04 * bound, bound))
        ax.set_title(f"{'ab'[col]}  {'Full' if stride == 1 else 'Stride10'}: direction / energy", loc='left')
        lower = axes[1, col]
        lower.set(xticks=[0, .5, 1], xlabel='Specified correction amplitude a', ylabel='Macro RMSE (pp)')
        lower.set_title(f"{'cd'[col]}  {'Full' if stride == 1 else 'Stride10'}: amplitude", loc='left'); lower.grid(alpha=.15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside upper center', ncol=3, frameon=False, fontsize=8)
    save(fig, 'figure_06_correction_diagnostics')
    (output / 'figure_manifest.json').write_text(json.dumps(dict(source_table_sha256=consumed,
        figures=[2, 3, 4, 5, 6], formats=['png', 'pdf', 'svg'], model_calls=0), indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.analysis, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
