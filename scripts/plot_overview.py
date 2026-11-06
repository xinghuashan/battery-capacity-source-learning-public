"""Draw a performance overview from the saved seed summaries only."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METHODS = ['Ridge', 'Ridge-CV', 'KRR', 'KRR-wide', 'B3', 'N01', 'M01',
           'M02', 'M03', 'M04', 'M06', 'M05-raw-CV', 'M05']
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#A07A00']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/overview')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(ROOT / 'results/seed_metrics.csv')
    data = data[data.scope == 'primary']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.titlesize': 10, 'axes.labelsize': 10})
    fig, axes = plt.subplots(2, 2, figsize=(7.48, 6.5))
    for row, stride in enumerate((1, 10)):
        for col, metric in enumerate(('rmse_pp', 'mae_pp')):
            ax = axes[row, col]
            ax.axhspan(11.5, 12.5, color='#EEF5FA')
            for index, method in enumerate(METHODS):
                values = data[(data.method == method) & (data.stride == stride)]
                if len(values) == 1:
                    ax.scatter(values.iloc[0][metric], index, s=22, facecolor='white', edgecolor='#182D44')
                else:
                    assert len(values) == 5
                    for entry in values.to_dict('records'):
                        seed = int(entry['seed'])
                        ax.scatter(entry[metric], index + (seed - 2) * .075, s=14, color=COLORS[seed])
                    ax.scatter(values[metric].mean(), index, marker='D', s=23, color='#182D44')
            ax.set_yticks(range(len(METHODS)), ['Raw-CV' if m == 'M05-raw-CV' else m for m in METHODS])
            ax.set_ylim(len(METHODS) - .4, -.6)
            ax.grid(axis='x', alpha=.18)
            ax.set_xlabel(('RMSE' if col == 0 else 'MAE') + ' (pp)')
            ax.set_title(f"{'abcd'[2 * row + col]}  {'Full' if stride == 1 else 'Stride10'}", loc='left')
    handles = [Line2D([0], [0], ls='none', marker='o', color=color, label=f'Seed {i}')
               for i, color in enumerate(COLORS)]
    handles += [Line2D([0], [0], ls='none', marker='D', color='#182D44', label='Mean'),
                Line2D([0], [0], ls='none', marker='o', mfc='white', mec='#182D44', label='Deterministic')]
    fig.legend(handles=handles, loc='upper center', ncol=7, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.14, right=.98, bottom=.08, top=.90, wspace=.43, hspace=.37)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(args.output / f'performance_overview.{extension}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
