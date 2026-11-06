"""Saved-prediction regression and coverage-failure tests; no model execution."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from battery_capacity.analysis import ROOT, analyze, paired_analysis, read_predictions
from battery_capacity.verify import verify


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = read_predictions([ROOT / 'results/paper_predictions.csv.gz'])

    def reject(self, frame, message):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.csv'
            frame.to_csv(path, index=False, float_format='%.17g')
            with self.assertRaisesRegex(ValueError, message):
                read_predictions([path])

    def test_all_paper_values(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'analysis'
            tables = analyze(self.data, output)
            self.assertTrue(verify(output)['passed'])
            self.assertEqual(len(tables['performance']), 52)
            self.assertEqual(len(tables['table4']), 4)

    def test_missing_seed_is_rejected(self):
        self.reject(self.data[~((self.data.method == 'M05') & (self.data.seed == 4))], 'Incomplete paper matrix')

    def test_duplicate_identity_is_rejected(self):
        import pandas as pd
        self.reject(pd.concat([self.data, self.data.iloc[[0]]]), 'Duplicate')

    def test_changed_truth_is_rejected(self):
        frame = self.data.copy()
        frame.loc[0, 'truth'] += .01
        self.reject(frame, 'alignment mismatch')

    def test_wrong_fold_is_rejected(self):
        frame = self.data.copy()
        frame.loc[0, 'fold'] = (frame.loc[0, 'fold'] + 1) % 3
        self.reject(frame, 'Fold/rank mismatch')

    def test_primary_suffix_is_checked(self):
        frame = self.data.copy()
        group = frame[(frame.method == 'M05') & (frame.stride == 10) & (frame.seed == 0)]
        early = group[~group.primary].index[0]
        late = group[group.primary].index[0]
        frame.loc[early, 'primary'], frame.loc[late, 'primary'] = True, False
        self.reject(frame, 'Primary-mask mismatch')

    def test_nonfinite_prediction_is_rejected(self):
        frame = self.data.copy()
        frame.loc[0, 'prediction'] = np.nan
        self.reject(frame, 'Nonfinite')

    def test_zero_error_interaction_remains_undefined(self):
        from battery_capacity.analysis import cell_errors
        cells = cell_errors(self.data)
        mask = (cells.method == 'M03') & (cells.stride == 10) & (cells.seed == 0) & (cells.cell == 'RW_battery-1.mat')
        cells.loc[mask, 'rmse_pp'] = 0.
        _, interaction, _ = paired_analysis(cells)
        row = interaction[(interaction.baseline == 'M03') & (interaction.seed == 0) & (interaction.cell == 'RW_battery-1.mat')]
        self.assertTrue(row.log_budget_interaction.isna().all())


if __name__ == '__main__':
    unittest.main()
