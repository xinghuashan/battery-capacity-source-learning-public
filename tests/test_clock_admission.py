"""Clock admission must reject unsupported evidence; no raw MAT data is read."""
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np


@unittest.skipUnless(find_spec('scipy') is not None, 'Clock admission checks require SciPy.')
class ClockAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only dependency absence skips these tests; other import errors propagate.
        from battery_capacity.preprocessing import admit_clock_scope
        cls.admit = staticmethod(admit_clock_scope)
        config = Path(__file__).resolve().parents[1] / 'config/accepted_clock_scope.json'
        cls.identity = json.loads(config.read_text(encoding='utf-8'))['cells']['RW_battery-1.mat']

    def reject(self, report, inventory, message=None):
        cell = {'q': np.zeros(self.identity['eligible_events'])}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            if message:
                context = self.assertRaisesRegex(AssertionError, message)
            else:
                context = self.assertRaises(AssertionError)
            with context:
                self.admit(Path('RW_battery-1.mat'), cell, report, [], [], inventory, output)
            self.assertFalse((output / 'RW_battery-1_clock_admission.json').exists())

    def test_changed_raw_fingerprint_is_rejected(self):
        report = {'raw_sha256': '0'*64, 'violations': [], 'passed': True}
        self.reject(report, [], 'Original raw fingerprint changed')

    def test_additional_violation_is_rejected(self):
        report = {'raw_sha256': self.identity['raw_sha256'], 'passed': False,
                  'violations': [{'raw_record': 1, 'failure': 'Reference missing charge/discharge'}]}
        self.reject(report, [], 'Unadmitted audit violation')

    def test_cross_record_reversal_is_rejected(self):
        report = {'raw_sha256': self.identity['raw_sha256'], 'violations': [], 'passed': False}
        inventory = [{'raw_record': 1, 'across_available_records_reversal': True,
                      'within_record_reversal': False}]
        self.reject(report, inventory)


if __name__ == '__main__':
    unittest.main()
