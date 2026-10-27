"""Release-level integrity and arithmetic checks, not scientific replications."""
import hashlib
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from repro.bct2_records import (ROOT, SEEDS, ARMS, REPORT_SHA256, T_DF3_975,
                               _json, _path, paired_summary, verify_bct2)


class BCT2RecordsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name) / 'bct2'
        shutil.copytree(ROOT / 'reference/bct2', self.directory)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_confirmation_and_not_pooled(self):
        result = verify_bct2(self.directory)
        self.assertTrue(result['passed'])
        self.assertEqual(result['parent_seeds'], list(SEEDS))
        self.assertEqual(result['original_score_receipts'], 8)
        self.assertFalse(result['pooled_with_BCT1'])
        self.assertEqual(result['new_model_evaluations'], 0)

    def test_exact_original_report_hash(self):
        raw = (self.directory / 'run/REPORT/FINAL_REPORT.json').read_bytes()
        self.assertEqual(len(raw), 2506)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), REPORT_SHA256)

    def test_missing_receipt_rejected(self):
        (self.directory / f'run/REPORT/{SEEDS[0]}_{ARMS[0]}.json').unlink()
        with self.assertRaises(ValueError):
            verify_bct2(self.directory)

    def test_receipt_byte_change_rejected(self):
        path = self.directory / f'run/REPORT/{SEEDS[0]}_{ARMS[0]}.json'
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaises(ValueError):
            verify_bct2(self.directory)

    def test_changed_summary_rejected(self):
        path = self.directory / 'run/REPORT/FINAL_REPORT.json'
        report = json.loads(path.read_bytes())
        report['interval95']['mean'] = 0.0
        path.write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            verify_bct2(self.directory)

    def test_unlisted_extra_record_rejected(self):
        (self.directory / 'additional_parent.json').write_text('{}')
        with self.assertRaises(ValueError):
            verify_bct2(self.directory)

    def test_origin_tampering_rejected(self):
        path = self.directory / 'ORIGIN.json'
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaises(ValueError):
            verify_bct2(self.directory)

    def test_protocol_derivative_has_distinct_bindings(self):
        origin = json.loads((self.directory / 'ORIGIN.json').read_bytes())
        item = next(r for r in origin['records'] if r['path'] == 'protocol.json')
        self.assertNotEqual(item['original_sha256'], item['public_sha256'])
        self.assertNotEqual(item['original_canonical_json_sha256'], item['public_canonical_json_sha256'])
        self.assertEqual(item['changed_string_fields'], ['/resources/blocked_gpu_until_GSC1_release'])

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):
            _json(b'{"seed":1,"seed":2}')

    def test_nonfinite_json_rejected(self):
        for constant in (b'NaN', b'Infinity', b'-Infinity'):
            with self.assertRaises(ValueError):
                _json(b'{"loss":' + constant + b'}')

    def test_unsafe_paths_rejected(self):
        for path in ('../outside', '/outside', 'run/../../outside', 'run\\outside'):
            with self.assertRaises(ValueError):
                _path(self.directory, path)

    def test_symlink_receipt_rejected(self):
        path = self.directory / f'run/REPORT/{SEEDS[0]}_{ARMS[0]}.json'
        copy = Path(self.temp.name) / 'receipt.json'
        copy.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(copy)
        with self.assertRaises(ValueError):
            verify_bct2(self.directory)

    def test_four_parent_summary_only(self):
        for values in ([], [-1.0] * 3, [-1.0] * 8, [math.nan] * 4, [True] * 4):
            with self.assertRaises(ValueError):
                paired_summary(values)

    def test_student_t_critical_value_against_scipy(self):
        from scipy.stats import t
        # The archived critical value and library inversion can differ at ~1e-13.
        self.assertAlmostEqual(T_DF3_975, t.ppf(0.975, 3), delta=1e-11)


if __name__ == '__main__':
    unittest.main()
