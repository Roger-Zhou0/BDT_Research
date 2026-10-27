from pathlib import Path
import hashlib,json,unittest
ROOT=Path(__file__).resolve().parents[1]

class PublicAnonymity(unittest.TestCase):
    def test_generic_home_path_detector(self):
        patterns=self._patterns()
        sample='/'+'home'+'/'+'synthetic_reader'+'/'+'file.json'
        self.assertIsNotNone(patterns['absolute_home'].search(sample))

    def test_generic_email_detector(self):
        sample='synthetic_user'+'@'+'example.invalid'
        self.assertIsNotNone(self._patterns()['email'].search(sample))

    def test_generic_notebook_detector(self):
        sample='https://'+'example.invalid'+'/'+'user'+'/'+'synthetic_reader'+'/files'
        self.assertIsNotNone(self._patterns()['private_notebook_url'].search(sample))

    def test_generic_credential_detector(self):
        sample='-----BEGIN '+'PRIVATE'+' KEY-----'
        self.assertIsNotNone(self._patterns()['credential_marker'].search(sample))

    def test_generic_patterns_allow_scientific_text(self):
        sample='Four fresh parents; BDT budgets and fixed model storage.'
        self.assertFalse(any(p.search(sample) for p in self._patterns().values()))

    @staticmethod
    def _patterns():
        import importlib.util
        spec=importlib.util.spec_from_file_location('public_checker',ROOT/'scripts/check_package.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module.privacy_patterns()

    def test_account_label_patch_preserves_origin_binding(self):
        origin=json.loads((ROOT/'reference/manuscript_v20/ORIGIN.json').read_text())
        rows={x['member']:x for x in origin['inputs']}
        rels=['reference/manuscript_v20/evidence/September22_Mechanism_Record_Audit.json','reference/manuscript_v20/evidence/oip1/authenticated_completion_metrics.json']
        for rel in rels:
            row=rows[rel]; self.assertEqual(row['transformation'],'identity/path/account-label derivative')
            self.assertEqual(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),row['public_sha256'])
            self.assertNotEqual(row['original_sha256'],row['public_sha256'])
