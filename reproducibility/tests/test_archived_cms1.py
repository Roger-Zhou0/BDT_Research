from pathlib import Path
import json,shutil,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repro.archived_cms1 import ROOT,recompute

class ArchivedCMS1(unittest.TestCase):
    def test_all_original_report_records(self):
        r=recompute();self.assertEqual(r['archived_report_files'],18);self.assertEqual(r['training_seed_units'],6)
        self.assertEqual(r['model_evaluations_performed'],0)
        self.assertAlmostEqual(r['contrasts']['bdt - adamw']['paired_95']['mean'],-.002696905,places=9)
        self.assertAlmostEqual(r['contrasts']['bdt - scrambled_matched']['paired_95']['mean'],-.006098807,places=9)
        self.assertEqual(r['contrasts']['bdt - adamw']['paired_95']['negative_signs'],6)
    def test_missing_cell_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cms';shutil.copytree(ROOT/'reference/cms1',p)
            (p/'seed_147/adamw_report.json').unlink()
            with self.assertRaises(FileNotFoundError):recompute(p)
    def test_edited_cell_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cms';shutil.copytree(ROOT/'reference/cms1',p)
            f=p/'seed_147/adamw_report.json';f.write_text(f.read_text()+' ')
            with self.assertRaises(RuntimeError):recompute(p)

if __name__=='__main__':unittest.main()
