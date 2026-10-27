from pathlib import Path
import sys,unittest,math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repro.results import report_results,allocation_results,t_df3,paired

class Results(unittest.TestCase):
    def test_final_report_bct1(self):
        r=report_results()['contrasts']['bct']['bdt_budget_gs - structured_budget_gs']
        self.assertAlmostEqual(r['mean'],-.000618126708781,places=12)
        self.assertAlmostEqual(r['lower'],-.000772291517,places=12)
        self.assertAlmostEqual(r['upper'],-.000463961901,places=12)
        self.assertEqual(r['negative_signs'],4)

    def test_residual_gap_remains(self):
        r=report_results()['contrasts']['bct']['bdt_budget_gs - combined']
        self.assertAlmostEqual(r['mean'],.000095988624,places=12)
        self.assertGreater(r['lower'],0)

    def test_dependent_cohorts_not_counted_as_replications(self):
        r=report_results();self.assertEqual(r['parent_units'],4)
        self.assertEqual(r['unique_checkpoint_cells'],40);self.assertEqual(r['logical_report_cells'],56)
        self.assertFalse(r['fresh_parent_confirmation']);self.assertFalse(r['ART1_both_primary_uppers_below_zero'])

    def test_retrospective_relative_not_absolute_reversal(self):
        r=allocation_results();self.assertEqual(r['observations'],240)
        self.assertEqual(r['full_minus_structured_positive_at_400'],24)
        self.assertEqual(r['full_minus_structured_negative_at_3000'],24)
        self.assertTrue(r['full_qD_itself_increases_400_to_3000_every_cell'])
        self.assertFalse(r['fixed_coefficient_or_schedule_ruled_out'])

    def test_t_inverse_against_scipy(self):
        from scipy.stats import t
        for level in (.9,.95,.975,.99):self.assertAlmostEqual(t_df3(level),float(t.ppf((1+level)/2,3)),places=9)

    def test_invalid_data_refused(self):
        with self.assertRaises(ValueError):paired([0,float('nan'),1,2])
        with self.assertRaises(ValueError):paired([1])

if __name__=='__main__':unittest.main()
