"""Synthetic Phase 3 regression tests; no private data dependencies."""
import unittest
import numpy as np
import pandas as pd
from trustworthy.episode_survival import (km_curves, recovery_risk_rows,
    fit_recovery_hazard, _markov_metrics, markov_matrix_bootstrap,
    simulate_markov_paths)

def episodes(rows):
    return pd.DataFrame([
        dict(episode_id=f'A::{r[0]}::{r[1]}',node='A',threshold_mgL=r[0],
        start_date_local=r[1],last_hypoxic_date_local=r[2],
        calendar_duration_days=(pd.Timestamp(r[2])-pd.Timestamp(r[1])).days+1,
        identifiable_onset=r[3],censored_left=not r[3],
        observed_recovery=r[4],censored_right=not r[4],
        provenance='observed_only') for r in rows])

def states(dates,labels):
    n=len(dates)
    return pd.DataFrame(dict(node=['A']*n,date=dates,state=labels,
      hours_lt2=[2 if s=='Severe' else 0 for s in labels],
      hours_lt4=[6 if s in ('Severe','Moderate') else 0 for s in labels],
      qc_suspect=[False]*n,provenance=['observed_only']*n))

def transitions():
    dates=pd.date_range('2026-07-01',periods=18,freq='D')
    a=['Normal','Normal','Moderate','Severe','Severe','Moderate',
       'Normal','Moderate','Moderate','Severe','Moderate','Normal',
       'Normal','Severe','Moderate','Normal','Severe','Normal']
    return pd.DataFrame(dict(node=['A']*17,date=dates[:-1],
        next_date=dates[1:],from_state=a[:-1],to_state=a[1:],
        provenance=['observed_only']*17))

class EpisodeSurvivalTests(unittest.TestCase):
    def test_km_censors_unknown_without_imaginary_recovery(self):
        e=episodes([(2,'2026-07-02','2026-07-03',True,True),
                    (2,'2026-07-06','2026-07-08',True,False)])
        curve,summ=km_curves(e,by_station=False)
        self.assertEqual(curve.n_at_risk.tolist(),[2,1])
        self.assertEqual(curve.n_recoveries.tolist(),[1,0])
        self.assertEqual(curve.n_censored.tolist(),[0,1])
        self.assertTrue(np.allclose(curve.survival_probability,[.5,.5]))
        self.assertEqual(summ.iloc[0].km_median_days,2)
        self.assertEqual(summ.iloc[0].n_right_censored,1)

    def test_km_only_censored_means_no_median(self):
        e=episodes([(4,'2026-07-01','2026-07-05',True,False)])
        _,summary=km_curves(e)
        self.assertFalse(bool(summary.iloc[0].km_median_reached))
        self.assertTrue(np.isnan(summary.iloc[0].km_median_days))

    def test_left_censored_not_in_onset_survival(self):
        e=episodes([(2,'2026-07-01','2026-07-05',False,True),
                    (2,'2026-07-11','2026-07-11',True,True)])
        _,s=km_curves(e)
        self.assertEqual(s.loc[s.scope.eq('pooled'),
                               'n_identifiable_onsets'].iloc[0],1)

    def test_risk_outcome_only_if_next_day_observed(self):
        e=episodes([(2,'2026-07-02','2026-07-03',True,True),
                    (2,'2026-07-06','2026-07-07',True,False)])
        s=states(pd.date_range('2026-07-01',periods=8,freq='D'),
          ['Normal','Severe','Severe','Moderate','Normal',
           'Severe','Severe','Unknown'])
        risk=recovery_risk_rows(e,s)
        self.assertEqual(len(risk),3)
        self.assertEqual(int(risk.next_day_recovery.sum()),1)
        self.assertEqual(risk.elapsed_days.tolist(),[1,2,1])
        self.assertEqual(risk.next_state.tolist(),['Severe','Moderate','Severe'])

    def test_recovery_mismatch_and_imputation_rejected(self):
        e=episodes([(2,'2026-07-02','2026-07-02',True,False)])
        s=states(['2026-07-01','2026-07-02','2026-07-03'],
                 ['Normal','Severe','Normal'])
        with self.assertRaisesRegex(ValueError,'censored end'):
            recovery_risk_rows(e,s)
        e.loc[0,'provenance']='gnn'
        with self.assertRaisesRegex(ValueError,'observed_only'):
            km_curves(e)

    def test_date_and_duration_agree(self):
        e=episodes([(4,'2026-07-02','2026-07-03',True,True)])
        e.loc[0,'calendar_duration_days']=3
        with self.assertRaisesRegex(ValueError,'duration'):
            km_curves(e)

    def test_stationary_markov_reference(self):
        matrix=np.array([[265,68,4],[59,311,59],[7,48,282]])
        metrics=_markov_metrics(matrix)
        self.assertAlmostEqual(metrics['expected_consecutive_severe_days'],
                               337/55)
        self.assertAlmostEqual(metrics['expected_severe_to_normal_days'],
                               16.500273,places=4)
        self.assertAlmostEqual(metrics['p_severe_within_7_days_from_normal'],
                               .316912,places=4)

    def test_markov_undefined_when_no_exit_or_denominator(self):
        a=np.array([[1,0,0],[0,1,0],[0,0,0]])
        self.assertTrue(np.isnan(_markov_metrics(a)['expected_severe_to_normal_days']))
        b=np.array([[2,0,0],[0,2,0],[0,0,2]])
        self.assertTrue(np.isnan(_markov_metrics(b)['expected_consecutive_severe_days']))

    def test_whole_block_bootstrap_reproducibility(self):
        x=transitions()
        a,draws=markov_matrix_bootstrap(x,31,schemes=('14day','month'),
                                        seed=7,save_draws=True)
        b,_=markov_matrix_bootstrap(x,31,schemes=('14day','month'),
                                    seed=7,save_draws=True)
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(len(draws),31*2*5)
        x.loc[0,'provenance']='gnn'
        with self.assertRaisesRegex(ValueError,'observed_only'):
            markov_matrix_bootstrap(x,10)

    def test_paths_reproducible_and_censored_at_horizon(self):
        x=transitions()
        a=simulate_markov_paths(x,40,schemes=('14day',),seed=55,
                                n_paths_per_matrix=3,horizon_days=30)
        b=simulate_markov_paths(x,40,schemes=('14day',),seed=55,
                                n_paths_per_matrix=3,horizon_days=30)
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue((a.n_hit_normal+a.n_censored_at_horizon==
                         a.n_trajectories).all())
        self.assertTrue(a.prob_hit_normal.between(0,1).all())

    def test_cloglog_hazard_fit_when_dependency_installed(self):
        try:
            import statsmodels
        except ImportError:
            self.skipTest('statsmodels optional')
        rng=np.random.default_rng(13)
        n=120
        risk=pd.DataFrame(dict(threshold_mgL=[2]*n,
          node=['A' if i%2==0 else 'B' for i in range(n)],
          episode_id=[f'e::{i}' for i in range(n)],
          log_elapsed_days=np.log(rng.integers(1,7,size=n)),
          hours_below_threshold=rng.uniform(.5,12,size=n),
          next_day_recovery=rng.binomial(1,.35,size=n).astype(bool)))
        res=fit_recovery_hazard(risk)
        self.assertEqual(set(res.term),
            {'intercept','log_elapsed_days','hours_below_per_6h'})
        self.assertTrue(np.isfinite(res.relative_hazard).all())

if __name__=='__main__':
    unittest.main()
