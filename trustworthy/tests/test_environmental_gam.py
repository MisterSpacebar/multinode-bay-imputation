"""Synthetic environmental GAM tests; never require private station measurements."""
import unittest
import numpy as np
import pandas as pd
from trustworthy.environmental_gam import (
    prepare_daily, severity_rows, recovery_rows, rolling_backtest,
    paired_scores, smooth_effect_curves, full_fit_summary,
)


def mock_daily(n=240, stations=('A','B')):
    rng=np.random.default_rng(3)
    dates=pd.date_range('2025-01-01',periods=n,freq='D')
    rows=[]
    for node in stations:
        for i,date in enumerate(dates):
            temp=25+3*np.sin(i/50)+rng.normal(0,.6)
            sal=24+2*np.cos(i/33)+rng.normal(0,.6)
            mean=5-.35*(temp-25)-.04*(sal-24)+rng.normal(0,.5)
            rows.append({'node':node,'date':date,'eligible':True,
                 'coverage':1.,'odo_min':mean-1.,'odo_mean':mean,
                 'temp_mean':temp,'sal_mean':sal,'n_zero_review':int(i==20)})
    return pd.DataFrame(rows)


def mock_risk(daily, threshold=2):
    rng=np.random.default_rng(4)
    rows=[]
    for node,group in daily.groupby('node'):
        group=group.sort_values('date').reset_index(drop=True)
        for i in range(12,len(group)-1):
            date=group.date.iloc[i]
            rows.append({'episode_id':f'{node}-{i//3}-{threshold}',
               'node':node,'date':date,'next_date':date+pd.Timedelta(days=1),
               'threshold_mgL':threshold,'log_elapsed_days':float(np.log(i%3+1)),
               'hours_below_threshold':float(2+(i%5)),
               'next_day_recovery':bool(rng.random()<.3),
               'qc_suspect':False,'provenance':'observed_only'})
    return pd.DataFrame(rows)


class EnvironmentalGAMTests(unittest.TestCase):
    def test_qc_coverage_and_no_gnn_as_observation(self):
        d=mock_daily(35,('A',))
        d.loc[0,'coverage']=.94
        d.loc[1,'eligible']=False
        d.loc[2,'temp_mean']=np.nan
        obs=prepare_daily(d)
        s=severity_rows(obs)
        self.assertEqual(len(s),32)
        self.assertTrue(s.provenance.eq('observed_only').all())
        self.assertTrue(bool(s.qc_suspect.any()))
        d['provenance']='gnn_imputed'
        with self.assertRaisesRegex(ValueError,'non-observed'):
            prepare_daily(d)

    def test_prior_day_trend_requires_true_adjacency(self):
        d=mock_daily(15,('A',))
        d=d.loc[d.date.ne(pd.Timestamp('2025-01-05'))].copy()
        obs=prepare_daily(d)
        self.assertTrue(np.isnan(obs.loc[obs.date.eq('2025-01-06'),'delta_do_mean_1d'].iloc[0]))
        self.assertTrue(np.isfinite(obs.loc[obs.date.eq('2025-01-07'),'delta_do_mean_1d'].iloc[0]))

    def test_recovery_features_do_not_join_tomorrow(self):
        d=mock_daily(45,('A',))
        r=mock_risk(d)
        obs=prepare_daily(d)
        features=recovery_rows(r,obs)
        self.assertTrue((features.next_date-features.date).dt.days.eq(1).all())
        row=features.iloc[0]
        old=obs.loc[obs.node.eq(row.node)&obs.date.eq(row.date),'temp_mean'].iloc[0]
        self.assertAlmostEqual(row.temp_mean,old)
        # The measured next-day DO must not be substituted for issue-day DO.
        self.assertAlmostEqual(row.odo_mean,obs.loc[obs.date.eq(row.date),'odo_mean'].iloc[0])
        r.loc[0,'next_date']=r.loc[0,'date']+pd.Timedelta(days=2)
        with self.assertRaisesRegex(ValueError,'immediately follow'):
            recovery_rows(r,obs)

    def test_recovery_complete_cases_preserve_matched_baseline(self):
        d=mock_daily(50,('A',))
        r=mock_risk(d)
        obs=prepare_daily(d)
        obs.loc[obs.date.eq(pd.Timestamp('2025-01-21')),'sal_mean']=np.nan
        features=recovery_rows(r,obs)
        self.assertLess(len(features),len(r))
        self.assertTrue(features[['temp_mean','sal_mean','delta_do_mean_1d']].notna().all().all())
        r.loc[0,'provenance']='gnn'
        with self.assertRaisesRegex(ValueError,'non-observed'):
            recovery_rows(r,obs)

    def test_duplicate_and_bad_boolean_rejected(self):
        d=mock_daily(20,('A',))
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            prepare_daily(pd.concat([d,d.iloc[[0]]]))
        d['eligible'] = d['eligible'].astype(object)
        d.loc[0,'eligible']='unknown'
        with self.assertRaisesRegex(ValueError,'boolean'):
            prepare_daily(d)

    def test_rolling_monthly_predictions_are_matched(self):
        d=prepare_daily(mock_daily(225))
        sev=severity_rows(d)
        pred,diag=rolling_backtest(sev,'odo_min',alpha=10,min_train=100)
        self.assertFalse(pred.empty)
        self.assertEqual(set(pred.model),{'baseline','environmental_gam'})
        for group in pred.groupby(['node','date']):
            self.assertEqual(len(group[1]),2)
        self.assertTrue((pred.test_month==pred.date.str[:7]).all())
        scores=paired_scores(pred)
        self.assertEqual(len(scores),2)
        self.assertTrue((scores.n_paired>0).all())
        self.assertTrue((pred.predicted.between(0,20)).all())

    def test_recovery_backtest_never_uses_incomplete_training_labels(self):
        obs=prepare_daily(mock_daily(230))
        risk=recovery_rows(mock_risk(mock_daily(230)),obs)
        pred,diag=rolling_backtest(risk,'next_day_recovery',min_train=100)
        self.assertFalse(pred.empty)
        self.assertTrue(pred.predicted.between(0,1).all())
        self.assertEqual(set(pred.threshold_mgL),{2})
        self.assertTrue((diag.loc[diag.status.eq('evaluable'),'n_train']>=100).all())
        self.assertEqual(len(paired_scores(pred)),2)

    def test_conditional_smooth_curves_finite_within_training_support(self):
        severity=severity_rows(prepare_daily(mock_daily(120)))
        effects=smooth_effect_curves(severity,'odo_min',points=11)
        self.assertEqual(len(effects),22)
        self.assertTrue(effects.fitted_response.between(0,20).all())
        self.assertTrue(effects.provenance.eq('observed_only').all())
        self.assertEqual(set(effects.predictor),{'temp_mean','sal_mean'})
        fits=full_fit_summary(severity,'odo_min')
        self.assertEqual(set(fits.model),{'baseline','environmental_gam'})


if __name__=='__main__':
    unittest.main()
