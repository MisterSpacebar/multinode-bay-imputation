"""Synthetic transition-warning tests: no private observations."""
import unittest
import numpy as np
import pandas as pd
from trustworthy.transition_warning import (
    prepare_states,warning_windows,validate_transitions,transition_matrix,
    markov_hit_probability,monte_carlo_reference,logistic_fit_predict,
    hindcast,evaluate,DO_FEATURES,ENV_FEATURES)


def synthetic(n=100,node='A',cycle=5):
    dates=pd.date_range('2026-01-01',periods=n,freq='D')
    # Normal -> Moderate -> Severe -> Normal in a repeated cycle
    labels=(np.array(['Normal','Moderate','Severe','Normal','Moderate']* ((n+4)//5))[:n] if cycle==5
        else np.array(['Severe' if i%cycle==2 else
            'Moderate' if i%3==1 else 'Normal' for i in range(n)]))
    states=pd.DataFrame(dict(node=[node]*n,date=dates,state=labels,
        hours_lt2=np.where(labels=='Severe',2.,0.),
        hours_lt4=np.where(labels=='Normal',0.,np.where(labels=='Moderate',3.,5.)),
        qc_suspect=False,provenance='observed_only'))
    daily=pd.DataFrame(dict(node=[node]*n,date=dates,eligible=True,
        coverage=1.,odo_mean=np.where(labels=='Normal',6.,np.where(labels=='Moderate',3.,1.)),
        odo_min=np.where(labels=='Normal',5.,np.where(labels=='Moderate',2.5,1.)),
        temp_mean=24+np.arange(n)%17/10.,sal_mean=26+np.arange(n)%11/10.))
    transitions=pd.DataFrame(dict(node=[node]*(n-1),date=dates[:-1],next_date=dates[1:],
        from_state=labels[:-1],to_state=labels[1:],provenance='observed_only'))
    return states,daily,transitions

class TransitionWarningTests(unittest.TestCase):
    def test_horizon_known_and_hit(self):
        states,daily,_=synthetic(14)
        out=warning_windows(states,daily)
        r=out.loc[out.issue_date.eq(pd.Timestamp('2026-01-01'))].set_index('horizon_days')
        self.assertEqual(r.loc[1,'label'],0)
        self.assertEqual(r.loc[3,'label'],1)
        self.assertEqual(r.loc[7,'label'],1)
        self.assertEqual(r.loc[3,'first_severe_date'],'2026-01-03')
        self.assertFalse(out.origin_state.eq('Severe').any())

    def test_missing_future_always_unscorable_even_after_hit(self):
        states,daily,_=synthetic(14)
        states.loc[5,'state']='Unknown';states.loc[5,['hours_lt2','hours_lt4']]=np.nan
        out=warning_windows(states,daily)
        r=out.loc[(out.issue_date.eq(pd.Timestamp('2026-01-01')))&out.horizon_days.eq(7)].iloc[0]
        self.assertTrue(pd.isna(r.label))
        self.assertEqual(r.status,'unknown_future')

    def test_calendar_gap_not_silently_skipped(self):
        states,daily,_=synthetic(14)
        states=states.drop(index=2)
        out=warning_windows(states,daily)
        r=out.loc[(out.issue_date.eq(pd.Timestamp('2026-01-01')))&out.horizon_days.eq(3)].iloc[0]
        self.assertTrue(pd.isna(r.label))

    def test_issue_day_provenance_and_qc(self):
        states,daily,_=synthetic(14)
        states.loc[0,'provenance']='gnn'
        with self.assertRaisesRegex(ValueError,'observed_only'):
            warning_windows(states,daily)
        states.loc[0,'provenance']='observed_only'
        daily.loc[0,'eligible']=False
        out=warning_windows(states,daily)
        self.assertEqual(out.iloc[0].status,'issue_qc_ineligible')

    def test_do_trend_strict_past_only(self):
        states,daily,_=synthetic(14)
        daily.loc[0,'odo_mean']=100  # historical outlier: future day must not leak
        out=warning_windows(states,daily)
        issue=out.loc[out.issue_date.eq(pd.Timestamp('2026-01-04'))].iloc[0]
        self.assertEqual(issue.delta_do_mean_1d,5.)
        # The first date has no prior-day data; no invented trend.
        self.assertTrue(pd.isna(out.loc[out.issue_date.eq(pd.Timestamp('2026-01-01')),
                                      'delta_do_mean_1d']).all())

    def test_transitions_reject_gaps_or_mismatch(self):
        states,daily,t=synthetic(14)
        validate_transitions(t,prepare_states(states))
        bad=t.copy();bad.loc[0,'next_date']=bad.loc[0,'next_date']+pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError,'adjacent'):
            validate_transitions(bad,prepare_states(states))
        bad=t.copy();bad.loc[0,'to_state']='Severe'
        with self.assertRaisesRegex(ValueError,'inconsistent'):
            validate_transitions(bad,prepare_states(states))

    def test_markov_hitting_not_only_state_at_horizon(self):
        p=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
        self.assertEqual(markov_hit_probability(p,'Normal',1),0)
        self.assertEqual(markov_hit_probability(p,'Normal',2),1)
        self.assertEqual(markov_hit_probability(p,'Normal',7),1)
        self.assertEqual(markov_hit_probability(p,'Moderate',1),1)
        with self.assertRaises(ValueError):
            markov_hit_probability(p,'Severe',1)

    def test_shrinkage_and_unknown_state_smoothing(self):
        _,_,t=synthetic(40)
        p=transition_matrix(t,station='A')
        self.assertEqual(p.shape,(3,3));self.assertTrue(np.allclose(p.sum(axis=1),1))
        other=transition_matrix(t,station='new_station')
        self.assertTrue(np.allclose(other.sum(axis=1),1))
        self.assertTrue((other>0).all())

    def test_simulation_close_to_analytic_and_seeded(self):
        p=np.array([[.6,.3,.1],[.2,.6,.2],[.1,.1,.8]])
        a=monte_carlo_reference(p,'Normal',25000,seed=123)
        b=monte_carlo_reference(p,'Normal',25000,seed=123)
        self.assertEqual(a,b)
        self.assertAlmostEqual(a['within_7d'],markov_hit_probability(p,'Normal',7),delta=.02)
        self.assertGreaterEqual(a['within_7d'],a['within_3d'])

    def test_logistic_fitting_does_not_need_future_features(self):
        s,d,t=synthetic(150)
        windows=warning_windows(s,d)
        rows=windows.loc[windows.horizon_days.eq(1)&windows.status.eq('evaluable')].copy()
        rows=rows.dropna(subset=ENV_FEATURES)
        train=rows.iloc[:60];test=rows.iloc[60:70]
        p=logistic_fit_predict(train,test,DO_FEATURES)
        self.assertEqual(len(p),len(test))
        self.assertTrue(((p>0)&(p<1)).all())

    def test_hindcast_month_boundary_no_7day_label_leak(self):
        s,d,t=synthetic(210,cycle=16)
        windows=warning_windows(s,d)
        preds,diag,mc=hindcast(windows,validate_transitions(t,prepare_states(s)),
                               min_train=12,min_events=2,n_paths=60)
        self.assertFalse(preds.empty)
        self.assertEqual(preds.groupby(['node','issue_date','horizon_days']).model.nunique().min(),3)
        self.assertFalse(mc.empty)
        first='2026-03'
        # No row whose label window ends in March is part of Feb's training for March.
        used=windows.loc[(windows.horizon_days==7)&(windows.status=='evaluable')&
                         (windows.window_end<pd.Timestamp('2026-03-01'))].dropna(subset=ENV_FEATURES)
        diag3=diag.loc[(diag.test_month==first)&(diag.horizon_days==7)]
        if len(diag3):
            self.assertEqual(diag3.iloc[0].n_train,len(used))
        scores=evaluate(preds)
        self.assertTrue(scores.n_paired.gt(0).all())

    def test_hindcast_requires_completed_transition_to_train_markov(self):
        s,d,t=synthetic(200)
        w=warning_windows(s,d)
        # Removing all transitions forces no evaluations even though model labels exist.
        pred,diag,_=hindcast(w,t.iloc[0:0],min_train=10,min_events=1,n_paths=0)
        self.assertTrue(pred.empty)

if __name__=='__main__':
    unittest.main()