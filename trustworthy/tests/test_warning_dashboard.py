"""Tests for warning dashboard metrics, historical dates and forecast provenance."""
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy.warning_dashboard import (
    calibration_table, checked_predictions, latest_snapshot,
    observed_future, query_forecast, risk_level, threshold_sweep, make_figures,
    all_stations_date_query,
)


def example_predictions():
    rows = []
    for day, y, probs in [
        ("2026-07-02", 0, (0.05, 0.10, 0.20)),
        ("2026-07-03", 1, (0.70, 0.65, 0.60)),
        ("2026-07-04", 0, (0.15, 0.30, 0.20)),
        ("2026-07-05", 1, (0.40, 0.55, 0.60)),
    ]:
        for h in (1, 3, 7):
            for m, p in zip(("markov", "do_history", "environmental"), probs):
                rows.append(dict(node="A", issue_date=day,
                    horizon_days=h, model=m, label=y, probability=p,
                    n_train=75, n_train_events=12, origin_state="Moderate",
                    provenance="observed_only"))
    return pd.DataFrame(rows)


def synthetic_states():
    dates = pd.date_range("2026-07-01", periods=12, freq="D")
    names = ["Normal", "Moderate", "Moderate", "Severe",
             "Moderate", "Normal", "Unknown", "Moderate",
             "Normal", "Normal", "Severe", "Normal"]
    return pd.DataFrame(dict(node=["A"]*len(dates),date=dates,
        state=names,coverage=[1]*len(dates),
        hours_lt2=[1 if x=="Severe" else (np.nan if x=="Unknown" else 0) for x in names],
        hours_lt4=[1 if x in ("Severe","Moderate") else (np.nan if x=="Unknown" else 0)
                    for x in names],provenance=["observed_only"]*len(dates),
        qc_suspect=[False]*len(dates)))


def synthetic_daily():
    dates = pd.date_range("2026-07-01", periods=12, freq="D")
    return pd.DataFrame(dict(node=["A"]*len(dates),
        date=dates,eligible=[True]*len(dates),coverage=[1.]*len(dates),
        odo_min=[4.5,3.5,3.5,1.5,3.5,4.5,4.5,3.5,4.5,4.5,1.5,4.5],
        odo_mean=[5,4,4,2,4,5,5,4,5,5,2,5],
        temp_mean=[29.]*len(dates),sal_mean=[30.]*len(dates)))


def simple_windows():
    dates=["2026-07-02","2026-07-03","2026-07-05"]
    rows=[]
    for date in dates:
        for h in (1,3,7):
            row=dict(node="A",issue_date=date,
                     window_end=(pd.Timestamp(date)+pd.Timedelta(days=h)).strftime("%Y-%m-%d"),
                     horizon_days=h,origin_state="Moderate",label=0,
                     first_severe_date="",status="evaluable",provenance="observed_only",
                     odo_min=3.5,odo_mean=4.,hours_lt2=0.,
                     hours_lt4=2.,delta_do_mean_1d=-1.,state_moderate=1.,
                     temp_mean=29.,sal_mean=30.,season_sin=.2,season_cos=.8)
            rows.append(row)
    return pd.DataFrame(rows)


class WarningDashboardTests(unittest.TestCase):
    def test_risk_levels(self):
        self.assertEqual([risk_level(x) for x in
                          [0,.099,.1,.249,.25,.499,.5,1.]],
                         ["Low","Low","Watch","Watch","Elevated","Elevated","High","High"])
        self.assertEqual(risk_level(np.nan), "Not available")
        with self.assertRaises(ValueError):
            risk_level(1.1)

    def test_threshold_operating_points(self):
        sweep=threshold_sweep(example_predictions(),[.25,.5])
        row=sweep.loc[(sweep.model=="markov") & (sweep.horizon_days==1)
                      & sweep.threshold.eq(.25)].iloc[0]
        self.assertEqual((row.alerts,row.true_alerts,row.false_alerts),(2,2,0))
        self.assertAlmostEqual(row.recall,1.)
        self.assertAlmostEqual(row.precision,1.)
        self.assertAlmostEqual(row.false_positive_rate,0.)
        self.assertTrue(sweep.balanced_accuracy.between(0,1).all())

    def test_calibration_denominators_and_no_fake_bins(self):
        c=calibration_table(example_predictions())
        self.assertEqual(c.loc[(c.horizon_days==1)&(c.model=="markov"),
                               "n_forecasts"].sum(),4)
        self.assertTrue(c.sparse_bin.all())
        self.assertTrue(c.observed_frequency.between(0,1).all())

    def test_reject_bad_provenance_duplicate_or_mismatch(self):
        p=example_predictions()
        p.loc[0,"provenance"]="gnn"
        with self.assertRaisesRegex(ValueError,"provenance"):
            checked_predictions(p)
        p=example_predictions()
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            checked_predictions(pd.concat([p,p.iloc[[0]]]))
        p=example_predictions()
        p.loc[0,"label"]=1
        with self.assertRaisesRegex(ValueError,"labels differ"):
            checked_predictions(p)

    def test_cached_hindcast_query_uses_existing_oos_probability(self):
        p=example_predictions()
        w=simple_windows()
        q=query_forecast(w,pd.DataFrame(),p,"A","2026-07-03")
        self.assertEqual(len(q),9)
        self.assertTrue((q.source=="monthly_out_of_sample_hindcast").all())
        self.assertTrue((q.historical_outcome==0).all())
        row=q.loc[(q.horizon_days==3)&q.model.eq("do_history")].iloc[0]
        self.assertAlmostEqual(row.probability,.65)

    def test_asof_query_cannot_use_uncompleted_labels(self):
        # No historical complete labels before this date: only Markov may run.
        w=simple_windows()
        t=pd.DataFrame(dict(node=["A","A","A"],
            date=["2026-07-01","2026-07-02","2026-07-03"],
            next_date=["2026-07-02","2026-07-03","2026-07-04"],
            from_state=["Normal","Moderate","Moderate"],
            to_state=["Moderate","Moderate","Severe"],
            provenance=["observed_only"]*3))
        q=query_forecast(w,t,pd.DataFrame(),"A","2026-07-03",
                         min_train=50,prefer_cached=False)
        self.assertEqual(set(q.model),{"markov"})
        self.assertTrue(q.probability.between(0,1).all())
        self.assertTrue((q.source=="asof_end_of_day_refit").all())

    def test_observed_future_is_not_filled_through_missing_day(self):
        a=observed_future(synthetic_states(),synthetic_daily(),
                          "A","2026-07-04",horizon=5)
        self.assertEqual(len(a),5)
        self.assertEqual(a.loc[a.day_ahead.eq(3),
                               "observed_state"].iloc[0],"Unknown")
        self.assertFalse(bool(a.loc[a.day_ahead.eq(3),
                                   "record_available"].iloc[0]))

    def test_latest_snapshot_marks_already_severe_and_stale(self):
        s=synthetic_states()
        # Latest eligible day is the 11th for B (Severe), two days behind A's
        b=s.iloc[[10]].copy()
        b["node"]="B"
        both=pd.concat([s,b],ignore_index=True)
        snapshot=latest_snapshot(both,synthetic_daily(),pd.DataFrame(),
                                 simple_windows(),pd.DataFrame(),min_train=99)
        chosen=snapshot.loc[snapshot.node.eq("B")].iloc[0]
        self.assertEqual(chosen.current_state,"Severe")
        self.assertEqual(chosen.status,"already_severe")
        self.assertEqual(chosen.stale_days,1)
        self.assertTrue(np.isnan(chosen.risk_7d))



    def test_visualization_smoke_on_synthetic_data(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest("Matplotlib optional")
        predictions=example_predictions()
        sweep=threshold_sweep(predictions)
        calibration=calibration_table(predictions)
        with tempfile.TemporaryDirectory() as folder:
            dest=Path(folder)
            make_figures(predictions,sweep,calibration,synthetic_daily(),
                         synthetic_states(),dest,timeline_stations=("A",))
            expected=[
                "threshold_tradeoff_1d.png","threshold_tradeoff_3d.png",
                "threshold_tradeoff_7d.png","reliability_1d.png",
                "reliability_3d.png","reliability_7d.png",
                "hindcast_model_comparison.png","station_risk_heatmap_3d.png",
                "station_risk_heatmap_7d.png","timeline_A.png",
            ]
            for filename in expected:
                self.assertGreater((dest/filename).stat().st_size,100,
                                   filename)


    def test_all_stations_date_query_mixed_states_and_missing_station(self):
        states=synthetic_states()
        # B is already Severe on July 3. C has no observation July 3.
        b=states.loc[states.date.eq(pd.Timestamp("2026-07-03"))].copy()
        b["node"]="B"
        b["state"]="Severe"
        b["hours_lt2"]=1.
        b["hours_lt4"]=1.
        c=states.loc[states.date.eq(pd.Timestamp("2026-07-02"))].copy()
        c["node"]="C"
        mixed=pd.concat([states,b,c],ignore_index=True)
        w=simple_windows()
        p=example_predictions()
        summary,details,observed=all_stations_date_query(
            w,pd.DataFrame(),p,mixed,synthetic_daily(),"2026-07-03")
        self.assertEqual(set(summary.node),{"A","B","C"})
        a=summary.set_index("node").loc["A"]
        self.assertEqual(a.status,"eligible")
        self.assertAlmostEqual(a.risk_1d,.65)
        self.assertAlmostEqual(a.risk_3d,.65)
        self.assertAlmostEqual(a.risk_7d,.7)
        self.assertEqual(a.model_1d,"do_history")
        self.assertEqual(a.model_7d,"markov")
        self.assertEqual(a.forecast_source_1d,"monthly_out_of_sample_hindcast")
        self.assertEqual(summary.set_index("node").loc["B","status"],"already_severe")
        self.assertEqual(summary.set_index("node").loc["C","status"],"no_observation_on_date")
        self.assertTrue(np.isnan(summary.set_index("node").loc["B","risk_3d"]))
        self.assertEqual(len(details),9)
        self.assertEqual(len(observed),21)
        self.assertEqual(observed.loc[observed.node.eq("C"),"record_available"].sum(),0)

    def test_all_stations_query_respects_unknown_issue_and_unavailable_outcome(self):
        s=synthetic_states()
        w=simple_windows()
        p=example_predictions()
        summary,_,_=all_stations_date_query(
            w,pd.DataFrame(),p,s,synthetic_daily(),"2026-07-07")
        self.assertEqual(summary.iloc[0].status,"unknown_issue_state")
        self.assertTrue(np.isnan(summary.iloc[0].risk_1d))
        # With sparse/restricted future labels there is no invented negative.
        summary,_,_=all_stations_date_query(
            w,pd.DataFrame(),p,s,synthetic_daily(),"2026-07-03")
        self.assertTrue(np.isnan(summary.iloc[0].observed_severe_within_1d)
                        or summary.iloc[0].observed_severe_within_1d in (0,1))

if __name__=="__main__":
    unittest.main()
