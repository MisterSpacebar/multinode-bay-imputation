"""Visualization, alert tradeoffs, latest snapshots and date-query console for warnings.

Consumes Phase 5 observed-only, time-held-out predictions. Historical date
queries reuse the actual monthly hindcast when available; unscored dates can
be refitted as of end-of-day without fitting on future outcomes. Future
observations are used *only* to assess what happened, never as predictors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy.environmental_gam import prepare_daily
from trustworthy.transition_warning import (
    HORIZONS, MODELS, DO_FEATURES, ENV_FEATURES, prepare_states,
    warning_windows, validate_transitions, transition_matrix,
    markov_hit_probability, logistic_fit_predict,
)

DEFAULT_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
BEST_MODEL = {1: "do_history", 3: "do_history", 7: "markov"}
LEVELS = ((0.10, "Low"), (0.25, "Watch"), (0.50, "Elevated"),
          (float("inf"), "High"))
CALIBRATION_EDGES = (0, .10, .20, .30, .40, .50, .65, .80, 1.0)


def risk_level(value):
    if not np.isfinite(value):
        return "Not available"
    if not 0 <= value <= 1:
        raise ValueError("Probabilities must be within 0..1")
    return next(label for cutoff, label in LEVELS if value < cutoff)


def checked_predictions(predictions):
    required = {"node", "issue_date", "horizon_days", "model",
                "label", "probability", "provenance"}
    if not required.issubset(predictions):
        raise ValueError(f"Missing hindcast columns: {sorted(required-set(predictions))}")
    p = predictions.copy()
    if p.provenance.isna().any() or not p.provenance.eq("observed_only").all():
        raise ValueError("Non-observed hindcast provenance")
    if p.node.isna().any():
        raise ValueError("Missing station name")
    p["node"] = p.node.astype(str)
    p["issue_date"] = pd.to_datetime(p.issue_date, errors="raise")
    if p.issue_date.isna().any() or p.issue_date.dt.tz is not None or not p.issue_date.dt.normalize().eq(p.issue_date).all():
        raise ValueError("Issue dates must be local calendar dates")
    if not p.horizon_days.isin(HORIZONS).all() or not p.model.isin(MODELS).all():
        raise ValueError("Unsupported warning horizon/model")
    p["probability"] = pd.to_numeric(p.probability, errors="coerce")
    p["label"] = pd.to_numeric(p.label, errors="coerce")
    if not p.probability.between(0, 1).all() or not p.label.isin((0, 1)).all():
        raise ValueError("Scored predictions require finite probabilities and binary outcomes")
    if p.duplicated(["node", "issue_date", "horizon_days", "model"]).any():
        raise ValueError("Duplicate station/day/horizon/model hindcast")
    groups = p.groupby(["node", "issue_date", "horizon_days"])
    if not groups.model.nunique().eq(len(MODELS)).all() or groups.label.nunique().gt(1).any():
        raise ValueError("Hindcast must have three matched models and identical labels")
    return p.sort_values(["node", "issue_date", "horizon_days", "model"]).reset_index(drop=True)


def threshold_sweep(predictions, thresholds=DEFAULT_THRESHOLDS):
    p = checked_predictions(predictions)
    grid = tuple(float(t) for t in thresholds)
    if not grid or any(not 0 < t < 1 for t in grid):
        raise ValueError("Alert thresholds must be between 0 and 1")
    rows = []
    for (h, model), g in p.groupby(["horizon_days", "model"], sort=True):
        y = g.label.to_numpy(dtype=int)
        prob = g.probability.to_numpy(float)
        for t in sorted(set(grid)):
            active = prob >= t
            tp = int(np.sum(active & (y == 1)))
            fp = int(np.sum(active & (y == 0)))
            tn = int(np.sum(~active & (y == 0)))
            fn = int(np.sum(~active & (y == 1)))
            precision = tp / (tp + fp) if tp + fp else np.nan
            recall = tp / (tp + fn) if tp + fn else np.nan
            specificity = tn / (tn + fp) if tn + fp else np.nan
            fpr = fp / (fp + tn) if fp + tn else np.nan
            rows.append(dict(horizon_days=int(h), model=model, threshold=t,
                n_forecasts=len(g), n_positive=int(y.sum()),
                alerts=tp + fp, true_alerts=tp, false_alerts=fp,
                missed_events=fn, true_negatives=tn,
                precision=precision, recall=recall,
                false_positive_rate=fpr,
                false_alarm_fraction=(fp/(tp+fp) if tp+fp else np.nan),
                specificity=specificity,
                f1=(2*precision*recall/(precision+recall)
                    if np.isfinite(precision) and np.isfinite(recall) and precision+recall else np.nan),
                balanced_accuracy=(.5*(recall+specificity)
                    if np.isfinite(recall) and np.isfinite(specificity) else np.nan),
                alert_fraction=(tp+fp)/len(g)))
    return pd.DataFrame(rows)


def calibration_table(predictions, edges=CALIBRATION_EDGES):
    p = checked_predictions(predictions)
    bins = np.asarray(edges, float)
    if not np.isclose(bins[0], 0) or not np.isclose(bins[-1], 1) or np.any(np.diff(bins) <= 0):
        raise ValueError("Calibration bins must cover 0..1 in increasing order")
    rows = []
    for (h, model), g in p.groupby(["horizon_days", "model"]):
        indexes = np.clip(np.searchsorted(bins, g.probability.to_numpy(float), side="right")-1,
                          0, len(bins)-2)
        for i in range(len(bins)-1):
            sub = g.iloc[np.where(indexes == i)[0]]
            if sub.empty:
                continue
            rows.append(dict(horizon_days=int(h), model=model,
                bin_start=float(bins[i]), bin_end=float(bins[i+1]),
                n_forecasts=len(sub), n_events=int(sub.label.sum()),
                mean_predicted=float(sub.probability.mean()),
                observed_frequency=float(sub.label.mean()),
                sparse_bin=bool(len(sub) < 10)))
    return pd.DataFrame(rows)


def _date(value):
    d = pd.Timestamp(value)
    if pd.isna(d) or d.tzinfo is not None or d != d.normalize():
        raise ValueError("Supply date as YYYY-MM-DD")
    return d


def _empty_query_result():
    return pd.DataFrame(columns=("node","issue_date","horizon_days","origin_state",
        "model","probability","alert_level","historical_outcome",
        "first_severe_date","n_train","n_train_events","source","status"))


def _outcome_lookup(windows, station, date):
    w = windows.copy()
    w["issue_date"] = pd.to_datetime(w.issue_date)
    s = w.loc[w.node.eq(station) & w.issue_date.eq(date)]
    return {int(r.horizon_days): r for r in s.itertuples(index=False)}


def query_forecast(windows, transitions, predictions, station, date,
                   alpha=10., prior=20., min_train=60, min_events=3,
                   prefer_cached=True):
    """Predict for an historical issue day and compare with observed future.

    Cached monthly-out-of-sample forecasts are preferred. Otherwise fit using
    only fully completed labels before *issue day* and transition destinations
    at or before issue day. The latter is valid at end-of-day issue time.
    """
    date = _date(date)
    w = windows.copy()
    w["issue_date"] = pd.to_datetime(w.issue_date)
    w["window_end"] = pd.to_datetime(w.window_end)
    w["node"] = w.node.astype(str)
    outcomes = _outcome_lookup(w, station, date)
    if not outcomes:
        return _empty_query_result()
    if any(row.origin_state not in ("Normal", "Moderate") for row in outcomes.values()):
        return _empty_query_result()
    cached = checked_predictions(predictions) if len(predictions) else _empty_query_result()
    saved = cached.loc[cached.node.eq(station) & cached.issue_date.eq(date)] if len(cached) else cached
    # A cached row is already an out-of-sample month-block hindcast.
    result = []
    if prefer_cached and len(saved) == 3*len(HORIZONS):
        for r in saved.itertuples(index=False):
            details = outcomes[int(r.horizon_days)]
            result.append(dict(node=station, issue_date=date.date().isoformat(),
                horizon_days=int(r.horizon_days), origin_state=r.origin_state,
                model=r.model, probability=float(r.probability),
                alert_level=risk_level(float(r.probability)),
                historical_outcome=(int(details.label) if details.status=="evaluable" else np.nan),
                first_severe_date=(details.first_severe_date if details.status=="evaluable" else ""),
                n_train=int(r.n_train), n_train_events=int(r.n_train_events),
                source="monthly_out_of_sample_hindcast", status=details.status))
        return pd.DataFrame(result)

    t = transitions.copy()
    t["next_date"] = pd.to_datetime(t.next_date)
    if "provenance" not in t or not t.provenance.eq("observed_only").all():
        raise ValueError("Transition history must be observed_only")
    history = t.loc[t.next_date.le(date)]
    if history.empty:
        return _empty_query_result()
    pmat = transition_matrix(history, station, prior)
    valid = w.status.eq("evaluable") & w.label.notna()
    numeric = w[list(ENV_FEATURES)].apply(pd.to_numeric, errors="coerce")
    complete = valid & np.isfinite(numeric).all(axis=1)
    rows_by_model = {model: [] for model in MODELS}
    for h in HORIZONS:
        if h not in outcomes:
            continue
        target = outcomes[h]
        issue = w.loc[w.node.eq(station) & w.issue_date.eq(date) &
                      w.horizon_days.eq(h)].copy()
        if not np.isfinite(issue[list(ENV_FEATURES)].apply(
                pd.to_numeric, errors="coerce").to_numpy(float)).all():
            continue
        train = w.loc[complete & w.horizon_days.eq(h)
                      & w.window_end.lt(date)].copy()
        events = int(train.label.sum())
        enough = (len(train) >= min_train and
                  min(events, len(train)-events) >= min_events and
                  station in set(train.node))
        guesses = {"markov": markov_hit_probability(pmat, target.origin_state, h)}
        if enough:
            for model, feats in (("do_history", DO_FEATURES),
                                 ("environmental", ENV_FEATURES)):
                try:
                    guesses[model] = float(logistic_fit_predict(
                        train, issue, feats, alpha)[0])
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    pass
        for model, prob in guesses.items():
            rows_by_model[model].append(dict(node=station, issue_date=date.date().isoformat(),
                horizon_days=h, origin_state=target.origin_state,
                model=model, probability=float(prob),
                alert_level=risk_level(float(prob)),
                historical_outcome=(int(target.label) if target.status=="evaluable" else np.nan),
                first_severe_date=(target.first_severe_date if target.status=="evaluable" else ""),
                n_train=len(train), n_train_events=events,
                source="asof_end_of_day_refit", status=target.status))
    # Enforce the logically nested horizon risks *within each model* without
    # seeing the future outcome. No comparison is made if the label is unknown.
    for model, records in rows_by_model.items():
        maximum = 0.
        for record in sorted(records, key=lambda r: r["horizon_days"]):
            maximum = max(maximum, record["probability"])
            record["probability"] = maximum
            record["alert_level"] = risk_level(maximum)
            result.append(record)
    return pd.DataFrame(result) if result else _empty_query_result()


def observed_future(states, daily, node, date, horizon=7):
    """Actual future daily observations are exclusively for evaluation."""
    start = _date(date)
    s = prepare_states(states)
    d = prepare_daily(daily)
    m = s.loc[s.node.eq(node), ["date","state","hours_lt2","coverage"]]
    m = m.merge(d[["node","date","odo_min","qc_valid"]].loc[
        d.node.eq(node)].drop(columns="node"), on="date", how="left",
        validate="one_to_one")
    indexed = m.set_index("date")
    dates = pd.date_range(start+pd.Timedelta(days=1), periods=horizon, freq="D")
    future = indexed.reindex(dates)
    result = future.reset_index().rename(columns={"index":"date"})
    result.insert(0,"node",node)
    result["date"] = result.date.dt.strftime("%Y-%m-%d")
    result["day_ahead"] = np.arange(1, horizon+1)
    result["observed_state"] = result.state.fillna("Unknown")
    result["observed_min_do_mgL"] = result.odo_min.where(result.qc_valid.eq(True))
    result["record_available"] = result.observed_state.isin(("Normal","Moderate","Severe"))
    return result[["node","date","day_ahead","observed_state",
                   "observed_min_do_mgL","hours_lt2","record_available"]]


def latest_snapshot(states, daily, transitions, windows, predictions=None,
                    alpha=10., prior=20., min_train=60, min_events=3):
    """Latest eligible *file* date per station, never labeled real-time data."""
    s = prepare_states(states)
    global_date = s.date.max()
    rows = []
    preds = predictions if predictions is not None else pd.DataFrame()
    for node, g in s.groupby("node", sort=True):
        known = g.loc[g.state.isin(("Normal", "Moderate", "Severe"))]
        if known.empty:
            rows.append(dict(node=node, issue_date="", current_state="Unknown",
                stale_days=np.nan, status="no_eligible_observed_day"))
            continue
        current = known.sort_values("date").iloc[-1]
        d = current.date
        item = dict(node=node, issue_date=d.date().isoformat(),
                    current_state=current.state, stale_days=int((global_date-d).days),
                    status="already_severe" if current.state=="Severe" else "eligible")
        if current.state in ("Normal","Moderate"):
            forecast = query_forecast(windows, transitions, preds, node, d, alpha,
                                      prior,min_train,min_events,prefer_cached=False)
            for h in HORIZONS:
                model = BEST_MODEL[h]
                rows_h = forecast.loc[forecast.horizon_days.eq(h) & forecast.model.eq(model)]
                item[f"risk_{h}d"] = (float(rows_h.probability.iloc[0])
                                      if len(rows_h) else np.nan)
                item[f"model_{h}d"] = model
            item["alert_level_3d"] = risk_level(item["risk_3d"])
            if not np.isfinite(item["risk_3d"]):
                item["status"] = "insufficient_history_or_features"
        else:
            for h in HORIZONS:
                item[f"risk_{h}d"] = np.nan
                item[f"model_{h}d"] = "not_applicable"
            item["alert_level_3d"] = "Already Severe"
        rows.append(item)
    return pd.DataFrame(rows)


def _plotting():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("NOTE: matplotlib not installed; CSV outputs remain available")
        return None


def make_figures(preds, sweep, calibrations, daily, states, directory,
                 timeline_stations=(), max_stations=3):
    plt = _plotting()
    if plt is None or preds.empty:
        return
    directory.mkdir(parents=True, exist_ok=True)
    p = checked_predictions(preds)
    model_colors = {"markov":"tab:blue", "do_history":"tab:orange",
                    "environmental":"tab:green"}

    for h in HORIZONS:
        g = sweep.loc[sweep.horizon_days.eq(h)]
        fig, axes = plt.subplots(1, 2, figsize=(11,4))
        for model, group in g.groupby("model"):
            group = group.sort_values("threshold")
            axes[0].plot(group.threshold,group.recall,marker="o",
                         label=model,color=model_colors[model])
            axes[0].plot(group.threshold,group.precision,linestyle="--",
                         color=model_colors[model],alpha=.75)
            axes[1].plot(group.threshold,group.false_alarm_fraction,
                         marker="o",label=model,color=model_colors[model])
        axes[0].set(title=f"{h}-day: recall (solid) and precision (dashed)",
                    xlabel="Alert cutoff",ylabel="Proportion",ylim=(0,1.05))
        axes[1].set(title="Fraction of issued alerts that were false",
                    xlabel="Alert cutoff",ylabel="False alert fraction",ylim=(0,1.05))
        for ax in axes:
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(directory/f"threshold_tradeoff_{h}d.png",dpi=160)
        plt.close(fig)

        g = calibrations.loc[calibrations.horizon_days.eq(h)]
        fig, ax = plt.subplots(figsize=(6,5))
        ax.plot([0,1],[0,1],linestyle=":",color="gray",label="Ideal calibration")
        for model, sub in g.groupby("model"):
            ax.plot(sub.mean_predicted,sub.observed_frequency,"o-",
                    label=model,color=model_colors[model])
        ax.set(title=f"{h}-day retrospective reliability (small bins unstable)",
               xlabel="Average forecast probability",ylabel="Observed severe frequency",
               xlim=(0,1),ylim=(0,1))
        ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(directory/f"reliability_{h}d.png",dpi=160)
        plt.close(fig)

    score = p.groupby(["horizon_days","model"]).apply(
        lambda g: np.mean((g.probability-g.label)**2),
        include_groups=False).rename("brier").reset_index()
    fig, ax = plt.subplots(figsize=(8,4))
    x = np.arange(3); width = .26
    for k, model in enumerate(MODELS):
        vals = [float(score.loc[(score.horizon_days==h)&(score.model==model),
                                "brier"].iloc[0]) for h in HORIZONS]
        ax.bar(x+(k-1)*width,vals,width,label=model,color=model_colors[model])
    ax.set_xticks(x,[f"{h} days" for h in HORIZONS])
    ax.set(ylabel="Hindcast Brier score (lower better)",title="Matched historical warning models")
    ax.legend();fig.tight_layout();fig.savefig(directory/"hindcast_model_comparison.png",dpi=160)
    plt.close(fig)

    best = p.loc[p.apply(lambda r: r.model==BEST_MODEL[int(r.horizon_days)],axis=1)]
    for h in (3,7):
        b = best.loc[best.horizon_days.eq(h)]
        if b.empty:
            continue
        mat = b.pivot(index="node",columns="issue_date",values="probability")
        mat = mat.sort_index()
        fig, ax = plt.subplots(figsize=(12,max(3, min(10,len(mat)*.38+1))))
        values = np.ma.masked_invalid(mat.to_numpy(float))
        im = ax.imshow(values,aspect="auto",interpolation="nearest",vmin=0,vmax=1)
        xt = np.unique(np.linspace(0,len(mat.columns)-1,min(9,len(mat.columns)),dtype=int))
        ax.set_xticks(xt,[mat.columns[i].strftime("%Y-%m-%d") for i in xt],
                      rotation=35,ha="right",fontsize=8)
        ax.set_yticks(range(len(mat.index)),[str(s)[-34:] for s in mat.index],fontsize=7)
        ax.set(title=f"{h}-day hindcast risk by station (missing cells = unscored)")
        fig.colorbar(im,ax=ax,label="Predicted P(any Severe within horizon)")
        fig.tight_layout()
        fig.savefig(directory/f"station_risk_heatmap_{h}d.png",dpi=160)
        plt.close(fig)

    d = prepare_daily(daily)
    s = prepare_states(states)
    if timeline_stations:
        nodes = list(timeline_stations)
    else:
        nodes = best.loc[best.horizon_days.eq(3)].node.value_counts().head(max_stations).index.tolist()
    for node in nodes:
        b = best.loc[best.node.eq(node)].copy()
        if b.empty:
            continue
        minimum, maximum = b.issue_date.min(), b.issue_date.max()
        daily_part = d.loc[d.node.eq(node) & d.date.between(minimum,maximum)]
        state_part = s.loc[s.node.eq(node) & s.date.between(minimum,maximum)]
        fig, axes = plt.subplots(2,1,figsize=(11,7),sharex=True,
                                 gridspec_kw={"height_ratios":[2,2]})
        axes[0].plot(daily_part.date,daily_part.odo_min,".-",
                     label="Observed minimum DO")
        axes[0].axhline(2,linestyle="--",color="tab:red",label="Severe threshold")
        axes[0].axhline(4,linestyle=":",color="tab:orange",label="Moderate threshold")
        severe=state_part.loc[state_part.state.eq("Severe")]
        axes[0].scatter(severe.date,np.full(len(severe),2.0),
                        color="tab:red",marker="x",s=30,label="Observed Severe day")
        axes[0].set(ylabel="DO (mg/L)",title=f"Observed oxygen and historical warning: {node}",
                    ylim=(0,max(8, min(20,float(daily_part.odo_min.max())+1))))
        for h in HORIZONS:
            g=b.loc[b.horizon_days.eq(h)].sort_values("issue_date")
            axes[1].plot(g.issue_date,g.probability,".-",alpha=.7,label=f"{h}-day ({BEST_MODEL[h]})")
        axes[1].set(ylabel="P(any Severe within horizon)",xlabel="Historical issue date",
                    ylim=(0,1.05))
        for ax in axes:
            ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.autofmt_xdate();fig.tight_layout()
        suffix=re.sub(r"[^a-zA-Z0-9_-]+","_",node)[-65:]
        fig.savefig(directory/f"timeline_{suffix}.png",dpi=160)
        plt.close(fig)


def _write_query(query, actual, out_dir):
    if len(query):
        query.to_csv(out_dir/"hindcast_date_query.csv",index=False)
    if len(actual):
        actual.to_csv(out_dir/"hindcast_actual_future.csv",index=False)


def show_query(station, date, windows, states, daily, transitions, preds,
               out_dir, alpha, prior, min_train, min_events):
    q = query_forecast(windows, transitions, preds, station, date,
                       alpha,prior,min_train,min_events)
    print(f"\nHistorical issue: {station} | {date}")
    if q.empty:
        print("No eligible Normal/Moderate issue-day at that station/date; cannot forecast.")
        return
    print(q[["horizon_days","model","probability","alert_level",
             "historical_outcome","status","source"]].to_string(index=False,
                  na_rep="unavailable",float_format=lambda v:f"{v:.3f}"))
    actual = observed_future(states,daily,station,date)
    print("\nObserved following seven calendar days (Unknown = unscorable):")
    print(actual[["day_ahead","date","observed_state","observed_min_do_mgL"]]
          .to_string(index=False,na_rep="missing",float_format=lambda v:f"{v:.2f}"))
    _write_query(q,actual,out_dir)
    print("\nA historical label is known only if ALL days in its horizon have eligible observations.")
    print("These are research hindcasts, not live warnings or verified alert thresholds.")


def main():
    ap=argparse.ArgumentParser(
        description="Hypoxia warning charts, latest observations and historical date queries")
    ap.add_argument("--predictions",type=Path,
                    default=Path("trustworthy_outputs/transition_warning/hindcast_predictions.csv"))
    ap.add_argument("--windows",type=Path,
                    default=Path("trustworthy_outputs/transition_warning/warning_windows.csv"))
    ap.add_argument("--daily",type=Path,
                    default=Path("trustworthy_outputs/daily_targets.csv"))
    ap.add_argument("--states",type=Path,
                    default=Path("trustworthy_outputs/episode_dynamics/station_day_states.csv"))
    ap.add_argument("--transitions",type=Path,
                    default=Path("trustworthy_outputs/episode_dynamics/observed_transitions.csv"))
    ap.add_argument("--out-dir",type=Path,
                    default=Path("trustworthy_outputs/transition_warning/dashboard"))
    ap.add_argument("--threshold-grid",nargs="+",type=float,default=list(DEFAULT_THRESHOLDS))
    ap.add_argument("--timeline-stations",nargs="*",default=[])
    ap.add_argument("--max-timeline-stations",type=int,default=3)
    ap.add_argument("--query-node",type=str)
    ap.add_argument("--query-date",type=str,help="Historical issue date YYYY-MM-DD")
    ap.add_argument("--interactive",action="store_true",
                    help="Prompt for station and historical date repeatedly")
    ap.add_argument("--no-figures",action="store_true")
    ap.add_argument("--no-snapshot",action="store_true")
    ap.add_argument("--alpha",type=float,default=10.)
    ap.add_argument("--station-prior",type=float,default=20.)
    ap.add_argument("--min-train",type=int,default=60)
    ap.add_argument("--min-events",type=int,default=3)
    args=ap.parse_args()
    if bool(args.query_node)!=bool(args.query_date):
        ap.error("--query-node and --query-date must be used together")
    if args.max_timeline_stations<0:
        ap.error("--max-timeline-stations must be nonnegative")
    inputs = {key:pd.read_csv(getattr(args,key))
              for key in ("predictions","windows","daily","states","transitions")}
    p = checked_predictions(inputs["predictions"])
    s = prepare_states(inputs["states"])
    t = validate_transitions(inputs["transitions"],s)
    w = inputs["windows"]
    if not {"node","issue_date","window_end","horizon_days","status",
            "origin_state","label",*ENV_FEATURES}.issubset(w):
        raise ValueError("warning_windows is missing required historical features")
    args.out_dir.mkdir(parents=True,exist_ok=True)
    sweep=threshold_sweep(p,args.threshold_grid)
    calibration=calibration_table(p)
    sweep.to_csv(args.out_dir/"warning_threshold_summary.csv",index=False)
    calibration.to_csv(args.out_dir/"calibration_summary.csv",index=False)
    # Model-specific threshold tradeoffs are in the summary; export a
    # separate file only for the selected prototype operating model/horizon.
    sweep.loc[sweep.apply(lambda r:r.model==BEST_MODEL[int(r.horizon_days)],axis=1)
              ].to_csv(args.out_dir/"warning_threshold_selected_models.csv",index=False)
    if not args.no_snapshot:
        snapshot=latest_snapshot(s,inputs["daily"],t,w,p,args.alpha,
                                 args.station_prior,args.min_train,args.min_events)
        snapshot.to_csv(args.out_dir/"latest_warning_snapshot.csv",index=False)
        print(f"Latest observed-file snapshot: {len(snapshot)} stations; never labeled live")
    if not args.no_figures:
        make_figures(p,sweep,calibration,inputs["daily"],s,
                     args.out_dir/"figures",args.timeline_stations,
                     args.max_timeline_stations)
    meta={key+"_sha256":hashlib.sha256(getattr(args,key).read_bytes()).hexdigest()
          for key in inputs}
    meta.update(dict(thresholds=args.threshold_grid,
        chosen_models=BEST_MODEL,alert_bands={"Low":"<10%","Watch":"10–<25%",
            "Elevated":"25–<50%","High":">=50%"},
        latest_definition="latest eligible per station in file; may be months old, not live",
        query_sources=["cached monthly out-of-sample hindcast","strict as-of end-of-day refit"],
        limits="retrospective, overlapping windows, threshold selection post hoc, calibration exploratory",
        provenance="observed_only"))
    (args.out_dir/"dashboard_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    if args.query_node:
        show_query(args.query_node,args.query_date,w,s,inputs["daily"],t,p,
                   args.out_dir,args.alpha,args.station_prior,args.min_train,args.min_events)
    if args.interactive:
        nodes = sorted(s.node.unique())
        print("\nInteractive historical hindcast lookup. Available station names:")
        for node in nodes:
            print(" ",node)
        while True:
            station=input("\nStation (blank to quit): ").strip()
            if not station:
                break
            if station not in nodes:
                print("Unknown station; copy a station name from the list.")
                continue
            date=input("Historical issue date (YYYY-MM-DD): ").strip()
            try:
                show_query(station,str(_date(date).date()),w,s,inputs["daily"],t,p,
                           args.out_dir,args.alpha,args.station_prior,args.min_train,args.min_events)
            except (ValueError,KeyError) as e:
                print("Unable to query:",e)
    print(f"Warning dashboard saved to {args.out_dir}")


if __name__=="__main__":
    main()
