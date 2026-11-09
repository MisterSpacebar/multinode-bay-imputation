"""Observed-only environmental GAM pilots for daily DO severity and next-day recovery.

Retrospective associations are NOT validated early-warning signals. The time-
blocked comparison uses only training outcomes completed before each test month.
No GNN reconstructed observations or regional forcing are used in this phase.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PRED_COLUMNS = ['outcome', 'threshold_mgL', 'node', 'date', 'test_month',
                'model', 'actual', 'predicted', 'qc_suspect', 'provenance']
SCORE_COLUMNS = ['outcome', 'threshold_mgL', 'model', 'n_paired', 'n_stations',
                 'n_test_months', 'n_positive', 'mae', 'rmse', 'brier',
                 'paired_baseline_score', 'skill_vs_baseline']
FIT_COLUMNS = ['outcome', 'threshold_mgL', 'model', 'n_rows', 'n_stations',
               'n_positive', 'aic', 'edf_total', 'converged', 'smoothing_alpha',
               'caution']
DAILY_FEATURES = ['temp_mean', 'sal_mean']


def _flag(x: pd.Series, field: str) -> pd.Series:
    values = x.astype('string').str.strip().str.lower().map(
        {'1': True, '0': False, 'true': True, 'false': False})
    if values.isna().any():
        raise ValueError(f'{field}: missing or invalid boolean')
    return values.astype(bool)


def _dates(x, field):
    result = pd.to_datetime(x, errors='raise')
    if result.isna().any() or result.dt.tz is not None or (result != result.dt.normalize()).any():
        raise ValueError(f'{field} must contain valid timezone-naive local dates')
    return result


def _season_terms(frame):
    doy = frame['date'].dt.dayofyear.to_numpy(float)
    # Calendar harmonics approximate season; not a cyclic penalized DOY smooth.
    return np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)


def prepare_daily(daily: pd.DataFrame, min_coverage=.95) -> pd.DataFrame:
    """QC-qualified records, with no implicit fill for absent temp or sal."""
    required = {'node','date','eligible','coverage','odo_min','odo_mean',
                'temp_mean','sal_mean'}
    if not required.issubset(daily):
        raise ValueError(f'daily_targets missing: {sorted(required-set(daily))}')
    if not 0 < min_coverage <= 1:
        raise ValueError('Invalid min_coverage')
    d = daily.copy()
    if 'provenance' in d and not d.provenance.eq('observed_only').all():
        raise ValueError('Daily table contains non-observed provenance')
    if d.node.isna().any():
        raise ValueError('Missing node')
    d['node'] = d.node.astype(str)
    d['date'] = _dates(d.date, 'daily date')
    if d.duplicated(['node','date']).any():
        raise ValueError('Duplicate station/date')
    d['eligible'] = _flag(d.eligible, 'eligible')
    for col in ['coverage','odo_min','odo_mean','temp_mean','sal_mean']:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    if ((d.coverage.notna()) & (~d.coverage.between(0,1))).any():
        raise ValueError('Coverage outside 0..1')
    d['qc_suspect'] = False
    for col in ['n_flatline','n_spike_suspect','n_zero_review']:
        if col in d:
            d['qc_suspect'] |= pd.to_numeric(d[col],errors='coerce').fillna(0).gt(0)
    d['qc_valid'] = d.eligible & d.coverage.ge(min_coverage) & d.odo_mean.notna()
    d = d.sort_values(['node','date']).reset_index(drop=True)
    # A trend requires adjacent eligible observed dates at the SAME station.
    previous = d.groupby('node').shift(1)
    adjacent = (d.date - previous.date).dt.days.eq(1)
    d['delta_do_mean_1d'] = (d.odo_mean - previous.odo_mean).where(
        d.qc_valid & previous.qc_valid.eq(True) & adjacent)
    d['provenance'] = 'observed_only'
    return d


def severity_rows(daily: pd.DataFrame, response='odo_min') -> pd.DataFrame:
    if response not in ('odo_min','odo_mean'):
        raise ValueError('Unsupported DO response')
    d = daily.loc[daily.qc_valid].copy()
    for f in DAILY_FEATURES+[response]:
        d = d.loc[np.isfinite(d[f])]
    d = d.loc[d[response].between(0,20)].copy()
    d['actual'] = d[response]
    d['threshold_mgL'] = np.nan
    d['outcome'] = response
    return d


def recovery_rows(risk: pd.DataFrame, daily: pd.DataFrame,
                  include_trend: bool = True) -> pd.DataFrame:
    """Observed issue-day features only; NEXT-day label is never joined as a feature.

    Train/test validation later requires completed next-day labels strictly
    earlier than the test fold's first issue date.
    """
    req = {'episode_id','node','date','next_date','threshold_mgL',
           'log_elapsed_days','hours_below_threshold','next_day_recovery',
           'provenance'}
    if not req.issubset(risk):
        raise ValueError(f'risk table missing: {sorted(req-set(risk))}')
    if not risk.provenance.eq('observed_only').all():
        raise ValueError('Recovery risk table contains non-observed values')
    r = risk.copy()
    if 'qc_suspect' not in r:
        r['qc_suspect'] = False
    else:
        r['qc_suspect'] = _flag(r.qc_suspect,'qc_suspect')
    r['date'] = _dates(r.date,'risk issue date')
    r['next_date'] = _dates(r.next_date,'risk next date')
    if not (r.next_date-r.date).dt.days.eq(1).all():
        raise ValueError('Next-day outcome date must immediately follow issue date')
    if r.duplicated(['episode_id','date','threshold_mgL']).any():
        raise ValueError('Duplicate episode-day')
    if not r.threshold_mgL.isin([2,4]).all():
        raise ValueError('Unrecognized threshold')
    r['next_day_recovery'] = _flag(r.next_day_recovery,'next_day_recovery')
    r = r.merge(daily.drop(columns=['provenance','qc_suspect'],errors='ignore'),
                on=['node','date'],how='left',validate='many_to_one',suffixes=('','_daily'))
    # The same-day complete-case population is shared by baseline and GAM.
    covariates = DAILY_FEATURES+['odo_mean','log_elapsed_days','hours_below_threshold']
    if include_trend:
        covariates.append('delta_do_mean_1d')
    r = r.loc[r.qc_valid.fillna(False)].copy()
    for f in covariates:
        r[f] = pd.to_numeric(r[f],errors='coerce')
        r = r.loc[np.isfinite(r[f])]
    r = r.loc[r.hours_below_threshold.between(0,24) & r.hours_below_threshold.gt(0)]
    r['hours_below_per_6h'] = r.hours_below_threshold/6
    r['actual'] = r.next_day_recovery.astype(int)
    r['outcome'] = 'next_day_recovery'
    r['provenance'] = 'observed_only'
    return r.reset_index(drop=True)


def _matrix(frame: pd.DataFrame, stations: list[str], outcome: str):
    from pandas.api.types import CategoricalDtype
    x = pd.DataFrame(index=range(len(frame)))
    x['intercept'] = 1.
    sin, cos = _season_terms(frame)
    x['season_sin'], x['season_cos'] = sin, cos
    if outcome == 'next_day_recovery':
        x['log_elapsed_days'] = frame.log_elapsed_days.to_numpy(float)
        x['hours_below_per_6h'] = frame.hours_below_per_6h.to_numpy(float)
    cat = frame.node.astype(CategoricalDtype(categories=stations))
    if cat.isna().any():
        raise ValueError('Unseen station in validation fold')
    for station in stations[1:]:
        x[f'station::{station}'] = cat.eq(station).to_numpy(dtype=float)
    return x


def _fit(train: pd.DataFrame, test: pd.DataFrame, outcome: str,
         model: str, alpha: float, include_trend: bool):
    try:
        import statsmodels.api as sm
        from statsmodels.gam.api import GLMGam, BSplines
    except ImportError as exc:
        raise RuntimeError('GAM requires statsmodels and scipy') from exc
    stations = sorted(train.node.unique().tolist())
    family = (sm.families.Binomial(link=sm.families.links.CLogLog())
              if outcome=='next_day_recovery' else sm.families.Gaussian())
    xtr = _matrix(train,stations,outcome)
    xte = _matrix(test,stations,outcome)
    if model == 'environmental_gam' and outcome=='next_day_recovery' and include_trend:
        xtr['delta_do_mean_1d'] = train.delta_do_mean_1d.to_numpy(float)
        xte['delta_do_mean_1d'] = test.delta_do_mean_1d.to_numpy(float)
    y = train.actual.to_numpy(float)
    if model == 'baseline':
        fitted = sm.GLM(y,xtr,family=family).fit(maxiter=150,disp=0)
        predicted = fitted.predict(xte)
    else:
        inputs = train[DAILY_FEATURES].copy()
        if any(inputs[f].nunique()<5 for f in DAILY_FEATURES):
            raise ValueError('Not enough unique environmental measurements for smooth')
        smoother=BSplines(inputs,df=[4,4],degree=[3,3],include_intercept=False)
        fitted=GLMGam(y,exog=xtr,smoother=smoother,alpha=[alpha,alpha],
                       family=family).fit(maxiter=150,disp=0)
        # Holdout values outside training knots are clipped, not used to
        # calculate new knots or to extrapolate unsupported smooths.
        new_smooth=test[DAILY_FEATURES].copy()
        for feature in DAILY_FEATURES:
            lo,hi=inputs[feature].min(),inputs[feature].max()
            new_smooth[feature]=new_smooth[feature].clip(lo+1e-8,hi-1e-8)
        predicted=fitted.predict(exog=xte,exog_smooth=new_smooth)
    predicted=np.asarray(predicted,dtype=float)
    if not np.isfinite(predicted).all():
        raise ValueError('Model returned non-finite predictions')
    if outcome=='next_day_recovery':
        predicted=np.clip(predicted,1e-6,1-1e-6)
    else:
        predicted=np.clip(predicted,0,20)
    return fitted,predicted


def rolling_backtest(data: pd.DataFrame, outcome: str, alpha=10.,
                     min_train=100, min_cases=8,
                     include_trend=True):
    """Expanding-origin validation by full test month, common complete cases.

    No target from an incomplete recovery window enters a training fold.
    A new station is not forecast until previously seen in training.
    """
    if alpha<=0 or min_train<20 or min_cases<1:
        raise ValueError('Invalid GAM/backtest configuration')
    if data.empty:
        return pd.DataFrame(columns=PRED_COLUMNS), pd.DataFrame()
    d = data.copy().sort_values(['date','node'])
    d['test_month'] = d.date.dt.to_period('M').astype(str)
    out, diagnostics=[],[]
    for month in sorted(d.test_month.unique()):
        first=pd.Period(month).start_time
        # Historical outcomes must already be completed by the fold start.
        train=d.loc[(d.next_date.lt(first) if outcome=='next_day_recovery'
                     else d.date.lt(first))].copy()
        test=d.loc[d.test_month.eq(month)].copy()
        stations=set(train.node.unique())
        test=test.loc[test.node.isin(stations)].copy()
        reason='evaluable'
        if len(train)<min_train or len(test)<min_cases:
            reason='insufficient_history_or_test'
        elif outcome=='next_day_recovery' and (
            train.actual.sum()<5 or len(train)-train.actual.sum()<5):
            reason='insufficient_recovery_classes'
        elif len(stations)<1:
            reason='no_seen_station'
        if reason=='evaluable':
            try:
                candidates={}
                for model in ('baseline','environmental_gam'):
                    _,p=_fit(train,test,outcome,model,alpha,include_trend)
                    candidates[model]=p
                for model,pred in candidates.items():
                    for row,p in zip(test.itertuples(index=False),pred):
                        out.append({'outcome':outcome,
                            'threshold_mgL': (int(row.threshold_mgL)
                                             if outcome=='next_day_recovery' else np.nan),
                            'node':row.node,'date':row.date.date().isoformat(),
                            'test_month':month,'model':model,
                            'actual':float(row.actual),'predicted':float(p),
                            'qc_suspect':bool(row.qc_suspect),
                            'provenance':'observed_only'})
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                reason=f'fit_unavailable:{type(exc).__name__}'
        diagnostics.append({'outcome':outcome,
            'threshold_mgL': int(d.threshold_mgL.iloc[0]) if outcome=='next_day_recovery' else np.nan,
            'test_month':month,'n_train':len(train),'n_test':len(test),
            'n_train_positive': int(train.actual.sum()) if outcome=='next_day_recovery' else np.nan,
            'status':reason})
    return pd.DataFrame(out,columns=PRED_COLUMNS),pd.DataFrame(diagnostics)


def paired_scores(predictions: pd.DataFrame):
    if predictions.empty:
        return pd.DataFrame(columns=SCORE_COLUMNS)
    keys=['outcome','node','date','test_month']
    if predictions.outcome.eq('next_day_recovery').any():
        keys.append('threshold_mgL')
    ref=predictions.loc[predictions.model=='baseline',keys+['predicted','actual']].rename(
        columns={'predicted':'base_predicted','actual':'baseline_actual'})
    candidate=predictions.loc[predictions.model=='environmental_gam'].merge(
        ref,on=keys,validate='one_to_one')
    if not np.allclose(candidate.actual,candidate.baseline_actual):
        raise ValueError('Paired prediction labels differ')
    results=[]
    for (outcome,threshold),g in candidate.groupby(['outcome','threshold_mgL'],dropna=False):
        for model,col in [('baseline','base_predicted'),('environmental_gam','predicted')]:
            y=g.actual.to_numpy(float); p=g[col].to_numpy(float)
            baseline=g.base_predicted.to_numpy(float)
            if outcome=='next_day_recovery':
                score=float(np.mean((y-p)**2))
                base=float(np.mean((y-baseline)**2))
            else:
                score=float(np.mean(np.abs(y-p)))
                base=float(np.mean(np.abs(y-baseline)))
            results.append({'outcome':outcome,'threshold_mgL':threshold,'model':model,
                'n_paired':len(g),'n_stations':g.node.nunique(),
                'n_test_months':g.test_month.nunique(),
                'n_positive':int(y.sum()) if outcome=='next_day_recovery' else np.nan,
                'mae':float(np.mean(np.abs(y-p))),
                'rmse':float(np.sqrt(np.mean((y-p)**2))),
                'brier':float(np.mean((y-p)**2)) if outcome=='next_day_recovery' else np.nan,
                'paired_baseline_score':base,
                'skill_vs_baseline':float(1-score/base) if base>0 else np.nan})
    return pd.DataFrame(results,columns=SCORE_COLUMNS)



def smooth_effect_curves(data: pd.DataFrame, outcome: str, alpha=10.,
                         include_trend=True, points=31):
    """Conditional fitted curves, not causal effects or confidence intervals.

    Other covariates are held at representative fixed values. Curves are
    truncated to the 5th-95th percentiles of observed predictor support.
    """
    columns=['outcome','threshold_mgL','predictor','grid_value',
             'fitted_response','reference_response','difference_from_reference',
             'reference_station','reference_date','n_fitted','provenance']
    if data.empty:
        return pd.DataFrame(columns=columns)
    from statsmodels.gam.api import GLMGam
    fitted, _=_fit(data,data,outcome,'environmental_gam',alpha,include_trend)
    stations=sorted(data.node.unique().tolist())
    # Choose the most represented station; other terms held at medians.
    station=data.node.value_counts().index[0]
    example=data.loc[data.node.eq(station)].iloc[[0]].copy().reset_index(drop=True)
    example['date']=pd.Timestamp('2025-07-01')
    for field in DAILY_FEATURES:
        example[field]=float(data[field].median())
    if outcome=='next_day_recovery':
        for field in ['log_elapsed_days','hours_below_per_6h']:
            example[field]=float(data[field].median())
        if include_trend:
            example['delta_do_mean_1d']=float(data.delta_do_mean_1d.median())
    def predict(rows):
        x=_matrix(rows,stations,outcome)
        if outcome=='next_day_recovery' and include_trend:
            x['delta_do_mean_1d']=rows.delta_do_mean_1d.to_numpy(float)
        return np.asarray(fitted.predict(exog=x,exog_smooth=rows[DAILY_FEATURES]),float)
    ref=float(predict(example)[0])
    rows=[]
    for predictor in DAILY_FEATURES:
        low,high=np.quantile(data[predictor],[.05,.95])
        if high<=low:
            continue
        grid=np.linspace(low,high,points)
        frame=pd.concat([example]*points,ignore_index=True)
        frame[predictor]=grid
        values=predict(frame)
        for value,response in zip(grid,values):
            rows.append({'outcome':outcome,
                'threshold_mgL':(int(data.threshold_mgL.iloc[0])
                                   if outcome=='next_day_recovery' else np.nan),
                'predictor':predictor,'grid_value':float(value),
                'fitted_response':float(response),'reference_response':ref,
                'difference_from_reference':float(response-ref),
                'reference_station':station,'reference_date':'2025-07-01',
                'n_fitted':len(data),'provenance':'observed_only'})
    return pd.DataFrame(rows,columns=columns)


def save_effect_figures(effects: pd.DataFrame, out: Path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib missing: smooth figures skipped')
        return
    if effects.empty:
        return
    out.mkdir(parents=True,exist_ok=True)
    for (outcome,threshold,predictor),g in effects.groupby(
            ['outcome','threshold_mgL','predictor'],dropna=False):
        fig,ax=plt.subplots(figsize=(7,4.2))
        ax.plot(g.grid_value,g.fitted_response)
        ax.set(xlabel=predictor.replace('_',' '),
               ylabel=('Next-day recovery probability' if outcome=='next_day_recovery'
                       else 'Fitted daily minimum DO (mg/L)'),
               title=(f'{outcome} / threshold {threshold:g}' if np.isfinite(threshold)
                      else 'Daily DO severity')+' | conditional GAM curve')
        ax.grid(alpha=.15);fig.tight_layout()
        suffix=(str(int(threshold)) if np.isfinite(threshold) else 'daily')
        fig.savefig(out/f'{outcome}_{suffix}_{predictor}.png',dpi=160)
        plt.close(fig)

def full_fit_summary(data,outcome,alpha=10,include_trend=True):
    output=[]
    if data.empty:
        return pd.DataFrame(columns=FIT_COLUMNS)
    for model in ('baseline','environmental_gam'):
        try:
            fitted,_=_fit(data,data,outcome,model,alpha,include_trend)
            edf=getattr(fitted,'edf',np.full(len(fitted.params),np.nan))
            output.append({'outcome':outcome,'threshold_mgL':(
                int(data.threshold_mgL.iloc[0]) if outcome=='next_day_recovery' else np.nan),
                'model':model,'n_rows':len(data),'n_stations':data.node.nunique(),
                'n_positive':int(data.actual.sum()) if outcome=='next_day_recovery' else np.nan,
                'aic':float(fitted.aic),'edf_total':float(np.sum(edf)) if np.isfinite(edf).all() else np.nan,
                'converged':bool(getattr(fitted,'converged',True)),
                'smoothing_alpha':alpha if model=='environmental_gam' else np.nan,
                'caution':'retrospective fit; partial nonlinear associations, not causation or warning alerts'})
        except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
            warnings.warn(f'{outcome}: {model} full fit unavailable: {exc}')
    return pd.DataFrame(output,columns=FIT_COLUMNS)


def main():
    parser=argparse.ArgumentParser(description='Observed-only daily DO severity and recovery GAM pilots')
    parser.add_argument('--daily',type=Path,default=Path('trustworthy_outputs/daily_targets.csv'))
    parser.add_argument('--risk',type=Path,default=Path('trustworthy_outputs/episode_dynamics/recovery_risk_days.csv'))
    parser.add_argument('--out-dir',type=Path,default=Path('trustworthy_outputs/environmental_gam'))
    parser.add_argument('--min-coverage',type=float,default=.95)
    parser.add_argument('--alpha',type=float,default=10.)
    parser.add_argument('--min-train',type=int,default=100)
    parser.add_argument('--no-trend',action='store_true')
    args=parser.parse_args()
    if not args.daily.exists() or not args.risk.exists():
        parser.error('Missing observed-only inputs; run quality and episode_survival first')
    daily=prepare_daily(pd.read_csv(args.daily),args.min_coverage)
    risk=pd.read_csv(args.risk)
    rec=recovery_rows(risk,daily,include_trend=not args.no_trend)
    severity=severity_rows(daily)
    parts=[];fit_summaries=[];diagnostics=[];curves=[]
    for outcome,data in [('odo_min',severity)]+[(
        'next_day_recovery',g) for _,g in rec.groupby('threshold_mgL')]:
        pred,diag=rolling_backtest(data,outcome,alpha=args.alpha,
           min_train=args.min_train,include_trend=not args.no_trend)
        parts.append(pred); diagnostics.append(diag)
        fit_summaries.append(full_fit_summary(data,outcome,args.alpha,not args.no_trend))
        if len(data)>=max(30,args.min_train):
            try:
                curves.append(smooth_effect_curves(data,outcome,args.alpha,not args.no_trend))
            except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
                warnings.warn(f'{outcome}: smooth effects unavailable: {exc}')
    predictions=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=PRED_COLUMNS)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    predictions.to_csv(args.out_dir/'time_block_predictions.csv',index=False)
    paired_scores(predictions).to_csv(args.out_dir/'paired_scores.csv',index=False)
    pd.concat(diagnostics,ignore_index=True).to_csv(args.out_dir/'fold_diagnostics.csv',index=False)
    pd.concat(fit_summaries,ignore_index=True).to_csv(args.out_dir/'full_fit_summary.csv',index=False)
    rec.to_csv(args.out_dir/'observed_recovery_features.csv',index=False)
    effect_table=pd.concat(curves,ignore_index=True) if curves else pd.DataFrame()
    effect_table.to_csv(args.out_dir/'gam_smooth_effects.csv',index=False)
    save_effect_figures(effect_table,args.out_dir/'figures')
    metadata={'daily_sha256':hashlib.sha256(args.daily.read_bytes()).hexdigest(),
        'risk_sha256':hashlib.sha256(args.risk.read_bytes()).hexdigest(),
        'min_coverage':args.min_coverage,'smoothing_alpha':args.alpha,
        'min_train':args.min_train,'use_prior_day_do_trend':not args.no_trend,
        'n_severity_complete':len(severity),'n_recovery_complete':len(rec),
        'n_recovery_raw':len(risk),'n_out_of_sample_predictions':len(predictions),
        'provenance':'observed_only',
        'warning':'Retrospective association and time-block diagnostic only; not an operational early-warning model',
        'forecast_qualification':'Full next-day labels must end BEFORE first issue date of test month'}
    (args.out_dir/'environmental_gam_metadata.json').write_text(
        json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(f"GAM pilot: {len(severity)} severity days, {len(rec)} recovery risk-days; {len(predictions)} paired out-of-sample prediction rows")


if __name__=='__main__':
    main()
