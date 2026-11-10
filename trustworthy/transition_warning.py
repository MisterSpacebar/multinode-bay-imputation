"""Observed-only 1/3/7-day severe-hypoxia transition warning prototype.

Predicts a FIRST severe (DO <2 mg/L) day anywhere in the next H eligible
calendar days, conditional on the issue-day state being Normal or Moderate.
All future days must have observed labels to evaluate a historical prediction.
Markov probabilities and Monte Carlo use only transitions observed before each
held-out month. Logistic features are same-day and past-day observations only.
This is a retrospective prototype, not a live alert or a causal model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from trustworthy.environmental_gam import prepare_daily
from trustworthy.episode_transitions import STATES

HORIZONS = (1, 3, 7)
DO_FEATURES = ('odo_min', 'odo_mean', 'hours_lt2', 'hours_lt4',
               'delta_do_mean_1d', 'state_moderate')
ENV_FEATURES = DO_FEATURES + ('temp_mean', 'sal_mean', 'season_sin', 'season_cos')
MODELS = ('markov', 'do_history', 'environmental')
WINDOW_COLUMNS = ('node','issue_date','window_end','horizon_days','origin_state',
    'label','first_severe_date','status','qc_suspect','provenance', *ENV_FEATURES)
PRED_COLUMNS = ('node','issue_date','window_end','test_month','horizon_days',
    'origin_state','model','label','probability','brier_error',
    'n_train','n_train_events','qc_suspect','provenance')
SCORE_COLUMNS = ('horizon_days','model','n_paired','n_events','n_non_events',
    'n_stations','n_test_months','brier','brier_skill_vs_markov',
    'mean_predicted_risk','observed_rate','precision_at_25pct',
    'recall_at_25pct','false_alarm_fraction_at_25pct','alerts_at_25pct')


def _bool(s, field):
    x=s.astype('string').str.strip().str.lower().map(
        {'true':True,'false':False,'1':True,'0':False})
    if x.isna().any():
        raise ValueError(f'{field}: missing or invalid boolean')
    return x.astype(bool)


def prepare_states(states):
    required={'node','date','state','hours_lt2','hours_lt4','provenance'}
    if not required.issubset(states):
        raise ValueError(f'state input missing {sorted(required-set(states))}')
    s=states.copy()
    if s.provenance.isna().any() or not s.provenance.eq('observed_only').all():
        raise ValueError('state input requires observed_only provenance')
    if s.node.isna().any():
        raise ValueError('missing station identity')
    s['node']=s.node.astype(str)
    s['date']=pd.to_datetime(s.date,errors='raise')
    if s.date.isna().any() or s.date.dt.tz is not None or not s.date.dt.normalize().eq(s.date).all():
        raise ValueError('state dates must be timezone-naive local days')
    if s.duplicated(['node','date']).any():
        raise ValueError('duplicate station/date state')
    if not s.state.isin((*STATES,'Unknown')).all():
        raise ValueError('invalid state')
    for c in ('hours_lt2','hours_lt4'):
        s[c]=pd.to_numeric(s[c],errors='coerce')
    known=s.state.isin(STATES)
    if ((known & (~s.hours_lt2.between(0,24) | ~s.hours_lt4.between(0,24))) |
        (known & s.hours_lt2.gt(s.hours_lt4+1e-8)) |
        (s.state.eq('Normal') & s.hours_lt4.gt(0)) |
        (s.state.eq('Moderate') & (s.hours_lt2.gt(0) | s.hours_lt4.le(0))) |
        (s.state.eq('Severe') & s.hours_lt2.le(0))).any():
        raise ValueError('state and observed threshold-hours disagree')
    s['qc_suspect']=_bool(s.qc_suspect,'qc_suspect') if 'qc_suspect' in s else False
    return s.sort_values(['node','date']).reset_index(drop=True)


def warning_windows(states, daily, horizons=HORIZONS, min_coverage=.95):
    """Conservative hindcast labels: all H future days known, no gap crossing.

    Issue-day Normal/Moderate only. Daily physical features may be absent;
    retain label window for exclusions and score only complete-case matches.
    """
    if not set(horizons).issubset(set(HORIZONS)) or not horizons:
        raise ValueError('supported horizons are 1, 3, 7')
    s=prepare_states(states)
    d=prepare_daily(daily,min_coverage=min_coverage)
    # Source QC is required for issue-day features. No join to future dates.
    features=['node','date','qc_valid','odo_min','odo_mean','temp_mean',
              'sal_mean','delta_do_mean_1d']
    join=d[features].rename(columns={'qc_valid':'issue_qc_valid'})
    s=s.merge(join,on=['node','date'],how='left',validate='one_to_one')
    s['state_moderate']=s.state.eq('Moderate').astype(float)
    doy=s.date.dt.dayofyear.to_numpy(float)
    s['season_sin']=np.sin(2*np.pi*doy/365.25)
    s['season_cos']=np.cos(2*np.pi*doy/365.25)
    output=[]
    for node,group in s.groupby('node',sort=True):
        g=group.set_index('date').sort_index()
        # Build full calendar to avoid confusing row adjacency with date adjacency.
        dates=pd.date_range(g.index.min(),g.index.max(),freq='D')
        g=g.reindex(dates)
        for date,origin in g.iterrows():
            if origin.state not in ('Normal','Moderate'):
                continue
            for h in horizons:
                end=date+pd.Timedelta(days=h)
                future=g.reindex(pd.date_range(date+pd.Timedelta(days=1),end,freq='D'))
                complete=future.state.isin(STATES).all()
                severe_days=future.index[future.state.eq('Severe')]
                label=int(len(severe_days)>0) if complete else np.nan
                status=('evaluable' if complete else 'unknown_future')
                if pd.isna(origin.issue_qc_valid) or not bool(origin.issue_qc_valid):
                    status='issue_qc_ineligible'
                values={c:origin.get(c,np.nan) for c in ENV_FEATURES}
                output.append(dict(node=node,issue_date=date,window_end=end,
                    horizon_days=h,origin_state=origin.state,label=label,
                    first_severe_date=(severe_days[0].date().isoformat()
                        if complete and len(severe_days) else ''),
                    status=status,qc_suspect=bool(origin.qc_suspect),
                    provenance='observed_only',**values))
    return pd.DataFrame(output,columns=WINDOW_COLUMNS)


def validate_transitions(transitions,states):
    required={'node','date','next_date','from_state','to_state','provenance'}
    if not required.issubset(transitions):
        raise ValueError(f'transitions missing {sorted(required-set(transitions))}')
    t=transitions.copy()
    if t.provenance.isna().any() or not t.provenance.eq('observed_only').all():
        raise ValueError('transitions require observed_only provenance')
    for c in ('date','next_date'):
        t[c]=pd.to_datetime(t[c],errors='raise')
        if t[c].isna().any() or t[c].dt.tz is not None:
            raise ValueError('transition dates must be local')
    if t.duplicated(['node','date']).any():
        raise ValueError('duplicate transitions')
    if not (t.next_date-t.date).dt.days.eq(1).all():
        raise ValueError('transitions must be adjacent local days')
    if not t.from_state.isin(STATES).all() or not t.to_state.isin(STATES).all():
        raise ValueError('transitions must have known states')
    lookup=states[['node','date','state']]
    from_check=t[['node','date','from_state']].merge(lookup,on=['node','date'],how='left',validate='many_to_one')
    if not from_check.from_state.eq(from_check.state).all():
        raise ValueError('transition origin inconsistent with state table')
    next_check=t[['node','next_date','to_state']].merge(
        lookup.rename(columns={'date':'next_date'}),on=['node','next_date'],how='left',validate='many_to_one')
    if not next_check.to_state.eq(next_check.state).all():
        raise ValueError('transition destination inconsistent with state table')
    return t


def transition_matrix(history,station=None,prior=20.):
    """Shrink station transition counts toward observed historical pooled rows."""
    if prior<0:
        raise ValueError('prior must be nonnegative')
    pooled=np.zeros((3,3),dtype=float)
    local=np.zeros((3,3),dtype=float)
    idx={k:i for i,k in enumerate(STATES)}
    for r in history.itertuples(index=False):
        i,j=idx[r.from_state],idx[r.to_state]
        pooled[i,j]+=1
        if station is not None and r.node==station:
            local[i,j]+=1
    # Weak, transparent Dirichlet smoothing for never-seen states.
    global_prob=(pooled+0.5)/(pooled.sum(axis=1,keepdims=True)+1.5)
    counts=local+prior*global_prob if station is not None else pooled+0.5
    if station is not None and prior==0:
        counts+=0.5
    return counts/counts.sum(axis=1,keepdims=True)


def markov_hit_probability(p,origin,horizon):
    """P(any Severe among the next H days | current state non-Severe)."""
    if origin not in ('Normal','Moderate') or horizon<1:
        raise ValueError('start in Normal/Moderate, horizon >=1')
    p=np.asarray(p,float)
    if p.shape!=(3,3) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1):
        raise ValueError('invalid Markov matrix')
    q=p[:2,:2]
    return float(np.clip(1-(np.linalg.matrix_power(q,horizon)@np.ones(2))[
        0 if origin=='Normal' else 1],0,1))


def monte_carlo_reference(p,origin,n_paths=2000,seed=2026):
    """Simulated seven-day trajectories; does not add forecast information."""
    if origin not in ('Normal','Moderate') or n_paths<1:
        raise ValueError('invalid simulation arguments')
    rng=np.random.default_rng(seed)
    state=np.full(n_paths,0 if origin=='Normal' else 1,dtype=np.int8)
    first=np.full(n_paths,0,dtype=np.int8)
    n_severe=np.zeros(n_paths,dtype=np.int8)
    for step in range(1,8):
        u=rng.random(n_paths)
        cumulative=np.cumsum(p[state],axis=1)
        state=(u[:,None]>cumulative[:,:-1]).sum(axis=1).astype(np.int8)
        n_severe+=(state==2)
        first[(first==0)&(state==2)]=step
    return {f'within_{h}d':float(np.mean((first>0)&(first<=h))) for h in HORIZONS} | {
        'mean_severe_days_7d':float(n_severe.mean()),
        'p_any_severe_7d':float(np.mean(first>0)),
        'median_first_severe_day_if_hit':float(np.median(first[first>0])) if (first>0).any() else np.nan,
    }


def logistic_fit_predict(train,test,features,alpha=10.):
    """One deterministic, standardized ridge logistic fit per monthly fold."""
    if alpha<=0:
        raise ValueError('alpha must be positive')
    stations=sorted(train.node.unique())
    if not test.node.isin(stations).all():
        raise ValueError('unseen station in test')
    def design(frame):
        x=frame[list(features)].to_numpy(float)
        x=np.column_stack([x]+[
            frame.node.eq(st).to_numpy(float) for st in stations[1:]])
        return x
    x,y,q=design(train),train.label.to_numpy(float),design(test)
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(q).all()):
        raise ValueError('nonfinite model inputs')
    if min(int(y.sum()),int(len(y)-y.sum()))<2:
        raise ValueError('insufficient event class')
    mu=x.mean(axis=0);sigma=x.std(axis=0);sigma[sigma<1e-8]=1.
    z=np.column_stack([np.ones(len(x)),(x-mu)/sigma])
    w=np.column_stack([np.ones(len(q)),(q-mu)/sigma])
    beta=np.zeros(z.shape[1]);rate=(y.sum()+.5)/(len(y)+1)
    beta[0]=np.log(rate/(1-rate))
    penalty=np.diag([0.]+[alpha]*(z.shape[1]-1))
    for _ in range(80):
        p=1/(1+np.exp(-np.clip(z@beta,-35,35)))
        v=np.maximum(p*(1-p),1e-8)
        step=np.linalg.solve(z.T@(z*v[:,None])+penalty,
                             z.T@(y-p)-penalty@beta)
        step=np.clip(step,-2,2)
        beta+=step
        if np.max(np.abs(step))<1e-7:
            break
    return np.clip(1/(1+np.exp(-np.clip(w@beta,-35,35))),1e-6,1-1e-6)


def hindcast(windows,transitions,alpha=10.,prior=20.,min_train=60,
             min_events=3,n_paths=1000,seed=2026):
    """Monthly expanding hindcast; all training labels complete BEFORE fold.

    Common-case scoring: return all 3 models only when both learned models
    can train and all issue-day features are available. Skips reported.
    """
    if min_train<10 or min_events<1 or n_paths<0:
        raise ValueError('invalid hindcast parameters')
    w=windows.copy()
    if w.empty:
        return pd.DataFrame(columns=PRED_COLUMNS),pd.DataFrame(),pd.DataFrame()
    w['issue_date']=pd.to_datetime(w.issue_date)
    w['window_end']=pd.to_datetime(w.window_end)
    w['test_month']=w.issue_date.dt.to_period('M').astype(str)
    valid=w.status.eq('evaluable')&w.label.notna()
    finite=np.isfinite(w[list(ENV_FEATURES)].apply(pd.to_numeric,errors='coerce')).all(axis=1)
    w['complete_case']=valid&finite
    preds=[];diagnostics=[];mc=[]
    for month in sorted(w.test_month.unique()):
        start=pd.Period(month).start_time
        # This is crucial for 7-day labels crossing month boundaries.
        history=transitions.loc[transitions.next_date.lt(start)]
        for horizon in HORIZONS:
            train=w.loc[w.complete_case & w.horizon_days.eq(horizon)
                        & w.window_end.lt(start)].copy()
            test=w.loc[w.complete_case & w.horizon_days.eq(horizon)
                       & w.test_month.eq(month) & w.node.isin(train.node.unique())].copy()
            events=int(train.label.sum())
            reason='evaluable'
            if len(train)<min_train or len(test)==0:
                reason='insufficient_history_or_test'
            elif events<min_events or len(train)-events<min_events:
                reason='insufficient_classes'
            elif history.empty:
                reason='no_past_transitions'
            if reason=='evaluable':
                try:
                    do=logistic_fit_predict(train,test,DO_FEATURES,alpha)
                    env=logistic_fit_predict(train,test,ENV_FEATURES,alpha)
                    matrices={node:transition_matrix(history,node,prior)
                              for node in test.node.unique()}
                    for i,row in enumerate(test.itertuples(index=False)):
                        markov=markov_hit_probability(matrices[row.node],row.origin_state,horizon)
                        for model,p in [('markov',markov),('do_history',do[i]),
                                        ('environmental',env[i])]:
                            preds.append(dict(node=row.node,issue_date=row.issue_date.date().isoformat(),
                               window_end=row.window_end.date().isoformat(),test_month=month,
                               horizon_days=horizon,origin_state=row.origin_state,
                               model=model,label=int(row.label),probability=float(p),
                               brier_error=float((p-row.label)**2),n_train=len(train),
                               n_train_events=events,qc_suspect=bool(row.qc_suspect),
                               provenance='observed_only'))
                    if n_paths and horizon==7:
                        for node,p in matrices.items():
                            for start_state in ('Normal','Moderate'):
                                # Reuse a fixed pseudorandom stream for a deterministic
                                # station/month/state-specific simulation, no hash().
                                local_seed=int(seed+sum(map(ord,f'{month}{node}{start_state}')))
                                sim=monte_carlo_reference(p,start_state,n_paths,local_seed)
                                mc.append(dict(test_month=month,node=node,
                                  origin_state=start_state,n_paths=n_paths,seed=local_seed,
                                  **sim,exact_p_1d=markov_hit_probability(p,start_state,1),
                                  exact_p_3d=markov_hit_probability(p,start_state,3),
                                  exact_p_7d=markov_hit_probability(p,start_state,7),
                                  provenance='observed_only'))
                except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
                    reason='fit_unavailable:'+type(exc).__name__
            diagnostics.append(dict(test_month=month,horizon_days=horizon,
                n_train=len(train),n_train_positive=events,n_test=len(test),
                n_history_transitions=len(history),status=reason))
    predictions=pd.DataFrame(preds,columns=PRED_COLUMNS)
    # These are nested event windows: risk within seven days cannot be less
    # than risk within three days at the same issue/model. This is a purely
    # issue-time postprocessing step (no outcome is consulted).
    if not predictions.empty:
        predictions=predictions.sort_values(['node','issue_date','model','horizon_days'])
        predictions['probability']=predictions.groupby(
            ['node','issue_date','model']).probability.cummax()
        predictions['brier_error']=(predictions.probability-predictions.label)**2
        predictions=predictions.sort_values(['issue_date','node','horizon_days','model']).reset_index(drop=True)
    return (predictions,pd.DataFrame(diagnostics),pd.DataFrame(mc))


def evaluate(preds,alert_threshold=.25):
    """Paired Brier and illustrative 25% alert operating point per horizon."""
    if not 0<alert_threshold<1:
        raise ValueError('alert_threshold must lie between 0 and 1')
    if preds.empty:
        return pd.DataFrame(columns=SCORE_COLUMNS)
    keys=['node','issue_date','horizon_days']
    counts=preds.groupby(keys).model.nunique()
    if not counts.eq(3).all() or preds.duplicated(keys+['model']).any():
        raise ValueError('unpaired models/duplicate predictions')
    if preds.groupby(keys).label.nunique().gt(1).any():
        raise ValueError('paired labels differ')
    rows=[]
    for horizon,g in preds.groupby('horizon_days'):
        baseline=float(g.loc[g.model.eq('markov'),'brier_error'].mean())
        for model,pred in g.groupby('model'):
            y=pred.label.to_numpy(int);p=pred.probability.to_numpy(float)
            alerted=p>=alert_threshold
            tp=int((alerted & (y==1)).sum());fp=int((alerted & (y==0)).sum())
            fn=int((~alerted & (y==1)).sum())
            brier=float(np.mean((p-y)**2))
            rows.append(dict(horizon_days=int(horizon),model=model,
               n_paired=len(pred),n_events=int(y.sum()),n_non_events=int((y==0).sum()),
               n_stations=pred.node.nunique(),n_test_months=pred.test_month.nunique(),
               brier=brier,brier_skill_vs_markov=float(1-brier/baseline) if baseline>0 else np.nan,
               mean_predicted_risk=float(p.mean()),observed_rate=float(y.mean()),
               precision_at_25pct=tp/(tp+fp) if tp+fp else np.nan,
               recall_at_25pct=tp/(tp+fn) if tp+fn else np.nan,
               false_alarm_fraction_at_25pct=fp/(tp+fp) if tp+fp else np.nan,
               alerts_at_25pct=int(alerted.sum())))
    return pd.DataFrame(rows,columns=SCORE_COLUMNS)


def plot_timeline(preds,out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if preds.empty:
        return
    # A representative observed station chosen by largest comparable 3d sample.
    x=preds.loc[preds.horizon_days.eq(3)]
    if x.empty:
        return
    node=x.node.value_counts().index[0]
    g=x.loc[x.node.eq(node)].copy()
    fig,ax=plt.subplots(figsize=(11,4.5))
    for model,part in g.groupby('model'):
        ax.plot(pd.to_datetime(part.issue_date),part.probability,
                '.-',alpha=.7,label=model)
    actual=g.loc[g.model.eq('markov')]
    ax.scatter(pd.to_datetime(actual.loc[actual.label.eq(1),'issue_date']),
               np.ones(int(actual.label.sum())),marker='x',color='black',
               label='Severe observed within 3d')
    ax.set(xlabel='Historical issue date',ylabel='P(severe within 3 days)',ylim=(-.04,1.08),
           title=f'3-day severe-hypoxia hindcast: {node}')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.autofmt_xdate();fig.tight_layout()
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'hindcast_3day_example.png',dpi=150)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description='Observed-only severe-transition warning hindcast')
    ap.add_argument('--daily',type=Path,default=Path('trustworthy_outputs/daily_targets.csv'))
    ap.add_argument('--states',type=Path,default=Path('trustworthy_outputs/episode_dynamics/station_day_states.csv'))
    ap.add_argument('--transitions',type=Path,default=Path('trustworthy_outputs/episode_dynamics/observed_transitions.csv'))
    ap.add_argument('--out-dir',type=Path,default=Path('trustworthy_outputs/transition_warning'))
    ap.add_argument('--min-coverage',type=float,default=.95)
    ap.add_argument('--alpha',type=float,default=10.)
    ap.add_argument('--station-prior',type=float,default=20.)
    ap.add_argument('--min-train',type=int,default=60)
    ap.add_argument('--min-events',type=int,default=3)
    ap.add_argument('--monte-carlo-paths',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--no-figures',action='store_true')
    args=ap.parse_args()
    data={key:pd.read_csv(getattr(args,key)) for key in ('daily','states','transitions')}
    states=prepare_states(data['states'])
    t=validate_transitions(data['transitions'],states)
    windows=warning_windows(states,data['daily'],min_coverage=args.min_coverage)
    preds,folds,mc=hindcast(windows,t,alpha=args.alpha,prior=args.station_prior,
        min_train=args.min_train,min_events=args.min_events,
        n_paths=args.monte_carlo_paths,seed=args.seed)
    scores=evaluate(preds)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for name,frame in [('warning_windows',windows),('hindcast_predictions',preds),
                       ('hindcast_scores',scores),('fold_diagnostics',folds),
                       ('monte_carlo_reference',mc)]:
        frame.to_csv(args.out_dir/f'{name}.csv',index=False)
    if not args.no_figures:
        plot_timeline(preds,args.out_dir/'figures')
    meta={key+'_sha256':hashlib.sha256(getattr(args,key).read_bytes()).hexdigest()
          for key in data}
    meta.update(dict(horizons=list(HORIZONS),target='first severe <2 mg/L sample within future H eligible days',
         min_coverage=args.min_coverage,ridge_alpha=args.alpha,station_prior=args.station_prior,
         min_train=args.min_train,min_events=args.min_events,n_monte_carlo_paths=args.monte_carlo_paths,
         seed=args.seed,n_hindcast_cases=int(len(preds)//3),
         eligibility='Normal/Moderate issue-day, all future days known',
         validation='monthly expanding, completed outcomes and transitions before test month',
         risk_definition='at least one future Severe, not continuous seven-day hypoxia',
         caveat='Research prototype; time-block hindcast, no operational calibration or causal guarantees',
         provenance='observed_only'))
    (args.out_dir/'warning_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f'Transition warning: {len(windows)} candidate windows; '
          f'{len(preds)//3} paired hindcasts, {len(mc)} Monte Carlo scenario rows')
    print('Output:',args.out_dir)

if __name__=='__main__':main()