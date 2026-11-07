"""
Does the FCM beat 'same as last month'? In-sample vs held-out, monthly data.

Weights are plain lag-1 ridge (the same fit as fcm.learn_fcm_weights, unscaled).
Held-out: fit on the first 70% of months, score on the last 30%, observed pairs
only. Min-max scaling uses the full record (minor leakage, same for all models).
Outputs: analysis/fcm_vs_baselines.csv, FCM/graphics/07_fcm_vs_persistence.png
"""
import io
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fcm_validate as fv
from fcm import _ridge

ROOT = Path(__file__).resolve().parent
OUT_PNG = ROOT / "FCM" / "graphics" / "07_fcm_vs_persistence.png"
OUT_CSV = ROOT / "analysis" / "fcm_vs_baselines.csv"
ALPHA = 1.5
TRAIN_FRAC = 0.7
FCM_COL, PERS_COL, MEAN_COL = "#c0501f", "#1e88e5", "#9e9e9e"


def fit_and_predict(a, train_rows):
    """Lag-1 ridge on rows [0, train_rows); returns predictions for every t+1."""
    cm = np.nanmean(a[:train_rows], axis=0)
    cm[np.isnan(cm)] = 0.5
    x = np.where(np.isnan(a), cm, a)
    nan = np.isnan(a)
    pred = np.full((len(a) - 1, a.shape[1]), np.nan)
    for j in fv.ENDO_IDXS:
        if nan[1:train_rows, j].mean() > 0.6:
            continue
        w = _ridge(x[:train_rows - 1], x[1:train_rows, j], ALPHA)
        pred[:, j] = np.clip(x[:-1] @ w, 0.0, 1.0)
    return pred


def score(a, obs_mask, pred, rows, train_mean):
    out = []
    for j in fv.ENDO_IDXS:
        m = rows & obs_mask[:-1, j] & obs_mask[1:, j] & ~np.isnan(pred[:, j])
        if m.sum() < 5:
            continue
        o = a[1:, j][m]

        def rmse(p):
            return float(np.sqrt(np.mean((o - p) ** 2)))

        out.append({"concept": fv.LABELS[j], "n": int(m.sum()),
                    "rmse_fcm": rmse(pred[m, j]), "rmse_persistence": rmse(a[:-1, j][m]),
                    "rmse_mean": rmse(np.full(o.shape, train_mean[j]))})
    return pd.DataFrame(out)


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        _, a, _, _, index, _, obs_mask = fv.prepare_data()
    T = len(a)
    cut = int(TRAIN_FRAC * T)
    rows_all = np.ones(T - 1, bool)
    rows_test = np.arange(T - 1) >= cut

    pred_in = fit_and_predict(a, T)
    pred_out = fit_and_predict(a, cut)
    d_in = score(a, obs_mask, pred_in, rows_all, np.nanmean(a, axis=0))
    d_out = score(a, obs_mask, pred_out, rows_test, np.nanmean(a[:cut], axis=0))
    for d in (d_in, d_out):
        d["skill_vs_persistence"] = 1 - d.rmse_fcm / d.rmse_persistence
    d_in["split"], d_out["split"] = "in-sample", "held-out (last 30%)"
    res = pd.concat([d_in, d_out])
    res.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(res.round(3).to_string(index=False))

    do = next(j for j in fv.ENDO_IDXS if fv.LABELS[j] == "ODO (mg/L)")
    m = rows_test & obs_mask[:-1, do] & obs_mask[1:, do] & ~np.isnan(pred_out[:, do])
    t = index[1:][m].to_timestamp()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), gridspec_kw={"width_ratios": [1.5, 1.2, 1.2]})
    a1, a2, a3 = axes
    a1.plot(t, a[1:, do][m], "o-", color="k", lw=1.2, ms=4, label="Observed")
    a1.plot(t, pred_out[m, do], "o--", color=FCM_COL, lw=1.2, ms=4, label="FCM (fit on earlier months)")
    a1.plot(t, a[:-1, do][m], "s:", color=PERS_COL, lw=1.2, ms=4, label="Persistence (last month)")
    a1.set_ylabel("DO mg/L (scaled 0-1)")
    a1.set_title("(a) DO, held-out months: persistence tracks closer than the FCM", fontsize=11)
    a1.legend(fontsize=9)
    a1.tick_params(axis="x", rotation=30)

    d = d_out.sort_values("rmse_persistence")
    y = np.arange(len(d))
    h = 0.27
    a2.barh(y + h, d.rmse_fcm, h, color=FCM_COL, label="FCM")
    a2.barh(y, d.rmse_persistence, h, color=PERS_COL, label="Persistence")
    a2.barh(y - h, d.rmse_mean, h, color=MEAN_COL, label="Training mean")
    a2.set_yticks(y)
    a2.set_yticklabels(d.concept, fontsize=9)
    a2.set_xlabel("Held-out RMSE (scaled 0-1, lower is better)")
    wins = int((d.skill_vs_persistence > 0).sum())
    a2.set_title(f"(b) Held-out: FCM beats persistence on {wins} of {len(d)}", fontsize=11)
    a2.legend(fontsize=9, loc="lower right")

    order = d_in.sort_values("skill_vs_persistence").concept.tolist()
    sk_in = d_in.set_index("concept").skill_vs_persistence.reindex(order)
    sk_out = d_out.set_index("concept").skill_vs_persistence.reindex(order)
    y = np.arange(len(order))
    a3.barh(y + 0.2, sk_in, 0.4, color="#80cbc4", label="In-sample")
    a3.barh(y - 0.2, sk_out, 0.4, color=FCM_COL, label="Held-out")
    a3.axvline(0, color="k", lw=0.8)
    a3.set_yticks(y)
    a3.set_yticklabels(order, fontsize=9)
    a3.set_xlabel("Skill vs persistence = 1 - RMSE_FCM / RMSE_persistence")
    a3.set_title("(c) Skill over 'same as last month'", fontsize=11)
    a3.legend(fontsize=9)

    n_neg = int((d_out.skill_vs_persistence <= 0).sum())
    fig.suptitle(f"FCM alone vs 'same as last month' on monthly data "
                 f"(held-out: no gain on {n_neg} of {len(d_out)} concepts)", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.005, f"{len(d_out)} modelled concepts; held-out = last {T - 1 - cut} monthly steps, "
             "observed pairs only. Depth, pressure and turbidity are not modelled (too sparse).",
             ha="center", fontsize=9.5, color="#444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT_PNG)


if __name__ == "__main__":
    main()
