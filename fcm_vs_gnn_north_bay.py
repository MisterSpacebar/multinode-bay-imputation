"""
North-bay dissolved-oxygen drivers, 2025-2026: observed-only FCM vs ST-GNN.

Same question asked of both models: what moves DO in the north bay, and how well
can DO be recovered from everything else?

FCM  - node-day panel (L1, L2, L6, p2 bay), observed days only, node-demeaned,
       lag-1 ridge weights between 6 concepts, with week-block bootstrap intervals.
GNN  - trained checkpoint, held-out tail of the record (last 20%), DO channels
       masked at every node, then (a) DO reconstruction error vs simple baselines
       and (b) counterfactual DO response to +1 C / +1 ppt at the observed inputs.

Outputs: analysis/fcm_vs_gnn_*.csv, visualizations/32_*.png, 33_*.png
"""
import io
import contextlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from preprocess import build_dataset, ALL_FEATURES, build_net_water_forcing
from model import BayImputationGNN, Normaliser
from train import WIN_LEN, D_MODEL, N_GAT, N_GRU, DEVICE, CKPT_DIR

ANALYSIS_DIR = Path("analysis")
VIZ_DIR = Path("visualizations")

FI = {f: i for i, f in enumerate(ALL_FEATURES)}
NODE_ORDER = ["L0", "L1", "L2", "L6", "L7", "p2 bay", "crest5"]   # build_dataset node order
NORTH = {"L1": 1, "L2": 2, "L6": 3, "p2 bay": 5}
NORTH_TEST = ["L1", "L2", "p2 bay"]
SPD = 288
MIN_OBS = int(0.8 * SPD)
CONCEPTS = ["net_water", "air_tmax", "temp", "sal", "do"]
LABELS = {"net_water": "Net water\n(rain - PET)", "air_tmax": "Air temp max",
          "temp": "Water temp", "sal": "Salinity", "do": "Dissolved oxygen"}
N_FORCING = 2
ALPHA = 2.0
N_BOOT = 400
RNG = np.random.default_rng(7)


def load():
    with contextlib.redirect_stdout(io.StringIO()):
        data = build_dataset()
    t = pd.DatetimeIndex(data["time_index"])
    if t.tz is not None:
        t = t.tz_localize(None)
    _, net_w = build_net_water_forcing(t)
    return data, t, np.asarray(net_w, dtype=float)


# ---------------------------------------------------------------------------
# FCM on the observed-only node-day panel
# ---------------------------------------------------------------------------

def daily_panel(X, t, net_w, tmax):
    codes, days = pd.factorize(t.floor("D"))
    nd = len(days)
    day_net = np.bincount(codes, weights=net_w, minlength=nd) / np.maximum(np.bincount(codes, minlength=nd), 1)
    day_tmax = np.bincount(codes, weights=tmax, minlength=nd) / np.maximum(np.bincount(codes, minlength=nd), 1)
    frames = []
    for name, n in NORTH.items():
        cols = {}
        for key, feat in (("temp", "temp_c"), ("sal", "sal_ppt"), ("do", "odo_mgL")):
            v = X[:, n, FI[feat]]
            ok = ~np.isnan(v)
            cnt = np.bincount(codes, weights=ok, minlength=nd)
            s = np.bincount(codes, weights=np.where(ok, v, 0.0), minlength=nd)
            cols[key] = np.where(cnt >= MIN_OBS, s / np.maximum(cnt, 1), np.nan)
        df = pd.DataFrame(cols, index=days)
        df["net_water"] = day_net
        df["air_tmax"] = day_tmax
        df["node"] = name
        frames.append(df)
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.reset_index().dropna(subset=CONCEPTS)
    for c in ("temp", "sal", "do"):
        panel[c] = panel[c] - panel.groupby("node")[c].transform("mean")
    return panel


def make_pairs(panel):
    nxt = panel[["node", "date"] + CONCEPTS].copy()
    nxt["date"] = nxt["date"] - pd.Timedelta(days=1)
    m = panel.merge(nxt, on=["node", "date"], suffixes=("", "_next"))
    return m


def ridge(X, y, alpha):
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)


def fit_weights(Xz, Yz):
    """W[i, j]: influence of concept i (at t) on concept j (at t+1), z-score units."""
    C = len(CONCEPTS)
    W = np.zeros((C, C))
    for j in range(N_FORCING, C):
        W[:, j] = ridge(Xz, Yz[:, j], ALPHA)
    return W


def fcm_analysis():
    data, t, net_w = load()
    X = data["X"]
    panel = daily_panel(X, t, net_w, data["temp_max"])
    pairs = make_pairs(panel)
    print(f"FCM panel: {len(panel)} node-days, {len(pairs)} lag-1 pairs, "
          f"nodes={panel.node.value_counts().to_dict()}")
    print(f"  date range {panel.date.min().date()} -> {panel.date.max().date()}")

    mu = panel[CONCEPTS].mean().values
    sd = panel[CONCEPTS].std().values
    Xz = (pairs[CONCEPTS].values - mu) / sd
    Yz = (pairs[[c + "_next" for c in CONCEPTS]].values - mu) / sd
    W = fit_weights(Xz, Yz)

    # contemporaneous (lag 0) DO model on the same days, drivers only
    drv = [CONCEPTS.index(c) for c in ("net_water", "air_tmax", "temp", "sal")]
    pz = (panel[CONCEPTS].values - mu) / sd
    w0 = ridge(pz[:, drv], pz[:, CONCEPTS.index("do")], ALPHA)

    # week-block bootstrap
    block = ((pairs["date"] - pairs["date"].min()).dt.days // 7).values
    ublocks = np.unique(block)
    idx_by_block = {b: np.where(block == b)[0] for b in ublocks}
    pblock = ((panel["date"] - panel["date"].min()).dt.days // 7).values
    pidx_by_block = {b: np.where(pblock == b)[0] for b in np.unique(pblock)}
    boot1, boot0 = [], []
    j_do = CONCEPTS.index("do")
    for _ in range(N_BOOT):
        pick = RNG.choice(ublocks, size=len(ublocks), replace=True)
        rows = np.concatenate([idx_by_block[b] for b in pick])
        boot1.append(ridge(Xz[rows], Yz[rows, j_do], ALPHA))
        ppick = RNG.choice(list(pidx_by_block), size=len(pidx_by_block), replace=True)
        prow = np.concatenate([pidx_by_block[b] for b in ppick])
        boot0.append(ridge(pz[prow][:, drv], pz[prow, j_do], ALPHA))
    boot1, boot0 = np.array(boot1), np.array(boot0)
    lo1, hi1 = np.percentile(boot1, [5, 95], axis=0)
    lo0, hi0 = np.percentile(boot0, [5, 95], axis=0)

    # edge table (all non-forcing targets) with bootstrap interval for DO targets only
    rows = []
    for i, src in enumerate(CONCEPTS):
        for j, tgt in enumerate(CONCEPTS):
            if j < N_FORCING:
                continue
            rows.append({"source": src, "target": tgt, "lag_days": 1, "weight_z": W[i, j],
                         "ci5": lo1[i] if tgt == "do" else np.nan,
                         "ci95": hi1[i] if tgt == "do" else np.nan})
    edges = pd.DataFrame(rows)
    edges.to_csv(ANALYSIS_DIR / "fcm_vs_gnn_fcm_edges.csv", index=False)

    # one-step hold-out: chronological last 20% of pairs vs persistence
    order = np.argsort(pairs["date"].values)
    cut = int(0.8 * len(order))
    tr, te = order[:cut], order[cut:]
    Wt = fit_weights(Xz[tr], Yz[tr])
    skill = []
    for tgt in ("temp", "sal", "do"):
        j = CONCEPTS.index(tgt)
        y = Yz[te, j]
        pred = Xz[te] @ Wt[:, j]
        pers = Xz[te, j]
        sst = np.sum((y - y.mean()) ** 2)
        skill.append({"target": tgt, "n_test": len(te),
                      "rmse_fcm": np.sqrt(np.mean((pred - y) ** 2)),
                      "rmse_persistence": np.sqrt(np.mean((pers - y) ** 2)),
                      "r2_fcm": 1 - np.sum((pred - y) ** 2) / sst,
                      "r2_persistence": 1 - np.sum((pers - y) ** 2) / sst})
    skill = pd.DataFrame(skill)
    skill.to_csv(ANALYSIS_DIR / "fcm_vs_gnn_fcm_onestep.csv", index=False)
    print("\nFCM one-step hold-out (z units, last 20% of pairs):")
    print(skill.round(3).to_string(index=False))

    # convert to raw-unit sensitivities for DO: mg/L per +1 degC and per +1 ppt
    sd_do = sd[j_do]
    raw = {}
    for key in ("temp", "sal"):
        i = CONCEPTS.index(key)
        k0 = drv.index(i)
        raw[key] = {
            "lag1": (W[i, j_do] * sd_do / sd[i], lo1[i] * sd_do / sd[i], hi1[i] * sd_do / sd[i]),
            "lag0": (w0[k0] * sd_do / sd[i], lo0[k0] * sd_do / sd[i], hi0[k0] * sd_do / sd[i]),
        }
    print("\nFCM DO sensitivity (mg/L per unit, 90% block-bootstrap):")
    for key, d in raw.items():
        for lag, (v, lo, hi) in d.items():
            print(f"  {key:5s} {lag}: {v:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return W, edges, raw, panel, pairs


# ---------------------------------------------------------------------------
# GNN: DO reconstruction with DO masked, and counterfactual sensitivity
# ---------------------------------------------------------------------------

def gnn_analysis():
    data, t, _ = load()
    X = data["X"]
    T, N, F = X.shape
    with open(CKPT_DIR / "normaliser.pkl", "rb") as f:
        norm: Normaliser = pickle.load(f)
    forcing = np.stack([data["rain"], data["temp_min"], data["temp_max"]], -1).astype(np.float32)
    ts = np.array([x.timestamp() for x in pd.DatetimeIndex(data["time_index"])], dtype=np.float64)
    X_norm = norm.transform(np.nan_to_num(X, nan=0.0)).astype(np.float32)
    true_mask = (~np.isnan(X)).astype(np.float32)

    model = BayImputationGNN(n_features=F, n_nodes=N, n_forcing=3, d_model=D_MODEL,
                             n_gat_layers=N_GAT, n_gru_layers=N_GRU).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_DIR / "best.pt", map_location=DEVICE))
    model.eval()
    ei = torch.tensor(data["edge_index"], dtype=torch.long).to(DEVICE)
    ew = torch.tensor(data["edge_weight"], dtype=torch.float32).to(DEVICE)

    split = int(0.8 * T)
    n_win = (T - split) // WIN_LEN
    starts = split + np.arange(n_win) * WIN_LEN
    print(f"\nGNN held-out: {t[split]} -> {t[-1]}, {n_win} windows of {WIN_LEN}")

    mean_do, std_do = float(norm.mean[FI["odo_mgL"]]), float(norm.std[FI["odo_mgL"]])
    do_idx = [FI["odo_mgL"], FI["odo_pct"]]
    pred_base = np.full((T, N), np.nan, dtype=np.float32)
    d_temp = np.full((T, N), np.nan, dtype=np.float32)
    d_sal = np.full((T, N), np.nan, dtype=np.float32)
    dT = 1.0 / float(norm.std[FI["temp_c"]])
    dS = 1.0 / float(norm.std[FI["sal_ppt"]])

    def run(xb, mb, fb, tb):
        with torch.no_grad():
            _, pred, _ = model(xb, mb, fb, ei, ew, tb)
        return pred[..., FI["odo_mgL"]] * std_do + mean_do

    BS = 16
    for b0 in range(0, n_win, BS):
        s = starts[b0:b0 + BS]
        sl = [np.arange(a, a + WIN_LEN) for a in s]
        xb = torch.tensor(np.stack([X_norm[i] for i in sl])).to(DEVICE)
        mb = torch.tensor(np.stack([true_mask[i] for i in sl])).to(DEVICE)
        fb = torch.tensor(np.stack([forcing[i] for i in sl])).to(DEVICE)
        tb = torch.tensor(np.stack([ts[i] for i in sl])).float().to(DEVICE)
        xb = xb.clone(); mb = mb.clone()
        for c in do_idx:
            xb[..., c] = 0.0
            mb[..., c] = 0.0
        base = run(xb, mb, fb, tb)
        xt = xb.clone(); xt[..., FI["temp_c"]] += mb[..., FI["temp_c"]] * dT
        xs = xb.clone(); xs[..., FI["sal_ppt"]] += mb[..., FI["sal_ppt"]] * dS
        pt, ps = run(xt, mb, fb, tb), run(xs, mb, fb, tb)
        for k, idx in enumerate(sl):
            pred_base[idx] = base[k].cpu().numpy()
            d_temp[idx] = (pt[k] - base[k]).cpu().numpy()
            d_sal[idx] = (ps[k] - base[k]).cpu().numpy()
    return data, t, split, pred_base, d_temp, d_sal, true_mask


def baselines_and_scores(data, t, split, pred_gnn, d_temp, d_sal, true_mask):
    X = data["X"]
    tmax = data["temp_max"].astype(float)
    hour = t.hour.values + t.minute.values / 60.0
    ang = hour / 24.0 * 2 * np.pi
    hs, hc = np.sin(ang), np.cos(ang)
    hbin = t.hour.values
    T = len(t)
    covered = np.zeros(T, bool)
    covered[split:] = ~np.isnan(pred_gnn[split:]).all(axis=1)

    rows, sens, preds = [], [], {}
    for name, n in zip(NODE_ORDER, range(len(NODE_ORDER))):
        do = X[:, n, FI["odo_mgL"]]; tp = X[:, n, FI["temp_c"]]; sl = X[:, n, FI["sal_ppt"]]
        ok = ~np.isnan(do) & ~np.isnan(tp) & ~np.isnan(sl)
        tr = ok & (np.arange(T) < split)
        te = ok & covered
        if te.sum() < 200:
            continue
        feats = np.column_stack([tp, sl, hs, hc, tmax])
        lin = np.full(T, np.nan)
        hm = np.full(T, np.nan)
        tm = np.nan
        coef_raw = (np.nan, np.nan)
        if tr.sum() >= 500:
            mu, sg = feats[tr].mean(0), feats[tr].std(0)
            sg[sg < 1e-8] = 1.0
            Z = (feats - mu) / sg
            ym = do[tr].mean()
            w = ridge(Z[tr], do[tr] - ym, 1.0)
            lin = Z @ w + ym
            coef_raw = (w[0] / sg[0], w[1] / sg[1])
            tm = ym
            for h in range(24):
                sel = tr & (hbin == h)
                if sel.sum() > 20:
                    hm[hbin == h] = do[sel].mean()
        y = do[te]
        sst = np.sum((y - y.mean()) ** 2)

        def score(p):
            p = p[te]
            if np.isnan(p).any():
                return (np.nan, np.nan, np.nan)
            e = p - y
            dfr = pd.DataFrame({"d": t[te].floor("D"), "e": e})
            dd = pd.DataFrame({"d": t[te].floor("D"), "p": p, "y": y}).groupby("d").agg(
                p=("p", "mean"), y=("y", "mean"), n=("p", "size"))
            dd = dd[dd.n >= 100]
            return (np.sqrt(np.mean(e ** 2)), 1 - np.sum(e ** 2) / sst,
                    np.sqrt(np.mean((dd.p - dd.y) ** 2)) if len(dd) else np.nan)

        g = pred_gnn[:, n].astype(float)
        for method, p in (("GNN (DO masked)", g), ("Linear ridge", lin),
                          ("Hour-of-day mean", hm), ("Train mean", np.full(T, tm))):
            r, r2, rd = score(p)
            rows.append({"node": name, "method": method, "n": int(te.sum()),
                         "rmse_mgL": r, "r2": r2, "daily_mean_rmse_mgL": rd})
        preds[name] = {"te": te, "gnn": g, "lin": lin, "hm": hm}
        sens.append({"node": name,
                     "gnn_per_degC": np.nanmean(d_temp[te & ~np.isnan(d_temp[:, n]), n]),
                     "gnn_per_ppt": np.nanmean(d_sal[te & ~np.isnan(d_sal[:, n]), n]),
                     "linear_per_degC": coef_raw[0], "linear_per_ppt": coef_raw[1]})
    return pd.DataFrame(rows), pd.DataFrame(sens), preds


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_fcm_map(W, edges, thresh=0.02):
    pos = {"net_water": (0.10, 0.72), "air_tmax": (0.10, 0.28), "sal": (0.50, 0.72),
           "temp": (0.50, 0.28), "do": (0.88, 0.50)}
    colour = {"net_water": "#fff0d9", "air_tmax": "#fff0d9", "temp": "#e6eefb",
              "sal": "#e6eefb", "do": "#e6eefb"}
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    bw, bh = 0.16, 0.13
    for c, (x, y) in pos.items():
        ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh,
                                    boxstyle="round,pad=0.01,rounding_size=0.02",
                                    fc=colour[c], ec="#33415c", lw=1.6, zorder=3))
        ci_ = CONCEPTS.index(c)
        keep = f"\npersistence {W[ci_, ci_]:.2f}" if ci_ >= N_FORCING else ""
        ax.text(x, y, LABELS[c] + keep, ha="center", va="center", fontsize=10.5, fontweight="bold", zorder=4)
    ci = {(r.source, r.target): (r.ci5, r.ci95) for r in edges.itertuples() if r.target == "do"}
    shown = 0
    for i, s in enumerate(CONCEPTS):
        for j, tg in enumerate(CONCEPTS):
            w = W[i, j]
            if i == j or j < N_FORCING or abs(w) < thresh:
                continue
            lo, hi = ci.get((s, tg), (np.nan, np.nan))
            stable = (not np.isnan(lo)) and (lo > 0 or hi < 0)
            known = not np.isnan(lo)
            col = "#00796b" if w > 0 else "#c0501f"
            (x0, y0), (x1, y1) = pos[s], pos[tg]
            rad = 0.12
            arr = FancyArrowPatch((x0, y0), (x1, y1), connectionstyle=f"arc3,rad={rad}",
                                  arrowstyle="-|>", mutation_scale=18,
                                  lw=1.5 + 40 * abs(w), color=col,
                                  ls="-" if w > 0 else (0, (5, 3)),
                                  alpha=1.0 if (stable or not known) else 0.35,
                                  shrinkA=52, shrinkB=52, zorder=2)
            ax.add_patch(arr)
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            tag = f"{w:+.2f}" + (" *" if stable else "")
            ax.text(mx, my + 0.025, tag, ha="center", fontsize=9, color=col, zorder=5,
                    bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.85))
            shown += 1
    ax.text(0.5, 0.97, "North-bay FCM from observed-only 2025-26 sensor data",
            ha="center", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.925, "Lag-1 day weights (z-score units) | nodes L1, L2, L6, p2 bay | "
            "node means removed | cross-links |w| >= %.2f; own-yesterday dominates" % thresh, ha="center", fontsize=10.5, color="#444")
    ax.text(0.5, 0.045, "* = 90% week-block bootstrap interval excludes 0 (DO-target edges only); "
            "faded = interval includes 0. Self-loops omitted.", ha="center", fontsize=9.5, color="#444")
    ax.text(0.5, 0.012, "Learned lagged associations, not causal effects. "
            "Sensors changed at several nodes in Jun 2026.", ha="center", fontsize=9.5, color="#c0501f")
    fig.savefig(VIZ_DIR / "32_fcm_north_bay_2025_26.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved 32_fcm_north_bay_2025_26.png ({shown} edges)")


def plot_comparison(scores, sens, fcm_raw, preds, t):
    fig = plt.figure(figsize=(17, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1, 1])
    a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[0, 1]); a3 = fig.add_subplot(gs[1, :])
    methods = ["GNN (DO masked)", "Linear ridge", "Hour-of-day mean", "Train mean"]
    cols = ["#d81b60", "#1e88e5", "#8d8d8d", "#cfcfcf"]
    nodes = list(scores["node"].unique())
    w = 0.2
    for k, (m, c) in enumerate(zip(methods, cols)):
        v = [scores[(scores.node == nd) & (scores.method == m)].rmse_mgL.values[0] for nd in nodes]
        a1.bar(np.arange(len(nodes)) + (k - 1.5) * w, np.nan_to_num(v), w, label=m, color=c)
        for x, val in zip(np.arange(len(nodes)) + (k - 1.5) * w, v):
            if np.isnan(val):
                a1.text(x, 0.03, "n/a", rotation=90, ha="center", va="bottom", fontsize=7, color="#555")
    a1.set_xticks(range(len(nodes)))
    a1.set_xticklabels([n + (" (north)" if n in NORTH_TEST else "") for n in nodes], fontsize=8)
    a1.set_ylabel("RMSE, mg/L (lower is better)")
    a1.set_title("(a) Recovering DO with DO masked, held-out Jun-Sep 2026\n"
                 "(L6 omitted: no held-out data)", fontsize=10.5)
    a1.legend(fontsize=8, loc="upper left")

    labels = ["FCM lag 1 day", "FCM same day", "Linear (5-min)", "GNN counterfactual"]
    xs = np.arange(2)
    bw = 0.2
    vals = {
        "FCM lag 1 day": [fcm_raw["temp"]["lag1"], fcm_raw["sal"]["lag1"]],
        "FCM same day": [fcm_raw["temp"]["lag0"], fcm_raw["sal"]["lag0"]],
    }
    nt = sens[sens.node.isin(NORTH_TEST)]
    lin_t, lin_s = nt.linear_per_degC.mean(), nt.linear_per_ppt.mean()
    gnn_t, gnn_s = nt.gnn_per_degC.mean(), nt.gnn_per_ppt.mean()
    colors = ["#00796b", "#80cbc4", "#1e88e5", "#d81b60"]
    for k, lab in enumerate(labels):
        if lab in vals:
            v = [x[0] for x in vals[lab]]
            err = np.array([[x[0] - x[1], x[2] - x[0]] for x in vals[lab]]).T
            a2.bar(xs + (k - 1.5) * bw, v, bw, yerr=err, color=colors[k], label=lab, capsize=3)
        else:
            v = [lin_t, lin_s] if lab.startswith("Linear") else [gnn_t, gnn_s]
            a2.bar(xs + (k - 1.5) * bw, v, bw, color=colors[k], label=lab)
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xticks(xs)
    a2.set_xticklabels(["Water temp (per +1 °C)", "Salinity (per +1 ppt)"])
    a2.set_ylabel("Change in DO, mg/L")
    a2.set_title("(b) What does each model say moves DO?\n(north nodes L1, L2, p2; FCM bars show 90% bootstrap)",
                 fontsize=10.5)
    a2.legend(fontsize=8)

    nd = "L2"
    p = preds[nd]
    te = p["te"]
    idx = np.where(te & (t >= pd.Timestamp("2026-09-01")))[0]
    if len(idx) > 0:
        i0 = idx[0]
        win = (t >= t[i0]) & (t < t[i0] + pd.Timedelta(days=7))
        sel = win & te
        a3.plot(t[sel], p["obs"][sel], ".", ms=2.5, color="k", label="Observed DO")
        a3.plot(t[sel], p["gnn"][sel], "-", lw=1.1, color="#d81b60", label="GNN (DO masked)")
        if not np.isnan(p["lin"][sel]).all():
            a3.plot(t[sel], p["lin"][sel], "-", lw=1.1, color="#1e88e5", label="Linear ridge")
            a3.plot(t[sel], p["hm"][sel], "-", lw=1.1, color="#8d8d8d", label="Hour-of-day mean")
        a3.axhline(4, color="#c0501f", ls="--", lw=1)
        a3.text(a3.get_xlim()[0], 4.1, " 4 mg/L", color="#c0501f", fontsize=9)
    a3.set_ylabel("DO, mg/L")
    a3.set_title(f"(c) Example week, {nd}, Sep 2026: temperature, salinity and neighbours alone "
                 f"vs. observed DO", fontsize=10.5)
    a3.legend(ncol=4, fontsize=9, loc="upper right")
    fig.suptitle("FCM vs GNN on the 2025-26 north-bay data: two different questions", fontsize=15,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(VIZ_DIR / "33_fcm_vs_gnn_north_bay.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  saved 33_fcm_vs_gnn_north_bay.png")


def main():
    ANALYSIS_DIR.mkdir(exist_ok=True)
    VIZ_DIR.mkdir(exist_ok=True)
    W, edges, fcm_raw, _, _ = fcm_analysis()
    data, t, split, pred_gnn, d_temp, d_sal, true_mask = gnn_analysis()
    scores, sens, preds = baselines_and_scores(data, t, split, pred_gnn, d_temp, d_sal, true_mask)
    for nm, p in preds.items():
        n = NODE_ORDER.index(nm)
        p["obs"] = data["X"][:, n, FI["odo_mgL"]]
    scores.to_csv(ANALYSIS_DIR / "fcm_vs_gnn_nowcast.csv", index=False)
    sens.to_csv(ANALYSIS_DIR / "fcm_vs_gnn_sensitivity.csv", index=False)
    print("\nDO reconstruction (DO masked) on held-out tail:")
    print(scores.round(3).to_string(index=False))
    print("\nSensitivity of DO (mg/L) per node:")
    print(sens.round(4).to_string(index=False))
    plot_fcm_map(W, edges)
    plot_comparison(scores, sens, fcm_raw, preds, t)


if __name__ == "__main__":
    main()
