"""
analysis_revision.py -- every StudentLife number, table and figure for the
Smart Health revision (SMHL-D-26-00918), computed from ONE set of training
records so figures and tables cannot disagree (Reviewer 1, Q1/Q2).

Inputs (repo root):
  data/cross_val_scores/*_<tag>_<config>_r<rep>.pkl   (from run_one.py)
  data/training_data/shuffled_splits/...all_students.pkl  (labels, onsets)

Outputs: results/<tag>/  (CSV tables, summary.json, fig*.pdf/png)

What it computes, mapped to reviewer comments
  R1-1/R2-2  stress trajectory s_t = sum_c c*p_t(c)  (max-prob kept for App. A)
  R2-1       null models: random change points, circularly shifted labels
  R1-5       onset-paired comparisons across models, student-cluster bootstrap
  R1-3       TOST equivalence on the paired mean difference + smallest margin
  R1 sugg.   PELT penalty x matching-window sensitivity grid
  R1 sugg.   exact p, Hodges-Lehmann shift and CI for the asymmetric loss
  R2-3       prediction-horizon runs: F1 and ΔT versus horizon
             fixed-k (Dynp) controlled segmentation, re-run on s_t

Usage:
  python3 juno/analysis_revision.py --tag rev
  python3 juno/analysis_revision.py --tag rev --quick     (fewer resamples)
"""
import argparse
import glob
import json
import os
import pickle
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats

DATA = ("data/training_data/shuffled_splits/"
        "training_date_normalized_shuffled_splits_"
        "select_features_no_prev_stress_all_students.pkl")
MAIN = ["base", "p1", "p3"]
NAMES = {"base": "LSTM baseline", "p1": "Personalised, 1 br.",
         "p3": "Personalised, 3 br.", "base_asym05": "Asymmetric, λ=0.5",
         "base_asym10": "Asymmetric, λ=1.0"}
PEN, WIN, JUMP, MIN_EX = 1.0, 72.0, 1, 5
RNG = np.random.default_rng(20261022)


# ───────────────────────── loading ─────────────────────────
def t_hours(key):
    p = key.split("_")
    return (datetime(2012, int(p[1]), int(p[2]), int(p[3]))
            - datetime(2012, 1, 1)).total_seconds() / 3600.0


def softmax(x):
    x = np.asarray(x, float)
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def load_runs(tag, labels):
    """-> seqs: list of dict(config, rep, fold, student, t, y, P), f1 rows."""
    seqs, f1 = [], []
    pat = re.compile(rf"_{re.escape(tag)}_(.+)_r(\d+)\.pkl$")
    for path in sorted(glob.glob("data/cross_val_scores/*.pkl")):
        m = pat.search(path)
        if not m:
            continue
        cfg, rep = m.group(1), int(m.group(2))
        with open(path, "rb") as f:
            folds = pickle.load(f)
        for fold, rec in enumerate(folds):
            if not rec or "val_ids" not in rec:
                continue
            mic = rec["val_f1"]["micro"]
            best = int(np.argmax(mic))
            f1.append(dict(config=cfg, rep=rep, fold=fold,
                           micro=mic[best], macro=rec["val_f1"]["macro"][best]))
            P = softmax(rec["outputs"][best])
            ids = rec["val_ids"]
            assert len(ids) == len(P), path
            by = defaultdict(list)
            for k, p in zip(ids, P):
                by[k.split("_")[0]].append((t_hours(k), labels[k], p))
            for s, rows in by.items():
                if len(rows) < MIN_EX:
                    continue
                rows.sort(key=lambda r: r[0])
                seqs.append(dict(config=cfg, rep=rep, fold=fold, student=s,
                                 t=np.array([r[0] for r in rows]),
                                 y=np.array([r[1] for r in rows], float),
                                 P=np.vstack([r[2] for r in rows])))
    return seqs, pd.DataFrame(f1)


# ───────────────────────── ΔT core ─────────────────────────
def trajectory(P, kind):
    if kind == "expected":
        return P @ np.arange(P.shape[1], dtype=float)
    if kind == "maxprob":
        return P.max(1)
    if kind == "phigh":
        return P[:, -1]
    raise ValueError(kind)


def change_points(x, method="pelt", param=PEN, jump=JUMP):
    """Indices where a new segment starts; the ruptures end sentinel is dropped."""
    x = x.reshape(-1, 1)
    if method == "pelt":
        bk = rpt.Pelt(model="rbf", min_size=2, jump=jump).fit(x).predict(pen=param)
    else:  # dynp with exactly `param` change points
        if len(x) < param + 2:
            return np.array([], int)
        bk = rpt.Dynp(model="rbf", min_size=1, jump=1).fit(x).predict(n_bkps=param)
    return np.array([b for b in bk if b < len(x)], int)


def onsets(t, y):
    i = np.where(np.diff(y) > 0)[0] + 1
    return t[i]


def match_cp(cp_t, on_t, win):
    """CP-centric (as in the paper): each CP -> nearest onset within win."""
    out = []
    if len(on_t) == 0:
        return out
    for c in cp_t:
        d = c - on_t
        j = np.argmin(np.abs(d))
        if abs(d[j]) <= win:
            out.append((on_t[j], d[j]))
    return out


def extract(seqs, kind="expected", method="pelt", param=PEN, win=WIN,
            jump=JUMP, configs=None):
    rows = []
    for s in seqs:
        if configs and s["config"] not in configs:
            continue
        x = trajectory(s["P"], kind)
        cp = change_points(x, method, param, jump)
        on = onsets(s["t"], s["y"])
        for o, d in match_cp(s["t"][cp], on, win):
            rows.append(dict(config=s["config"], rep=s["rep"], fold=s["fold"],
                             student=s["student"], onset=o, dt=d))
    return pd.DataFrame(rows, columns=["config", "rep", "fold", "student",
                                       "onset", "dt"])


def summarise(df, by="config"):
    out = []
    for c, g in df.groupby(by):
        per_rep = g.groupby("rep")["dt"].apply(lambda v: 100 * (v < 0).mean())
        d = g["dt"].to_numpy()
        out.append(dict(config=c, N=len(d), mean=d.mean(), median=np.median(d),
                        std=d.std(ddof=1), iqr=np.subtract(*np.percentile(d, [75, 25])),
                        ant=100 * (d < 0).mean(), ant_rep_mean=per_rep.mean(),
                        ant_rep_sd=per_rep.std(ddof=1), tie0=100 * (d == 0).mean(),
                        n_onsets=g[["fold", "student", "onset"]].drop_duplicates().shape[0],
                        n_students=g["student"].nunique()))
    return pd.DataFrame(out)


# ───────────────────── statistics helpers ─────────────────────
def cluster_boot(units, stat, B, cluster="student"):
    """Resample clusters (students) with replacement; returns B statistics."""
    groups = {k: g for k, g in units.groupby(cluster)}
    keys = np.array(list(groups))
    res = np.empty(B)
    for b in range(B):
        pick = RNG.choice(keys, len(keys), replace=True)
        res[b] = stat(pd.concat([groups[k] for k in pick]))
    return res


def hodges_lehmann(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) * len(b) > 4e6:  # subsample for memory
        a = RNG.choice(a, 2000); b = RNG.choice(b, 2000)
    return np.median(np.subtract.outer(a, b))


def onset_table(seqs, configs, kind="expected", param=PEN, win=WIN, jump=JUMP):
    """Onset-centric ΔT: for every onset, nearest CP within win (NaN if none),
    then the median across repetitions. Unit = (fold, student, onset)."""
    rows = []
    for s in seqs:
        if s["config"] not in configs:
            continue
        cp_t = s["t"][change_points(trajectory(s["P"], kind), "pelt", param, jump)]
        for o in onsets(s["t"], s["y"]):
            d = cp_t - o
            v = d[np.argmin(np.abs(d))] if len(d) and np.min(np.abs(d)) <= win else np.nan
            rows.append(dict(config=s["config"], rep=s["rep"], fold=s["fold"],
                             student=s["student"], onset=o, dt=v))
    df = pd.DataFrame(rows)
    return (df.groupby(["config", "fold", "student", "onset"])["dt"]
              .median().unstack("config").reset_index())


def paired(wide, a, b, B, margin):
    u = wide.dropna(subset=[a, b]).copy()
    u["d"] = u[a] - u[b]
    if len(u) < 5:
        return dict(pair=f"{a}-{b}", n_onsets=len(u))
    w = stats.wilcoxon(u["d"], zero_method="zsplit") if (u["d"] != 0).any() else None
    boot = cluster_boot(u, lambda g: g["d"].mean(), B)
    lo90, hi90 = np.percentile(boot, [5, 95])
    lo95, hi95 = np.percentile(boot, [2.5, 97.5])
    p_tost = max((boot <= -margin).mean(), (boot >= margin).mean())
    return dict(pair=f"{a}-{b}", n_onsets=len(u), n_students=u["student"].nunique(),
                mean_diff=u["d"].mean(), median_diff=u["d"].median(),
                ci95_lo=lo95, ci95_hi=hi95, ci90_lo=lo90, ci90_hi=hi90,
                wilcoxon_p=(w.pvalue if w else 1.0),
                tost_margin=margin, tost_p_boot=p_tost,
                equivalent=bool(lo90 > -margin and hi90 < margin),
                smallest_equiv_margin=max(abs(lo90), abs(hi90)))


# ───────────────────────── nulls ─────────────────────────
def nulls(seqs, cfg, kind, B, win=WIN):
    """Observed vs two nulls, for anticipation rate, median ΔT and the share
    of change points within ±24 h of an onset (alignment)."""
    S = [s for s in seqs if s["config"] == cfg]
    pre = []
    for s in S:
        cp = change_points(trajectory(s["P"], kind))
        pre.append((s, cp, onsets(s["t"], s["y"])))

    def summ(pairs, ncp):
        d = np.array([x[1] for x in pairs])
        if len(d) == 0:
            return (np.nan, np.nan, np.nan)
        return (100 * (d < 0).mean(), np.median(d), 100 * (np.abs(d) <= 24).sum() / max(ncp, 1))

    obs_pairs, ncp = [], 0
    for s, cp, on in pre:
        obs_pairs += match_cp(s["t"][cp], on, win); ncp += len(cp)
    obs = summ(obs_pairs, ncp)

    out = {}
    for name in ("random_cp", "shifted_labels"):
        res = []
        for _ in range(B):
            pairs = []
            for s, cp, on in pre:
                n = len(s["t"])
                if name == "random_cp":
                    k = min(len(cp), n - 1)
                    rcp = np.sort(RNG.choice(np.arange(1, n), k, replace=False)) if k else np.array([], int)
                    pairs += match_cp(s["t"][rcp], on, win)
                else:
                    y = np.roll(s["y"], RNG.integers(1, n))
                    pairs += match_cp(s["t"][cp], onsets(s["t"], y), win)
            res.append(summ(pairs, ncp))
        res = np.array(res)
        out[name] = dict(
            ant_mean=np.nanmean(res[:, 0]), ant_lo=np.nanpercentile(res[:, 0], 2.5),
            ant_hi=np.nanpercentile(res[:, 0], 97.5),
            med_mean=np.nanmean(res[:, 1]), align_mean=np.nanmean(res[:, 2]),
            align_lo=np.nanpercentile(res[:, 2], 2.5), align_hi=np.nanpercentile(res[:, 2], 97.5),
            p_ant_lower=(np.sum(res[:, 0] <= obs[0]) + 1) / (B + 1),
            p_align_higher=(np.sum(res[:, 2] >= obs[2]) + 1) / (B + 1))
    return dict(config=cfg, obs_ant=obs[0], obs_median=obs[1], obs_align=obs[2], **{
        f"{n}_{k}": v for n, d in out.items() for k, v in d.items()})


# ───────────────────────── figures ─────────────────────────
def figures(out, dt_main, wide, null_df, grid, hz, asym, dyn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 150})
    cols = {"base": "#1f77b4", "p1": "#ff7f0e", "p3": "#2ca02c"}

    def save(fig, name):
        fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(out, name + ".png"), bbox_inches="tight", dpi=300)
        plt.close(fig)

    # Fig 2: histograms (numbers in panel titles come from the same table)
    fig, ax = plt.subplots(1, 3, figsize=(10, 3), sharey=True, constrained_layout=True)
    bins = np.arange(-72, 76, 6)
    for a, c in zip(ax, MAIN):
        d = dt_main.loc[dt_main.config == c, "dt"].to_numpy()
        a.hist(d[d < 0], bins=bins, color="#2e7d32", alpha=.8, label="ΔT < 0")
        a.hist(d[d >= 0], bins=bins, color="#c62828", alpha=.8, label="ΔT ≥ 0")
        a.axvline(np.median(d), color="k", ls="--", lw=1)
        a.set_title(f"{NAMES[c]}\nN = {len(d)}, anticipation {100*(d<0).mean():.1f}%")
        a.set_xlabel("ΔT (hours)")
    ax[0].set_ylabel("Matched pairs"); ax[0].legend(frameon=False, fontsize=8)
    save(fig, "fig2_delta_t_hist")

    # Fig 3: KDE
    fig, a = plt.subplots(figsize=(5, 3), constrained_layout=True)
    xs = np.linspace(-72, 72, 400)
    for c in MAIN:
        d = dt_main.loc[dt_main.config == c, "dt"].to_numpy()
        a.plot(xs, stats.gaussian_kde(d)(xs), color=cols[c], label=NAMES[c])
    a.axvline(0, color="grey", lw=.8); a.set_xlabel("ΔT (hours)"); a.set_ylabel("Density")
    a.legend(frameon=False, fontsize=8)
    save(fig, "fig3_delta_t_kde")

    # Fig 4: paired differences with 95% CI and equivalence margin
    fig, a = plt.subplots(figsize=(5, 2.6), constrained_layout=True)
    pr = pd.DataFrame([p for p in wide if "mean_diff" in p])
    y = np.arange(len(pr))
    a.errorbar(pr["mean_diff"], y, xerr=[pr["mean_diff"] - pr["ci95_lo"],
               pr["ci95_hi"] - pr["mean_diff"]], fmt="o", color="k", capsize=3)
    m = pr["tost_margin"].iloc[0]
    a.axvspan(-m, m, color="#bbdefb", alpha=.6, label=f"±{m:g} h margin")
    a.axvline(0, color="grey", lw=.8)
    a.set_yticks(y, [p.replace("base", "Baseline").replace("p1", "1 br.").replace("p3", "3 br.").replace("-", " − ") for p in pr["pair"]])
    a.set_xlabel("Paired mean ΔT difference (hours), 95% cluster-bootstrap CI")
    a.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "fig4_paired_differences")

    # Fig 5: fixed-k control
    fig, ax = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
    for i, c in enumerate(MAIN):
        g = dyn[dyn.config == c]
        ax[0].errorbar(g["k"] + (i - 1) * .08, g["ant_rep_mean"], yerr=g["ant_rep_sd"],
                       fmt="o-", color=cols[c], capsize=3, label=NAMES[c])
    ax[0].set_xlabel("Change points per subject (k)"); ax[0].set_ylabel("Anticipation rate (%)")
    ax[0].set_xticks([1, 2, 3]); ax[0].legend(frameon=False, fontsize=8)
    for c in MAIN:
        g = dyn[dyn.config == c]
        ax[1].plot(g["k"], g["median"], "o-", color=cols[c])
    ax[1].set_xlabel("Change points per subject (k)"); ax[1].set_ylabel("Median ΔT (hours)")
    ax[1].set_xticks([1, 2, 3])
    save(fig, "fig5_fixed_k")

    # Fig 6: asymmetric loss
    if asym is not None and len(asym):
        fig, ax = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
        order = ["base", "base_asym05", "base_asym10"]
        dd = [dt_main_all.loc[dt_main_all.config == c, "dt"] for c in order if c in set(dt_main_all.config)]
        ax[0].boxplot(dd, showfliers=False)
        ax[0].set_xticks(range(1, len(dd) + 1), ["Symmetric", "λ = 0.5", "λ = 1.0"][:len(dd)])
        ax[0].axhline(0, color="grey", lw=.8); ax[0].set_ylabel("ΔT (hours)")
        ax[1].bar(range(len(asym)), asym["macro_f1"], yerr=asym["macro_f1_sd"], capsize=3,
                  color=["#90a4ae", "#4db6ac", "#00897b"][:len(asym)])
        ax[1].set_xticks(range(len(asym)), ["Symmetric", "λ = 0.5", "λ = 1.0"][:len(asym)])
        ax[1].set_ylabel("Macro-F1"); ax[1].set_ylim(asym["macro_f1"].min() - .05, asym["macro_f1"].max() + .03)
        save(fig, "fig6_asymmetric_loss")

    # Fig 7: nulls
    fig, ax = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
    x = np.arange(len(null_df))
    for j, (k, lab) in enumerate([("random_cp", "Random change points"), ("shifted_labels", "Shifted labels")]):
        ax[0].errorbar(x + (j - .5) * .25, null_df[f"{k}_ant_mean"],
                       yerr=[null_df[f"{k}_ant_mean"] - null_df[f"{k}_ant_lo"],
                             null_df[f"{k}_ant_hi"] - null_df[f"{k}_ant_mean"]],
                       fmt="s", capsize=3, label=f"Null: {lab}", alpha=.8)
        ax[1].errorbar(x + (j - .5) * .25, null_df[f"{k}_align_mean"],
                       yerr=[null_df[f"{k}_align_mean"] - null_df[f"{k}_align_lo"],
                             null_df[f"{k}_align_hi"] - null_df[f"{k}_align_mean"]],
                       fmt="s", capsize=3, alpha=.8)
    ax[0].plot(x, null_df["obs_ant"], "k*", ms=10, label="Observed")
    ax[1].plot(x, null_df["obs_align"], "k*", ms=10)
    for a in ax:
        a.set_xticks(x, [NAMES[c].replace(", ", "\n") for c in null_df["config"]])
    ax[0].set_ylabel("Anticipation rate (%)"); ax[0].legend(frameon=False, fontsize=7)
    ax[1].set_ylabel("Change points within ±24 h of an onset (%)")
    save(fig, "fig7_null_models")

    # Fig 8: horizon
    if hz is not None and len(hz):
        fig, ax = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
        for fam, col in (("base", cols["base"]), ("p3", cols["p3"])):
            g = hz[hz.family == fam].sort_values("h")
            ax[0].errorbar(g["h"], g["micro"], yerr=g["micro_sd"], fmt="o-", color=col, capsize=3, label=NAMES[fam])
            ax[1].plot(g["h"], g["ant"], "o-", color=col)
        ax[0].set_xlabel("Forecast horizon (EMA responses ahead)"); ax[0].set_ylabel("Micro-F1")
        ax[1].set_xlabel("Forecast horizon (EMA responses ahead)"); ax[1].set_ylabel("Anticipation rate (%)")
        for a in ax: a.set_xticks([0, 1, 2, 3])
        ax[0].legend(frameon=False, fontsize=8)
        save(fig, "fig8_horizon")

    # Fig 9: sensitivity grid (anticipation rate per model)
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8), constrained_layout=True, sharey=True)
    for a, c in zip(ax, MAIN):
        g = grid[grid.config == c].pivot(index="pen", columns="win", values="ant")
        im = a.imshow(g.values, cmap="viridis", vmin=grid["ant"].min(), vmax=grid["ant"].max(), aspect="auto")
        a.set_xticks(range(g.shape[1]), [f"{w:g}" for w in g.columns])
        a.set_yticks(range(g.shape[0]), [f"{p:g}" for p in g.index])
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                a.text(j, i, f"{g.values[i, j]:.0f}", ha="center", va="center", color="w", fontsize=7)
        a.set_title(NAMES[c]); a.set_xlabel("Matching window (h)")
    ax[0].set_ylabel("PELT penalty")
    fig.colorbar(im, ax=ax, label="Anticipation rate (%)", shrink=.9)
    save(fig, "fig9_sensitivity")


# ───────────────────────── main ─────────────────────────
def main():
    global dt_main_all
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rev")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--margin", type=float, default=6.0,
                    help="TOST equivalence margin in hours")
    ap.add_argument("--data", default=DATA)
    a = ap.parse_args()
    B = 200 if a.quick else 2000
    out = os.path.join("results", a.tag); os.makedirs(out, exist_ok=True)

    with open(a.data, "rb") as f:
        raw = pickle.load(f)
    labels = {k: float(v[-1]) for k, v in raw["data"].items()}
    seqs, f1 = load_runs(a.tag, labels)
    print(f"{len(seqs)} subject-fold sequences, configs: {sorted(set(s['config'] for s in seqs))}")
    if f1.empty:
        raise SystemExit(f"No training records found for tag '{a.tag}' in "
                         "data/cross_val_scores/. Did the training jobs finish? "
                         "Check logs/*.err.")
    J = {}

    # T1 reproduction F1
    t1 = f1.groupby("config").agg(micro=("micro", "mean"), micro_sd=("micro", "std"),
                                  macro=("macro", "mean"), macro_sd=("macro", "std"),
                                  n_folds=("micro", "size")).reset_index()
    t1.to_csv(f"{out}/T1_f1.csv", index=False); print(t1.round(4))

    # T2 main ΔT, s_t; and App. A max-prob; and original ruptures defaults (jump=5)
    cfgs_main = [c for c in ["base", "p1", "p3", "base_asym05", "base_asym10"]
                 if c in set(f1.config)]
    dt_main_all = extract(seqs, "expected", configs=cfgs_main)
    dt_main = dt_main_all[dt_main_all.config.isin(MAIN)]
    dt_main_all.to_csv(f"{out}/dt_pairs_expected.csv", index=False)
    t2 = summarise(dt_main_all); t2.to_csv(f"{out}/T2_dt_expected.csv", index=False)
    print(t2.round(2))
    tA = pd.concat([summarise(extract(seqs, "maxprob", configs=MAIN)).assign(traj="maxprob"),
                    summarise(extract(seqs, "phigh", configs=MAIN)).assign(traj="P(high)"),
                    summarise(extract(seqs, "maxprob", configs=MAIN, jump=5)).assign(traj="maxprob, jump=5 (as submitted)")])
    tA.to_csv(f"{out}/TA_trajectory_sensitivity.csv", index=False)

    if not set(MAIN) <= set(f1.config):
        raise SystemExit(f"Stopping after Tables 1-2: comparisons need all of {MAIN}; "
                         f"found {sorted(set(f1.config))}. Re-run when training is complete.")

    # pooled W and MW (kept for continuity, now secondary)
    rows = []
    for i, x in enumerate(MAIN):
        for y in MAIN[i + 1:]:
            dx = dt_main.loc[dt_main.config == x, "dt"]; dy = dt_main.loc[dt_main.config == y, "dt"]
            rows.append(dict(pair=f"{x}-{y}", W=stats.wasserstein_distance(dx, dy),
                             MW_p=stats.mannwhitneyu(dx, dy).pvalue))
    pd.DataFrame(rows).to_csv(f"{out}/T2b_pooled_W_MW.csv", index=False)

    # T3 paired + TOST (R1-3, R1-5)
    wide = onset_table(seqs, MAIN)
    pairs = [paired(wide, x, y, B, a.margin) for i, x in enumerate(MAIN) for y in MAIN[i + 1:]]
    pd.DataFrame(pairs).to_csv(f"{out}/T3_paired_tost.csv", index=False)
    print(pd.DataFrame(pairs).round(3).to_string())

    # T4 fixed-k
    dyn = []
    for k in (1, 2, 3):
        s = summarise(extract(seqs, "expected", "dynp", k, configs=MAIN)); s["k"] = k; dyn.append(s)
    dyn = pd.concat(dyn); dyn.to_csv(f"{out}/T4_fixed_k.csv", index=False)

    # T5 nulls (R2-1)
    null_df = pd.DataFrame([nulls(seqs, c, "expected", B // 2) for c in MAIN])
    null_df.to_csv(f"{out}/T5_nulls.csv", index=False); print(null_df.round(2).T)

    # T6 sensitivity grid
    grid = []
    for pen in (0.5, 1.0, 2.0, 5.0):
        for win in (24.0, 48.0, 72.0, 96.0):
            s = summarise(extract(seqs, "expected", "pelt", pen, win, configs=MAIN))
            s["pen"], s["win"] = pen, win; grid.append(s)
    grid = pd.concat(grid); grid.to_csv(f"{out}/T6_sensitivity_grid.csv", index=False)

    # T7 asymmetric loss (exact p, HL shift, cluster CI, paired by onset)
    asym = None
    if {"base_asym05", "base_asym10"} & set(f1.config):
        rows = []
        wide_a = onset_table(seqs, ["base", "base_asym05", "base_asym10"])
        for c in ["base", "base_asym05", "base_asym10"]:
            if c not in set(f1.config):
                continue
            d = dt_main_all.loc[dt_main_all.config == c, "dt"]
            r = dict(config=c, N=len(d), mean=d.mean(), median=d.median(), ant=100 * (d < 0).mean(),
                     macro_f1=f1.loc[f1.config == c, "macro"].mean(),
                     macro_f1_sd=f1.loc[f1.config == c, "macro"].std())
            if c != "base":
                b = dt_main_all.loc[dt_main_all.config == "base", "dt"]
                both = dt_main_all[dt_main_all.config.isin(["base", c])]
                boot = cluster_boot(both, lambda g: hodges_lehmann(
                    g.loc[g.config == c, "dt"], g.loc[g.config == "base", "dt"]), B // 4)
                fb = f1[f1.config == "base"].set_index(["rep", "fold"])["macro"]
                fc = f1[f1.config == c].set_index(["rep", "fold"])["macro"]
                jj = fb.index.intersection(fc.index)
                pr = paired(wide_a, c, "base", B, a.margin)
                r.update(MW_p=stats.mannwhitneyu(d, b).pvalue, HL=hodges_lehmann(d, b),
                         HL_lo=np.percentile(boot, 2.5), HL_hi=np.percentile(boot, 97.5),
                         paired_mean=pr.get("mean_diff"), paired_lo=pr.get("ci95_lo"),
                         paired_hi=pr.get("ci95_hi"),
                         macro_gain=(fc[jj] - fb[jj]).mean() * 100,
                         macro_wilcoxon_p=stats.wilcoxon(fc[jj], fb[jj]).pvalue)
            rows.append(r)
        asym = pd.DataFrame(rows); asym.to_csv(f"{out}/T7_asymmetric.csv", index=False)
        print(asym.round(3).to_string())

    # T8 horizon (R2-3); onsets always from the CURRENT labels
    hz = None
    if any(c.endswith("_h1") for c in set(f1.config)):
        rows = []
        for fam in ("base", "p3"):
            for h in (0, 1, 2, 3):
                c = fam if h == 0 else f"{fam}_h{h}"
                if c not in set(f1.config):
                    continue
                d = extract(seqs, "expected", configs=[c])["dt"]
                g = f1[f1.config == c]
                rows.append(dict(family=fam, h=h, config=c, micro=g.micro.mean(), micro_sd=g.micro.std(),
                                 macro=g.macro.mean(), N=len(d), median=d.median(),
                                 ant=100 * (d < 0).mean()))
        hz = pd.DataFrame(rows); hz.to_csv(f"{out}/T8_horizon.csv", index=False); print(hz.round(3))

    figures(out, dt_main, pairs, null_df, grid, hz, asym, dyn)
    J.update(n_sequences=len(seqs), margin_h=a.margin, B=B, pen=PEN, win=WIN, jump=JUMP)
    json.dump(J, open(f"{out}/summary.json", "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
