"""
horizon_followup.py -- confidence intervals and null models for the
prediction-horizon runs (Smart Health revision, Section 6.7).

The main analysis reported only point estimates for these runs. This adds,
for the baseline and three-branch model at each horizon h = 0..3:
  * anticipation rate, mean, median and IQR of ΔT, with 95% CIs from a
    bootstrap that resamples students
  * the alignment statistic (change points within ±24 h of an onset)
  * both null models (random change points, shifted labels), 1,000 draws
  * the change from h = 0 in anticipation rate and in mean ΔT, with a 95%
    student-bootstrap CI (unpaired: horizon runs have different folds)
ΔT is always measured against the CURRENT-label onsets.

Usage (repo root, venv active):
  python3 juno/horizon_followup.py --tag rev
Output: results/<tag>/T8b_horizon_inference.csv, T8c_horizon_vs_h0.csv,
        dt_pairs_horizon.csv, fig8b_horizon_nulls.pdf   (about 10 minutes)
"""
import argparse, os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_revision as A


def boot_students(df, stat, B):
    groups = {k: g["dt"].to_numpy() for k, g in df.groupby("student")}
    keys = list(groups)
    out = np.empty(B)
    for b in range(B):
        pick = A.RNG.choice(len(keys), len(keys), replace=True)
        out[b] = stat(np.concatenate([groups[keys[i]] for i in pick]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rev")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    B = 200 if a.quick else 2000
    out = os.path.join("results", a.tag); os.makedirs(out, exist_ok=True)
    with open(A.DATA, "rb") as f:
        raw = pickle.load(f)
    labels = {k: float(v[-1]) for k, v in raw["data"].items()}
    seqs, f1 = A.load_runs(a.tag, labels)

    ant = lambda v: 100 * np.mean(v < 0)
    rows, diffs, allpairs, boots = [], [], [], {}
    for fam in ("base", "p3"):
        for h in (0, 1, 2, 3):
            cfg = fam if h == 0 else f"{fam}_h{h}"
            if cfg not in set(f1.config):
                continue
            d = A.extract(seqs, "expected", configs=[cfg])
            d["family"], d["h"] = fam, h
            allpairs.append(d)
            v = d["dt"].to_numpy()
            ba = boot_students(d, ant, B); bm = boot_students(d, np.mean, B)
            boots[(fam, h)] = (ba, bm)
            nl = A.nulls(seqs, cfg, "expected", B // 2)
            rows.append(dict(
                family=fam, h=h, config=cfg, reps=d["rep"].nunique(), N=len(v),
                students=d["student"].nunique(),
                ant=ant(v), ant_lo=np.percentile(ba, 2.5), ant_hi=np.percentile(ba, 97.5),
                mean=v.mean(), mean_lo=np.percentile(bm, 2.5), mean_hi=np.percentile(bm, 97.5),
                median=np.median(v), iqr=np.subtract(*np.percentile(v, [75, 25])),
                tie0=100 * np.mean(v == 0), align=nl["obs_align"],
                rand_ant=nl["random_cp_ant_mean"], rand_ant_lo=nl["random_cp_ant_lo"],
                rand_ant_hi=nl["random_cp_ant_hi"],
                p_ant_above_rand=1 - nl["random_cp_p_ant_lower"] + 1 / (B // 2 + 1),
                rand_align=nl["random_cp_align_mean"], p_align_above_rand=nl["random_cp_p_align_higher"],
                shift_ant=nl["shifted_labels_ant_mean"], shift_ant_lo=nl["shifted_labels_ant_lo"],
                shift_ant_hi=nl["shifted_labels_ant_hi"],
                shift_align=nl["shifted_labels_align_mean"],
                p_align_above_shift=nl["shifted_labels_p_align_higher"]))
        # change from h = 0: students are resampled JOINTLY, i.e. the same
        # bootstrap sample of students is applied to both runs, so that the
        # dependence between results from the same students is respected
        base0 = next((d for d in allpairs if d["family"].iloc[0] == fam and d["h"].iloc[0] == 0), None)
        for h in (1, 2, 3):
            dh = next((d for d in allpairs if d["family"].iloc[0] == fam and d["h"].iloc[0] == h), None)
            if base0 is None or dh is None:
                continue
            g0 = {k: g["dt"].to_numpy() for k, g in base0.groupby("student")}
            gh = {k: g["dt"].to_numpy() for k, g in dh.groupby("student")}
            keys = sorted(set(g0) | set(gh)); empty = np.array([])
            da, dm = [], []
            for _ in range(B):
                pick = A.RNG.choice(len(keys), len(keys), replace=True)
                v0 = np.concatenate([g0.get(keys[i], empty) for i in pick])
                vh = np.concatenate([gh.get(keys[i], empty) for i in pick])
                if len(v0) and len(vh):
                    da.append(ant(vh) - ant(v0)); dm.append(vh.mean() - v0.mean())
            r0 = next(r for r in rows if r["family"] == fam and r["h"] == 0)
            rh = next(r for r in rows if r["family"] == fam and r["h"] == h)
            diffs.append(dict(
                family=fam, h=h, n_students=len(keys),
                d_ant=rh["ant"] - r0["ant"], d_ant_lo=np.percentile(da, 2.5), d_ant_hi=np.percentile(da, 97.5),
                d_mean=rh["mean"] - r0["mean"], d_mean_lo=np.percentile(dm, 2.5), d_mean_hi=np.percentile(dm, 97.5)))
    t = pd.DataFrame(rows); t.to_csv(f"{out}/T8b_horizon_inference.csv", index=False)
    dd = pd.DataFrame(diffs); dd.to_csv(f"{out}/T8c_horizon_vs_h0.csv", index=False)
    pd.concat(allpairs).to_csv(f"{out}/dt_pairs_horizon.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(t.round(2).to_string()); print(dd.round(2).to_string())

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 3.0), constrained_layout=True, sharey=True)
    for a_, fam, name in zip(ax, ("base", "p3"), ("LSTM baseline", "Personalised, 3 branches")):
        g = t[t.family == fam]
        a_.fill_between(g.h, g.rand_ant_lo, g.rand_ant_hi, color="#56B4E9", alpha=.4, label="Null: random change points")
        a_.fill_between(g.h, g.shift_ant_lo, g.shift_ant_hi, color="#CC79A7", alpha=.4, label="Null: shifted labels")
        a_.errorbar(g.h, g.ant, yerr=[g.ant - g.ant_lo, g.ant_hi - g.ant], fmt="k*-", ms=9, capsize=3, label="Observed (95% CI)")
        a_.set_title(name, loc="left"); a_.set_xticks([0, 1, 2, 3]); a_.set_xlabel("Forecast horizon (EMA responses ahead)")
    ax[0].set_ylabel("Anticipation rate (%)")
    h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    fig.savefig(f"{out}/fig8b_horizon_nulls.pdf", bbox_inches="tight"); fig.savefig(f"{out}/fig8b_horizon_nulls.png", dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
