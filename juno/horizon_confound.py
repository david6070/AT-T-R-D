"""
horizon_confound.py -- separates the effect of forecast training from the
effect of the changed evaluation sequences (Smart Health revision, Sec. 6.7).

The forecast runs (h = 1, 2, 3) are validated on different sequences from the
h = 0 runs, so a raw comparison mixes two things. No retraining is needed to
separate them:

  A. SAME-SEQUENCE COMPARISON. Every example has an out-of-fold prediction
     from the h = 0 model of the same repetition (each example is in exactly
     one validation fold). For each validation sequence of a forecast run we
     rebuild the h = 0 model's trajectory on exactly the same examples, run
     the same change-point detection, and compare. Sequences and onsets are
     identical, so the comparison is paired by onset, with students
     resampled for the intervals. Repetitions 0-4 of the h = 0 run are used
     to match the five forecast repetitions.
       caveat: within one sequence the h = 0 predictions come from the five
       fold-models of that repetition, each out-of-fold for its examples.

  B. NULL FOR THE MEAN. Mean ΔT under random change points, for each run's
     own sequences (the main analysis only gave the null anticipation rate).

Usage (repo root, venv active):   python3 juno/horizon_confound.py --tag rev
Output: results/<tag>/T8d_same_sequence.csv, T8e_null_mean.csv,
        dt_pairs_same_sequence.csv            (about 10 minutes)
"""
import argparse, glob, os, pickle, re, sys
from collections import defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_revision as A

WIN, NCOMP = A.WIN, 6


def load_keyed(tag):
    """-> {(config, rep): [ {fold, ids, P} ... ]}, P at the peak-F1 epoch."""
    out = defaultdict(list)
    pat = re.compile(rf"_{re.escape(tag)}_(.+)_r(\d+)\.pkl$")
    for path in sorted(glob.glob("data/cross_val_scores/*.pkl")):
        m = pat.search(path)
        if not m:
            continue
        with open(path, "rb") as f:
            folds = pickle.load(f)
        for fold, rec in enumerate(folds):
            if not rec or "val_ids" not in rec:
                continue
            best = int(np.argmax(rec["val_f1"]["micro"]))
            out[(m.group(1), int(m.group(2)))].append(
                dict(fold=fold, ids=list(rec["val_ids"]), P=A.softmax(rec["outputs"][best])))
    return out


def sequences(ids, prob_of, labels):
    """Group example keys by student, sort by time; prob_of maps key -> probs."""
    by = defaultdict(list)
    for k in ids:
        if k in prob_of:
            by[k.split("_")[0]].append((A.t_hours(k), labels[k], prob_of[k]))
    for s, rows in by.items():
        if len(rows) < A.MIN_EX:
            continue
        rows.sort(key=lambda r: r[0])
        yield s, np.array([r[0] for r in rows]), np.array([r[1] for r in rows], float), np.vstack([r[2] for r in rows])


def qint(a, level):
    return np.percentile(a, [100 * level / 2, 100 * (1 - level / 2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rev")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    B = 300 if a.quick else 5000
    out = os.path.join("results", a.tag); os.makedirs(out, exist_ok=True)
    with open(A.DATA, "rb") as f:
        raw = pickle.load(f)
    labels = {k: float(v[-1]) for k, v in raw["data"].items()}
    runs = load_keyed(a.tag)
    ant = lambda v: 100 * np.mean(v < 0)

    cp_rows, on_rows, null_rows = [], [], []
    for fam in ("base", "p3"):
        # out-of-fold h = 0 predictions, per repetition: key -> probs
        p0 = {}
        for (cfg, rep), folds in runs.items():
            if cfg == fam:
                p0[rep] = {k: p for fd in folds for k, p in zip(fd["ids"], fd["P"])}
        for h in (1, 2, 3):
            cfg = f"{fam}_h{h}"
            reps = sorted(r for (c, r) in runs if c == cfg and r in p0)
            if not reps:
                continue
            null_means = [[] for _ in range(B // 5)]
            for rep in reps:
                for fd in runs[(cfg, rep)]:
                    ph = dict(zip(fd["ids"], fd["P"]))
                    for (s, t, y, Ph), (_, _, _, P0) in zip(sequences(fd["ids"], ph, labels),
                                                             sequences(fd["ids"], p0[rep], labels)):
                        on = A.onsets(t, y)
                        for model, P in (("forecast", Ph), ("present", P0)):
                            cp = A.change_points(A.trajectory(P, "expected"))
                            cpt = t[cp]
                            for o, d in A.match_cp(cpt, on, WIN):
                                cp_rows.append(dict(family=fam, h=h, model=model, rep=rep, fold=fd["fold"],
                                                    student=s, onset=o, dt=d))
                            for o in on:
                                dd = cpt - o
                                v = dd[np.argmin(np.abs(dd))] if len(dd) and np.min(np.abs(dd)) <= WIN else np.nan
                                on_rows.append(dict(family=fam, h=h, model=model, rep=rep, fold=fd["fold"],
                                                    student=s, onset=o, dt=v))
                            if model == "forecast":       # null for the mean, own sequences
                                n = len(t); k = min(len(cp), n - 1)
                                for b in range(len(null_means)):
                                    if k and len(on):
                                        r = np.sort(A.RNG.choice(np.arange(1, n), k, replace=False))
                                        null_means[b] += [d for _, d in A.match_cp(t[r], on, WIN)]
            nm = np.array([np.mean(v) for v in null_means if len(v)])
            null_rows.append(dict(family=fam, h=h, null_mean=nm.mean(), null_lo=np.percentile(nm, 2.5),
                                  null_hi=np.percentile(nm, 97.5)))

    cp_df = pd.DataFrame(cp_rows); on_df = pd.DataFrame(on_rows)
    cp_df.to_csv(f"{out}/dt_pairs_same_sequence.csv", index=False)

    res = []
    for (fam, h), g in cp_df.groupby(["family", "h"]):
        f_, p_ = g[g.model == "forecast"], g[g.model == "present"]
        # (1) change-point-centric, students resampled jointly
        gf = {k: v["dt"].to_numpy() for k, v in f_.groupby("student")}
        gp = {k: v["dt"].to_numpy() for k, v in p_.groupby("student")}
        keys = sorted(set(gf) | set(gp)); e = np.array([]); dm, da = [], []
        for _ in range(B):
            pk = A.RNG.choice(len(keys), len(keys), replace=True)
            vf = np.concatenate([gf.get(keys[i], e) for i in pk]); vp = np.concatenate([gp.get(keys[i], e) for i in pk])
            if len(vf) and len(vp):
                dm.append(vf.mean() - vp.mean()); da.append(ant(vf) - ant(vp))
        dm, da = np.array(dm), np.array(da)
        # (2) onset-paired (median over repetitions), students resampled
        w = (on_df[(on_df.family == fam) & (on_df.h == h)]
             .groupby(["model", "fold", "student", "onset"])["dt"].median().unstack("model").dropna().reset_index())
        w["d"] = w["forecast"] - w["present"]
        gs = {k: v["d"].to_numpy() for k, v in w.groupby("student")}; ks = sorted(gs)
        bp = np.array([np.concatenate([gs[ks[i]] for i in A.RNG.choice(len(ks), len(ks), replace=True)]).mean()
                       for _ in range(B)])
        nr = next(r for r in null_rows if r["family"] == fam and r["h"] == h)
        res.append(dict(
            family=fam, h=h, students=len(keys),
            N_forecast=len(f_), N_present=len(p_),
            mean_forecast=f_.dt.mean(), mean_present=p_.dt.mean(),
            median_forecast=f_.dt.median(), median_present=p_.dt.median(),
            ant_forecast=ant(f_.dt), ant_present=ant(p_.dt),
            d_mean=f_.dt.mean() - p_.dt.mean(), d_mean_lo=qint(dm, .05)[0], d_mean_hi=qint(dm, .05)[1],
            d_mean_loB=qint(dm, .05 / NCOMP)[0], d_mean_hiB=qint(dm, .05 / NCOMP)[1],
            d_ant=ant(f_.dt) - ant(p_.dt), d_ant_lo=qint(da, .05)[0], d_ant_hi=qint(da, .05)[1],
            d_ant_loB=qint(da, .05 / NCOMP)[0], d_ant_hiB=qint(da, .05 / NCOMP)[1],
            paired_onsets=len(w), paired_mean=w.d.mean(), paired_lo=qint(bp, .05)[0], paired_hi=qint(bp, .05)[1],
            paired_loB=qint(bp, .05 / NCOMP)[0], paired_hiB=qint(bp, .05 / NCOMP)[1],
            paired_lo90=qint(bp, .10)[0], paired_hi90=qint(bp, .10)[1],
            null_mean_forecast=nr["null_mean"], null_lo=nr["null_lo"], null_hi=nr["null_hi"]))
    t = pd.DataFrame(res); t.to_csv(f"{out}/T8d_same_sequence.csv", index=False)
    pd.DataFrame(null_rows).to_csv(f"{out}/T8e_null_mean.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 50)
    print("\nSAME-SEQUENCE COMPARISON (forecast-trained minus present-state, identical sequences)")
    print(t[["family", "h", "students", "N_forecast", "N_present", "mean_forecast", "mean_present",
             "median_forecast", "median_present", "ant_forecast", "ant_present"]].round(2).to_string(index=False))
    print("\nDIFFERENCES: 95% interval [lo, hi], Bonferroni interval [loB, hiB]")
    print(t[["family", "h", "d_mean", "d_mean_lo", "d_mean_hi", "d_mean_loB", "d_mean_hiB",
             "d_ant", "d_ant_lo", "d_ant_hi", "d_ant_loB", "d_ant_hiB"]].round(2).to_string(index=False))
    print("\nONSET-PAIRED mean difference (hours)")
    print(t[["family", "h", "paired_onsets", "paired_mean", "paired_lo", "paired_hi", "paired_loB", "paired_hiB",
             "paired_lo90", "paired_hi90"]].round(2).to_string(index=False))
    print("\nRANDOM-CHANGE-POINT NULL for mean ΔT (forecast runs' own sequences)")
    print(t[["family", "h", "mean_forecast", "null_mean_forecast", "null_lo", "null_hi"]].round(2).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
