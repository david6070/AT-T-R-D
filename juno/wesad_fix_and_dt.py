"""
wesad_fix_and_dt.py -- WESAD for the Smart Health revision.

WHY THIS EXISTS
  In the reference implementation (wesad/calm_net.py) the evaluation
  DataLoader is built with shuffle=True, and the saved fold records hold only
  y_preds / y_trues -- no file names, so no time index. Any trajectory built
  from those records is a random permutation of the subject's windows, and
  "onsets" in it are artefacts of the shuffle. The submitted WESAD ΔT numbers
  (66 pairs, 33.3%) must therefore be regenerated or withdrawn.

TWO SUB-COMMANDS (run inside the repo's wesad/ directory)

  python3 ../juno/wesad_fix_and_dt.py patch
      * calm_net.py : eval DataLoader shuffle=True -> shuffle=False
      * main.py     : test file names sorted chronologically
                      (subject, window index), and saved in each record
      Originals are kept as *.pre_rev. Then retrain as before:
      python3 -m main personalize loocv

  python3 ../juno/wesad_fix_and_dt.py dt --records data/exp_res/personalize
      ΔT on the stress-class probability p_t(1), full PELT penalty scan,
      matched pairs AND distinct onsets, ties at ΔT = 0 reported separately,
      onset-clustered bootstrap CIs, and a random-change-point null.
      Writes results/wesad/*.csv and fig_wesad.pdf
"""
import argparse
import glob
import os
import pickle
import re
import shutil

import numpy as np
import pandas as pd

WIN = 10          # matching window, in 0.25 Hz windows (as in the paper)
PENS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
RNG = np.random.default_rng(7)


def patch():
    def edit(path, old, new):
        s = open(path).read()
        if new in s:
            print(f"{path}: already patched"); return
        assert s.count(old) == 1, f"{path}: anchor not found: {old!r}"
        if not os.path.exists(path + ".pre_rev"):
            shutil.copy2(path, path + ".pre_rev")
        open(path, "w").write(s.replace(old, new)); print(f"{path}: patched")

    edit("calm_net.py",
         "eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)",
         "eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)")
    edit("main.py",
         '    eval_dataset = WESAD(split["test_fnames"], sample_root=sample_root, covs_path=covs_path)',
         '    test_fnames = sorted(split["test_fnames"], key=lambda f: (f.split("_")[0], int(f.split("_")[2])))\n'
         '    eval_dataset = WESAD(test_fnames, sample_root=sample_root, covs_path=covs_path)')
    edit("main.py",
         '            "y_trues": record["last_pred"]["y_trues"],\n        }',
         '            "y_trues": record["last_pred"]["y_trues"],\n'
         '            "fnames": list(test_fnames),\n        }')


def load(records_dir):
    seqs = []
    for p in sorted(glob.glob(os.path.join(records_dir, "*_record_*"))):
        r = pickle.load(open(p, "rb"))
        if "fnames" not in r:
            raise SystemExit(f"{p} has no fnames: it was produced by the "
                             "unpatched (shuffled) pipeline and cannot be used.")
        df = pd.DataFrame(dict(f=r["fnames"], y=r["y_trues"],
                               p1=np.asarray(r["y_preds"])[:, 1]))
        df["subject"] = df.f.str.split("_").str[0]
        df["idx"] = df.f.str.split("_").str[2].astype(int)
        for s, g in df.groupby("subject"):
            g = g.sort_values("idx")
            seqs.append(dict(subject=s, t=g.idx.to_numpy(float),
                             y=(g.y.to_numpy() == 1).astype(int), x=g.p1.to_numpy()))
    return seqs


def dt_pairs(seqs, pen, random_cp=False):
    import ruptures as rpt
    rows = []
    for s in seqs:
        n = len(s["x"])
        bk = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(s["x"].reshape(-1, 1)).predict(pen=pen)
        cp = np.array([b for b in bk if b < n], int)
        if random_cp and len(cp):
            cp = np.sort(RNG.choice(np.arange(1, n), len(cp), replace=False))
        on = s["t"][np.where(np.diff(s["y"]) > 0)[0] + 1]
        for c in s["t"][cp]:
            if len(on) == 0:
                break
            d = c - on; j = np.argmin(np.abs(d))
            if abs(d[j]) <= WIN:
                rows.append(dict(subject=s["subject"], onset=on[j], dt=d[j]))
    return pd.DataFrame(rows, columns=["subject", "onset", "dt"])


def onset_boot(df, B=2000):
    """Anticipation rate with the ONSET as resampling unit (pairs sharing an
    onset are not independent)."""
    if df.empty:
        return np.nan, np.nan
    g = [x["dt"].to_numpy() for _, x in df.groupby(["subject", "onset"])]
    res = [100 * np.mean(np.concatenate([g[i] for i in RNG.integers(0, len(g), len(g))]) < 0)
           for _ in range(B)]
    return np.percentile(res, [2.5, 97.5])


def dt(records_dir):
    out = "results/wesad"; os.makedirs(out, exist_ok=True)
    seqs = load(records_dir)
    print(f"{len(seqs)} subject sequences")
    rows = []
    for pen in PENS:
        d = dt_pairs(seqs, pen)
        lo, hi = onset_boot(d)
        null = [100 * (dt_pairs(seqs, pen, True)["dt"] < 0).mean() for _ in range(200)]
        v = d["dt"].to_numpy()
        rows.append(dict(pen=pen, pairs=len(v), onsets=d.groupby(["subject", "onset"]).ngroups,
                         subjects=d.subject.nunique(), mean=v.mean() if len(v) else np.nan,
                         median=np.median(v) if len(v) else np.nan,
                         pct_before=100 * (v < 0).mean() if len(v) else np.nan,
                         pct_at=100 * (v == 0).mean() if len(v) else np.nan,
                         pct_after=100 * (v > 0).mean() if len(v) else np.nan,
                         ant_ci_lo=lo, ant_ci_hi=hi,
                         null_ant_mean=np.nanmean(null), null_lo=np.nanpercentile(null, 2.5),
                         null_hi=np.nanpercentile(null, 97.5)))
    t = pd.DataFrame(rows); t.to_csv(f"{out}/T_wesad_penalty_scan.csv", index=False)
    print(t.round(2).to_string())

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
    ax[0].errorbar(t.pen, t.pct_before, yerr=[t.pct_before - t.ant_ci_lo, t.ant_ci_hi - t.pct_before],
                   fmt="o-", capsize=3, color="k", label="Observed (onset-clustered 95% CI)")
    ax[0].fill_between(t.pen, t.null_lo, t.null_hi, color="#90caf9", alpha=.5, label="Random change points")
    ax[0].set_xscale("log"); ax[0].set_xlabel("PELT penalty ρ"); ax[0].set_ylabel("ΔT < 0 (%)")
    ax[0].legend(frameon=False, fontsize=7)
    ax[1].bar(range(len(t)), t.onsets, color="#607d8b", label="Distinct onsets")
    ax[1].plot(range(len(t)), t.pairs, "ko-", label="Matched pairs")
    ax[1].set_xticks(range(len(t)), [f"{p:g}" for p in t.pen]); ax[1].set_xlabel("PELT penalty ρ")
    ax[1].set_ylabel("Count"); ax[1].legend(frameon=False, fontsize=7)
    fig.savefig(f"{out}/fig_wesad.pdf"); fig.savefig(f"{out}/fig_wesad.png", dpi=300)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["patch", "dt"])
    ap.add_argument("--records", default="data/exp_res/personalize")
    a = ap.parse_args()
    patch() if a.cmd == "patch" else dt(a.records)
