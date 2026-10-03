"""joint_horizon_boot.py -- recompute Table 8 (change from h = 0) with students
resampled jointly, from the matched-pair files alone (no model outputs needed).
Usage: python3 joint_horizon_boot.py <dt_pairs_expected.csv> <dt_pairs_horizon.csv>"""
import sys, numpy as np, pandas as pd
rng = np.random.default_rng(20261022); B = 5000
main = pd.read_csv(sys.argv[1]); hz = pd.read_csv(sys.argv[2])
ant = lambda v: 100 * np.mean(v < 0)
rows = []
for fam in ("base", "p3"):
    d0 = main[main.config == fam]
    g0 = {str(k): g["dt"].to_numpy() for k, g in d0.groupby("student")}
    for h in (1, 2, 3):
        dh = hz[(hz.family == fam) & (hz.h == h)]
        gh = {str(k): g["dt"].to_numpy() for k, g in dh.groupby("student")}
        keys = sorted(set(g0) | set(gh)); e = np.array([]); da, dm = [], []
        for _ in range(B):
            pick = rng.choice(len(keys), len(keys), replace=True)
            v0 = np.concatenate([g0.get(keys[i], e) for i in pick]); vh = np.concatenate([gh.get(keys[i], e) for i in pick])
            if len(v0) and len(vh): da.append(ant(vh) - ant(v0)); dm.append(vh.mean() - v0.mean())
        rows.append(dict(family=fam, h=h, students=len(keys),
            d_ant=ant(dh.dt) - ant(d0.dt), d_ant_lo=np.percentile(da, 2.5), d_ant_hi=np.percentile(da, 97.5),
            d_mean=dh.dt.mean() - d0.dt.mean(), d_mean_lo=np.percentile(dm, 2.5), d_mean_hi=np.percentile(dm, 97.5)))
print(pd.DataFrame(rows).round(2).to_string(index=False))
