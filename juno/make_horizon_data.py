"""
make_horizon_data.py -- label-shifted copies of the StudentLife pickle for
the prediction-horizon experiment (Reviewer 2, comment 3).

For horizon h, each example's label is replaced by the label of the same
student's h-th next EMA response (chronological order of compound keys
"<student>_<month>_<day>_<hour>"). The last h examples of each student
have no future label and are dropped. Features are unchanged, so a model
trained on horizon_h.pkl is trained to forecast h responses ahead.

Also writes data/horizon/horizon_gaps.csv with the median and IQR of the
time gap (hours) between an example and its target response.

Usage (repo root):  python3 make_horizon_data.py
"""
import os
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np

SRC = ("data/training_data/shuffled_splits/"
       "training_date_normalized_shuffled_splits_"
       "select_features_no_prev_stress_all_students.pkl")
OUT = "data/horizon"


def t_hours(key):
    p = key.split("_")
    return (datetime(2012, int(p[1]), int(p[2]), int(p[3]))
            - datetime(2012, 1, 1)).total_seconds() / 3600.0


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(SRC, "rb") as f:
        data = pickle.load(f)

    by_student = defaultdict(list)
    for k in data["data"]:
        by_student[k.split("_")[0]].append(k)
    for s in by_student:
        by_student[s].sort(key=t_hours)

    rows = ["horizon,n_examples,median_gap_h,q25_gap_h,q75_gap_h"]
    for h in (1, 2, 3):
        new = dict(data)
        new["data"] = {}
        gaps = []
        for s, keys in by_student.items():
            for i in range(len(keys) - h):
                k, k_future = keys[i], keys[i + h]
                rec = list(data["data"][k])
                rec[-1] = data["data"][k_future][-1]   # label is always last
                new["data"][k] = type(data["data"][k])(rec) \
                    if isinstance(data["data"][k], tuple) else rec
                gaps.append(t_hours(k_future) - t_hours(k))
        # carry any other top-level fields; drop stale split info if present
        for key in list(new):
            if key not in ("data",) and isinstance(new[key], (list, dict)) \
                    and key.lower().startswith(("train", "val", "test")):
                new.pop(key)
        new["horizon"] = h
        path = os.path.join(OUT, f"horizon_{h}.pkl")
        with open(path, "wb") as f:
            pickle.dump(new, f)
        g = np.asarray(gaps)
        rows.append(f"{h},{len(new['data'])},{np.median(g):.1f},"
                    f"{np.percentile(g,25):.1f},{np.percentile(g,75):.1f}")
        print(f"h={h}: {len(new['data'])} examples, median gap "
              f"{np.median(g):.1f} h -> {path}")
    with open(os.path.join(OUT, "horizon_gaps.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
