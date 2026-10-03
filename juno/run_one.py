"""
run_one.py -- one (configuration, repetition) StudentLife training run for
the Smart Health revision (SMHL-D-26-00918).

Wraps the reference implementation's run_exp() without editing its source:
  * redirects the data file (horizon-shifted copies) via --data
  * stores val_ids in every fold record, so the ΔT analysis never has to
    reconstruct splits (the old analysis did, which is fragile)
  * dispatches to the asymmetric-loss trainer when --asym > 0
  * --epochs overrides the epoch count (use 2 for a smoke test)

Run from the repository root:
  python3 -u run_one.py --config base --rep 0
  python3 -u run_one.py --config base --rep 0 --epochs 2 --tag smoke
"""
import argparse
import os
import sys
sys.path.insert(0, os.getcwd())   # repo root, so "import src..." works

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = ("data/training_data/shuffled_splits/"
                "training_date_normalized_shuffled_splits_"
                "select_features_no_prev_stress_all_students.pkl")

# name -> (job_name, num_branches, groups, asym_lambda, horizon)
CONFIGS = {
    "base":        ("lstm",     1, "all_in_one",   0.0, 0),
    "p1":          ("calm_net", 1, "one_for_each", 0.0, 0),
    "p3":          ("calm_net", 3, "one_for_each", 0.0, 0),
    "base_asym05": ("lstm",     1, "all_in_one",   0.5, 0),
    "base_asym10": ("lstm",     1, "all_in_one",   1.0, 0),
}
for h in (1, 2, 3):
    CONFIGS[f"base_h{h}"] = ("lstm",     1, "all_in_one",   0.0, h)
    CONFIGS[f"p3_h{h}"]   = ("calm_net", 3, "one_for_each", 0.0, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", default="rev")
    args = ap.parse_args()

    job, nbr, groups, lam, h = CONFIGS[args.config]
    data_path = DEFAULT_DATA if h == 0 else f"data/horizon/horizon_{h}.pkl"
    if not os.path.exists(data_path):
        sys.exit(f"missing {data_path} (run make_horizon_data.py first)")
    os.makedirs("data/cross_val_scores", exist_ok=True)

    import numpy as np
    import torch
    seed = 1000 + args.rep
    np.random.seed(seed)
    torch.manual_seed(seed)

    import src.experiments.run_exp as RE
    import src.utils.train_val_utils as TVU

    # asymmetric trainer lives in the same namespace as train_and_val
    exec(open(os.path.join(HERE, "asym_train_and_val.py")).read(), TVU.__dict__)

    captured = {}
    orig_read = RE.read_data

    def read_data(path):
        if path.endswith(os.path.basename(DEFAULT_DATA)):
            path = data_path
        d = orig_read(path)
        if "data" in d and "raw" not in captured:
            captured["raw"] = d
        return d
    RE.read_data = read_data

    orig_cfg = RE.get_config

    def get_config(*a, **k):
        c = orig_cfg(*a, **k)
        if args.epochs is not None:
            c["training_params"]["epochs"] = args.epochs
        return c
    RE.get_config = get_config

    orig_tv = TVU.train_and_val

    def train_and_val(data, model_params, training_params, **kw):
        if lam > 0:
            rec = TVU.train_and_val_asym(
                data=data, model_params=model_params,
                training_params=training_params,
                raw_data=captured["raw"], asym_lambda=lam, **kw)
        else:
            rec = orig_tv(data=data, model_params=model_params,
                          training_params=training_params, **kw)
        rec["val_ids"] = list(data["val_ids"])
        rec["meta"] = dict(config=args.config, rep=args.rep, job=job,
                           branches=nbr, groups=groups, asym_lambda=lam,
                           horizon=h, data=data_path, seed=seed)
        return rec
    RE.train_and_val = train_and_val

    remark = f"{args.tag}_{args.config}_r{args.rep}"
    print(f"config={args.config} job={job} branches={nbr} groups={groups} "
          f"lambda={lam} horizon={h} rep={args.rep} -> remark {remark}",
          flush=True)
    RE.run_exp(job, "5fold", nbr, 0, groups, remark)


if __name__ == "__main__":
    main()
