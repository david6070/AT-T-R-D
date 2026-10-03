#!/bin/bash
# One-time setup on Juno. Run from the directory where you want the repo.
set -euo pipefail
module load gnu14 2>/dev/null || true
git clone https://github.com/Information-Fusion-Lab-Umass/personalized-stress-prediction.git sl_rev
cd sl_rev
mkdir -p data/training_data/shuffled_splits data/cross_val_scores data/horizon logs
cp -r ../juno .           # this folder
source /work/dod220001/tmlcn/venv/bin/activate
pip install -q torch einops tqdm scikit-learn scipy ruptures matplotlib pandas pyyaml tabulate sqlalchemy bokeh
echo
echo "Now copy the preprocessed pickle from the repo's GitHub Release into:"
echo "  sl_rev/data/training_data/shuffled_splits/"
echo "then run:  python3 juno/make_horizon_data.py"
