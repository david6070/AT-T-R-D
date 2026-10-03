#!/bin/bash
# StudentLife retraining for the Smart Health revision.
# One array task = one (configuration, repetition) = 5 folds x 200 epochs.
# 80 tasks total (see tasks.txt). Submit from the repository root:
#   sbatch --array=1-80%20 juno/train_array.sbatch
# Smoke test first:
#   sbatch --array=1 --time=00:30:00 --export=ALL,EPOCHS=2,TAG=smoke juno/train_array.sbatch
#SBATCH --job-name=sl_rev
#SBATCH --output=logs/sl_rev_%A_%a.out
#SBATCH --error=logs/sl_rev_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
##SBATCH --partition=<GPU partition, if Juno has one>
##SBATCH --gres=gpu:1

set -euo pipefail
module load gnu14 2>/dev/null || true
source /work/dod220001/tmlcn/venv/bin/activate

export PYTHONPATH=$PWD:${PYTHONPATH:-}
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" juno/tasks.txt)
CFG=$(echo "$LINE" | cut -d' ' -f1)
REP=$(echo "$LINE" | cut -d' ' -f2)
EXTRA=""
[ -n "${EPOCHS:-}" ] && EXTRA="--epochs ${EPOCHS}"
echo "task ${SLURM_ARRAY_TASK_ID}: ${CFG} rep ${REP} on $(hostname) $(date)"
python3 -u juno/run_one.py --config "$CFG" --rep "$REP" --tag "${TAG:-rev}" $EXTRA
echo "done $(date)"
