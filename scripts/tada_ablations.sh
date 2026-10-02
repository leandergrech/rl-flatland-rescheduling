#!/usr/bin/env bash
# Step 4: train and evaluate every ablation, prioritised as the brief asks (B and YIELD_TO first).
# Each run: 60 iterations x 8 medium episodes with malfunctions (half the main run), capped at 55
# minutes; the baseline row is the main run's 60-iteration checkpoint (model_it60.pt).
#   bash scripts/tada_ablations.sh            # all
#   bash scripts/tada_ablations.sh B1 noyield # a subset
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
declare -A ARGS=(
  [B1]="--budget 1"
  [B4]="--budget 4"
  [noyield]="--no-yield"
  [M4]="--M 4"
  [M16]="--M 16"
  [shaping]="--shaping"
  [tree]="--features slack+tree"
)
RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(B1 noyield B4 M4 M16 shaping tree)
for n in "${RUNS[@]}"; do
  $PY scripts/tada_train.py --name "$n" --iterations 60 ${ARGS[$n]}
  $PY scripts/tada_evaluate.py --name "$n"
done
if [ -f data/tada/checkpoints/main/model_it60.pt ] && [ ! -f data/tada/results/main_it60.json ]; then
  $PY scripts/tada_evaluate.py --name main --model model_it60.pt --out main_it60
fi
