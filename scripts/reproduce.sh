#!/usr/bin/env bash
# Reproduce the stored results.
#
#   bash scripts/reproduce.sh          # quick check, about 10 min on 8 workers: re-evaluate every stored
#                                      # baseline on the small and medium held-out seeds and compare with
#                                      # data/results (deterministic metrics must match exactly)
#   FULL=1 bash scripts/reproduce.sh   # retrain PPO, imitation (BC, BC+PPO) and PPO-tree, then evaluate
#                                      # everything on all four scenarios (about 3.5 h on 8 workers)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
if [ -x .venv/bin/python ] && [ "$PY" = "python" ]; then PY=.venv/bin/python; fi

echo "== scenario set and statistics"
$PY scripts/evaluate.py --stats

if [ "${FULL:-0}" = "1" ]; then
  echo "== full reproduction: training"
  $PY scripts/train.py ppo --budget-s 1800
  rm -f data/demos/or_demos.npz
  $PY scripts/train.py imitation --budget-s 1800
  $PY scripts/train.py ppo_tree --budget-s 1800
  echo "== full reproduction: evaluation"
  for p in or shortest_path reactive_avoid ppo bc bc_ppo ppo_tree; do
    $PY scripts/evaluate.py --policy "$p"
  done
  $PY scripts/evaluate.py --table
  exit 0
fi

echo "== quick reproduction on small + medium held-out seeds"
OUT=.runs/repro
rm -rf "$OUT"; mkdir -p "$OUT"
for f in data/results/*.json; do
  p=$(basename "$f" .json)
  case "$p" in
    or|shortest_path|reactive_avoid|ppo|bc|bc_ppo|ppo_tree) ;;
    *) continue ;;
  esac
  $PY scripts/evaluate.py --policy "$p" --scenarios small medium --out "$OUT/$p.json" > "$OUT/$p.log"
  echo "   $p done"
done
$PY scripts/check_reproduction.py "$OUT"
