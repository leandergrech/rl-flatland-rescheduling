# rl-flatland-rescheduling

**Literature review: <https://leandergrech.github.io/rl-flatland-rescheduling/>**

Can multi-agent reinforcement learning close the gap to operations research on railway
rescheduling? Flatland (SBB, Deutsche Bahn, SNCF, AIcrowd; now the Flatland Association) asks
trains on a grid-drawn network to reach their targets within timetable windows, without sharing
cells, while random malfunctions stop them. OR has won every official round: in NeurIPS 2020 the
best OR entry got 98.6% of trains home and the best RL entry 78.5%, and at ECML 2026 the best
planner scored 20.84 against 9.42 for the RL baseline. This repository is a verified literature
review of that gap (2019 to 2026) and a CPU-sized codebase to work on it. It has four fixed mid-size
scenarios, an OR reference that reproduces the core of the winning planners, a parameter-shared
PPO baseline, and an imitation-then-PPO baseline, all evaluated by one harness on held-out seeds
with and without malfunctions.

**Scope:** 30×30 to 100×100 grids with 10 to 100 trains. Every baseline trains and evaluates on a
laptop CPU in under an hour. The competitions went up to 314×314 cells and 6,256 trains; see
[limitations](docs/05-limitations.md).

## Quick start

```bash
git clone https://github.com/leandergrech/rl-flatland-rescheduling.git
cd rl-flatland-rescheduling
python3.12 -m venv .venv && source .venv/bin/activate      # or: uv venv --python 3.12 .venv
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch, optional
pip install -e ".[dev]"
pytest                                   # 17 tests, about 20 s
bash scripts/reproduce.sh                # re-evaluate stored baselines on small+medium, compare with data/results
mkdocs serve                             # the literature review locally
```

Then open `notebooks/01-explore.ipynb`.

## Results

RESULTS_TABLE_PLACEHOLDER

## Layout

```
docs/                 literature review (MkDocs Material), published to GitHub Pages
src/rl_flatland/      scenarios.py (fixed scenarios + seeds), graph.py, deadlock.py, observations.py,
                      env.py (decision wrapper), evaluation.py, metrics.py, plotting.py, policies.py,
                      baselines/or_planner.py, baselines/ppo.py, baselines/imitation.py, baselines/rl_policy.py
scripts/              train.py, evaluate.py, reproduce.sh, check_reproduction.py
notebooks/            01-explore, 02-baseline, 03-first-experiment
data/                 scenarios/, results/, checkpoints/, demos/ (all committed, < 20 MB), fetch.py
tests/                env sanity tests and a smoke test per baseline
```

## Reproducing

- `bash scripts/reproduce.sh`: exports the scenario set, re-evaluates every stored baseline on the
  small and medium held-out seeds, and checks that arrival rate, normalised reward and deadlock
  counts match `data/results/` exactly (about 10 minutes on 8 workers).
- `FULL=1 bash scripts/reproduce.sh`: retrains PPO, imitation and PPO-tree (30-minute budget each),
  then evaluates everything on all four scenarios.
- `python scripts/evaluate.py --table` prints the stored summary.

## Licence

MIT for the code in this repository (see `LICENSE`). Third-party licences are listed in
[docs/07-references.md](docs/07-references.md). No third-party solver code is included; the OR
reference was written from the published papers.
