---
icon: fl/repeat
---

# :fl-repeat: Reproducing the results

Everything on this site runs on a laptop CPU. The stored episodes, checkpoints and logs are in the
repository, so the quick checks below re-run the evaluation and compare it with the stored numbers
exactly; the full runs retrain every model. Commands run from the repository root.

## Environment

| | Version used | Pinned |
|---|---|---|
| Python | CPython 3.12.3 | `>=3.10` in `pyproject.toml`; CI uses 3.12 |
| flatland-rl | 4.3.0 | exactly |
| torch | 2.14.1, CPU build | `>=2.2` |
| numpy | 1.26.4 | `<2` (flatland-rl requires it) |
| mkdocs / mkdocs-material | 1.6.1 / 9.7.7 | exactly |
| Machine | 12th Gen Intel Core i7-1260P (16 logical CPUs), Linux 6.8, 8 worker processes | – |

The full list, with licences, is under
[References](../07-references.md#software-environment-verified-by-installation).

```bash
git clone https://github.com/leandergrech/rl-flatland-rescheduling.git
cd rl-flatland-rescheduling
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch, optional
pip install -e ".[dev]"
```

## Quick checks

| Command | What it checks | Time on the machine above |
|---|---|---|
| `pytest` | environment sanity, a smoke test per baseline, the dispatcher's executor and window, the figure palette | about 1 minute |
| `bash scripts/reproduce.sh` | re-evaluates every stored baseline on the small and medium held-out seeds and checks that trains arrived, normalised reward, deadlocks and episode length match `data/results/` exactly | about 10 minutes |
| `python scripts/tada_verify.py --workers 8` | the dispatcher's executor reproduces the OR planner exactly on all 80 evaluation episodes, and 200 episodes of random clearances never deadlock or violate a reservation | – |
| `python scripts/make_lab_data.py`, then `node scripts/check_lab.mjs` | the browser port against flatland-rl on the 80 Lab maps: distances, five recorded policies, the live planner, the dispatcher's clearances (any Node 18+); the fixtures it writes (about 430 MB) are not committed | export about 28 min for large and xlarge alone; check about 8.5 min |
| `mkdocs build --strict` | this site, with every internal link and anchor checked | under a minute |

## What is exact and what is not

- **Evaluation is deterministic.** A seed fixes the map, timetable and breakdowns; learned policies act
  greedily from stored checkpoints; the planner has no wall-clock cap. Re-running an evaluation
  should give the same arrivals, rewards and deadlocks. `scripts/check_reproduction.py` tests
  exactly that, and passed on the machine above for all 280 episodes of the quick check.
- **Wall-clock times are not.** Times per step and training times depend on the machine and on
  what else it runs; the stored runs shared the CPU with other jobs
  ([compute](../research/methodology.md#compute)). Compare them only within one table.
- **Retraining is approximate.** Training budgets are wall-clock limits (30 minutes for the
  baselines; a 55-minute cap for the dispatcher), so another machine completes a different number
  of iterations and produces a different checkpoint. Expect numbers of the same order, not the same
  digits. Every table on this site evaluates the stored checkpoints.

## Full runs

**Baselines** (about 3.5 hours on 8 workers): retrain PPO, imitation and PPO with the tree
observation, then evaluate everything on all four scenarios.

```bash
FULL=1 bash scripts/reproduce.sh
python scripts/evaluate.py --table        # print the summary from data/results/
```

**The learned dispatcher** (about 1 hour per training run): every command, from verification to
the continuous variant, is listed on [TADA on rails](../08-tada-dispatcher.md#reproduce).

## Rebuilding the tables and figures

The first three read the stored episodes only; `failure_modes.py` re-runs the held-out seeds.

```bash
python scripts/make_report.py             # results tables and figures, README table
python scripts/make_overview_figures.py   # home and methods overview figures, data/analysis/overview.json
python scripts/tada_report.py             # every table and figure on TADA on rails
python scripts/failure_modes.py --workers 8   # where failed trains end up (re-runs the held-out seeds)
mkdocs serve                              # the site at http://127.0.0.1:8000
```

The generated sections of each page sit between `<!-- NAME:START -->` and `<!-- NAME:END -->`
markers, so a script replaces only its own tables. Every figure is written in a light and a dark
version.

## What is stored

| Path | Contents |
|---|---|
| `data/scenarios/` | the scenario set (flatland-rl version, generator settings, seed splits) and its statistics |
| `data/results/` | one row per evaluation episode for each baseline; run logs with timings and machine load |
| `data/checkpoints/` | each learned baseline's weights, `config.json` with its hyperparameters, training log and wall-clock |
| `data/demos/` | the planner demonstrations used for behaviour cloning |
| `data/tada/` | the dispatcher's checkpoints, training logs, evaluation rows and verification results |
| `data/analysis/` | derived summaries (overview, failure modes) |
| `docs/assets/lab/` | the Lab's maps and recorded episodes |
