# For Leander: an on-ramp

You already know the RL side. This page maps what you know onto Flatland, flags what is new, and
gives a two-week plan with one reading and one coding task per day. Everything it links to is on
this site or in this repository.

## What transfers from your work

**From TADA and air traffic control.**

- *Conflict detection and resolution under separation constraints.* A Flatland cell is a
  separation minimum on a 1-D track: two trains may never share one, just as two aircraft may
  never violate separation. The closest ATC analogue is not en-route conflict resolution but
  **arrival sequencing in terminal airspace**: which of two flows goes first through a shared
  resource, decided ahead of time, with delays propagating to followers. That is exactly TADA's
  airspace ([SESAR project page](https://www.sesarju.eu/projects/TADA)).
- *RL against a strong classical baseline.* In ATC, RL policies are measured against geometric
  resolvers such as the modified-voltage-potential method in BlueSky. The honest TU Delft finding
  is that stand-alone RL "cannot yet match" geometric methods, while RL tuning the geometric
  method's parameters helps ([Ribeiro, Ellerbroek & Hoekstra 2022](https://api.openalex.org/works/doi:10.3390/aerospace9120847)).
  Flatland has the same structure, with PP/SIPP/LNS in the role of the geometric resolver.
  "RL that tunes or repairs the planner" is the most promising line on both sides.
- *Multi-agent RL for separation.* Brittain & Wei's PPO with centralised learning, decentralised
  execution and attention over neighbours
  ([2019](https://arxiv.org/abs/1905.01303); [2021](https://arxiv.org/abs/2003.08353)) is
  architecturally the same as JBR_HSE's Flatland winner (PPO plus attention-based communication,
  [Laurent et al. 2021, §4.2](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)).
- *The same funding line.* AI4REALNET (Horizon Europe, 6 M€) uses Flatland for rail and BlueSky
  for air traffic side by side, and maintains both in one GitHub organisation
  ([ai4realnet.eu](https://ai4realnet.eu/); [github.com/AI4REALNET](https://github.com/AI4REALNET)).
  A result that transfers between the two is exactly what that community is looking for.

**From the CERN crystal-alignment and accelerator work.**

- *Sample efficiency and honest baselines* carry over, but the constraint flips. Flatland's
  simulator is cheap, deterministic apart from malfunctions, and exactly known, so samples are not
  scarce. Wall-clock per decision and planning time are, because the competitions cap them
  (10 s per step, 10 minutes of planning, 2 hours in total in Flatland 3,
  [evaluation page](https://flatland-association.github.io/flatland-book/challenges/flatland3/eval.html)).
- *Sim-to-real* does not arise inside the benchmark. It returns the moment you ask whether a
  Flatland policy means anything for a real railway: no train length, no braking curves, no
  signalling headways (see [05-limitations](05-limitations.md#sim-to-real)).

**From the Semablu super-resolution work.** Grid-structured inputs invite CNNs, and Flatland does
offer a global grid observation (h×w×16 transitions plus agent channels,
[Flatland book](https://flatland-association.github.io/flatland-book/environment/observation_builder/provided_observations.html)).
But the network is sparse (about 9 to 11% of cells carry track in our scenarios) and the relevant
structure is a graph, so the strong entries use trees, graph features or attention over trains,
not convolutions over the grid.

**From teaching.** Flatland is a clean MARL coordination example for your Master's course: a known
model, an exact OR baseline that crushes naive RL, and a failure mode (deadlock) students can
see.

## What is new for you

1. **The MAPF toolbox**: prioritised planning, safe intervals, conflict-based search, large
   neighbourhood search, and order-preserving execution (MCP). Read [02-primer](02-primer.md#the-mapf-toolkit) first.
2. **Irreversibility as the dominant failure.** In ATC there is always a manoeuvre (a hold, a
   heading change). On a single track there is none. A learned policy has to be conservative
   about commitments, not about proximity.
3. **A semi-MDP with a varying number of agents.** Trains appear at departure and vanish at arrival,
   and each acts only at cell exits. [04-designs](04-designs.md#ppo-with-parameter-sharing) shows
   how the PPO here handles it.
4. **Competition scoring.** A score is a sum of per-episode normalised rewards over episodes solved
   before a time limit, and evaluation stops once fewer than 25% of trains arrive. Speed and
   robustness count as much as per-episode quality.
5. **Rail vocabulary**: timetable, slack, knock-on delay, block section, passing loop. See the
   [primer](02-primer.md#what-a-dispatcher-does).

## Two-week study plan

Each day is about 2 hours of reading and 2 hours of coding. Commands assume `source .venv/bin/activate`
in the repository root.

| Day | Read | Do |
|---|---|---|
| 1 | [01-problem](01-problem.md) and [02-primer](02-primer.md); [Laurent et al. 2021](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf) §1–3 (pp. 275–282) | `pip install -e .[dev] && pytest`; run `notebooks/01-explore.ipynb` end to end; change the seed and scenario and watch where the shortest-path policy deadlocks |
| 2 | [Flatland book: agent states](https://flatland-association.github.io/flatland-book/environment/environment/agent.html) and [actions](https://flatland-association.github.io/flatland-book/environment/environment/actions.html); flatland's [`agent_chains.py`](https://github.com/flatland-association/flatland-rl/blob/v4.3.0/flatland/envs/agent_chains.py) docstring | Write a 30-line script that puts two trains head-on on one segment and confirm `rl_flatland.deadlock.find_deadlocked` flags them; then build a 3-train ring and check it rotates |
| 3 | [Stern et al. 2019](https://arxiv.org/abs/1906.08291) §2–3; [Silver 2005](https://ojs.aaai.org/index.php/AIIDE/article/view/18726) | Read `src/rl_flatland/baselines/or_planner.py` top to bottom; plot a time-space diagram (x: path index, y: time) of 5 planned trains through one shared corridor |
| 4 | [Li et al. 2021](https://ojs.aaai.org/index.php/ICAPS/article/download/15994/15805/19487), all of it | Ablate ordered execution: make `_will_move` ignore the visit order (timing only) and count deadlocks on the medium malfunction seeds (`python scripts/evaluate.py --policy or --scenarios medium --splits test_malfunction`) |
| 5 | [Chen et al. 2023](https://arxiv.org/abs/2306.06455); [Andreica 2021](https://arxiv.org/abs/2111.07876) | Compare the four priority orders individually (`PrioritizedPlannerPolicy(orderings=("slack",), n_random_orderings=0)` etc.) on 20 training seeds of `large`; tabulate arrival rate and total lateness |
| 6 | JBR_HSE in [Laurent et al. 2021, §4.2](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf); [Roost et al. 2020](https://arxiv.org/html/2004.13439) | Read `env.py` and `baselines/ppo.py`; run `python scripts/train.py ppo --budget-s 600 --out-suffix _d6` and plot `train_log.jsonl`; compare with the stored 30-minute run |
| 7 | [Jiang et al. 2023 (TreeLSTM)](https://arxiv.org/pdf/2210.12933) §3–4; [MAMBA](https://arxiv.org/abs/2205.15023) §3 | Train `ppo_tree` and `ppo` for 10 minutes each on `small` only (edit `train_mix`) and compare learning curves; note what the tree observation lacks (no timetable) |
| 8 | [Mohanty et al. 2020](https://arxiv.org/abs/2012.05893) §4–5 (imitation); [Maze-Flatland](https://arxiv.org/html/2605.10257v1) §III–IV | Load `data/demos/or_demos.npz`, split BC accuracy by decision type (off-map departure vs switch vs pre-switch) and find where cloning fails |
| 9 | [ECML 2026 pages](https://flatland-association.github.io/flatland-book/challenges/ecml2026.html) and the [winner's README](https://github.com/darshanmakwana412/ecml2026); [flatland-baselines](https://github.com/flatland-association/flatland-baselines) | Re-score the stored OR and PPO rollouts with `ECML2026Rewards` (pass `rewards=ECML2026Rewards()` to `RailEnv` in `scenarios.make_env`) and see how the ranking changes when STOP and collisions are priced |
| 10 | [Brittain & Wei 2019](https://arxiv.org/abs/1905.01303); [Ribeiro et al. 2022](https://api.openalex.org/works/doi:10.3390/aerospace9120847); [05-limitations](05-limitations.md) and [06-open-questions](06-open-questions.md) | Pick the first experiment (the default is below) and fill in TODOs 1–2 of `notebooks/03-first-experiment.ipynb` |
| 11–12 | Whatever the experiment needs | Run it: code, a 30-minute training budget, evaluation on the held-out and malfunction seeds |
| 13 | [Atzmon et al. 2020 (robust MAPF)](https://jair.org/index.php/jair/article/view/11734) | Add the result to a copy of the results table in [04-designs](04-designs.md#results-on-our-scenarios); decide whether it is worth scaling |
| 14 | — | Write a one-page note: hypothesis, result, what it would take at competition scale |

## The first experiment, in three lines

See [06-open-questions](06-open-questions.md) for the ranking. The default first experiment is
the one set up in `notebooks/03-first-experiment.ipynb`: **learn when to break the planned order
after a malfunction**. The OR reference keeps every cell's visiting order fixed, so one broken train
holds up everyone planned behind it. A small policy that decides, at the next switch, whether a
waiting train may overtake the delayed one (with a replan to keep things deadlock-free) attacks the
one place where RL should beat plan-and-repair.
