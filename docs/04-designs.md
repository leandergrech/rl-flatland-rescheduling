# 4. Solution designs side by side

This page compares the designs that produced the numbers in [03-timeline](03-timeline.md), then the
baselines in this repository and how they perform on our mid-size scenarios.

## Published designs

"n/r" means not reported in any source we could open.

### Operations research and heuristics

| System | Core algorithm | What it sees | How it acts | Compute | Reported result |
|---|---|---|---|---|---|
| **Andreica, 2019 winner** ([arXiv:2111.07876](https://arxiv.org/abs/2111.07876); [interview](https://www.aicrowd.com/blogs/flatland-mugurel)) | Per-train shortest path in a time-expanded graph (cell, time), trains planned in priority order, fast trains first. After a malfunction, paths are updated to keep each cell's visiting order. | Full state | Native actions read off the timed paths | Rewritten from Python to C++ for speed | 99% of trains routed on average |
| **An_Old_Driver, NeurIPS 2020 winner** ([Li et al. 2021](https://ojs.aaai.org/index.php/ICAPS/article/download/15994/15805/19487)) | PP with SIPP; LNS (3 adaptive neighbourhoods, 4 parallel threads); MCP execution; partial replanning after malfunctions; lazy planning (plan some trains up front, the rest during execution); simulated annealing to split the time budget across test levels | Full state. The team skipped building flatland's own observation, which sped the simulator up by more than 6× on 556-train instances | Native actions from the plan, gated by MCP | 4 CPUs; evaluated on an AMD Opteron 63xx cloud instance with 64 GB | Score 297.507, 98.5% success on 362 instances; 3,256 trains solved in 704 s. Stage-by-stage: PP+A*+MCP 282.6 → SIPP 285.4 → +LNS 289.1 → +partial replanning 291.9 → +lazy planning 297.5 ([Laurent et al. 2021, §4.1](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)) |
| **An_Old_Driver, Flatland 3 winner** ([Chen et al. 2023](https://arxiv.org/abs/2306.06455)) | MAPF-LNS with slack-first priorities (ties: fast first), a delay-based LNS neighbourhood, partial replanning at fixed intervals (every T/20) | Full state | As above | AMD Opteron 63xx, 32 GB | 135.47 at close (145 of 150 instances). Each component: 123.966 → 124.227 (slack order) → 124.432 (delay neighbourhood) → 125.175 (periodic replanning) on their local benchmark |
| **v4-sipp-locks, ECML 2026 winner** ([repo](https://github.com/darshanmakwana412/ecml2026)) | Prioritized SIPP, tightest slack first; trains held off-map before departure; "overstay rights" (a train physically in a cell keeps it, and stale reservations are invalidated); directional locks on the 18 longest single-track corridors | Full state | Plan precomputed, re-planned on malfunction; avoids STOP actions (priced at 250 × speed by the ECML reward) | n/r | 20.8429, against 12.4925 for the runner-up ([results](https://flatland-association.github.io/flatland-book/challenges/ecml2026/post-competition_analysis.html)) |
| **Deadlock-avoidance heuristic** ([flatland-baselines](https://github.com/flatland-association/flatland-baselines)) | Follow shortest path; move only if enough free cells separate you from every opposing train on your path | Full state | Native | Trivial | 10.7320 in ECML 2026 |

### Reinforcement learning

| System | Algorithm and network | Observation | Action design | Reward | Training | Reported result |
|---|---|---|---|---|---|---|
| **JBR_HSE, NeurIPS 2020 RL winner** ([Laurent et al. 2021, §4.2](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)) | PPO; actor and critic [256, 128]; self-attention over messages from neighbouring trains | Depth-3 tree plus agent features, including a random "handle" in [0, 1] as a priority | Native; a separate supervised classifier gates departures (threshold 0.92, decaying) | \(0.01\,\Delta d - 5\,\text{deadlock} + 10\,\text{arrival}\) | Small environments; hardware n/r | 214.150, 78.5% arrived; communication "by far the most impactful" change |
| **Netcetera, 2nd RL** ([§4.3](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)) | Ape-X DQN (RLlib), dueling | Depth-1 tree plus a global conflict graph; graph colouring sets priorities | Native | Shaped | Curriculum | 181.497, 88.1% arrived |
| **MARMot-Lab-NUS, 4th RL** ([§4.4](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)) | A3C on PRIMAL | 27 hand-built features | Decision-cell masking plus handcrafted "traffic lights" limiting entry to clusters of switches | Shaped | n/r | 127.912, 66.4% arrived |
| **Roost et al. 2020** ([arXiv:2004.13439](https://arxiv.org/html/2004.13439)) | A3C, FC(128-64-64) + LSTM(64) | Tree | Reduced decision space; 5 learned communication actions | Shaped | Curriculum | 2-train switch: 47% → 95% with communication |
| **Mohanty et al. 2020 baselines** ([arXiv:2012.05893](https://arxiv.org/abs/2012.05893)) | Ape-X, PPO, CCPPO, MARWIL (RLlib) | Tree; global density map | Native | Flatland 2020 reward | Imitation data from a 2019 OR solution | 25×25, 5 trains: PPO 81.33%, Ape-X + 25% imitation 86% test completion |
| **TreeLSTM** ([Jiang et al. 2023](https://arxiv.org/pdf/2210.12933)) | PPO; child-sum TreeLSTM over a deep path tree (depth > 10, built in C++), then 3 self-attention blocks across trains; value summed over trains | Per-train attributes plus deep tree | Native 5 actions | Shared across trains: env reward + arrival bonus + departure term − deadlock penalty, weights tuned per phase | 6-phase curriculum at 50 → 80 → 100 → 200 trains; one phase took 5 days; hardware n/r | 125.3 on the 15 Flatland 3 stages, 66.4% arrived |
| **MAMBA** ([Egorov & Shpilman 2022](https://arxiv.org/abs/2205.15023)) | Model-based: discrete world model (32×32 categorical latents) plus attention communication; PPO in imagination | Agent-local | Native | Env reward + JBR_HSE's shaping | 300k to 1.5M env steps | More sample-efficient than a JBR_HSE re-implementation at 5 and 10 trains |
| **Maze-Flatland** ([Castagna et al. 2026](https://arxiv.org/html/2605.10257v1)) | Two policies: departure scheduling {dispatch, wait} and routing {follow plan, alternative, stop}, both behaviour-cloned from a fixed-budget MCTS planner | Global and conflict features for departures; tree collapsed to the next decision point for routing | Hierarchical, masked | Arrival ratio, deadlock-free | Iterative BC on 10, 20 and 50-train scenarios; hardware n/r | Up to 80 trains: "nearly doubling" arrivals against Greedy, PP, Deadlock Avoidance and TreeLSTM; deadlocks ≤ 5% |
| **ECML 2026 organisers' RL baseline** ([repo](https://github.com/dynamik1703/ecml2026-starterkit)) | Masked PPO / rerank policy with a deadlock-avoidance fallback on dense station clusters | Custom lightweight builder | Masked | n/r | n/r | 9.4231 in ECML 2026 |

### What makes the OR entries strong

1. **They use the model.** The simulator is deterministic apart from malfunctions, and the
   planners exploit that fully: they reserve cell-time slots and know exactly when a train will
   arrive. An RL policy with a local tree observation has to infer, from a few features, what the
   planner reads off the reservation table.
2. **Deadlock-freedom by construction.** Order-preserving execution (MCP) turns "never deadlock"
   from something learned into a structural guarantee, even under malfunctions. Every OR winner
   has this in some form. JBR_HSE's departure classifier, MARMot's traffic lights and
   Maze-Flatland's dispatch policy are all attempts to recover it by other means.
3. **Anytime improvement within the clock.** LNS keeps improving a feasible plan for as long as the
   time budget allows, and the 2020 winner even optimised how to spend the budget across test
   levels. The score is a sum over episodes solved before an 8-hour or 2-hour limit, so speed turns
   directly into score.
4. **Good priorities.** Slack first and fast trains first are cheap and strong, and they are
   exactly the kind of decision a learned model could improve (see [06](06-open-questions.md)).

### What made the RL entries work at all

Every RL entry above that scored well used most of these: decision-cell masking or a reduced action
set; reward shaping on distance progress with deadlock penalties; an explicit mechanism for
departure timing (classifier, traffic lights, dispatch policy); some form of priority signal;
communication or attention across trains; a curriculum over the number of trains.

## Baselines in this repository

All code is in `src/rl_flatland/`. Every baseline runs through the same harness
(`evaluation.run_episode`), on the same four scenarios and the same held-out seeds (10 without and
10 with malfunctions per scenario, see `data/scenarios/scenarios.json`).

### Environment wrapper and observations

- `scenarios.py` fixes four `sparse_rail_generator` scenarios (small 30×30/10 trains, medium
  50×50/30, large 80×80/60, xlarge 100×100/100) with the Flatland 3 mixed speed profile (a quarter
  each at 1, 1/2, 1/3, 1/4). Training uses seeds 0 to 999, evaluation 1000 to 1009 (no
  malfunctions) and 2000 to 2009 (malfunction rate 1/1000 per train-step, 20 to 50 steps).
- `env.DecisionEnv` asks a train for a decision only when it is ready to depart, at a facing
  switch, or on the cell before a switch. Elsewhere it keeps rolling. On the medium scenario that
  is 29% of rail configurations (notebook 01). The action set is WAIT / GO along the shortest
  branch / GO along the alternative branch, with the last masked where there is no alternative.
- `observations.CompactObs` (28 features): own state, speed, malfunction, remaining distance,
  timetable slack, time, and for each branch a 30-cell look-ahead giving distance to the nearest
  opposing train, number of opposing trains, distance to the train ahead, whether the next cell is
  occupied, whether the opposing train is broken down, and the length of the single-track stretch.
- `observations.TreeObs` (260 features): flatland's depth-2 `TreeObsForRailEnv` with a 20-step
  shortest-path predictor, flattened and normalised as in the 2020 starter kits, plus 8 own
  features.

### OR reference: PP + SIPP + ordered execution

`baselines/or_planner.py`, written from the papers (no code copied). At reset it plans every train
off-map-to-target with SIPP on a reservation table that encodes flatland 4.3.0's exact conflict
model: one train per cell, no swaps, following allowed, k steps per cell for speed 1/k, and a
target cell held only in the arrival step. It tries 4 priority orders (fast first, slack first,
earliest departure, shortest trip) and 4 random ones, and keeps the plan that routes the most
trains, then the least lateness. During execution a train enters a cell only when every train
planned into that cell before it has left, never earlier than its planned time, and never while
broken down. A queue of trains may move up together, and so may a ring of three or more (flatland's
MotionCheck allows both). Trains that could not be planned are retried every 10 steps against the
current reservations.

This matches the "PP + SIPP + MCP" stage of the 2020 winner (285.4 of their final 297.5 in
[Laurent et al. 2021, §4.1](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)),
plus priority restarts. Not included: LNS, partial replanning after malfunctions, lazy planning.
It is plain Python, and still plans 100 trains in about 2 s.

### PPO with parameter sharing

`baselines/ppo.py`. One actor-critic, two tanh layers of 128 units (about 21k parameters with the compact
observation, 50k with the tree), shared by all trains. Each train is an independent trajectory in a semi-MDP: rewards between two of its
decisions are discounted inside the transition, and GAE uses \(\gamma^{\tau}\) and
\((\gamma\lambda)^{\tau}\) for the variable gap \(\tau\) (\(\gamma = 0.99\), \(\lambda = 0.95\)). Reward is JBR_HSE's shaping. Rollouts:
16 episodes per iteration on 8 worker processes, half small and half medium, half with
malfunctions. Then 4 epochs of clipped PPO (clip 0.2, entropy 0.01, lr 3e-4) on minibatches of 2048
decisions. Training stops after a 30-minute wall-clock budget. Evaluation is greedy.

#### Why a custom PPO trainer

Stable-Baselines3 assumes one agent per env step with a fixed step length. Here the set of trains
that must act changes every step, and each train's step length is variable. Wrapping that into
SB3's vectorised single-agent interface would need a padding scheme and would still get the
discounting wrong. The custom trainer is 250 lines and does the semi-MDP bookkeeping explicitly.

### Imitation, then PPO

`baselines/imitation.py`. The OR reference is run on 60 small and 60 medium training seeds (half
with malfunctions). At every decision point of the RL wrapper, the planner's action is labelled
in the same 3-action space: WAIT if it holds the train, otherwise the branch its planned next
cell lies on. Off-map WAIT labels are subsampled to 20%. A behaviour-cloning policy (BC) is trained
with cross-entropy for 15 epochs, then fine-tuned with PPO for 30 minutes (lr 1e-4, entropy 0.003,
3 value-only warm-up iterations). A BC loss on the demonstrations is added and annealed from
weight 1 to 0 by half-way. The demonstrations are committed in `data/demos/or_demos.npz`.

### Two rule-based references

- `ShortestPathPolicy`: everybody departs at once and follows their shortest path, with no
  coordination. It measures how much coordination matters.
- `ReactiveAvoidPolicy`: on the RL wrapper's decisions and compact features, go along the shortest
  branch if no opposing train is within 30 cells, else the alternative if it is clear, else wait.
  It measures how much the compact observation gives you without learning.

## Results on our scenarios

Results are being generated; this section is completed in the next commit.
