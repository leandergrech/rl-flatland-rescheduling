---
icon: fl/warning
---

# :fl-warning: Limitations: what fails, and by how much

## The RL-versus-OR gap in the official rounds

| Round | Best OR | Best RL | Gap | Source |
|---|---|---|---|---|
| NeurIPS 2020 | 297.507, 98.6% arrived, 363 envs | 214.150, 78.5% arrived, 336 envs | 20.1 points of arrival rate; RL ranked 8th overall in Round 2 | [Laurent et al. 2021, Table 1](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf); [Li et al. 2021](https://ojs.aaai.org/index.php/ICAPS/article/download/15994/15805/19487) |
| Flatland 3 (2021) | 135.47 at close; 141.0 and 88.0% arrived on the current leaderboard | 27.868, 38.6% arrived | Score ratio 0.21; the winners matched the best RL score in 3 minutes of their 2-hour budget | [Chen et al. 2023](https://arxiv.org/abs/2306.06455); [Jiang et al. 2023, Table 7](https://arxiv.org/pdf/2210.12933) |
| Flatland 3 stages, post hoc | 141.0, 88.0% | TreeLSTM 125.3, 66.4% | 21.6 points of arrival rate | [Jiang et al. 2023, Table 7](https://arxiv.org/pdf/2210.12933) |
| ECML 2026 | 20.8429 | 9.4231 (organisers' baseline; the only RL-track entry listed) | Score ratio 0.45 | [post-competition analysis](https://flatland-association.github.io/flatland-book/challenges/ecml2026/post-competition_analysis.html) |

The organisers' 2020 verdict still holds: "RL solutions are still a considerable distance away
from OR based solutions" ([Laurent et al. 2021, §6](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)).

![Best RL relative to best OR in each round, and arrival rates](assets/figures/rounds-light.svg#only-light)
![Best RL relative to best OR in each round, and arrival rates](assets/figures/rounds-dark.svg#only-dark)

*The table above as a chart, plus this repository's small and xlarge scenarios for scale. Sources as in the table.*

## On our mid-size scenarios

Details and all metrics are in [Designs and results](04-designs.md#results-on-our-scenarios).

<!-- README_RESULTS:START -->
| Policy | small 30×30, 10 trains | medium 50×50, 30 | large 80×80, 60 | xlarge 100×100, 100 | train + eval (min) |
|---|---|---|---|---|---|
| OR: PP+SIPP+ordered execution | 100.0 / 100.0 | 99.3 / 98.0 | 98.2 / 93.8 | 95.7 / 89.1 | 0.0 + 1.1 |
| PPO, compact obs | 60.0 / 60.0 | 35.0 / 36.0 | 14.3 / 14.2 | 7.5 / 7.5 | 30.3 + 6.7 |
| BC from OR | 39.0 / 41.0 | 18.0 / 19.7 | 13.0 / 15.2 | 7.9 / 8.2 | 0.5 + 2.7 |
| BC then PPO | 63.0 / 61.0 | 23.7 / 20.7 | 13.3 / 15.3 | 6.4 / 6.9 | 31.2 + 1.4 |
| PPO, tree obs | 25.0 / 22.0 | 24.7 / 22.7 | 16.5 / 16.3 | 16.0 / 15.0 | 31.3 + 22.0 |
| Reactive rule | 82.0 / 76.0 | 46.3 / 45.7 | 25.3 / 22.3 | 11.1 / 9.8 | 0.0 + 3.5 |
| Shortest path, no coordination | 36.0 / 36.0 | 24.7 / 23.7 | 8.3 / 9.2 | 5.7 / 5.2 | 0.0 + 0.8 |

Arrival rate in % on the 10 held-out seeds per scenario, without / with malfunctions (same seeds, rate 1/1000 per train-step, 20 to 50 steps). Mean over episodes. Wall-clock on a ThinkPad i7-1260P (16 threads) with 8 worker processes, while two unrelated training jobs shared the CPU (1-minute load average median 23, range 11 to 44, logged in data/results/run_log/).

<!-- README_RESULTS:END -->

After a 30-minute training budget on a laptop CPU, the best learned policy is 37 points of
arrival rate behind the OR reference on small, 64 on medium, 82 on large and 80 on xlarge. From
medium up, that is larger than both the 20-point gap of NeurIPS 2020 and the 49-point gap between the best Flatland 3 OR and
RL entries (88.0% against 38.6%, [Jiang et al. 2023, Table 7](https://arxiv.org/pdf/2210.12933)),
for two reasons. First, the budget is minutes rather than days. Second, the
learned policies here have none of the structural fixes the competitive RL entries used (departure
gating, priorities, communication). The failure modes are the ones the literature describes:
behaviour cloning walks into deadlocks (up to 30.9 trains per xlarge episode), and PPO trades
deadlocks for gridlock, with 79 of 86 unarrived trains on 4 medium seeds waiting on the map at the
horizon. The OR reference, meanwhile, loses 1.3 to 6.6 points to malfunctions from medium to xlarge.
That loss is the one clear opening for learning at this scale.

### Where the trains that do not arrive end up

At the end of every held-out episode without malfunctions, each train is in one of four places:
arrived, stuck on the map (still waiting at the horizon, but not in a deadlock), deadlocked, or
never departed. Computed by `python scripts/failure_modes.py` into
[`data/analysis/failure_modes.json`](https://github.com/leandergrech/rl-flatland-rescheduling/blob/main/data/analysis/failure_modes.json).

![Where trains end up at the horizon, per policy and scenario](assets/figures/failure-modes-light.svg#only-light)
![Where trains end up at the horizon, per policy and scenario](assets/figures/failure-modes-dark.svg#only-dark)

*Share of all trains in each end state, summed over the 10 held-out seeds. Colours follow the status scale (good, warning, serious, critical) and are always labelled.*

<!-- FAILURE_MODES:START -->
| Policy | Scenario | Arrived | Stuck on the map | Never departed | Deadlocked |
|---|---|---|---|---|---|
| OR: PP+SIPP+ordered execution | small | 100.0% | 0.0% | 0.0% | 0.0% |
| OR: PP+SIPP+ordered execution | medium | 99.3% | 0.0% | 0.7% | 0.0% |
| OR: PP+SIPP+ordered execution | large | 98.2% | 0.0% | 1.8% | 0.0% |
| OR: PP+SIPP+ordered execution | xlarge | 95.7% | 0.0% | 4.3% | 0.0% |
| PPO, compact obs | small | 60.0% | 37.0% | 3.0% | 0.0% |
| PPO, compact obs | medium | 35.0% | 52.7% | 9.3% | 3.0% |
| PPO, compact obs | large | 14.3% | 65.2% | 16.5% | 4.0% |
| PPO, compact obs | xlarge | 7.5% | 61.4% | 23.5% | 7.6% |
| BC from OR | small | 39.0% | 40.0% | 10.0% | 11.0% |
| BC from OR | medium | 18.0% | 28.0% | 34.7% | 19.3% |
| BC from OR | large | 13.0% | 25.8% | 38.3% | 22.8% |
| BC from OR | xlarge | 7.9% | 24.8% | 42.7% | 24.6% |
| BC then PPO | small | 63.0% | 25.0% | 4.0% | 8.0% |
| BC then PPO | medium | 23.7% | 44.3% | 23.7% | 8.3% |
| BC then PPO | large | 13.3% | 49.2% | 29.3% | 8.2% |
| BC then PPO | xlarge | 6.4% | 47.0% | 38.4% | 8.2% |
| PPO, tree obs | small | 25.0% | 21.0% | 54.0% | 0.0% |
| PPO, tree obs | medium | 24.7% | 7.3% | 67.3% | 0.7% |
| PPO, tree obs | large | 16.5% | 14.0% | 69.2% | 0.3% |
| PPO, tree obs | xlarge | 16.0% | 14.7% | 68.3% | 1.0% |
| Reactive rule | small | 82.0% | 15.0% | 3.0% | 0.0% |
| Reactive rule | medium | 46.3% | 37.0% | 12.7% | 4.0% |
| Reactive rule | large | 25.3% | 50.7% | 13.3% | 10.7% |
| Reactive rule | xlarge | 11.1% | 58.6% | 24.0% | 6.3% |
| Shortest path, no coordination | small | 36.0% | 28.0% | 0.0% | 36.0% |
| Shortest path, no coordination | medium | 24.7% | 13.0% | 2.7% | 59.7% |
| Shortest path, no coordination | large | 8.3% | 33.7% | 6.8% | 51.2% |
| Shortest path, no coordination | xlarge | 5.7% | 21.9% | 10.7% | 61.7% |

<!-- FAILURE_MODES:END -->

Each method fails in its own way, and the way it fails says what it is missing:

- **OR reference:** it only loses trains it never dispatches (0.7% on medium up to 4.3% on
  xlarge). These are trains it could not plan to arrive within the horizon, so it keeps them off the
  network rather than let them block others. Nothing it dispatches gets stuck.
- **PPO, compact observation:** it gridlocks. Most of the trains that do not arrive are on the map
  at the horizon (52.7% of all trains on medium, 61.4% on xlarge), waiting at switches, while true
  deadlocks stay at 3.0 to 7.6%. It learned that entering a contested segment is dangerous, not
  who should go first.
- **PPO, tree observation:** it mostly never dispatches (67.3 to 69.2% of trains from medium up).
  That is the safe, useless end of the same trade-off, and it explains its low deadlock counts.
- **Behaviour cloning:** it deadlocks (11.0% of trains on small, 24.6% on xlarge) and also leaves
  many trains undispatched. It copies the planner's actions without the reservations behind them.
- **Shortest path, no coordination:** mostly deadlock (36.0% of trains on small, 51.2 to 61.7%
  from medium up), as expected with no coordination.

## Where RL fails, mechanically

1. **Deadlocks it walks into.** A shared policy that sees 30 cells ahead cannot know that a train
   will appear from a station at the other end of a 40-cell single-track link. JBR_HSE's answer was
   a separate departure classifier gating entry into the network, and MARMot's was handcrafted
   traffic lights ([Laurent et al. 2021, §4](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)).
   Both are hand-built partial substitutes for a reservation table.
2. **Departure timing.** Most of the value is in *when* a train enters the network, and that
   decision has the longest-delayed consequences. Maze-Flatland split it into its own policy for
   exactly this reason ([Castagna et al. 2026](https://arxiv.org/html/2605.10257v1)).
3. **Scale generalisation.** Policies trained at tens of trains lose accuracy at hundreds.
   TreeLSTM's arrival rate falls from 94.3% at 7 trains to 34.9% at 400 on the Flatland 3 stages,
   after a 6-phase curriculum up to 200 trains
   ([Jiang et al. 2023, Table 6](https://arxiv.org/pdf/2210.12933)).
4. **Sample cost.** TreeLSTM spent 5 days on one curriculum phase. MAMBA's point was sample
   efficiency: with a learned world model it beat a JBR_HSE re-implementation at budgets of 300k
   and 1M environment steps with 5 and 10 trains, and it was only tested at 5 to 15 trains
   ([Egorov & Shpilman 2022](https://arxiv.org/abs/2205.15023)).

## Where OR fails, and by how much

OR is not solved either, which is the opening for learning:

- **Malfunctions.** The ECML 2026 winner delivered 100% of trains on the clean levels but 56 to 78%
  per level with malfunctions, and 15.6% on the level with infrastructure disruptions, according
  to its README ([v4-sipp-locks](https://github.com/darshanmakwana412/ecml2026)).
  Partial replanning was worth 19.9% less flowtime for the 2020 winner
  ([Li et al. 2021](https://ojs.aaai.org/index.php/ICAPS/article/download/15994/15805/19487)),
  so repair matters, and it is where the OR entries spend their engineering effort.
- **Knock-on delay from order-preserving execution.** MCP keeps the plan's order in every cell, so a
  broken train holds up everyone planned behind it. On our large scenario (seed 1000, malfunctions
  on), ordered execution holds trains for 4,349 train-steps behind late or broken trains (4,036
  late, 313 broken), against 742 steps of waiting the plan itself contains ([`notebooks/03-first-experiment.ipynb`](https://github.com/leandergrech/rl-flatland-rescheduling/blob/main/notebooks/03-first-experiment.ipynb)).
- **Density.** At high density some trains cannot be planned at all within the horizon and are never
  dispatched. The ECML 2026 runner-up does this deliberately ("abandon unplaceable",
  [v5-dispatcher](https://github.com/Avinash837/ecml2026-starterkit/tree/v5-dispatcher-submission)).
- **Engineering cost.** The winning systems stack several search components and careful engineering:
  Andreica rewrote his planner in C++ ([interview](https://www.aicrowd.com/blogs/flatland-mugurel)),
  and the 2020 winner used simulated annealing just to allocate its time budget across test levels
  ([Li et al. 2021](https://ojs.aaai.org/index.php/ICAPS/article/download/15994/15805/19487)). A
  learned policy that is competitive at a fraction of that engineering would already be useful.

## Partial observability

Every OR entry reads the full state and plans with the exact simulator. Every RL entry in
[Designs and results](04-designs.md) gives each train a local view: a tree to depth 1 to 3, or depth over 10
for TreeLSTM. Communication (JBR_HSE's attention, MAMBA's messages, TreeLSTM's cross-train
attention) is how RL entries claw back global information. It was JBR_HSE's single most effective
change ([Laurent et al. 2021, §4.2](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)).
A fair comparison would give RL the full state too. Almost nobody does, because the per-train
observation is what makes parameter sharing scale.

## Safety

Flatland prevents collisions by construction (MotionCheck), so "safety" here means not deadlocking
and not stranding trains. OR's ordered execution *guarantees* deadlock-freedom given a
collision-free plan. No RL method on Flatland has a guarantee. Maze-Flatland reports deadlocks
at or below 5% ([Castagna et al. 2026](https://arxiv.org/html/2605.10257v1)); shielding MARL with
safety contracts has been tried on Flatland, but its Flatland numbers could not be verified
([Adalat et al. 2026](https://arxiv.org/abs/2606.14130)). Any deployment story needs a shield. The
natural one is ordered execution itself: RL proposes, the MCP layer disposes.

## Sim-to-real

Flatland abstracts away train length, braking curves, signalling headways, crews, rolling stock and
passenger connections (see [02-primer](how/1-network.md#real-maps)).
Its networks are procedurally generated, its timetables come from a formula, and its malfunctions
are memoryless Poisson onsets with uniform durations
([`malfunction_generators.py`](https://github.com/flatland-association/flatland-rl/blob/v4.3.0/flatland/envs/malfunction_generators.py)).
No published Flatland policy has been tested on real operations. The ECML 2026 edition moved
closer with a single realistic topology, intermediate stops with time windows and hidden disruption
parameters ([ECML 2026](https://flatland-association.github.io/flatland-book/challenges/ecml2026.html)).
AI4REALNET's human-in-the-loop framing is the most direct path to real use
([ai4realnet.eu](https://ai4realnet.eu/)). DISPLIB offers OR-style dispatching instances as a
different bridge ([Kloster et al. 2025](https://arxiv.org/abs/2509.12254)).

## Data

There is no operational data in Flatland: every instance is generated. Imitation learning therefore
imitates planners, not dispatchers. That caps imitation at the planner's quality unless RL
fine-tuning adds something, which is what the imitation-then-PPO baseline here tests. The ECML 2026
competition hid its line and malfunction parameters from competitors
([level config](https://flatland-association.github.io/flatland-book/challenges/ecml2026/levelconfig.html)),
which makes overfitting to the generator harder and is the right direction.

## What changes at competition scale

| Aspect | Here (≤ 100×100, ≤ 100 trains) | Competition scale (up to 314×314, 6,256 trains) |
|---|---|---|
| Planning time | Our Python PP+SIPP plans 100 trains in 2 to 7 s (under a shared CPU) | The 2020 winner needed C++, LNS, lazy planning and parallel search to plan thousands of trains within minutes ([Li et al. 2021](https://ojs.aaai.org/index.php/ICAPS/article/download/15994/15805/19487)) |
| Simulator cost | 0.5 ms (small) to 3.6 ms (xlarge) per env step with a dummy observation builder, on a shared CPU | Observation building dominates: the 2020 winner saw a more than 6× speed-up from skipping it on 556-train instances |
| RL inference | ~1 ms per step for 100 trains with a 21k-parameter MLP | Per-step inference over thousands of trains inside a 10 s step limit rules out large per-train models without batching |
| Density and horizon | Horizons of 129 to 2,021 steps, ≤ 100 trains | More trains per corridor; the planner starts to leave trains unplanned; RL deadlocks cascade |
| Scoring | Mean over fixed seeds | Sum over episodes solved before a time limit, stopping when < 25% of trains arrive; speed buys score |
| Training budget | 30 minutes per learned baseline | TreeLSTM: 5 days for one curriculum phase ([Jiang et al. 2023](https://arxiv.org/pdf/2210.12933)) |
