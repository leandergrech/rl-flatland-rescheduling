# 5. Limitations: what fails, and by how much

## The RL-versus-OR gap in the official rounds

| Round | Best OR | Best RL | Gap | Source |
|---|---|---|---|---|
| NeurIPS 2020 | 297.507, 98.6% arrived, 363 envs | 214.150, 78.5% arrived, 336 envs | 20.1 points of arrival rate; RL ranked 8th overall in Round 2 | [Laurent et al. 2021, Table 1](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf); [Li et al. 2021](https://ojs.aaai.org/index.php/ICAPS/article/download/15994/15805/19487) |
| Flatland 3 (2021) | 135.47 at close; 141.0 and 88.0% arrived on the current leaderboard | 27.868, 38.6% arrived | Score ratio 0.21; the winners matched the best RL score in 3 minutes of their 2-hour budget | [Chen et al. 2023](https://arxiv.org/abs/2306.06455); [Jiang et al. 2023, Table 7](https://arxiv.org/pdf/2210.12933) |
| Flatland 3 stages, post hoc | 141.0, 88.0% | TreeLSTM 125.3, 66.4% | 21.6 points of arrival rate | [Jiang et al. 2023, Table 7](https://arxiv.org/pdf/2210.12933) |
| ECML 2026 | 20.8429 | 9.4231 (organisers' baseline; the only RL-track entry listed) | Score ratio 0.45 | [post-competition analysis](https://flatland-association.github.io/flatland-book/challenges/ecml2026/post-competition_analysis.html) |

The organisers' 2020 verdict still holds: "RL solutions are still a considerable distance away
from OR based solutions" ([Laurent et al. 2021, §6](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)).

## On our mid-size scenarios

OURS_PLACEHOLDER

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
  broken train holds up everyone planned behind it. On our large scenario (seed 2000), ordered
  execution adds 1,725 held train-steps behind late or broken trains, on top of 1,240 steps of
  waiting the plan itself contains ([`notebooks/03-first-experiment.ipynb`](https://github.com/leandergrech/rl-flatland-rescheduling/blob/main/notebooks/03-first-experiment.ipynb)).
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
[04-designs](04-designs.md) gives each train a local view: a tree to depth 1 to 3, or depth over 10
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
passenger connections (see [02-primer](02-primer.md#flatlands-abstraction-and-what-it-leaves-out)).
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
