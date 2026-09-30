# 6. Open questions you could attack

Ranked by expected value per unit of effort for you: an RL researcher with ATC multi-agent experience,
one laptop-class CPU for prototyping, and access to a cluster for scaling. Effort assumes this
repository as the starting point. Each entry says why it is open, what a first paper would show,
and what it would take.

RANKING_NOTE_PLACEHOLDER

## 1. Learned repair: when to break the planned order after a malfunction

**Why open.** Every OR winner keeps a plan's cell-visiting order after disruptions (MCP) and then
spends engineering effort on replanning: LNS-based partial replanning in 2020 and 2021
([Li et al. 2021](https://ojs.aaai.org/index.php/ICAPS/article/download/15994/15805/19487);
[Chen et al. 2023](https://arxiv.org/abs/2306.06455)), and "overstay rights" plus corridor locks in
the ECML 2026 winner ([repo](https://github.com/darshanmakwana412/ecml2026)). Malfunctions are
still where OR loses most: the ECML 2026 winner fell from 100% delivered on clean levels to 56 to
78% with malfunctions. On our scenarios the OR reference loses 3.7 to 6.1 points of arrival rate to
malfunctions on medium to xlarge, and knock-on waiting behind late trains exceeds the waiting the
plan contains (1,725 against 1,240 train-steps on large, seed 2000, [notebook 03](https://github.com/leandergrech/rl-flatland-rescheduling/blob/main/notebooks/03-first-experiment.ipynb)). No published RL work targets
the repair decision on top of a planner. This is the place where the brief's "RL beats OR on
malfunction response" hypothesis can actually be tested.

**First paper.** A gate that, when a train is held behind a late or broken train, chooses "keep
order" or "overtake and re-plan this train". Trained as a contextual bandit with counterfactual
rewards from cloned environments (the simulator is deterministic between malfunctions). Show: X%
of the arrivals lost to malfunctions recovered, zero deadlocks, Y ms per decision, against the OR
reference and against "always re-plan" as an oracle-ish upper bound.

**Effort.** 4 to 6 weeks: 1 week to make the planner re-plan a train from its current position,
1 week for counterfactual data, 1 week for the gate, 1 to 2 weeks for evaluation at 30 to 100
trains and a Flatland 3-stage run. `notebooks/03-first-experiment.ipynb` is the scaffold.

## 2. A learned departure dispatcher on top of planned routing

**Why open.** When a train enters the network is the decision with the longest-delayed
consequences. JBR_HSE gated it with a supervised classifier
([Laurent et al. 2021, §4.2](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)),
Maze-Flatland gave it its own policy
([Castagna et al. 2026](https://arxiv.org/html/2605.10257v1)), and the 2020 winner's "lazy
planning" (plan only some trains up front) was its single largest late gain, from 291.9 to 297.5
([Laurent et al. 2021, §4.1](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)).
Nobody has combined a learned dispatcher with a planner for routing.

**First paper.** RL chooses which waiting train to release next, and PP+SIPP routes it against
current reservations. Compare with plan-everything-at-once and with slack-ordered release, at
matched planning time.

**Effort.** 3 to 5 weeks. The environment wrapper already separates departure decisions.

## 3. Learned priorities for prioritized planning

**Why open.** PP quality depends almost entirely on the order
([Ma et al. 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4758)). The winners used simple
keys (fast first, slack first) plus random restarts or LNS
([Chen et al. 2023](https://arxiv.org/abs/2306.06455)). A learned ranking that matches best-of-K
restarts in one pass trades training compute for planning time, which the competition clock
rewards.

**First paper.** Train a per-train scoring network (features: slack, speed, corridor overlap with
other trains' shortest paths, departure time) by imitating the best of K random orderings. Report
the arrivals and lateness of one learned ordering against best-of-K, as a function of K and
planning time.

**Effort.** 2 to 3 weeks. `or_planner.plan_all` already takes an ordering.

## 4. Imitation at scale: DAgger from the planner

**Why open.** Imitation from OR has helped RL on Flatland since 2020
([Mohanty et al. 2020](https://arxiv.org/abs/2012.05893): 86% with 25% imitation data against 81%
for PPO on 5-train maps), and Maze-Flatland's policies are cloned from MCTS. But plain behaviour
cloning compounds its errors, and nobody has shown a cloned policy generalising from tens to
hundreds of trains. DAgger needs an expert that can answer "what would you do here?" from states
the learner reached, which means re-planning from arbitrary states. That is the same planner
extension as question 1.

**First paper.** DAgger with PP+SIPP as the expert, on this repo's decision interface. Train at 30
trains, test at 60 and 100, and measure the gap to the expert against the number of expert queries.

**Effort.** 3 to 4 weeks, shared with question 1's planner work.

## 5. Shielded RL: deadlock-freedom by construction

**Why open.** No RL method on Flatland guarantees anything; Maze-Flatland reports ≤ 5% deadlocks.
Ordered execution is a ready-made shield: any proposed move that would violate the planned visiting
order is replaced by a wait. Contract-based shielding has touched Flatland
([Adalat et al. 2026](https://arxiv.org/abs/2606.14130)), but without numbers we could verify.

**First paper.** PPO acting through an MCP shield over a coarse plan, with the shield's
interventions as a cost signal. Show zero deadlocks and how far learning can improve on the plan
it is shielded by.

**Effort.** 4 to 6 weeks.

## 6. Sequencing through shared resources, rail and air together

**Why open.** Terminal-airspace arrival sequencing and junction ordering on rail are the same
decision: which flow goes first through a shared resource, taken ahead of time, with knock-on
delays. AI4REALNET funds both testbeds, Flatland and BlueSky, in one project
([ai4realnet.eu](https://ai4realnet.eu/)), and on both sides stand-alone RL trails the classical
method while hybrids help ([Ribeiro et al. 2022](https://api.openalex.org/works/doi:10.3390/aerospace9120847)).

**First paper.** One repair or priority policy architecture (question 1 or 3), trained separately on
Flatland and on a BlueSky terminal-area scenario, with a shared feature vocabulary (slack, conflict
horizon, queue length), and a transfer test.

**Effort.** 2 to 3 months; the natural bridge to TADA and an AI4REALNET collaboration.

## 7. RL as a fast replanner at competition scale

**Why open.** The score is a sum over episodes solved within a time limit, so speed is score. At
thousands of trains a per-decision learned policy is O(1) per train, while replanning grows with
density. TreeLSTM ran up to 425 trains but needed 2,330 s at the largest stage
([Jiang et al. 2023, Table 6](https://arxiv.org/pdf/2210.12933)).

**First paper.** Wall-clock-matched comparison of a learned repair policy against LNS partial
replanning at 500 to 3,000 trains.

**Effort.** 2+ months and cluster time, plus a competition-scale planner (C++ or the 2020 winner's
code under its academic licence, which forbids use in later Flatland challenges).
