---
icon: fl/idea
---

# :fl-idea: Open questions you could attack

Ranked by expected value per unit of effort for you: an RL researcher with ATC multi-agent experience,
one laptop-class CPU for prototyping, and access to a cluster for scaling. Effort assumes this
repository as the starting point. Each entry says why it is open, what a first paper would show,
and what it would take.

The ranking follows from the results in [Designs and results](04-designs.md#results-on-our-scenarios). On
their own, learned policies here are 37 to 82 points of arrival rate behind the planner after a
30-minute training budget on a laptop CPU, while the planner loses 1.3 to 6.6 points to
malfunctions from medium to xlarge. So the openings that attach learning to a planner (1 to 4) rank above those that try to replace it (5 to 7).

![The seven openings by estimated effort, in rank order](assets/figures/openings-light.svg#only-light)
![The seven openings by estimated effort, in rank order](assets/figures/openings-dark.svg#only-dark)

*Effort ranges from the sections below, in rank order. "2–3 months" and "2+ months" are drawn as 9 to 13 weeks.*

## Update: what the TADA dispatcher settled and what it opened

[Page 8](08-tada-dispatcher.md) puts a learned, windowed dispatcher on top of the OR reference
(TADA's structure, transplanted from air traffic control). Three things changed:

- **Question 5 (shielded RL) is half answered.** A re-timed reservation table under the planner
  made every offered edit safe: no deadlock or reservation violation in any episode, including
  200 episodes of random edits ([step 1](08-tada-dispatcher.md#step-1-the-executor-reproduces-main-exactly))
  and the continuous runs that inject up to 400 trains. The other half, how far learning improves on the
  plan it is shielded by, came out close to zero after 55 minutes of training on 8 cores. On held-out medium the
  learned layer ties the executor, and its only gains are a few trains on large and xlarge with
  malfunctions ([results](08-tada-dispatcher.md#results-on-medium),
  [generalisation](08-tada-dispatcher.md#generalisation-without-retraining)).
- **Question 1 (learned repair) is sharpened, not closed.** On the harder training maps, editing
  plans at all (even near-randomly) delivered more trains than the executor; what PPO learned was
  to stop those edits costing delay, not which edits to make
  ([training curves](08-tada-dispatcher.md#results-on-medium)). Credit assignment is the
  bottleneck: a decision point on most env steps, almost all of them no-ops, and a reward at the
  horizon. That favours question 1's design (decide only when a train is actually held behind a
  late one, and score the choice with counterfactual rollouts) over PPO on a dense decision stream.
- **A new, cheap opening: a relevance window.** The context window's rules are a safety filter, so
  the window is full on most steps ([what did not work](08-tada-dispatcher.md#what-did-not-work)).
  TADA worked because its rule picked the aircraft that mattered next. The rail equivalent would admit
  only trains whose plan just slipped behind a late or broken train, which is question 1's trigger.
  It needs no new machinery on the branch, about one to two weeks.

The ranking below is unchanged. The branch makes questions 1 and 5 cheaper: the executor already
replans a train from its current position, which question 1 needs and which this page priced at a
week.

## Update: related work checked (2026-10-02)

A search for learning on top of planners ([Learning on top of a planner](literature/hybrids.md))
changes the starting point of four questions, not their ranking:

- **Question 1** has a close Flatland precedent. A 2022 thesis learned which wait or reroute
  resolution to apply to delay conflicts on a conflict-free timetable, and lost to simple
  baselines ([Leichthammer 2022](07-references.md#leichthammer2022)). It also has a strong
  non-learned baseline in multi-robot execution: online, deadlock-free re-ordering by search or
  optimisation ([Feng et al. 2024](07-references.md#feng2024);
  [Berndt et al.](07-references.md#berndt2023)).
- **Question 3** has positive prior results outside rail: learned priority orders for
  prioritized planning ([Zhang et al. 2022](07-references.md#zhang2022)), and, with PPO and
  attention, about 25% higher warehouse throughput than random priorities
  ([Zheng et al. 2026](07-references.md#zheng2026)). Flatland with malfunctions is still
  untested.
- **Question 4**: the 2020 winner's planner has been used as an expert queried during training,
  at 5 to 15 trains ([Bourgeat et al. 2026](07-references.md#bourgeat2026)).
- **Question 5**: the dispatcher of page 8 is a planner-computed preemptive shield in the sense
  of [Alshiekh et al. 2018](07-references.md#alshiekh2018).

## 1. Learned repair: when to break the planned order after a malfunction

**Why open.** Every OR winner keeps a plan's cell-visiting order after disruptions (MCP) and then
spends engineering effort on replanning: LNS-based partial replanning in 2020 and 2021
([Li et al. 2021](https://ojs.aaai.org/index.php/ICAPS/article/download/15994/15805/19487);
[Chen et al. 2023](https://arxiv.org/abs/2306.06455)), and "overstay rights" plus corridor locks in
the ECML 2026 winner ([repo](https://github.com/darshanmakwana412/ecml2026)). Malfunctions are
still where OR loses most: the ECML 2026 winner fell from 100% delivered on clean levels to 56 to
78% with malfunctions. On our scenarios the OR reference loses 1.3, 4.3 and 6.6 points of arrival
rate to malfunctions on medium, large and xlarge (same seeds, malfunctions on and off), and knock-on
waiting behind late or broken trains is almost six times the waiting the plan itself contains
(4,349 against 742 train-steps on large, seed 1000, [notebook 03](https://github.com/leandergrech/rl-flatland-rescheduling/blob/main/notebooks/03-first-experiment.ipynb)). The one
published learned repair layer on Flatland, a 2022 thesis, resolved delay conflicts on a
conflict-free timetable with DQN and did not beat waiting or a heuristic
([Leichthammer 2022](07-references.md#leichthammer2022)). In multi-robot execution, the same
re-ordering decision is solved by search without learning
([Feng et al. 2024](07-references.md#feng2024)). This is where the hypothesis that RL can beat OR
on malfunction response can actually be tested.

**First paper.** A gate that, when a train is held behind a late or broken train, chooses "keep
order" or "overtake and re-plan this train". Trained as a contextual bandit with counterfactual
rewards from cloned environments (the simulator is deterministic between malfunctions). Show: X%
of the arrivals lost to malfunctions recovered, zero deadlocks, Y ms per decision, against the OR
reference, against "always re-plan" as an oracle-ish upper bound, and against a non-learned
re-ordering search in the style of switchable-edge search.

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
rewards. It works in multi-robot path finding: a ranking learned from the best of many runs
([Zhang et al. 2022](07-references.md#zhang2022)), and PPO-learned priority orders with about 25%
higher warehouse throughput than random ones ([Zheng et al. 2026](07-references.md#zheng2026)).
Nobody has tried it on rail with malfunctions.

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
extension as question 1. The 2020 winner's planner has been used this way inside a world model,
at 5 to 15 trains: 80.8% arrived at 10 trains against the planner's 98.0%
([Bourgeat et al. 2026, Table 1](07-references.md#bourgeat2026)).

**First paper.** DAgger with PP+SIPP as the expert, on this repo's decision interface. Train at 30
trains, test at 60 and 100, and measure the gap to the expert against the number of expert queries.

**Effort.** 3 to 4 weeks, shared with question 1's planner work.

## 5. Shielded RL: deadlock-freedom by construction

**Why open.** No RL method on Flatland guarantees anything; Maze-Flatland reports ≤ 5% deadlocks.
Ordered execution is a ready-made shield: any proposed move that would violate the planned visiting
order is replaced by a wait. Safe-RL calls this kind of check a shield
([Alshiekh et al. 2018](07-references.md#alshiekh2018);
[ElSayed-Aly et al. 2021](07-references.md#elsayed2021) for the multi-agent case). Contract-based shielding has touched Flatland
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
