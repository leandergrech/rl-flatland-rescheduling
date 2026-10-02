---
icon: fl/method
---

# :fl-method: Methods and evaluation

!!! abstract "In short"

    Every controller in this project is evaluated by one harness, on the same four Flatland
    scenarios and the same 10 held-out seeds per scenario, once without and once with breakdowns:
    80 episodes per controller. Policies act greedily at evaluation. Results are means over seeds
    with standard errors, and every episode is stored in `data/`. This page states the protocol,
    what each controller is and where it comes from, what the metrics mean, which statements are
    guarantees and which are observations, and why the competition numbers quoted elsewhere are
    context, not comparisons.

## Research questions

1. **How far is reinforcement learning from operations research on mid-size Flatland maps, at a
   CPU budget?** Answered by evaluating a reproduction of the winning planners' core against
   policies learned from scratch, on identical maps ([Designs and results](../04-designs.md)).
2. **Can a learned layer on top of the planner recover what the planner loses to breakdowns,
   without giving up its safety?** Answered by a learned dispatcher that may only edit the plan
   through checked clearances ([TADA on rails](../08-tada-dispatcher.md)).

Neither question was framed as a hypothesis with a pre-registered test; the comparisons below are
descriptive, with standard errors over seeds.

## Benchmark and scenarios

All experiments use **flatland-rl 4.3.0** with its sparse rail generator, the Flatland 3 mixed
speed profile (a quarter of the trains each at speed 1, 1/2, 1/3 and 1/4) and Flatland 3's
`DefaultRewards` (`src/rl_flatland/scenarios.py`).

| Scenario | Grid | Trains | Cities | Horizon \(T\) on the held-out seeds |
|---|---|---|---|---|
| small | 30×30 | 10 | 2 | 129 to 226 |
| medium | 50×50 | 30 | 4 | 439 to 902 |
| large | 80×80 | 60 | 6 | 1,010 to 1,450 |
| xlarge | 100×100 | 100 | 8 | 1,116 to 2,021 |

Horizons from [`data/scenarios/scenario_stats.json`](https://github.com/leandergrech/rl-flatland-rescheduling/blob/main/data/scenarios/scenario_stats.json)
(see [Trains, timetables and breakdowns](../how/2-trains.md#what-the-numbers-look-like)).

**Seeds.** Training uses seeds 0 to 999. Evaluation uses seeds 1000 to 1009 for every scenario,
twice: **without malfunctions** and **with malfunctions** (rate 1/1000 per train-step, duration 20
to 50 steps). Switching malfunctions on does not change a seed's network, trains or timetable, and
flatland draws breakdowns independently of the policy, so every controller meets the same
breakdowns on the same seed. Comparisons between controllers, and between the two conditions, are
therefore paired.

## Controllers compared

| Controller | Provenance | Decides | Trained on | Training budget | Code |
|---|---|---|---|---|---|
| OR planner: prioritized planning + SIPP + ordered execution | **reproduced** from published descriptions of the 2019 and 2020 winners (no code copied) | a timed path per train at reset; trains keep each cell's planned order | – | none | [`baselines/or_planner.py`](https://github.com/leandergrech/rl-flatland-rescheduling/blob/main/src/rl_flatland/baselines/or_planner.py) |
| PPO, compact observation | **implemented**, standard method | WAIT / GO / GO-alternative at decision cells | small and medium, seeds 0–999, half with malfunctions | 30 min wall-clock, 8 processes | [`baselines/ppo.py`](https://github.com/leandergrech/rl-flatland-rescheduling/blob/main/src/rl_flatland/baselines/ppo.py) |
| PPO, tree observation | **implemented**, standard method | same | same | 30 min wall-clock, 8 processes | same |
| Behaviour cloning from the planner, then PPO | **implemented**, standard method | same | same, plus 120 planner episodes | 30 min wall-clock, 8 processes | [`baselines/imitation.py`](https://github.com/leandergrech/rl-flatland-rescheduling/blob/main/src/rl_flatland/baselines/imitation.py) |
| Reactive rule; shortest path | **implemented** reference heuristics | enter a segment only if no opposing train is on it; or always follow the shortest path | – | none | [`policies.py`](https://github.com/leandergrech/rl-flatland-rescheduling/blob/main/src/rl_flatland/policies.py) |
| Learned dispatcher on the planner ("TADA on rails") | **original**: adapted from the author's air-traffic project to rail | HOLD / YIELD_TO / REROUTE clearances, checked against a live reservation table | medium only, seeds 0–999, malfunctions on | 115 PPO iterations × 8 episodes, 55 minutes on 8 processes | [`tada/`](https://github.com/leandergrech/rl-flatland-rescheduling/tree/main/src/rl_flatland/tada) |

Hyperparameters are stored next to each checkpoint (`data/checkpoints/<run>/config.json`,
`data/tada/checkpoints/<run>/config.json`). The baselines and their design choices are described in
[Designs and results](../04-designs.md#baselines-in-this-repository); the dispatcher's in
[TADA on rails](../08-tada-dispatcher.md#architecture).

**Evaluation is greedy**: learned policies take the arg-max of their masked action distribution, so
a stored checkpoint gives the same episodes every time. The OR planner is deterministic (its
optional wall-clock cap is off, because a cap makes the plan depend on machine load).

## Metrics

- **Arrival rate**: the share of trains that reach their target by the horizon. The primary
  metric here: it is what a railway would count, and it is not affected by how the score is
  normalised.
- **Normalised reward**: Flatland 3's episode score, \(1 + \frac{1}{NT}\sum_i \max(R_i, -T)\),
  where \(R_i\) is the train's lateness, cancellation or remaining-delay penalty
  ([The problem](../01-problem.md#objective)). It is 1 when every train is on time, but it is
  forgiving: a missing train costs only its travel time divided by \(NT\). Policies that deliver
  6% of trains still score above 0.5, so the two metrics must be read together.
- **Deadlocked trains** at the end of the episode (`rl_flatland.deadlock.find_deadlocked`).
- **Wall-clock per step and setup time**, which depend on the machine and are only comparable
  within a table.

![Arrival rate against normalised reward for every policy and scenario](../assets/figures/overview-metrics-light.svg#only-light)
![Arrival rate against normalised reward for every policy and scenario](../assets/figures/overview-metrics-dark.svg#only-dark)

*Means over the 10 held-out seeds without malfunctions; each point is one policy on one scenario.
Points far above the dashed line are policies whose score looks better than their arrivals.
Source: [`data/results/`](https://github.com/leandergrech/rl-flatland-rescheduling/tree/main/data/results),
figure by `python scripts/make_overview_figures.py`.*

**Uncertainty.** Tables report the mean over 10 seeds ± the standard error of that mean. Ten seeds
resolve large differences (the planner against learned policies) but not differences of one or two
trains: for the learned dispatcher against the planner, the paired per-seed differences are given
instead of a significance claim ([TADA on rails, results](../08-tada-dispatcher.md#results-on-medium)).

## Compute

All runs used one laptop CPU: a 12th Gen Intel Core i7-1260P (16 logical CPUs), Linux 6.8, with 8
worker processes ([software environment](../07-references.md#software-environment-verified-by-installation)).
The main baselines were trained while unrelated jobs shared the CPU, and during the dispatcher runs
the processor's clock varied between 400 MHz and 2.9 GHz (logged per iteration). Wall-clock
figures are therefore upper bounds, and the dispatcher's runs were budgeted by iterations (equal
samples), with a 55-minute cap.

## Guarantees and observations

| Statement | Kind | Conditions and evidence |
|---|---|---|
| Executing a collision-free plan in each cell's planned order cannot deadlock, however late trains are | **guarantee**, under assumptions | the plan is collision-free under flatland's conflict model, every train follows it, and breakdowns only delay; argument in [Executing a plan](../how/5-execution.md#keep-the-order-not-the-times). It says nothing about trains the planner could not route: those never depart and are charged as cancellations. |
| The OR planner never deadlocked | **observation** | 0 deadlocked trains in all 80 evaluation episodes ([results](../04-designs.md#results-on-our-scenarios)) |
| A clearance cannot create a deadlock | **by construction, not proven** | each clearance is a SIPP search against the live reservation table and is committed only if it succeeds; observed 0 deadlocks and 0 reservation violations in 200 random-clearance episodes (28,705 edits) and in every dispatcher evaluation ([step 1](../08-tada-dispatcher.md#step-1-the-executor-reproduces-main-exactly)) |
| The learned dispatcher improves on the planner | **not supported** at 10 seeds | ties on medium; a few trains gained on large and xlarge with malfunctions, within seed-to-seed variation ([results](../08-tada-dispatcher.md#generalisation-without-retraining)) |
| The dispatcher generalises from medium to other sizes | **observation** on the evaluated seeds only | evaluated without retraining on 10 seeds each of small, large and xlarge |
| The in-browser Lab reproduces flatland-rl | **observation** | identical step by step on the 40 exported maps, five recorded policies each, the live planner and 302 recorded clearances ([how it is checked](../how/7-lab.md#how-faithful-it-is)); untested on other maps or flatland versions |

## Controlled experiments and competition results

The site quotes competition results (for example 98.6% against 78.5% of trains arrived in NeurIPS
2020) to show where the field stands. They are **not comparable** with this project's numbers:

- **Different instances and scale.** The competitions evaluated hundreds of instances up to
  314×314 cells and 6,256 trains; this project uses four fixed mid-size scenarios, 10 seeds each.
- **Different scoring.** A competition score is the sum of normalised rewards over the episodes
  solved before an evaluation time limit, with early stopping when too few trains arrive; here each
  episode is scored on its own.
- **Different budgets.** Competition entries were developed over weeks and could plan for minutes
  per instance on dedicated hardware; the learned baselines here train for 30 minutes of wall-clock on one laptop CPU.
- **Different code.** The OR planner here reproduces the core of the winners (no large
  neighbourhood search, no partial replanning, Python instead of C++).

What carries over is the direction of the comparison, not its size
([Timeline](../03-timeline.md), [Limitations](../05-limitations.md#what-changes-at-competition-scale)).

## Reproducing

Commands, versions and run times are on [Reproducibility](../implementation/reproduce.md).
