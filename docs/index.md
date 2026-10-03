---
icon: fl/home
hide:
  - toc
---

<div class="fl-hero" markdown>

<p class="fl-eyebrow">A research investigation by Leander Grech · flatland-rl 4.3.0 · CPU only</p>

# Can learning beat planning at train rescheduling?

Flatland is the railway scheduling benchmark of SBB, Deutsche Bahn and SNCF. Trains must reach
their targets inside timetable windows on a shared network, one train per track cell, while random
breakdowns stop them. Planners from operations research (OR) have won every official round. This
project **reproduces** the core of the winning planner, **compares** it with reinforcement-learning
(RL) policies on the same maps and seeds, and **tests an original idea**: a learned dispatcher that
may only edit the planner's plan through checked clearances.

<div class="fl-stats">
<div class="fl-stat"><b>99.3%</b><span>trains arrived on medium maps (30 trains) with the reproduced OR planner</span></div>
<div class="fl-stat"><b>35%</b><span>the best policy learned from scratch on the same maps, after a 30-minute training budget</span></div>
<div class="fl-stat"><b>+1 / 300</b><span>net trains gained by the learned dispatcher over the planner, medium with breakdowns: no measurable gain</span></div>
<div class="fl-stat"><b>80 / 80</b><span>held-out maps, up to 100×100 with 100 trains, on which the in-browser simulator matches flatland-rl step for step</span></div>
</div>

<div class="fl-actions">
<a class="fl-action" href="#see-a-deadlock-and-how-a-plan-avoids-it">Try the deadlock demo</a>
<a class="fl-action fl-action--quiet" href="#what-was-found">Read the findings</a>
<a class="fl-action fl-action--quiet" href="research/methodology/">How it was evaluated</a>
</div>

</div>

<nav class="fl-paths" aria-label="Three ways into this site">
  <a class="fl-path" href="01-problem/">
    <span class="fl-path__step">1</span>
    <span class="fl-path__text"><b>Understand the problem</b><span>Flatland stated formally, six illustrated chapters with playable models, and the Scheduling Lab.</span></span>
  </a>
  <a class="fl-path" href="research/methodology/">
    <span class="fl-path__step">2</span>
    <span class="fl-path__text"><b>Explore the experiments</b><span>Methods, baselines and results, the learned dispatcher, limitations and open questions.</span></span>
  </a>
  <a class="fl-path" href="implementation/architecture/">
    <span class="fl-path__step">3</span>
    <span class="fl-path__text"><b>Inspect the implementation</b><span>How the code is organised, how each number is produced, and how to reproduce it.</span></span>
  </a>
</nav>

## See a deadlock, and how a plan avoids it

Two trains approach each other on single track with one passing loop. Flatland lets a train move
only into a free cell, forbids two trains from swapping cells, and never lets a train reverse on
open track. Choose what each train does, then press **Play**.

1. Leave both trains on **run straight through** and press **Play**. They meet nose to nose on
   single track. Neither can move again: a **deadlock**, permanent for the rest of the episode.
2. Press **Reset**, set train A to **take the loop and wait**, and play again. A waits in the loop
   until B has passed, and both trains arrive.
3. Now put **both** trains on **take the loop**. Each detour makes sense on its own, but together
   the trains meet head-on inside the loop. Avoiding a deadlock means deciding **who yields to
   whom**, for both trains at once, before they meet.

<div class="fl-widget" data-widget="deadlock" data-title="Interactive: two trains, one passing loop">
<p class="fl-fallback">This demo needs JavaScript, which is switched off or did not load. The
figures below show the same situation: a head-on deadlock on real maps, and how a planner orders
the trains to avoid it.</p>
</div>

**What a reservation-based planner does instead.** The OR planner settles the order before anyone
moves. It plans the trains one at a time. Each train's timed path is booked in a **reservation
table** of cells and time intervals, and later trains are routed only through the gaps that remain.
In the demo, if A is planned first, B's plan waits at its station until A has cleared the single
track. During execution every cell is entered in its planned order, so a breakdown delays the
trains behind it but cannot trap them (the conditions for this guarantee are on
[Methods](research/methodology.md#guarantees-and-observations)).

![Three time-space diagrams of a single-track corridor: the two forbidden conflicts, a prioritized plan in which B waits for A, and ordered execution holding B behind a broken-down A](assets/figures/time-space-light.svg#only-light)
![Three time-space diagrams of a single-track corridor: the two forbidden conflicts, a prioritized plan in which B waits for A, and ordered execution holding B behind a broken-down A](assets/figures/time-space-dark.svg#only-dark)

*Position on the vertical axis, time on the horizontal. (a) The two conflicts flatland forbids.
(b) Prioritized planning: A's cell-times are reserved first, so B's earliest safe plan waits.
(c) A breaks down for 6 steps; ordered execution holds B until A has left the shared track. The
red mark is the deadlock that entering at B's planned time would have caused.
Chapters: [Conflicts and deadlock](how/3-deadlock.md), [Planning with reservations](how/4-planning.md),
[Executing a plan under disruption](how/5-execution.md).*

The same thing on the project's real maps, in the browser:

[Shortest paths deadlock (small, seed 1004)](how/7-lab.md?map=small-1004-n&preset=shortest_path&t=32&train=6){ .fl-try } [The planner on the same map](how/7-lab.md?map=small-1004-n&preset=or&t=32){ .fl-try } [Be the dispatcher](how/7-lab.md?map=medium-1005-m&preset=you&t=60){ .fl-try }

## Why compare learning with planning

Flatland was set up to test whether RL could match planners at railway rescheduling. It has not
yet done so in any official round:

<div class="gap-tiles">
  <a class="gap-tile" href="http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf" title="Laurent et al. 2021, Table 1">
    <span class="gap-round">NeurIPS 2020</span>
    <span class="gap-metric">trains arrived</span>
    <span class="gap-row"><span class="gap-who">OR</span><span class="gap-val or">98.6%</span></span>
    <span class="gap-bar"><span class="or" style="width:98.6%"></span></span>
    <span class="gap-row"><span class="gap-who">RL</span><span class="gap-val rl">78.5%</span></span>
    <span class="gap-bar"><span class="rl" style="width:78.5%"></span></span>
  </a>
  <a class="gap-tile" href="https://arxiv.org/abs/2306.06455" title="Chen et al. 2023">
    <span class="gap-round">Flatland 3, 2021</span>
    <span class="gap-metric">competition score</span>
    <span class="gap-row"><span class="gap-who">OR</span><span class="gap-val or">135.5</span></span>
    <span class="gap-bar"><span class="or" style="width:100%"></span></span>
    <span class="gap-row"><span class="gap-who">RL</span><span class="gap-val rl">27.9</span></span>
    <span class="gap-bar"><span class="rl" style="width:20.6%"></span></span>
  </a>
  <a class="gap-tile" href="https://flatland-association.github.io/flatland-book/challenges/ecml2026/post-competition_analysis.html" title="ECML 2026 post-competition analysis">
    <span class="gap-round">ECML 2026</span>
    <span class="gap-metric">competition score</span>
    <span class="gap-row"><span class="gap-who">OR</span><span class="gap-val or">20.84</span></span>
    <span class="gap-bar"><span class="or" style="width:100%"></span></span>
    <span class="gap-row"><span class="gap-who">RL</span><span class="gap-val rl">9.42</span></span>
    <span class="gap-bar"><span class="rl" style="width:45.2%"></span></span>
  </a>
  <a class="gap-tile" href="04-designs/#results-on-our-scenarios" title="This project, xlarge scenario, no malfunctions">
    <span class="gap-round">This project, 100×100</span>
    <span class="gap-metric">trains arrived, 100 trains</span>
    <span class="gap-row"><span class="gap-who">OR</span><span class="gap-val or">95.7%</span></span>
    <span class="gap-bar"><span class="or" style="width:95.7%"></span></span>
    <span class="gap-row"><span class="gap-who">RL</span><span class="gap-val rl">16.0%</span></span>
    <span class="gap-bar"><span class="rl" style="width:16%"></span></span>
  </a>
</div>

*Best OR and best RL entry per round. The rounds differ in instances, scoring and budgets, so the
tiles are not comparable with each other or with this project's controlled runs
([why](research/methodology.md#controlled-experiments-and-competition-results)).*

The question stays interesting because the two approaches fail differently. A planner is only as
good as its plan: when a train breaks down, it keeps the planned order and passes the delay on to
every train behind. A learned policy can react to the current state at every step, but it must
discover by trial and error that a deadlock many steps ahead is caused by a choice made now. The
open hope for RL is where plans break: responding to breakdowns, replanning fast, and generalising
to unseen maps. This project
measures how far each approach gets on identical maps, and tries one way to combine them.

## What was built

<div class="grid cards" markdown>

-   <span class="fl-tag fl-tag--reproduced">Reproduced</span>

    **The OR planner.** Prioritized planning with safe-interval path planning (SIPP), executed in
    planned cell order: the core of the 2019 and 2020 winners, rewritten from their published
    descriptions (no code copied).

    [:octicons-arrow-right-24: Design and results](04-designs.md#or-reference-pp-sipp-ordered-execution)

-   <span class="fl-tag fl-tag--implemented">Implemented and compared</span>

    **Six baselines on the same harness.** PPO with parameter sharing on a compact and on a tree
    observation, behaviour cloning from the planner, cloning then PPO, and two rule-based
    references, all trained and evaluated on a laptop CPU.

    [:octicons-arrow-right-24: Baselines](04-designs.md#baselines-in-this-repository)

-   <span class="fl-tag fl-tag--original">Original</span>

    **TADA on rails.** The author's air-traffic dispatcher design moved to rail: a learned policy
    that may HOLD, YIELD or REROUTE trains, where every edit is re-planned against a live
    reservation table before it is accepted. A 2022 thesis tried a close variant on Flatland,
    with the same outcome ([related work](literature/hybrids.md)).

    [:octicons-arrow-right-24: The experiment](08-tada-dispatcher.md)

-   <span class="fl-tag fl-tag--original">Original</span>

    **The Scheduling Lab.** Flatland, the planner and the dispatcher ported to JavaScript and
    checked step for step against flatland-rl on 80 held-out maps, up to 100 trains. Replay every policy, break
    trains, or dispatch yourself.

    [:octicons-arrow-right-24: Open the Lab](how/7-lab.md)

</div>

Around them: a [literature review](literature/mapf.md) of the methods behind the winners and of
every result from 2019 to 2026, verified against the sources, and an
[evaluation protocol](research/methodology.md) shared by every controller: four fixed scenarios,
10 held-out seeds each, with and without breakdowns.

## What was found

![Bar chart of trains arrived per scenario. The planner and the learned dispatcher on top of it deliver 96 to 100 percent without malfunctions and 89 to 100 percent with them; policies learned from scratch and the reactive rule deliver 25 to 82 percent on small and 8 to 16 percent on xlarge](assets/figures/overview-arrival-light.svg#only-light)
![Bar chart of trains arrived per scenario. The planner and the learned dispatcher on top of it deliver 96 to 100 percent without malfunctions and 89 to 100 percent with them; policies learned from scratch and the reactive rule deliver 25 to 82 percent on small and 8 to 16 percent on xlarge](assets/figures/overview-arrival-dark.svg#only-dark)

*Trains arrived, mean over the 10 held-out seeds of each scenario; whiskers show ± one standard
error. Lower panel: the same seeds with breakdowns. The dispatcher (hatched) was trained on medium
only and evaluated on the other sizes without retraining. Behaviour cloning (with and without PPO)
and the shortest-path rule are left out for legibility; all seven baselines are in
[Designs and results](04-designs.md#results-on-our-scenarios). Numbers:
[`data/analysis/overview.json`](https://github.com/leandergrech/rl-flatland-rescheduling/blob/main/data/analysis/overview.json),
from `python scripts/make_overview_figures.py`.*

<div class="fl-findings" markdown>

1. **The reproduced planner gets almost every train home.** It delivers 100, 99.3, 98.2 and 95.7%
   of trains on the small, medium, large and xlarge held-out seeds, and had no deadlocked trains in
   any of its 80 evaluation episodes. Breakdowns cost it 0, 1.3, 4.3 and 6.6 points.
   <span class="fl-evidence">Evidence: [results tables](04-designs.md#results-on-our-scenarios)</span>
2. **Policies learned from scratch stay far behind at this budget.** The best learned policy per
   scenario delivers 63.0, 35.0, 16.5 and 16.0%, and a reactive rule beats all of them on small,
   medium and large. This describes a 30-minute training budget and these architectures, not RL
   in general. <span class="fl-evidence">Evidence: [results](04-designs.md#what-the-numbers-say),
   [limitations](05-limitations.md#on-our-mid-size-scenarios)</span>
3. **A learned layer on top of the planner was safe but did not measurably help.** No evaluation
   of the dispatcher ended in a deadlock, and neither did 200 episodes of random edits through the
   same checks. It ties the planner
   on every scenario without breakdowns. With breakdowns it gains a net 1 train of 300 on medium,
   1 of 600 on large and 5 of 1,000 on xlarge, at unchanged normalised reward: within seed-to-seed
   variation. <span class="fl-evidence">Evidence: [TADA on rails](08-tada-dispatcher.md#results-on-medium),
   [generalisation](08-tada-dispatcher.md#generalisation-without-retraining)</span>
4. **The score flatters failing policies.** Flatland's normalised reward charges a missing train
   only its travel time, so PPO scores 0.724 on medium while delivering 35% of trains. Arrival rate
   is the primary metric here. <span class="fl-evidence">Evidence: [metrics](research/methodology.md#metrics)</span>
5. **What is left to gain is the order after a breakdown.** The planner keeps the pre-breakdown
   order, so on large seed 1000 trains wait almost six times longer behind late or broken trains
   than the plan itself asks (4,349 against 742 train-steps).
   <span class="fl-evidence">Evidence: [knock-on delay](how/5-execution.md#what-it-costs-knock-on-delay)</span>

</div>

??? info "How to read these numbers"

    **Evaluation conditions.** Four scenarios, from 30×30 cells with 10 trains to 100×100 with 100.
    Each controller runs on the same 10 held-out seeds (1000 to 1009) per scenario, once without
    and once with breakdowns (rate 1/1000 per train-step, 20 to 50 steps each). A seed fixes the
    map, the timetable and the breakdowns, so every controller faces the same episodes. Learned
    policies act greedily. Full protocol: [Methods and evaluation](research/methodology.md).

    **Two metrics.** *Arrival rate* is the share of trains that reach their target before the
    episode ends. *Normalised reward* is flatland's score: 1 when every train is on time, but a
    train that never arrives costs only its travel time, so the score stays high when many trains
    fail. Medium, no breakdowns:

    | Controller | Trains arrived | Normalised reward |
    |---|---|---|
    | OR planner (reproduced) | 99.3% | 0.991 |
    | Learned dispatcher on the planner (original) | 99.3% | 0.990 |
    | Reactive rule (no learning) | 46.3% | 0.766 |
    | PPO, compact observation (learned from scratch) | 35.0% | 0.724 |

    **Uncertainty.** Ten seeds per scenario separate the planner from learned policies by many
    standard errors, but they cannot resolve differences of one or two trains. For the dispatcher
    against the planner the site reports paired per-seed differences, not significance tests.

    **Competition numbers** are quoted for context only; they come from other instances, scoring
    rules and budgets.

## Open questions

The results point at learning *with* the planner rather than instead of it. The three highest-ranked
openings, each with what a first paper would show and an effort estimate, are in
[Open questions](06-open-questions.md):

1. **Learned repair**: decide when to break the planned order after a breakdown, and score the
   choice with counterfactual rollouts.
2. **A learned departure dispatcher** on top of planned routing.
3. **Learned priorities** for prioritized planning.

Two things this project cannot answer yet: whether any of it holds at competition scale (up to
314×314 cells and 6,256 trains, [what changes](05-limitations.md#what-changes-at-competition-scale)),
and whether a longer training budget changes the ranking
([what longer training would likely change](04-designs.md#what-longer-training-would-likely-change)).

## About

<div class="fl-about" markdown>

**Leander Grech** is a researcher in reinforcement learning for physical control systems, leads
the TADA project on RL for air traffic control, and is lead author on RL-based bent-crystal
alignment at CERN. This project asks whether the dispatcher design from air traffic carries over
to rail, and tests it against the strongest classical baseline available.

[:octicons-arrow-right-24: About the author and this project](about.md) · [:octicons-mark-github-16: Source on GitHub](https://github.com/leandergrech/rl-flatland-rescheduling)

</div>
