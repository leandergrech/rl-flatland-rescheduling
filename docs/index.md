---
icon: fl/home
hide:
  - toc
---

<div class="fl-hero" markdown>

# RL for Flatland train rescheduling

A personal literature review and a runnable, CPU-sized codebase on one question: **can multi-agent
reinforcement learning close the gap to operations research on Flatland train rescheduling, or beat
it somewhere that matters?** Flatland is the railway scheduling benchmark of SBB, Deutsche Bahn and
SNCF, now run by the Flatland Association: trains on a grid-drawn network must reach their targets
inside timetable windows, never sharing a cell, while breakdowns stop them for 20 to 50 steps.

<div class="fl-stats">
<div class="fl-stat"><b>99.3%</b><span>trains home on medium for this repo's OR planner (PP + SIPP + ordered execution)</span></div>
<div class="fl-stat"><b>35%</b><span>the best policy learned from scratch on the same maps, after 30 CPU minutes</span></div>
<div class="fl-stat"><b>98.3%</b><span>a learned dispatcher on top of the planner, with breakdowns (planner alone: 98.0%)</span></div>
<div class="fl-stat"><b>40 / 40</b><span>held-out maps on which the in-browser flatland of the Lab matches flatland-rl step for step</span></div>
</div>

</div>

In the NeurIPS 2020 round the best OR entry got 98.6% of trains home and the best RL entry 78.5%
([Laurent et al. 2021, Table 1](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)).
Six years later OR still wins: the ECML 2026 challenge ended with 20.84 for the best planning entry
against 9.42 for the RL baseline
([results](https://flatland-association.github.io/flatland-book/challenges/ecml2026/post-competition_analysis.html)).

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
  <a class="gap-tile" href="04-designs/#results-on-our-scenarios" title="This repository, xlarge scenario">
    <span class="gap-round">This repo, 100×100</span>
    <span class="gap-metric">trains arrived, 100 trains</span>
    <span class="gap-row"><span class="gap-who">OR</span><span class="gap-val or">95.7%</span></span>
    <span class="gap-bar"><span class="or" style="width:95.7%"></span></span>
    <span class="gap-row"><span class="gap-who">RL</span><span class="gap-val rl">16.0%</span></span>
    <span class="gap-bar"><span class="rl" style="width:16%"></span></span>
  </a>
</div>

!!! note "Scope"
    The code targets **mid-size grids: 30×30 to 100×100 cells, 10 to 100 trains**, so every
    baseline trains and evaluates on a laptop CPU in under an hour. The competitions went up to
    314×314 cells and 6,256 trains. [Limitations](05-limitations.md#what-changes-at-competition-scale)
    covers what changes at that scale.

[Open the Scheduling Lab](how/7-lab.md?map=medium-1003-m&preset=or&t=120){ .fl-try } [Be the dispatcher](how/7-lab.md?map=medium-1005-m&preset=you&t=60){ .fl-try } [Watch a deadlock happen](how/7-lab.md?map=small-1004-n&preset=shortest_path&t=25&train=6){ .fl-try }

## What this repo found

1. **A planner gets almost every train home.** The OR reference, which reproduces the core of the
   2019 and 2020 winners, delivers 100, 99.3, 98.2 and 95.7% of trains on the small, medium, large and
   xlarge held-out seeds, never deadlocks, and loses 0, 1.3, 4.3 and 6.6 points to breakdowns
   ([Designs and results](04-designs.md#results-on-our-scenarios)).
2. **Learning from scratch does not come close on a CPU budget.** After 30 CPU minutes the best
   learned policy per scenario delivers 63, 35, 16.5 and 16% of trains; a one-line reactive rule
   beats it on every scenario but xlarge ([Designs and results](04-designs.md#results-on-our-scenarios),
   [Limitations](05-limitations.md)).
3. **Learning on top of the planner is safe but adds little, so far.** A windowed dispatcher in the
   style of TADA, which only edits the plan through checked clearances, never deadlocked in any run,
   ties the planner on medium, and gains a few trains with breakdowns on large and xlarge
   ([TADA on rails](08-tada-dispatcher.md)).
4. **What is left to gain is the order after a breakdown.** The ranked openings, with effort
   estimates, are in [Open questions](06-open-questions.md).

## How to read this site

<div class="grid cards" markdown>

-   :fl-onramp:{ .lg .middle } **For Leander**

    ---

    What transfers from TADA and accelerator RL, what is new, and a two-week plan.

    [:octicons-arrow-right-24: Start here](for-leander.md)

-   :fl-primer:{ .lg .middle } **How it works**

    ---

    Six chapters from the rail network to learning on top of a plan, each with a widget you can
    play with, and the **Scheduling Lab**: flatland in your browser, on this repo's held-out maps.

    [:octicons-arrow-right-24: The mechanisms](02-primer.md) · [:fl-lab: The Lab](how/7-lab.md)

-   :fl-problem:{ .lg .middle } **The problem**

    ---

    Flatland stated formally: configurations, timetable, reward, constraints, deadlock.

    [:octicons-arrow-right-24: The MDP](01-problem.md)

-   :fl-results:{ .lg .middle } **Designs and results**

    ---

    How the winning and the RL designs work, and every baseline of this repo on the same seeds.

    [:octicons-arrow-right-24: The results](04-designs.md)

-   :fl-tada:{ .lg .middle } **TADA on rails**

    ---

    A learned, windowed dispatcher on top of the planner: verification, results, ablations, a
    continuous variant, and what did not work.

    [:octicons-arrow-right-24: The experiment](08-tada-dispatcher.md)

-   :fl-warning:{ .lg .middle } **Limitations**

    ---

    What fails, by how much, and what changes at competition scale.

    [:octicons-arrow-right-24: The caveats](05-limitations.md)

-   :fl-idea:{ .lg .middle } **Open questions**

    ---

    Seven ranked research openings with effort estimates, and what the dispatcher settled.

    [:octicons-arrow-right-24: What to do next](06-open-questions.md)

-   :fl-books:{ .lg .middle } **Literature**

    ---

    The MAPF toolkit behind the winners, who got which number when, every source, and a glossary.

    [:octicons-arrow-right-24: The toolkit](literature/mapf.md) · [Timeline](03-timeline.md) · [References](07-references.md)

</div>

## The codebase in one paragraph

`rl_flatland` wraps flatland-rl 4.3.0 with four fixed scenarios and seed splits, a decision
interface that asks a train for an action only where it matters (departure, and before or at a
switch), a 28-feature compact observation and flatland's tree observation. Baselines, all evaluated
by the same harness on the same held-out seeds with and without malfunctions:

- an **OR reference** reproducing the core of the 2019 and 2020 winners (prioritized planning with
  safe-interval search, executed in planned cell order so it cannot deadlock);
- **PPO** with parameter sharing over trains, written as a semi-MDP;
- **imitation then RL**: behaviour cloning on the OR planner's decisions, then PPO fine-tuning;
- two rule-based references (uncoordinated shortest paths, and a reactive heuristic).

Quick start, results table and reproduction commands are in the
[README](https://github.com/leandergrech/rl-flatland-rescheduling#readme).
