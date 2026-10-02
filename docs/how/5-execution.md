---
icon: fl/execution
---

# :fl-execution: Executing a plan under disruption

!!! abstract "The question"

    A train breaks down and the plan's times are now wrong for everyone behind it. What keeps the
    other trains safe, and what does that cost?

## Keep the order, not the times

A plan made at step 0 cannot know when trains will break down. Executing it by the clock ("go when
your planned time comes") fails at the first breakdown: a late train still holds a segment that an
on-time train believes is free, and on single track they meet head-on. The fix used by the 2019
and 2020 winners is to throw away the times and keep the **order**: a train may enter a cell only
after every train that was planned to use that cell before it has left
([Li et al. 2021](https://ojs.aaai.org/index.php/ICAPS/article/download/15994/15805/19487), the
"minimum communication policy"; [Andreica 2021](https://arxiv.org/abs/2111.07876)).

Why that cannot deadlock: draw one node per (train, planned cell visit), an edge along each train's
path, and an edge from "train j leaves cell x" to "train i enters x" whenever j was planned into x
before i. In a collision-free plan every edge points forward in planned time, so the graph has no
cycle, and some train can always move. A breakdown makes trains wait; it never traps them.

<div class="fl-widget" data-widget="mcp" data-title="Interactive: the same plan, a breakdown, two ways to execute it"></div>

[Ordered execution through the breakdowns of medium seed 1003](7-lab.md?map=medium-1003-m&preset=or&t=40&train=6){ .fl-try } [Follow one late train on the time–space diagram](7-lab.md?map=medium-1003-m&preset=or&t=250&train=7){ .fl-try }

## What it costs: knock-on delay

The order chosen before the breakdown may now be a bad order: an on-time train waits for a late
one that was planned to go first. That is the cost the OR reference pays. On the held-out seeds it
loses 1.3, 4.3 and 6.6 points of arrival rate to malfunctions on medium, large and xlarge, against
the same seeds without them ([Designs and results](../04-designs.md#results-on-our-scenarios)), and
on large seed 1000 the knock-on waiting behind late or broken trains is almost six times the waiting
the plan itself contains (4,349 against 742 train-steps,
[notebook 03](https://github.com/leandergrech/rl-flatland-rescheduling/blob/main/notebooks/03-first-experiment.ipynb)).

## Re-timing and repair

Two steps make the order adjustable without losing the guarantee. Both are implemented in this
repository's dispatcher ([TADA on rails](../08-tada-dispatcher.md)) and run in the Lab:

- **Re-timing.** Every step, recompute each future visit's earliest time from where the trains
  are now (breakdowns included) and the planned order of every cell. That is a longest-path
  computation over the order graph above; the result is the plan the trains are actually
  following, and a **live reservation table** built from it shows which cell-times are really free.
- **Repair by clearances.** Change the order for a few trains by re-planning them against the live
  table: hold a train a few steps (HOLD), let a partner go first through the cells they share
  (YIELD_TO), or forbid the planned branch at the next facing switch (REROUTE). Each is a SIPP
  search that is committed only if it succeeds, so a repair can make things slower but never unsafe.

The winners went further with large neighbourhood search: delete and re-plan the paths of the
trains a breakdown affects, keep the result if it is better (worth 19.9% less flowtime on 261
instances for the 2020 winner, [Li et al. 2021](https://ojs.aaai.org/index.php/ICAPS/article/download/15994/15805/19487)).
This repository's planner does not; deciding *when* such a repair pays is
[Open question 1](../06-open-questions.md#1-learned-repair-when-to-break-the-planned-order-after-a-malfunction).

[Be the dispatcher on a hard map (medium, seed 1005)](7-lab.md?map=medium-1005-m&preset=you&t=60){ .fl-try } [Watch the learned dispatcher's 58 clearances there](7-lab.md?map=medium-1005-m&preset=tada&t=60){ .fl-try }

!!! tip "What it means for the agent"

    - **Safety can be separated from performance.** Ordered execution makes any plan
      deadlock-free under delays. A learned layer that only proposes changes to the order, and
      lets the planner check them, cannot cause a deadlock, which turns a safe-exploration
      problem into an ordinary optimisation problem (page 8 runs exactly this).
    - **The opportunity is narrow.** What is left to gain is the knock-on delay from orders that
      became bad, a few points of arrival rate at most on these scenarios.
    - **Timing matters as much as content.** A clearance only makes sense at a decision instant
      of the train it concerns, which is why the dispatcher's masks check both.

??? question "Check yourself"

    1. In the widget, break train A early and switch to "go at the planned time". Why do two trains
       end up head-on? *B's planned time to enter the shared segment arrives while A, late, is still
       in it from the other side; neither checks who is actually there.*
    2. Under ordered execution, can a breakdown make a train arrive *earlier*? *No: the order and
       the planned times are lower bounds; a breakdown can only push visits later.*
    3. Why is the reservation table rebuilt from re-timed times instead of the original plan?
       *Because a late train still holds cells its original intervals say are free; planning a
       repair against stale intervals could put another train head-on against it.*
