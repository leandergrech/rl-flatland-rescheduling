---
icon: fl/primer
---

# :fl-primer: How it works: start here

!!! abstract "In short"

    - Six short chapters take you from **the rail network** to **learning on top of a plan**, one
      mechanism each, and end in the **Scheduling Lab**: flatland itself, running in your browser
      on this repository's held-out maps, with every policy's recorded episodes, the planner live,
      and you as the dispatcher.
    - Every chapter follows the same pattern: a question, a widget you can play with, the rule,
      **what it means for the agent**, buttons that open the Lab at the right moment, and a short
      self-check.
    - The widgets and the Lab run a JavaScript port of flatland-rl 4.3.0 and of this repository's
      planner that reproduces flatland step for step on all 40 small and medium held-out maps
      ([how it is checked](how/7-lab.md#how-faithful-it-is)).
    - Where the mechanisms come from is in [The MAPF toolkit](literature/mapf.md); every term and
      formula is on the [glossary and formula sheet](literature/glossary.md).

You know PPO, value functions and multi-agent credit assignment. This part covers what a railway
operations person would take for granted, then the planning machinery the winning Flatland entries
are built from, at the depth an RL researcher needs to read the results critically. Formal
definitions are in [The problem](01-problem.md).

## What a dispatcher does

A railway runs to a **timetable**: for every train, a route and a planned time at every station
and junction. Real operations drift from it all the time. A door fault holds a train for two
minutes, a signal failure closes a line, a freight train leaves late. The dispatcher's job is
**rescheduling** (also called real-time traffic management or train dispatching). Given the
current positions and delays, they decide a new plan using four levers:

| Lever | Real railway | Flatland action |
|---|---|---|
| Re-timing | hold a train at a station or signal | `STOP_MOVING`, or not departing yet |
| Re-ordering | change which train goes first through a junction or single-track section | who waits and who moves at a switch |
| Re-routing | send a train over a different track or platform | `MOVE_LEFT` / `MOVE_RIGHT` at a facing switch |
| Cancellation | drop a service | never depart (penalised as a cancellation) |

Delay spreads. A late train holds the track, so the trains behind it and the trains waiting to
cross its path are delayed in turn. These **knock-on delays** are why a two-minute fault can wreck
an evening peak, and why the order in which trains pass a shared resource matters more than any
individual train's speed.

## Why rail coordination is harder than roads or air

A car can overtake. An aircraft can change altitude or heading and pass above another. A train
can do neither: it runs on a fixed track, cannot pass another train except where the track
splits, and (in Flatland) can only reverse at a dead end. Three consequences follow.

1. **The resource is the track.** Real railways cut the line into **block sections** that
   interlocking and signalling keep to one train at a time. A Flatland cell is exactly such a
   block, and `MotionCheck` plays the interlocking.
2. **Right-of-way decisions are made at switches, long before the conflict.** Once two trains are
   on a single-track segment facing each other, no action can separate them. The decision that
   prevents it has to be taken at the switch before the segment. In RL terms, a fatal action's
   consequence appears many steps later and is irreversible.
3. **Deadlock is the dominant failure.** Two trains nose to nose on one track is the simplest
   case. A ring of trains, each waiting for the next one's block, is the general case. In real
   railways, interlocking logic and dispatchers avoid it by reserving whole routes before a train
   enters. The Flatland winners avoid it by planning ([chapters 4 and 5](how/4-planning.md)).


## The chapters

<div class="grid cards" markdown>

-   :fl-network:{ .lg .middle } **[The rail network](how/1-network.md)**

    ---

    Cells, headings and the 16 bits that make track; facing switches, the only places to choose.
    *Widgets: the cell types; drive a train.*

-   :fl-train:{ .lg .middle } **[Trains, timetables and breakdowns](how/2-trains.md)**

    ---

    Speed classes, earliest departure and latest arrival, the seven-state train, and breakdowns that
    hit every policy alike. *Widget: four speed classes and a breakdown.*

-   :fl-deadlock:{ .lg .middle } **[Conflicts and deadlock](how/3-deadlock.md)**

    ---

    What MotionCheck forbids, why two trains nose to nose stay there forever, and how deadlocks are
    counted. *Widget: two trains, one passing loop.*

-   :fl-planner:{ .lg .middle } **[Planning with reservations](how/4-planning.md)**

    ---

    Timed paths, safe intervals, prioritized planning, and why the order is everything.
    *Widget: plan four trains on a time–distance diagram.*

-   :fl-execution:{ .lg .middle } **[Executing a plan under disruption](how/5-execution.md)**

    ---

    Keep the order, not the times; what breakdowns cost; re-timing and repair by clearances.
    *Widget: one plan, one breakdown, two ways to execute.*

-   :fl-learning:{ .lg .middle } **[Learning on top of the plan](how/6-learning.md)**

    ---

    The learning problem, the reward and its normalisation, why RL finds this hard, and two ways to
    put learning in. *Widget: from per-train outcomes to the score.*

-   :fl-lab:{ .lg .middle } **[The Scheduling Lab](how/7-lab.md)**

    ---

    All of the above on 40 real maps: replay every policy, run the planner, break trains, and
    dispatch yourself.

</div>
