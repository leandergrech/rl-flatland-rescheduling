---
icon: fl/planner
---

# :fl-planner: Planning with reservations

!!! abstract "The question"

    How does a planner give thirty trains conflict-free timed paths in a fraction of a second, and
    why does the order in which it plans them matter so much?

## A plan is a list of timed cells

A **timed path** says, for one train, which cell it enters at which step: (configuration, entry
time) pairs from its start to its target. A set of timed paths is collision-free if no two trains
hold the same cell at the same step and no two swap cells in one step: exactly the two things
`MotionCheck` forbids (chapter 3). A **reservation table** records, for every cell, the time
intervals already promised to some train, plus the reserved moves between neighbouring cells, so a
new path can be checked against everything planned so far (`ReservationTable` in
`rl_flatland/baselines/or_planner.py`).

## Safe intervals: SIPP

Between its reservations, each cell has **safe intervals**: maximal stretches of time when it is
free. Safe-interval path planning searches over (configuration, safe interval) instead of
(configuration, step), and stores for each the earliest time the train can be there
([Phillips & Likhachev 2011](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/ICRA.2011.5980306)).
Waiting is implicit: a train can stay in a cell for as long as that cell's safe interval lasts, so
the search never has to try "wait one more step" explicitly. The heuristic is \(k_i\) times the
remaining shortest-path distance, which never overestimates, because a train needs \(k_i\) steps
per cell. The 2020 winner's switch from space-time A* to SIPP cut its runtime by up to 4 times
([Li et al. 2021](https://ojs.aaai.org/index.php/ICAPS/article/download/15994/15805/19487)).

## Prioritized planning

Plan the trains one at a time. Each gets its earliest-arrival path around the cell-times already
reserved by the trains planned before it, and then its own path is reserved
([Erdmann & Lozano-Pérez 1987](https://dspace.mit.edu/handle/1721.1/5602);
[Silver 2005](https://ojs.aaai.org/index.php/AIIDE/article/view/18726)). That is one single-train
search per train, which is why it scales. The price is that it is incomplete: an early train can
take a path that leaves no room for a later one, and the result depends almost entirely on the order
([Ma et al. 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4758)).

<div class="fl-widget" data-widget="sipp" data-title="Interactive: prioritized planning on a corridor with two passing loops"></div>

Main's OR reference tries eight orders (fast trains first, least slack first, earliest departure
first, shortest trip first, then four random permutations) and keeps the plan that routes the most
trains, then has the least lateness, then the earliest arrivals. The Lab shows the table for every
map. On medium seed 1000 without breakdowns, the four rule-based orders route 22, 20, 29 and 20 of
the 30 trains; the third random order routes all 30, and all 30 arrive. On medium seed 1003 the
orders route between 27 and 30, and the plan kept arrives with normalised reward 0.995 instead of
0.929 for the worst order. (The Lab's planner is checked to produce exactly main's paths on all 80
held-out maps; these numbers come from its planner panel.)

[The eight orders on medium seed 1000](7-lab.md?map=medium-1000-n&preset=or&t=120){ .fl-try } [Plan with "least slack first" only](7-lab.md?map=medium-1000-n&preset=or&order=slack&t=120){ .fl-try } [Seed 1005: even the best order gets 28 of 30 home](7-lab.md?map=medium-1005-n&preset=or&t=200){ .fl-try }

Across the 10 held-out seeds, this planner delivers 99.3% of trains on medium and 95.7% on xlarge
without breakdowns ([Designs and results](../04-designs.md#results-on-our-scenarios)). Its plan
takes half a second on medium and seven on xlarge in Python
([or.json](https://github.com/leandergrech/rl-flatland-rescheduling/blob/main/data/results/or.json),
`setup_s`).

!!! tip "What it means for the agent"

    - **The order is the policy.** Given an order, prioritized planning is deterministic and fast.
      Learning a good order (a per-train score) is a much smaller problem than learning to drive
      every train, and it is open ([Open questions, 3](../06-open-questions.md#3-learned-priorities-for-prioritized-planning)).
    - **Plans are global.** A planned path is safe because it was checked against every other
      train's reservations, not because the train looked around. A learned policy that only sees a
      local window has to recover that information or work on top of a planner.
    - **Incompleteness shows up as unroutable trains.** A train that finds no path never departs
      and is charged as a cancellation; it is not a deadlock, but it costs as much.

??? question "Check yourself"

    1. Why can SIPP skip "wait one step" actions? *Because a state is a whole safe interval: the
       train may stay in the cell anywhere inside it, so the search only asks when it can leave.*
    2. In the widget, put D first. What happens to A? *A has to wait in a loop for D (and B) to
       pass; its arrival moves later, and it may become late.*
    3. Switch the widget to "every train for itself". Why do the paths cross? *Each train was
       planned with an empty table, so nothing stopped two of them from using the same single
       track at the same time.*
