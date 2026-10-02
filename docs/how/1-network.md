---
icon: fl/network
---

# :fl-network: The rail network

!!! abstract "The question"

    What is a Flatland map, as data, and where on it can a train actually decide anything?

## Cells, headings and transitions

A Flatland map is a grid of cells. A cell with track is a **block section**: at most one train may
be in it at any time. What makes a cell a piece of railway is a 16-bit number. For each of the four
**headings** a train can have inside the cell (the direction it was travelling when it entered), four
bits say which exits are open: north, east, south or west
([Mohanty et al. 2020, §2.2](https://arxiv.org/abs/2012.05893)). Plain track opens one exit per
heading, a curve turns it, a switch opens two. A train's position is therefore a
**configuration** (row, column, heading), and the map is a directed graph over configurations
([The problem](../01-problem.md#infrastructure)).

<div class="fl-widget" data-widget="cells" data-title="Interactive: one cell, its 16 bits, and where a train may go"></div>

Heading matters. A train that enters a simple switch from the branch side has exactly one way on;
a train that enters it from the stem has two. The same cell is a choice for one train and plain
track for another.

## Facing switches: the only places to choose

A train chooses its route only at a **facing switch**, a cell where its heading opens two exits.
Everywhere else its next cell is fixed, and the only decision is *when* to move. Trains cannot
reverse except at a **dead end**, which sends them back the way they came. This repository's RL
wrapper asks a train for an action only at its departure, at a facing switch, or on the last cell
before any switch cell, the last place it can stop before entering a shared resource
(`RailGraph.is_decision_config`); everywhere else the train keeps rolling.

<div class="fl-widget" data-widget="drive" data-title="Interactive: drive one train (a passing loop on single track)"></div>

Flatland has five actions: `DO_NOTHING`, `MOVE_LEFT`, `MOVE_FORWARD`, `MOVE_RIGHT`, `STOP_MOVING`.
The environment interprets them: a left or right turn that the track does not offer becomes
`MOVE_FORWARD`, and a train that has stopped needs a move action to start again
(`RailGridTransitionMap._check_action_new`, flatland-rl 4.3.0). The widget prints how flatland read
each action.

## Real maps

Flatland's sparse rail generator places cities (clusters of parallel station tracks) and joins
them with one or a few rail lines. Most cells are empty, and every segment between two switches
carries trains in both directions, so it behaves as single track. Every map in the
[Scheduling Lab](7-lab.md) is a held-out map of this repository's small (30×30, 10 trains) or
medium (50×50, 30 trains) scenario.

| Real world | Flatland |
|---|---|
| Block section | grid cell (one train at a time) |
| Plain track, curve | cell with one transition per heading |
| Turnout (switch), crossing, slip | cell with several transitions: simple switch, diamond crossing, single slip, double slip, symmetric switch ([Mohanty et al. 2020, §2.2](https://arxiv.org/abs/2012.05893)) |
| Station platform | cell inside a "city", used as start or target |
| Passing loop, double track | parallel rails between cities (`max_rails_between_cities`) |
| Timetable | earliest departure \(ED_i\), latest arrival \(LA_i\) per train |
| Train categories (passenger vs freight) | speed \(1/k\), \(k \in \{1,2,3,4\}\) steps per cell |
| Disturbances (door fault, loco failure) | malfunctions: Poisson onsets, 20 to 50 step durations in the benchmark settings |
| Interlocking | `MotionCheck`: no shared cells, no swaps, queues may close up |
| Dispatcher | your policy |

Left out: train length (a Flatland train occupies one cell; [Atzmon et al. 2019](https://ojs.aaai.org/index.php/SOCS/article/view/18515)
model trains occupying several cells), acceleration and braking curves (flatland-rl 4.x has
optional acceleration deltas, off in the benchmarks), signalling headways, crews, rolling stock
rotations, passenger connections, and infrastructure closures (the ECML 2026 level 6 added
"infrastructure disruptions" ([level config](https://flatland-association.github.io/flatland-book/challenges/ecml2026/levelconfig.html))).

![A real small-scenario map with switch cells, decision cells, targets and trains](../assets/figures/map-decision-cells-light.svg#only-light)
![A real small-scenario map with switch cells, decision cells, targets and trains](../assets/figures/map-decision-cells-dark.svg#only-dark)

*The small scenario, seed 1000, drawn from flatland's transition map. Grey lines are track, boxes are switch or crossing cells, amber dots are the cells where this repo's RL wrapper asks a train for a decision (departure, facing switch, or the last cell before a switch), squares are target stations and black dots are trains part-way through an OR-reference run. Each track segment between two switches carries trains in both directions, so it behaves as single track.*

[Open the Lab on a small map](7-lab.md?map=small-1000-n&preset=or&t=60){ .fl-try } [A medium map with 30 trains](7-lab.md?map=medium-1003-n&preset=or&t=150){ .fl-try }

!!! tip "What it means for the agent"

    - **Most steps are not decisions.** A train has a real choice only at a few cells, and the
      choice that matters is usually *whether to go now*, not which way. That is why every serious
      RL entry asked for actions only at decision cells and why this repository's wrapper does
      ([Laurent et al. 2021, §3](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)).
    - **Configurations, not cells.** Distance to target, legal moves and conflicts all depend on
      the heading. Any observation that drops the heading loses which way a train can go.
    - **No reverse gear.** Except at dead ends, a train that has passed a switch cannot undo the
      choice. Mistakes are permanent, which is the root of the next two chapters.

??? question "Check yourself"

    1. Why does a configuration need the heading, if a train occupies one cell? *Because the open
       exits depend on the heading: the same cell can be a switch for a train entering from the
       stem and plain track for one entering from a branch.*
    2. In the drive widget, what does `MOVE_LEFT` do on plain track? *Flatland reads it as
       `MOVE_FORWARD`; only at a facing switch with a left branch is it a turn.*
    3. Why does the wrapper also ask for a decision on the cell *before* a switch, not just on the
       switch? *Because that is the last cell where the train can stop outside the shared resource;
       once inside the switch it already blocks it.*
