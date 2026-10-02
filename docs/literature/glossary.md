---
icon: fl/glossary
---

# :fl-glossary: Glossary and formulas

Every term the site uses, then every formula, each with where it lives in flatland-rl 4.3.0 or in
this repository. Notation follows [The problem](../01-problem.md).

## Glossary

| Term | Meaning | Where to see it |
|---|---|---|
| **Block section** | A stretch of track that holds one train at a time; in flatland, one grid cell. | [The rail network](../how/1-network.md) |
| **Cell code** | 16 bits per cell: for each of the four headings, which of the four exits are open. | [The rail network](../how/1-network.md#cells-headings-and-transitions) |
| **Heading** | The direction a train was travelling when it entered its cell (N, E, S, W). | [The rail network](../how/1-network.md) |
| **Configuration** | (row, column, heading): a train's position for every purpose that matters. | [The problem](../01-problem.md#infrastructure) |
| **Facing switch** | A cell where the train's heading opens two exits: the only place a route is chosen. | [The rail network](../how/1-network.md#facing-switches-the-only-places-to-choose) |
| **Trailing switch** | The same cell entered from a branch: one way on, no choice. | [The rail network](../how/1-network.md) |
| **Dead end** | A cell that sends a train back the way it came; the only place a train can reverse. | [The rail network](../how/1-network.md) |
| **Decision cell** | Where this repository's wrapper asks for an action: departure, a facing switch, or the last cell before any switch. | `RailGraph.is_decision_config` |
| **Speed class** | Maximum speed \(1/k\), \(k \in \{1,2,3,4\}\): \(k\) steps per cell. | [Trains](../how/2-trains.md#speed-classes) |
| **Cell exit** | The last step a train spends in a cell; the only step at which it can move on. | [Trains](../how/2-trains.md) |
| **Earliest departure** \(ED_i\) | No departure before this step. | [Trains](../how/2-trains.md#the-timetable) |
| **Latest arrival** \(LA_i\) | Arriving later is penalised step by step. | [Trains](../how/2-trains.md#the-timetable) |
| **Slack** | \(LA_i - ED_i - \tau_i\): steps a train can lose and still be on time. | [Trains](../how/2-trains.md#the-timetable) |
| **Horizon** \(T\) | The episode's last step; trains not arrived by then are charged. | [The problem](../01-problem.md#trains-and-timetable) |
| **Malfunction (breakdown)** | A train frozen for a number of steps; drawn per train per step, independently of the policy. | [Trains](../how/2-trains.md#breakdowns) |
| **MotionCheck** | Flatland's conflict resolution in each step: one train per cell, no swaps, following allowed, rings rotate. | [Conflicts and deadlock](../how/3-deadlock.md#one-cell-one-train-motioncheck) |
| **Swap** | Two neighbouring trains entering each other's cells in one step: forbidden. | [Conflicts and deadlock](../how/3-deadlock.md) |
| **Deadlock** | Trains that can never move again, typically two head-on on single track and everything queued behind them. | [Conflicts and deadlock](../how/3-deadlock.md#deadlock) |
| **Timed path** | A train's planned (configuration, entry time) list. | [Planning](../how/4-planning.md#a-plan-is-a-list-of-timed-cells) |
| **Reservation table** | Per cell, the time intervals promised to trains, plus reserved moves. | [Planning](../how/4-planning.md) |
| **Safe interval** | A maximal time window in which a cell is free. | [Planning](../how/4-planning.md#safe-intervals-sipp) |
| **SIPP** | Safe-interval path planning: A* over (configuration, safe interval). | [The MAPF toolkit](mapf.md) |
| **Prioritized planning (PP)** | Plan trains one by one in a priority order, each around the reservations of the earlier ones. | [Planning](../how/4-planning.md#prioritized-planning) |
| **Ordered execution (MCP)** | Enter a cell only after every train planned through it earlier has left; deadlock-free under delays. | [Execution](../how/5-execution.md#keep-the-order-not-the-times) |
| **Switchable order** | Executing a plan while allowed to change which of two agents passes a shared cell first, keeping the dependency graph acyclic (switchable action dependency graphs, temporal plan graphs). | [Learning on top of a planner](hybrids.md#strand-2-re-ordering-a-plan-after-delays-multi-robot-path-finding) |
| **Shield (preemptive)** | A check that removes unsafe actions before a learned policy chooses; here the planner computes it for each clearance. | [Learning on top of a planner](hybrids.md#strand-5-air-traffic-control-and-shielded-rl) |
| **Knock-on delay** | Delay passed from a late train to trains that wait for it. | [Execution](../how/5-execution.md#what-it-costs-knock-on-delay) |
| **Re-timing** | Recomputing every planned visit's earliest feasible time from the current state and the planned order. | [Execution](../how/5-execution.md#re-timing-and-repair) |
| **Clearance** | A dispatcher's edit to the plan: HOLD, YIELD_TO a partner, or REROUTE. | [TADA on rails](../08-tada-dispatcher.md) |
| **Context window** | The trains a dispatcher may act on at a step: rules (a) to (d), least slack first, at most M. | [TADA on rails](../08-tada-dispatcher.md#context-window) |
| **Arrival rate** | Fraction of trains that reach their target by \(T\). | [The problem](../01-problem.md#objective) |
| **Normalised reward** | Flatland's episode score in [0, 1]; 1 means every train on time. | [Learning](../how/6-learning.md#the-reward) |
| **Cancellation** | A train that never departed; charged its shortest-path travel time. | [Learning](../how/6-learning.md#the-reward) |

## Formulas

**Shortest-path time and slack.** With \(\ell_i\) the shortest-path length in cells,

\[
\tau_i = k_i\,\ell_i, \qquad \text{slack}_i = LA_i - ED_i - \tau_i .
\]

**Timetable** (flatland-rl 4.3.0 `timetable_generators.py`). Travel allowance and horizon:

\[
LA_i = ED_i + \big\lceil 1.3\,\tau_i + 0.2\,\bar\tau \big\rceil, \qquad
T = \min\Big( \big\lceil 1.5 \max_i \tau_i \big\rceil + 0.2\,\bar\tau,\; 24\,(W + H + N/C) \Big),
\qquad \bar\tau = \tfrac1N \textstyle\sum_i \tau_i .
\]

**Per-train reward** (`DefaultRewards`, \(\phi = 1\), \(\pi = 0\), \(\nu = 0\)). With \(A_i\) the
arrival step and \(d_i(\cdot)\) the distance to target in cells,

\[
R_i =
\begin{cases}
\min(LA_i - A_i,\; 0) & \text{arrived} \\
-\phi\,\big(k_i\,(d_i(s_i) + 1) + \pi\big) & \text{never departed} \\
\min\big(-\nu,\; LA_i - T - k_i\,(d_i(c_i(T)) + 1)\big) & \text{on the map at } T
\end{cases}
\]

The "+1" is flatland's: its shortest path counts both the start and the target cell
(`DistanceMap._reconstruct_shortest_path`), which the Lab's port reproduces.

**Normalised reward.**

\[
\bar R = 1 + \frac{1}{N\,T}\sum_{i=1}^{N} \max(R_i, -T) \;\in [0, 1].
\]

**SIPP heuristic.** From configuration \(c\), \(h(c) = k_i\, d_i(c)\): admissible because a train
needs \(k_i\) steps per cell.

**Re-timing** (`TadaExecutor.retime`). For every future visit \(v\), its earliest feasible entry
time is the least fixed point of

\[
E_v = \max\Big( \mathrm{lb}_v,\; \max_{(u \to v)} E_u + w_{uv} \Big),
\]

where \(\mathrm{lb}_v\) is the planned time (and, for a train's next visit, the earliest it can
physically move, breakdowns included), train-path edges have \(w = k_i\), and cell-order edges
(the next visitor of a cell after the previous one leaves) have \(w = 0\), or \(1\) after a train
that ends its journey in that cell. Rotations of three or more trains make zero-weight cycles, so it
is solved by relaxation.

**Window slack** (the dispatcher's ranking). \(\text{slack}_i(t) = LA_i - \big(t + k_i\, d_i(c_i(t))\big)\).
