# 1. The control problem

Flatland is a discrete-time simulator of trains on a rail network drawn on a grid. You control
every train. Each train must get from its start station to its target station inside a time
window, without ever sharing a cell with another train. Trains cannot pass each other on plain
track and cannot reverse except at dead ends. The dispatcher decides when each train departs,
which branch it takes at every switch, and when it waits. That dispatcher is the policy.

This page states the problem as precisely as the flatland-rl 4.3.0 source defines it, first as a
multi-agent sequential decision problem, then as the multi-agent path-finding (MAPF) problem the
winning OR entries actually solve.

!!! note "Scope of this repository"
    Everything here runs on **mid-size grids: 30×30 to 100×100 cells with 10 to 100 trains**
    (the four scenarios in `src/rl_flatland/scenarios.py`), so that each baseline trains and
    evaluates on a laptop CPU in under an hour. The competitions went far beyond that. The
    NeurIPS 2020 evaluation used square grids from 25×25 up to 314×314 cells and 1 to 6,256 trains
    ([Laurent et al. 2021, App. B, p.296](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)).
    Flatland 3 ran up to 425 trains on 158×158 cells
    ([Jiang et al. 2023, Table 6](https://arxiv.org/pdf/2210.12933)), and the ECML 2026 challenge
    up to 532 trains on a 120×150 map
    ([level config](https://flatland-association.github.io/flatland-book/challenges/ecml2026/levelconfig.html)).
    [05-limitations](05-limitations.md#what-changes-at-competition-scale) covers what changes at that scale.

## Infrastructure

The rail network is a grid of \(H \times W\) cells. Each cell stores 16 bits: for each of the 4
headings a train can enter with, 4 bits say which exit directions are allowed
([Mohanty et al. 2020, §2.2](https://arxiv.org/abs/2012.05893)). From that we get a directed
graph whose nodes are **configurations** \(c = (r, q, d)\): cell row \(r\), column \(q\), and
heading \(d \in \{N, E, S, W\}\), the direction the train was travelling when it entered the cell.
An edge \(c \to c'\) exists when a train in \(c\) may move into the neighbouring cell of \(c'\) with
heading \(d'\). Plain track has one successor per configuration, a facing switch has two, and a dead
end sends the train back the way it came. Heading matters because a train that entered a switch
from the "wrong" side has no choice at all.

## Trains and timetable

There are \(N\) trains. Train \(i\) has:

- a start configuration \(s_i\) (a station cell and heading) and a target cell \(g_i\);
- a maximum speed \(v_i = 1/k_i\) cells per step, with \(k_i \in \{1,2,3,4\}\) in the Flatland 3
  speed profile, so it spends exactly \(k_i\) steps in every cell (checked empirically in this repo
  against flatland-rl 4.3.0);
- an earliest departure \(ED_i\) and a latest arrival \(LA_i\).

The timetable generator in flatland-rl 4.3.0
([`timetable_generators.py`](https://github.com/flatland-association/flatland-rl/blob/v4.3.0/flatland/envs/timetable_generators.py))
sets the horizon from the slowest train's shortest-path time \(\tau_i = k_i \, \ell_i\), where
\(\ell_i\) is the shortest-path length in cells:

\[
T = \min\Big( \big\lceil 1.5 \max_i \tau_i \big\rceil + 0.2\,\bar\tau,\; 24\,(W + H + N/C) \Big),
\qquad \bar\tau = \tfrac1N \textstyle\sum_i \tau_i ,
\]

with \(C\) the number of cities. Each train gets a travel allowance
\(\lceil 1.3\,\tau_i + 0.2\,\bar\tau \rceil\), \(ED_i\) is drawn uniformly from the window that still
lets it arrive by \(0.95\,T\), and \(LA_i = ED_i +\) allowance. On the held-out seeds of our four
scenarios \(T\) ranges from 129 to 2,021 steps (see [02-primer](02-primer.md#what-the-numbers-look-like)).

## State

At step \(t\) the full state is:

- \(t\) itself (the timetable makes the problem non-stationary);
- per train: its state-machine state
  \(\sigma_i \in \{\text{WAITING}, \text{READY\_TO\_DEPART}, \text{MALFUNCTION\_OFF\_MAP}, \text{MOVING}, \text{STOPPED}, \text{MALFUNCTION}, \text{DONE}\}\)
  ([Flatland book, agent](https://flatland-association.github.io/flatland-book/environment/environment/agent.html)),
  its configuration \(c_i\) (undefined while off the map), its progress through the current cell
  (a fraction in multiples of \(1/k_i\)), and its remaining malfunction steps \(m_i\);
- the static data: the graph, all \((s_i, g_i, k_i, ED_i, LA_i)\).

The simulator exposes all of this, so a centralised planner sees the full state. What it cannot
see is the future: when the next malfunction will hit and how long it will last. For RL, each train
is usually given only a local view (a tree of the track ahead), which turns the problem into a
decentralised, partially observed Markov game.

## Actions

Each train picks one of 5 actions per step
([book, actions](https://flatland-association.github.io/flatland-book/environment/environment/actions.html)):
0 `DO_NOTHING`, 1 `MOVE_LEFT`, 2 `MOVE_FORWARD`, 3 `MOVE_RIGHT`, 4 `STOP_MOVING`. An action only
matters at a **decision instant**: when the train is ready to depart, or when it is at the exit
point of its current cell. For a train at speed \(1/k\), that is one step in \(k\). Between
decision instants the train keeps rolling. So each train lives in a semi-MDP with state-dependent
decision times, which matters for RL credit assignment and discounting (see [04-designs](04-designs.md)).

## Dynamics

Given the joint action, one step does the following (`RailEnv.step` in
[`rail_env.py`](https://github.com/flatland-association/flatland-rl/blob/v4.3.0/flatland/envs/rail_env.py)):

1. **Malfunctions.** Each train independently breaks down with probability
   \(p = 1 - e^{-\lambda}\), where \(\lambda\) is the malfunction rate, for a duration drawn
   uniformly from \(\{d_{\min}, \dots, d_{\max}\} + 1\) steps
   ([`malfunction_generators.py`](https://github.com/flatland-association/flatland-rl/blob/v4.3.0/flatland/envs/malfunction_generators.py)).
   A broken train cannot move, and one that breaks while off the map cannot depart. This is the
   only randomness after reset.
2. **Desired moves.** Each train at a decision instant computes where its action would take it.
3. **Conflict resolution** (`MotionCheck` in
   [`agent_chains.py`](https://github.com/flatland-association/flatland-rl/blob/v4.3.0/flatland/envs/agent_chains.py)).
   Each cell is a resource for one train at a time. Two trains that try to swap cells are both
   stopped (a head-on collision is impossible, but so is passing). If two trains want the same
   cell, the lower index wins, and a train that wants a cell whose holder is not leaving is
   stopped. A train *may* enter a cell in the same step its holder leaves it, so queues of trains
   move together, and a ring of three or more trains can rotate.
4. **State machine.** States update, and a train that enters its target cell becomes DONE and is
   removed from the grid in the same step.

Everything except step 1 is deterministic. That is why this repo has no model-based RL baseline:
the model is the simulator, and it is known exactly.

## Objective

Flatland 3 scores each episode with the environment's `DefaultRewards`
([`rewards.py`](https://github.com/flatland-association/flatland-rl/blob/v4.3.0/flatland/envs/rewards.py);
formula restated in [Jiang et al. 2023, §2](https://arxiv.org/pdf/2210.12933)). Train \(i\) gets
a single end-of-journey term:

\[
R_i =
\begin{cases}
\min(LA_i - A_i,\; 0) & \text{arrived at step } A_i \\
-\phi\,(\tau_i + \pi) & \text{never departed (cancellation)} \\
\min\big(-\nu,\; LA_i - T - k_i\,\ell_i(T)\big) & \text{on the map but not arrived at } T
\end{cases}
\]

where \(\ell_i(T)\) is its remaining shortest-path distance at the horizon, and flatland-rl 4.3.0
defaults are \(\phi = 1\), \(\pi = 0\), \(\nu = 0\) (the third case is the train's "current delay").
The episode score is the normalised reward

\[
\bar R = 1 + \frac{1}{N\,T}\sum_{i=1}^{N} \max(R_i, -T) \;\in [0, 1],
\]

which is 1.0 when every train arrives on time
([Flatland 3 evaluation](https://flatland-association.github.io/flatland-book/challenges/flatland3/eval.html)).
A competition score is the sum of \(\bar R\) over all episodes solved before the evaluation stops.
Alongside it, everyone reports the **arrival rate**, the fraction of trains that reach their target
by \(T\).

The NeurIPS 2020 round used a simpler reward: \(-1\) per train per step until arrival, plus \(+1\)
to every train if all arrived, normalised the same way with
\(T = 8\,(W + H + \lfloor N/C \rfloor)\)
([Laurent et al. 2021, §2.3](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)).
The ECML 2026 challenge kept the structure but raised the stakes: collision factor 250, minimum
not-arrived penalty 100, missed intermediate stop 50, cancellation factor 5
([ECML 2026 evaluation](https://flatland-association.github.io/flatland-book/challenges/ecml2026/eval.html)).

## Constraints

- **Hard, enforced by the simulator:** one train per cell, no swaps, no departure before
  \(ED_i\), \(k_i\) steps per cell, no reversing except at dead ends.
- **Soft, priced by the reward:** arrival by \(LA_i\); in the ECML 2026 variant, serving
  intermediate stops inside their windows.
- **Computational, enforced by the evaluator:** Flatland 3 allowed 10 minutes of planning before
  the first step, 10 seconds per step and 2 hours per full evaluation, on 4 CPU cores with 15 GB
  RAM ([Flatland 3 evaluation](https://flatland-association.github.io/flatland-book/challenges/flatland3/eval.html)).
  ECML 2026 allowed 30 minutes per scenario and 5 hours in total
  ([ECML 2026 evaluation](https://flatland-association.github.io/flatland-book/challenges/ecml2026/eval.html)).
  Real dispatching has similar limits: a rescheduling decision that arrives after the train has
  passed the switch is useless.

## Deadlock: the failure that matters

Because trains can neither pass nor reverse, two trains that meet head-on on a single-track
segment can never move again, and every train queued behind them is stuck too. A deadlock is
absorbing and it spreads, because other trains pile up behind the blocked ones. Most of the gap
between RL and OR in the competitions comes from how each method avoids deadlocks
(see [05-limitations](05-limitations.md)). `rl_flatland.deadlock.find_deadlocked` counts the trains
in head-on cores plus those queued behind them, and every result table in this repo reports that
count.

## Two equivalent views

**As a Markov game.** \(N\) agents, shared (or per-train) reward, horizon \(T\), per-agent decisions
only at decision instants, action masks from the graph, a known deterministic transition kernel
apart from malfunctions. Global optimum: the joint policy that maximises \(\mathbb{E}[\bar R]\).
The hard parts for RL are that the reward is terminal and sparse (one term per train per
episode), horizons run to hundreds or thousands of steps, credit must be split across up to
thousands of trains, and one bad decision (entering a single-track segment) can be irreversibly
fatal many steps later.

**As MAPF.** Find one timed path per train through the time-expanded configuration graph such that
no two paths share a cell at the same step or swap cells. The planner then executes those paths
and repairs them when malfunctions shift the timing. Flatland adds four twists to classical MAPF
([Stern et al. 2019](https://arxiv.org/abs/1906.08291)): directed motion with headings, per-train
speeds, time windows, and trains that vanish at their target. Finding the fastest joint solution
is NP-hard ([Jiang et al. 2023, §1](https://arxiv.org/pdf/2210.12933)), so all practical entries
use prioritised or large-neighbourhood heuristics (see [04-designs](04-designs.md)).

## What "solved" would mean

At the scale of this repo, the OR reference already gets 100% of trains home on the small and
medium scenarios without malfunctions (see [04-designs](04-designs.md#results-on-our-scenarios)).
So "solved" cannot mean arrival rate alone. A useful definition has three parts:

1. **Parity:** a learned policy that matches the OR reference's arrival rate and normalised reward
   on held-out seeds, with and without malfunctions, at comparable compute per decision.
2. **Where RL should win:** measurably better recovery after malfunctions than a plan-and-repair
   method, lower wall-clock time per decision at scale, or generalisation to unseen networks
   without re-planning from scratch.
3. **At competition scale:** the same at 100×100 and beyond with hundreds of trains, which no
   published RL method has shown. The best post-hoc RL result on the Flatland 3 stages reached a
   score of 125.3 with 66.4% of trains arrived, against 141.0 and 88.0% for the winning OR entry
   ([Jiang et al. 2023, Table 7](https://arxiv.org/pdf/2210.12933)).
