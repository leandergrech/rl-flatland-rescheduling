# 2. Domain primer: railway dispatching for an RL researcher

You know PPO, value functions and multi-agent credit assignment. This page covers what a railway
operations person would take for granted, then the path-finding toolkit the winning Flatland
entries are built from. Formal definitions are in [01-problem](01-problem.md).

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
   enters. The Flatland winners avoid it by planning (see below).

## Flatland's abstraction, and what it leaves out

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

## What the numbers look like

Generated networks are sparse. On our 10 held-out seeds per scenario (computed by
`python scripts/evaluate.py --stats`, stored in [`data/scenarios/scenario_stats.json`](https://github.com/leandergrech/rl-flatland-rescheduling/blob/main/data/scenarios/scenario_stats.json), flatland-rl
4.3.0; the last two columns are the median over seeds of each seed's median train):

| Scenario | Grid | Trains | Cities | Rail cells (mean) | Switch cells (mean) | Horizon \(T\) (min to max) | Shortest-path time \(\tau\) | Slack \(LA - ED - \tau\) |
|---|---|---|---|---|---|---|---|---|
| small | 30×30 | 10 | 2 | 103 | 25 | 129 to 226 | 60 steps | 36 steps |
| medium | 50×50 | 30 | 4 | 332 | 57 | 439 to 902 | 127 steps | 76 steps |
| large | 80×80 | 60 | 6 | 562 | 82 | 1,010 to 1,450 | 200 steps | 114 steps |
| xlarge | 100×100 | 100 | 8 | 886 | 108 | 1,116 to 2,021 | 264 steps | 147 steps |

Only about 11% of cells carry track at 30×30 and 9% at 100×100, so trains share a small number of
corridors between cities. The median train's slack is about 0.6 times its own shortest-path travel
time: enough to wait once or twice at a junction, not enough to wait behind a long queue.

The 2020 competition used malfunction rates between 0 and 1/250 per train-step with durations of
20 to 50 steps
([Laurent et al. 2021, §2.2 and App. B](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)).
Our malfunction splits use a rate of 1/1000 per train-step with the same durations. So a train on
the map for 500 steps expects about 0.5 breakdowns.

## Timetable slack and priorities

The single most useful derived quantity is a train's **slack**:

\[
\text{slack}_i = LA_i - ED_i - \tau_i, \qquad \tau_i = k_i\,\ell_i ,
\]

the number of steps it can lose and still arrive on time. The Flatland 3 winner ordered trains by
slack, tightest first, breaking ties by speed
([Chen et al. 2023, §3](https://arxiv.org/abs/2306.06455)). So did the ECML 2026 winner
([repo](https://github.com/darshanmakwana412/ecml2026)). The 2019 winner put fast trains first,
because a fast train that has already left the network cannot be delayed by later disruptions
([Andreica interview](https://www.aicrowd.com/blogs/flatland-mugurel)). In RL terms, slack is the
natural feature for a learned priority, and giving each train a notion of priority was one of the
things that separated the better RL entries (JBR_HSE fed each agent a random "handle" as a
priority feature, [Laurent et al. 2021, §4.2](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)).

## The MAPF toolkit

Multi-agent path finding (MAPF) asks for one collision-free path per agent on a shared graph
([Stern et al. 2019](https://arxiv.org/abs/1906.08291)). These are the pieces the winning Flatland
entries are made of, in the order you need them.

**Time-expanded graph and conflicts.** Copy the configuration graph once per time step. A path is
then a sequence of (configuration, time) pairs. Two paths have a *vertex conflict* if they occupy
the same cell at the same time, and an *edge (swap) conflict* if they exchange cells in one step.
Flatland's `MotionCheck` forbids exactly these two. Following into a cell that is being vacated in
the same step is allowed.

**Prioritised planning (PP).** Order the agents. Plan each one's shortest path in the time-expanded
graph, treating the cells and times already used by higher-priority agents as blocked, then
reserve its path in a reservation table
([Erdmann & Lozano-Pérez 1987](https://dspace.mit.edu/handle/1721.1/5602);
[Silver 2005](https://ojs.aaai.org/index.php/AIIDE/article/view/18726)). It is fast (one
single-agent search per agent) and incomplete: a bad order can make a later agent unplannable. The
quality depends almost entirely on the order, hence all the work on priorities
([Ma et al. 2019, PBS](https://ojs.aaai.org/index.php/AAAI/article/view/4758)).

**Safe-interval path planning (SIPP).** Instead of one search node per time step, give each cell
its list of *safe intervals*, the maximal time windows in which it is free. The search state
becomes (configuration, safe interval), and the node's value is the earliest time you can be there
([Phillips & Likhachev 2011](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/ICRA.2011.5980306)).
Waiting is implicit. On Flatland, replacing space-time A* with SIPP cut the 2020 winner's runtime
by up to 4 times ([Li et al. 2021](https://ojs.aaai.org/index.php/ICAPS/article/download/15994/15805/19487)).

**Conflict-based search (CBS).** The optimal method: search a tree of constraint sets, where each
node plans all agents independently and branches on the first conflict by forbidding it for one
agent or the other ([Sharon et al. 2015](https://digitalcommons.du.edu/computer_science_faculty/7/)).
It is exact and exponential in the number of conflicts, so it does not scale to hundreds of trains.
PP is CBS that only ever explores one branch.

**Large neighbourhood search (LNS).** Start from any solution (for example PP). Repeatedly pick a
small group of agents, delete their paths, re-plan them with PP against everyone else, and keep
the result if the total cost went down
([Li et al. 2021, MAPF-LNS](https://github.com/Jiaoyang-Li/MAPF-LNS);
[Li et al. 2022, MAPF-LNS2](https://ojs.aaai.org/index.php/AAAI/article/view/21266)). It is anytime,
and on Flatland it cut flowtime by an average of 12.4% on 312 of 400 instances
([Li et al. 2021](https://ojs.aaai.org/index.php/ICAPS/article/download/15994/15805/19487)).

**Execution under delays: the minimum communication policy (MCP).** A plan is a list of timed
paths, but a malfunction shifts one train's timing and the plan becomes infeasible. MCP throws
away the times and keeps the *order*. A train may enter a cell only after every train that was
planned to use that cell earlier has left it. This "avoids deadlocks by stopping some trains to
maintain the ordering with which each train visits each cell"
([Li et al. 2021](https://ojs.aaai.org/index.php/ICAPS/article/download/15994/15805/19487)). The
same rule is in the 2019 winner's write-up
([Andreica 2021](https://arxiv.org/abs/2111.07876)).

Why it works: draw a dependency graph with one node per (train, path step). Add an edge along each
train's path, and an edge from "train \(j\) leaves cell \(x\)" to "train \(i\) enters \(x\)" whenever
\(j\) was planned into \(x\) before \(i\). In a collision-free plan every edge points forward in
planned time, so the graph is acyclic and some train can always move. Delays make trains wait but
never trap them. The price is that the plan's order, chosen before the malfunction, may now be a
bad order. Hence **partial replanning**: re-run PP or LNS for the trains affected by a
malfunction. That was worth 19.9% less flowtime on 261 instances for the 2020 winner
([Li et al. 2021](https://ojs.aaai.org/index.php/ICAPS/article/download/15994/15805/19487)).

**Robust plans.** An alternative to repair is to plan with slack built in, so that any delay of up
to \(k\) steps is absorbed ([Atzmon et al. 2020](https://jair.org/index.php/jair/article/view/11734)).
Nobody in the Flatland competitions relied on this. Buffers cost throughput, and malfunctions of
20 to 50 steps are too long to buffer.

**Reactive heuristics.** Without a plan, a train can still avoid the worst by checking, before
entering a segment, that no opposing train is on it. The flatland-baselines deadlock-avoidance
policy moves a train only if enough free cells separate it from every opposing train on its
shortest path ([flatland-baselines](https://github.com/flatland-association/flatland-baselines)).
It scored 10.73 in ECML 2026, against 20.84 for the winning planner
([results](https://flatland-association.github.io/flatland-book/challenges/ecml2026/post-competition_analysis.html)).

## Why RL finds this hard

Put in your vocabulary:

- **Sparse, delayed reward.** Each train gets one reward term, at arrival or at the horizon, after
  hundreds of steps. Every RL entry shaped the reward. JBR_HSE used
  \(0.01\,\Delta d - 5\cdot\text{deadlocked} + 10\cdot\text{arrived}\), where \(\Delta d\) is
  progress along the shortest path
  ([Laurent et al. 2021, §4.2](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)).
- **Irreversible, delayed failure.** Entering a single-track segment is safe or fatal depending on
  where other trains will be 10 to 50 steps later. That looks less like a control problem and more
  like safe exploration with an absorbing failure state.
- **Joint coordination.** Two trains at opposite ends of a segment both do well by entering if the
  other waits, and both lose everything if neither waits. That is a coordination game at every
  single-track segment, in the Hanabi sense rather than the StarCraft micro sense.
- **Semi-MDP timing.** Decisions happen only at cell exits (one step in \(k\) for a train of speed
  \(1/k\)) and only a few are real choices: most cells are plain track. Nearly every entry masked
  "non-decision" cells ([Laurent et al. 2021, §3](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)).
- **Varying agent count and density.** Trains appear at their departure time and vanish at their
  target, and a policy trained at 10 trains meets 100 at test time.
- **The competition clock.** OR planners exploit the full state and the exact simulator. An RL
  policy that only sees a local tree is solving a harder, partially observed problem, under the
  same per-step time limit.

The flip side, and the reason this is still worth your time, is in [06-open-questions](06-open-questions.md).
