---
icon: fl/dispatcher
---

# :fl-dispatcher: The MAPF toolkit

!!! abstract "What this page is"

    The literature the winning Flatland entries are built from, in the order you need it: the
    time-expanded graph, prioritized planning, safe-interval search, conflict-based search, large
    neighbourhood search, and the execution rule that keeps a plan deadlock-free under delays. The
    chapters under *How it works* let you play with each mechanism
    ([planning](../how/4-planning.md), [execution](../how/5-execution.md)); this page says who
    introduced it and what it was worth on Flatland.

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

![Time-space diagrams: conflicts, a prioritized plan, and ordered execution under a malfunction](../assets/figures/time-space-light.svg#only-light)
![Time-space diagrams: conflicts, a prioritized plan, and ordered execution under a malfunction](../assets/figures/time-space-dark.svg#only-dark)

*Schematic time-space diagrams of a 12-cell single-track corridor: position on the vertical axis, time on the horizontal. (a) The two conflicts flatland forbids. (b) Prioritized planning: A is planned first and its cell-times are reserved, so B's earliest safe plan waits at its station. (c) A breaks down for 6 steps. Ordered execution holds B until A has left the last shared cell, so B arrives late but never meets A. Entering at B's planned time would have created the head-on deadlock marked in red.*

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
