# RL for Flatland train rescheduling

A personal literature review and a runnable, CPU-sized codebase on one question: **can
multi-agent reinforcement learning close the gap to operations research on Flatland train
rescheduling, or beat it somewhere that matters?**

Flatland is the railway scheduling benchmark built by SBB, Deutsche Bahn and SNCF with AIcrowd,
now run by the Flatland Association. Trains on a grid-drawn rail network must reach their targets
within timetable windows, without ever sharing a cell, while random malfunctions stop trains for
20 to 50 steps. In the NeurIPS 2020 round the best OR entry got 98.6% of trains home (score
297.507), and the best RL entry 78.5% (score 214.150)
([Laurent et al. 2021, Table 1](http://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf)).
Six years later, OR still wins: the ECML 2026 challenge ended with 20.84 for the best planning
entry against 9.42 for the RL baseline
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
    314×314 cells and 6,256 trains. [05-limitations](05-limitations.md#what-changes-at-competition-scale)
    covers what changes at that scale.

## How to read this

| If you want | Read |
|---|---|
| A personal on-ramp: what transfers from ATC and accelerator RL, a two-week plan | [For Leander](for-leander.md) |
| The problem stated formally | [1. The problem](01-problem.md) |
| Railway dispatching and the MAPF toolkit at the depth an RL researcher needs | [2. Domain primer](02-primer.md) |
| Who got which number when, 2019 to 2026 | [3. Timeline](03-timeline.md) |
| How the winning and the RL designs work, and our baselines' results | [4. Solution designs](04-designs.md) |
| What fails and by how much | [5. Limitations](05-limitations.md) |
| Ranked research openings with effort estimates | [6. Open questions](06-open-questions.md) |
| Every source, verified | [7. References](07-references.md) |

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
