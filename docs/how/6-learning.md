---
icon: fl/learning
---

# :fl-learning: Learning on top of the plan

!!! abstract "The question"

    Where does a learned policy fit into all this, what feedback does it get, and why has it so far
    lost to planners?

## The learning problem in one paragraph

Each train is an agent. It acts only at decision cells (chapter 1), so the problem is a semi-MDP
whose decision points come at different times for different trains (chapter 2). It observes either
a hand-built vector (this repository's 28-feature compact observation) or flatland's tree
observation, which walks the track ahead of the train branch by branch. Every train shares one
policy, and the reward is flatland's score, which arrives once per train, at arrival or at the
horizon ([The problem](../01-problem.md#objective)).

## The reward

Flatland 3's `DefaultRewards` gives each train one term: how late it arrived, or, if it never
departed, minus its shortest-path travel time (a cancellation), or, if it is still on the map at
the horizon, how late it would be if it drove straight on from there. The benchmark score averages
these over trains and the horizon, capped at \(-T\) per train, and adds 1, so that 1.0 means every
train arrived on time.

<div class="fl-widget" data-widget="reward" data-title="Interactive: from per-train outcomes to the normalised reward"></div>

The score is forgiving in a way that matters for comparisons: a missing train costs only its
travel time divided by \(N T\), so a policy can lose a tenth of its trains and still score above
0.9. That is why every table in this repository reports the arrival rate next to the normalised
reward.

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



## Two ways to put learning in

**Replace the planner.** Main's learned baselines drive every train directly: PPO with parameter
sharing, behaviour cloning from the planner, and cloning followed by PPO. After 30 CPU minutes the
best of them delivers 35% of trains on medium against the planner's 99.3%
([Designs and results](../04-designs.md#results-on-our-scenarios)). In the Lab, its typical failure
is not a head-on deadlock but a standstill: on small seed 1000 it departs nine of ten trains and
every one of them is still stopped on the map at the horizon, waiting at switches for trains that
are waiting for it.

**Sit on top of the planner.** Page 8's dispatcher lets the planner keep every train safe and
learns only when to change the plan: a context window of the trains that matter, and clearances
(HOLD, YIELD_TO, REROUTE) that the planner checks before applying (chapter 5). It cannot deadlock,
it matches the planner on medium, and its gains are a few trains on large and xlarge with
malfunctions ([TADA on rails](../08-tada-dispatcher.md)).

[PPO stands still on small seed 1000](7-lab.md?map=small-1000-n&preset=ppo&t=150){ .fl-try } [PPO jams on medium seed 1000](7-lab.md?map=medium-1000-n&preset=ppo&t=76&train=9){ .fl-try } [The learned dispatcher's clearances (medium seed 1005)](7-lab.md?map=medium-1005-m&preset=tada&t=120){ .fl-try }

!!! tip "What it means for the agent"

    - **Learn the part the planner does badly.** Planning is fast, safe and nearly optimal on
      these scenarios; what it does badly is react to breakdowns with a better order. That is a
      small, well-posed decision with counterfactual rewards available, not a whole-network
      control problem ([Open questions](../06-open-questions.md)).
    - **Shape carefully, or not at all.** Every RL entry shaped the reward; the dispatcher on page
      8 did best with a dense slack-based shaping term, but not significantly at 10 seeds.
    - **Report arrivals.** A score near 1 can hide cancelled trains.

??? question "Check yourself"

    1. Thirty trains, horizon 600: one train never departs, and its trip is 120 steps. What does
       that cost the normalised reward? *120 / (30 × 600) ≈ 0.0067.*
    2. Why does a policy that stops every train before the first switch avoid deadlocks yet score
       badly? *Stopped trains never arrive; at the horizon each is charged as late by its remaining
       travel time, and the arrival rate collapses.*
    3. What does the dispatcher on page 8 learn, exactly? *Which train in its window to act on, with
       which clearance and partner, and whether to issue another; the planner decides how.*
