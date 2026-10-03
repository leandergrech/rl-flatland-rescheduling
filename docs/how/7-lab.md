---
hide:
  - toc
icon: fl/lab
---

# :fl-lab: The Scheduling Lab

!!! abstract "What this is"

    Flatland itself, running in your browser on all 80 held-out maps of this repository: the four
    scenarios from 30×30 with 10 trains to 100×100 with 100 trains, 10 seeds each, every seed with
    malfunctions off and on. It is not a model of flatland: it is
    a JavaScript port of flatland-rl 4.3.0's environment and of this repository's OR planner, checked
    to reproduce flatland step for step on every one of those maps ([how faithful it
    is](#how-faithful-it-is)). Presets are grouped by **who dispatches**. *Recorded* presets replay
    exactly the actions a policy took in flatland-rl. *Live* presets run here: the OR planner, and
    you, issuing clearances on top of it. Breakdowns you add and clearances you issue re-run the
    episode from the start, so you can try an idea, watch what it does to every train, and take it
    back.

<div class="fl-widget" data-widget="lab" data-title="The Scheduling Lab">
<p class="fl-fallback">The Lab runs in JavaScript, which is switched off or did not load. Static
views of the same maps and policies are in <a href="../3-deadlock/">Conflicts and deadlock</a> and
<a href="../../04-designs/#results-on-our-scenarios">Designs and results</a>.</p>
</div>

## How to use it

- **Above: pick a map and who dispatches.** Small (30×30, 10 trains), medium (50×50, 30), large
  (80×80, 60) or xlarge (100×100, 100); a held-out seed; and malfunctions off or on. Every seed has
  both: the same map and timetable, once without breakdowns and once with flatland's breakdowns
  for that seed (rate 1/1000 per train-step, 20 to 50 steps each). The story under the presets lists the episode's breakdowns, its first
  deadlock, its first late arrival and its end; click a time to jump there.
- **The side pane stays in view** while the panels scroll past it:
    - *Playback*: play, step, speed, the time slider, and an event strip (breakdowns in red, a
      deadlock in dark red, clearances in amber, arrivals in green; click to jump). Keys: space
      plays or pauses, ← and → step one step (with shift, ten).
    - *Controller*: whether the preset is live or recorded. For the planner, the priority order
      it plans with (main's planner tries eight and keeps the best). For you, the HOLD length and
      the clearances you have issued, each removable.
    - *Disturbance*: for live presets, break the selected train now, for 5 to 50 steps.
    - *Outcome*: arrivals, normalised reward, deadlocked and late trains, and what flatland-rl
      recorded for the same controller on the same map.
- **The panels**, from the top: the network with every train (click one to select it; its route
  ahead is dashed and its target boxed); the dispatcher's window (for you and the learned
  dispatcher); every train through the episode, one row each; a time–space diagram along the
  selected train's route; trains arrived over time for every controller on this map; and the
  orders the planner tried.
- **Links** open the Lab in a given state, for example
  `?map=medium-1003-m&preset=or&t=120&train=4`, or with a breakdown you added,
  `&brk=4@60x20` (train 4 at step 60 for 20 steps), or with your clearances,
  `&preset=you&you=118:7:1` (step 118, train 7, HOLD; 2 is YIELD_TO with a partner as a fourth
  number, 3 is REROUTE). The address bar updates as you go, so a link always reproduces what you see.

## Who dispatches

| Preset | Who decides | Runs | What it is |
|---|---|---|---|
| OR planner | a plan: PP + SIPP, ordered execution | live, this page | main's `PrioritizedPlannerPolicy`, ported line for line |
| You dispatch | you, on top of the planner | live, this page | the planner plus HOLD / YIELD_TO / REROUTE clearances, each checked by a speculative SIPP search against the live reservation table (`TadaExecutor`) |
| TADA dispatcher | a learned policy on top of the planner | recorded in flatland-rl | page 8's main checkpoint, greedy; its window and clearances are shown |
| PPO (main) | a learned policy, no planner | recorded in flatland-rl | main's best learned baseline, compact observation |
| Shortest path | every train for itself | recorded in flatland-rl | depart at the earliest time, follow the shortest path |
| Reactive rule | a local rule | recorded in flatland-rl | enter a segment only if no opposing train is on it |

Recorded runs are open loop here: they are the actions taken in flatland, and adding a breakdown
would not change them, so breakdowns can only be added to the live presets.

## Guided experiments

Each takes a few minutes. Predict first, then look.

1. **A deadlock in slow motion.** [Shortest paths on small seed 1004](7-lab.md?map=small-1004-n&preset=shortest_path&t=25&train=6).
   Step forward to t = 32. Which decision, how many steps earlier, made it inevitable? Now load
   [the planner on the same map](7-lab.md?map=small-1004-n&preset=or&t=25&train=6): where does train
   6 wait instead?
2. **Order matters.** [The planner on medium seed 1000](7-lab.md?map=medium-1000-n&preset=or&t=0)
   with the best of eight orders, then with [least slack first](7-lab.md?map=medium-1000-n&preset=or&order=slack&t=0).
   *Predict* how many trains each routes (the planner panel tells you), and look at which trains go
   missing in the timeline.
3. **What a breakdown costs.** [Medium seed 1003 with malfunctions](7-lab.md?map=medium-1003-m&preset=or&t=26&train=6):
   train 6 breaks down at t = 26. Select a train queued behind it and follow it on the time–space
   diagram. Compare the arrivals curve with the [same map without malfunctions](7-lab.md?map=medium-1003-n&preset=or&t=26&train=6).
4. **Break it yourself.** On [medium seed 1003 without malfunctions](7-lab.md?map=medium-1003-n&preset=or&t=100),
   select a train on a single-track section and break it for 50 steps. Does any train deadlock?
   (It cannot: ordered execution.) How many arrive late?
5. **Be the dispatcher.** [You dispatch on medium seed 1005](7-lab.md?map=medium-1005-m&preset=you&t=60),
   the hardest of the medium maps. The planner alone gets 26 of 30 home there. Pause where a late
   train holds up others, and try a YIELD_TO or a HOLD from the window. Can you beat 26? The dashed
   line in the arrivals chart is you.
6. **What the learned dispatcher did.** [The same map, TADA dispatcher](7-lab.md?map=medium-1005-m&preset=tada&t=60).
   Its clearances are amber dots on the timeline. Where did it act, and did it help? (Flatland
   recorded 27 of 30 for it on this map.)
7. **How learned policies fail.** [PPO on small seed 1000](7-lab.md?map=small-1000-n&preset=ppo&t=60):
   no deadlock, yet no train arrives. Watch where the trains stop, and compare with
   [the reactive rule](7-lab.md?map=small-1000-n&preset=reactive_avoid&t=60), which gets 9 of 10 home with
   one line of logic.
8. **A rule that is not enough.** [The reactive rule on medium seed 1005](7-lab.md?map=medium-1005-n&preset=reactive_avoid&t=170&train=20):
   five trains jam at t = 187. Its check looks one segment ahead; why was that too late?
9. **A hundred trains.** [The planner on xlarge seed 1003](7-lab.md?map=xlarge-1003-n&preset=or&t=0)
   gets 97 of 100 trains home. [With malfunctions on](7-lab.md?map=xlarge-1003-m&preset=or&t=0)
   it gets 78, the largest loss to breakdowns of any held-out map. Find the breakdowns on the event
   strip and follow the queue behind one on the time–space diagram. Then compare
   [the learned dispatcher on the same map](7-lab.md?map=xlarge-1003-m&preset=tada&t=0) (82, its
   largest gain anywhere) with [xlarge seed 1008](7-lab.md?map=xlarge-1008-m&preset=tada&t=0),
   where it got 84 against the planner's 87.

## How faithful it is {#how-faithful-it-is}

**What is ported.** `docs/javascripts/flatland-core.js` ports flatland-rl 4.3.0's `RailEnv.step`:
the transition grid and how actions are interpreted, the speed counters, the seven-state train state
machine, `MotionCheck` (including the order in which CPython iterates its set of trains stopped in
a swap, which decides the outcome when several are stopped at once), breakdowns, and the
Flatland 3 `DefaultRewards` with the benchmark's normalisation. Breakdowns are not re-drawn: flatland
draws one per train per step from its random stream whatever the trains do, so
`scripts/make_lab_data.py` records the draws and the Lab replays them. The same file ports main's
OR planner (distance maps, prioritized planning with SIPP, the eight orders, ordered execution);
`docs/javascripts/tada-core.js` ports page 8's dispatcher executor and its context window.

**How it is checked.** `node scripts/check_lab.mjs` (after `python scripts/make_lab_data.py`, which
writes the fixtures) checks, on each of the 80 maps: every train's distance-to-target from every
configuration; Python's successor order where a train has a choice; the five recorded episodes,
replayed step for step (state, position, heading, progress through the cell, speed and breakdown
counter of every train, then arrivals and normalised reward); the planner run live here against
main's planner (identical paths, identical episode); and the learned dispatcher's 7,173 recorded
clearances re-applied through the ported executor, with identical windows at every step and an
identical episode. Result: **80 of 80 maps match flatland-rl**, and the 400 recorded episodes
(five policies on 80 maps) have exactly the outcomes stored in `data/`.

**What is not ported.** The learned policies' networks: PPO and the TADA dispatcher are replays of
what they did in flatland, not policies running here. Flatland's observation builders, which
nothing in the Lab needs. Rendering is this page's own.

**Speed.** A map is fetched only when you pick it (about 45 KB compressed for a 100×100 map).
Loading a 100-train map and replaying every controller on it takes about a second. *You dispatch*
rebuilds the dispatcher's window at every step, so on large and xlarge maps each re-run takes a few
seconds.

See also [the glossary and formula sheet](../literature/glossary.md).
