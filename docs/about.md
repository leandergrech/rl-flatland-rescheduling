---
icon: fl/author
---

# :fl-author: About this project

## The author

**Leander Grech** is a researcher in reinforcement learning for physical control systems. Current
work includes leading [TADA](https://www.sesarju.eu/projects/TADA), a SESAR 3 JU exploratory
research project on RL for terminal air traffic control; lead authorship on RL-based bent-crystal
alignment at CERN; satellite image super-resolution for the Earth-observation start-up Semablu; and
teaching a Master's course in AI and machine learning.

[:octicons-mark-github-16: github.com/leandergrech](https://github.com/leandergrech) · [:octicons-repo-16: This repository](https://github.com/leandergrech/rl-flatland-rescheduling)

## Why this project

Air traffic control and railway dispatching share a core problem: several vehicles need the same
resource (a stretch of airspace, a single-track section), and someone must decide who goes first,
ahead of time, while delays pass from one vehicle to the next. In both fields RL is measured
against strong classical methods, and in both the classical methods still lead
([air traffic](for-leander.md#what-transfers-from-your-work), [rail](05-limitations.md)). Flatland
is the cleanest public benchmark for the rail side, with published planners from every round
since 2019 to compare against.

The project therefore asks two questions, in order. How far is RL from the planners on mid-size
maps, at a budget one laptop can afford? And can the dispatcher design from TADA, a learned policy
that edits a plan instead of replacing it, recover what the planner loses to breakdowns without
giving up its safety? [Methods and evaluation](research/methodology.md) states how both were
tested.

## What was done

| Part | Kind | Where |
|---|---|---|
| Literature review of Flatland 2019 to 2026 and the MAPF methods behind the winners, every number checked against its source | review | [Literature](literature/mapf.md), [Timeline](03-timeline.md), [References](07-references.md) |
| Four fixed mid-size scenarios with seed splits, a decision wrapper and two observations around flatland-rl 4.3.0 | implemented | [Architecture](implementation/architecture.md) |
| OR planner: prioritized planning, SIPP and ordered execution, written from the published descriptions of the 2019 and 2020 winners | reproduced | [Designs and results](04-designs.md#or-reference-pp-sipp-ordered-execution) |
| PPO (two observations), behaviour cloning, cloning then PPO, two rule-based references, one evaluation harness | implemented and compared | [Designs and results](04-designs.md#baselines-in-this-repository) |
| A learned windowed dispatcher on top of the planner, with a re-timed live reservation table and checked clearances; verification, ablations, generalisation and a continuous-traffic variant | original (a 2022 thesis tried a close variant: [related work](literature/hybrids.md)) | [TADA on rails](08-tada-dispatcher.md) |
| The Scheduling Lab: flatland, the planner and the dispatcher in the browser, checked against flatland-rl step for step | original | [The Scheduling Lab](how/7-lab.md) |
| Chapters with playable models, from the rail network to learning on top of a plan | explanation | [How it works](02-primer.md) |

## Lessons

These follow from the results on this site; the evidence is linked.

- **Build the strong baseline first.** The reproduced planner set a bar (95.7 to 100% of trains
  arrived) that no learned policy came near, and it made the learned results readable
  ([results](04-designs.md#results-on-our-scenarios)).
- **Choose the metric before the method.** Flatland's normalised reward stays above 0.5 for
  policies that deliver 6% of trains; arrival rate does not hide that
  ([metrics](research/methodology.md#metrics)).
- **Safety by construction was cheap; improvement was not.** Checking every edit against a live
  reservation table kept the planner's record of no deadlocks through every run, including 200
  episodes of random edits. Learning which edits help did not measurably beat the plan
  ([TADA on rails](08-tada-dispatcher.md#what-did-not-work)).
- **Dense decisions dilute credit.** A decision on most steps, nearly all of them "carry on", and a
  reward at the end of the episode: PPO learned to make its edits cost no delay, not which edits
  to make ([results on medium](08-tada-dispatcher.md#results-on-medium)).
- **Search for prior work before building, not after.** The related-work search ran after the
  dispatcher experiment and found a 2022 thesis that had tried a close variant on Flatland, with
  the same null result ([related work](literature/hybrids.md)).
- **Verify a port step by step.** The browser simulator matched flatland-rl only after reproducing
  details such as the order in which Python iterates a set
  ([how the Lab is checked](how/7-lab.md#how-faithful-it-is)).

## How this site was made

Code, experiments and text were developed with an AI coding assistant (Claude Code) working under
the author's direction; the commit history records each step. Every result of this project quoted on the site is
computed from episodes stored in [`data/`](https://github.com/leandergrech/rl-flatland-rescheduling/tree/main/data)
by a script in [`scripts/`](https://github.com/leandergrech/rl-flatland-rescheduling/tree/main/scripts),
and [Reproducing the results](implementation/reproduce.md) lists the commands.

The [study notes](for-leander.md) written for the author at the start of the project map ideas from
air traffic, accelerator control and remote sensing onto Flatland, with a two-week reading and
coding plan.

## Citing

```bibtex
@software{grech2026flatland,
  author = {Grech, Leander},
  title  = {{RL for Flatland train rescheduling}: a reproduced planner, learned baselines and a learned dispatcher},
  year   = {2026},
  url    = {https://github.com/leandergrech/rl-flatland-rescheduling}
}
```

The code is MIT-licensed. Third-party licences are listed under
[References](07-references.md#software-environment-verified-by-installation).
