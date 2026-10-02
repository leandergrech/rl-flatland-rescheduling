# STATUS
repo: https://github.com/leandergrech/rl-flatland-rescheduling
pages: https://leandergrech.github.io/rl-flatland-rescheduling/
step: 7
state: done
updated: 2026-09-30T22:20:29Z
blockers: none
next: none; theme and visuals revision published
log:
- 2026-09-30T16:08:06Z step 0 done: gh logged in as leandergrech (repo, workflow scopes); git user.name "Leander Grech"; repo folder empty
- 2026-09-30T16:08:38Z step 1 done: skeleton committed, public repo created and pushed
- 2026-09-30T16:18:17Z step 2 done: flatland-rl 4.3.0 + torch 2.14.1+cpu + SB3 2.9.0 in .venv (Python 3.12.3 via uv); random rollouts run at 30x30/10 up to 100x100/100 trains (2.5 to 33.6 ms/step)
- 2026-09-30T16:40:14Z step 3 done: 07-references (7 research reports, all sources opened), 01-problem, 02-primer, 03-timeline written; OR planner prototype working (small/medium 100% arrival)
- 2026-09-30T17:11:08Z step 4 progress: OR reference (PP+SIPP+ordered execution) evaluated on 80 held-out episodes: arrival 100/99.3/98.2/95.7% (small/medium/large/xlarge, no malfunctions), 100/96.3/93.2/89.6% with malfunctions, 0 deadlocks; CPU shared with two sibling training jobs (load ~30)
- 2026-09-30T17:58:33Z step 4 progress: PPO (30 min, 8 workers) greedy arrival 60/35/14/7.5% small/medium/large/xlarge vs OR 100/99.3/98.2/95.7%; PPO avoids head-on deadlocks but gridlocks (trains wait at switches); fixed value warm-up bug and relaunched imitation
- 2026-09-30T19:33:17Z step 4 done: all baselines trained and evaluated (560 episodes); arrival small/medium/large/xlarge, no malfunctions: OR 100/99.3/98.2/95.7, best learned 63.0/35.0/16.5/16.0; every baseline trains+evaluates in <60 min (max 53.3, PPO-tree)
- 2026-09-30T19:33:17Z step 5 done: docs 04-06, for-leander, index, README with generated results tables and figures; notebooks 01-03 executed
- 2026-09-30T19:48:18Z step 6 done: Pages enabled (workflow build), deploy succeeded after re-dispatch (first run stuck in 'waiting'); site and all 8 doc pages return HTTP 200; math and Mermaid render
- 2026-09-30T19:48:18Z step 7 done: all four checks pass (pip install -e .[dev]; pytest 19 passed; bash scripts/reproduce.sh 280/280 episodes match; mkdocs build --strict); small commits throughout, no force-push
- 2026-09-30T22:20:29Z revision: 'Departure board' theme chosen by Leander; 13 themed light/dark figures + 7 Mermaid diagrams added; failure-mode analysis (280 episodes); OR planner made deterministic (ordering time budget off by default); pytest 22 passed, reproduce.sh 280/280 match, mkdocs --strict ok

## TADA dispatcher
branch: feat/tada-dispatcher
step: 7
state: done
updated: 2026-10-01T04:40:28Z
blockers: none for the work; site deployment needs the user's choice
next: user decision: publish the branch docs (Pages deploys only from main: merge, or allow feat/tada-dispatcher in the github-pages environment)
log:
- 2026-09-30T22:56:40Z step 0: branch feat/tada-dispatcher created. Brief correction: on the current paired malfunction split the OR reference is 100/98.0/93.8/89.1% (not 96.3/93.2/89.6, which were the old unpaired seeds 2000-2009); malfunction gap is 0/1.3/4.4/6.6 points small/medium/large/xlarge, so medium has the smallest non-zero gap. Training stays on medium as specified (CPU ceiling); large/xlarge covered by step 5. Design note: with ordered execution a malfunction never makes a plan order-infeasible, only late, so replanning is triggered by clearances only; this is what makes step 1's exact reproduction possible.
- 2026-10-01T00:00:20Z step 1 in progress: executor, window, env, policy, PPO committed. PROCEED-only matches main's OR exactly on all 20 small and 20 medium episodes and large seeds 1000-1002; first full run reported 48 field mismatches, all on large/xlarge (re-running with per-episode output). Fixed: YIELD_TO planned the second train against a table missing the first train's current cell (caught by retime as an order cycle); find_deadlocked flags plan-scheduled 2x2 rotations (now exempt, counted as rotation_flags); terminations now charge the horizon penalty to unfinished trains; NaN bootstrap on empty windows (PyTorch MHA fast path). CPU was pinned at 400 MHz (thermal) from ~01:00 to ~01:55 UTC+2; now ~2.2 GHz.
- 2026-10-01T00:38:48Z steps 1-2 done. PROCEED-only = main's OR on 80/80 held-out episodes (arrived, deadlocked, steps, normalised reward to 1e-9; data/tada/verify.json, commit 5c667ed). 200 random-clearance episodes: 28,705 committed edits, 0 deadlocks/violations/deviations. Earlier 48 mismatches were all early terminations: main's find_deadlocked flags plan-scheduled 2x2 rotations (12 large/xlarge episodes); exempted. Window occupancy (no malfunctions, M=8): mean 5.9/6.6/7.3/7.5 trains, full on 47/71/86/90% of steps small..xlarge (docs/assets/figures/tada-occupancy-*.svg). Executor-only on medium through the loop: 99.3/98.0% arrival (data/tada/results/executor.json).
- 2026-10-01T01:58:26Z step 3 done. Main run: 115/120 iterations (cap 55 min, CPU 400-2900 MHz), entropy coef 0.001 (a first run at 0.01 drifted: entropy 1.55->2.2, edits x3, stopped at it 31, kept as checkpoints/main_ent01_stopped). Held-out medium, greedy: learned 99.3/98.3% arrival vs executor 99.3/98.0% (paired: net +0/+1 train of 300; nr -0.0007+-0.0005 / +0.0009+-0.0014), 0 terminations, 11 ms/step vs 8.5; greedy clearances 97.5% PROCEED, 2.4% HOLD. Training maps (executor 95.9%): stochastic policy +0.84+-0.10 pt arrival vs executor on the same maps, nr gap -0.0037 -> -0.0005 over training. main's PPO baseline: 35.0/36.0%. Continuous executor-only sweep done (rate 0.02-0.4): no deadlocks, saturates ~145 arrivals/1000 steps, queue grows from 0.2.
- 2026-10-01T03:00:03Z steps 5-6 done; step 7 drafted. Generalisation (medium-trained, greedy): small 100/100% (=executor), large 98.2/94.0% (exec 98.2/93.8; paired +2/-1 trains with malf), xlarge 95.7/89.6% (exec 95.7/89.1; +9/-4), 0 terminations. Continuous (rates 0.02-0.4, 3 seeds, 1000 steps): learned = executor within noise at every rate; neither deadlocks (0/42); both <50% on time from 0.15, queue from 0.2, peak ~144 arrivals/1000 steps. Ablations so far (60 it): B1 99.3/98.7%, noyield 99.3/98.0%, main@60 99.3/98.3%. docs/08 and 06 updated; mkdocs build --strict passes.
- 2026-10-01T04:40:28Z all steps done. Ablations (60 it, medium, malf): executor 98.0%, main@60 98.3, B1 98.7, B4 98.3, noyield 98.0, M4 98.3, M16 98.3, shaping 99.0 (+3 trains, nr +0.0036+-0.0030, n.s.), tree 98.3; no-malf all 99.3%; 0 terminations anywhere. Gains are mostly one train on seed 1005 (6 of 7 runs). docs/08-tada-dispatcher.md complete (generated by scripts/tada_report.py), 06-open-questions updated, mkdocs build --strict passes, pytest 34 passed.

## Docs overhaul (Scheduling Lab)
branch: feat/docs-lab (from feat/tada-dispatcher)
step: done
state: done
updated: 2026-10-02T12:15:19Z
blockers: publishing needs main (the github-pages environment deploys only from main): merge feat/docs-lab, or allow the branch
next: user decision on merging / deploying
log:
- 2026-10-02T12:15:19Z Docs restructured after rl-tokamak-rampup's strategy: rail icon set, nav in four groups (Overview, The benchmark, How it works, Literature), prev/next pager, collapsible page sections, departure-board hero with stat tiles, Open-the-Lab buttons. How it works: hub + six chapters (network, trains, deadlock, planning, execution, learning), each with a widget, 'what it means for the agent' and a self-check; Literature: MAPF toolkit (moved from the primer), timeline, references, glossary and formulas. Scheduling Lab: JavaScript port of flatland-rl 4.3.0 RailEnv + main's OR planner + the TADA executor (docs/javascripts/flatland-core.js, tada-core.js), 40 held-out small/medium maps with 5 recorded policies (scripts/make_lab_data.py); node scripts/check_lab.mjs: 40/40 maps match flatland-rl step for step (distances, successor order, 200 replays, 40 live planner runs, 302 dispatcher clearances). mkdocs build --strict passes; pytest 34 passed.

