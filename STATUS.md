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
