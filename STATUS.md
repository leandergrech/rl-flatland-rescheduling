# STATUS
repo: https://github.com/leandergrech/rl-flatland-rescheduling
pages: not yet live
step: 5
state: running
updated: 2026-09-30T19:33:17Z
blockers: none
next: step 6: deploy MkDocs to GitHub Pages, confirm 200; final checks (pytest, reproduce.sh, mkdocs build --strict, pip install -e .[dev])
log:
- 2026-09-30T16:08:06Z step 0 done: gh logged in as leandergrech (repo, workflow scopes); git user.name "Leander Grech"; repo folder empty
- 2026-09-30T16:08:38Z step 1 done: skeleton committed, public repo created and pushed
- 2026-09-30T16:18:17Z step 2 done: flatland-rl 4.3.0 + torch 2.14.1+cpu + SB3 2.9.0 in .venv (Python 3.12.3 via uv); random rollouts run at 30x30/10 up to 100x100/100 trains (2.5 to 33.6 ms/step)
- 2026-09-30T16:40:14Z step 3 done: 07-references (7 research reports, all sources opened), 01-problem, 02-primer, 03-timeline written; OR planner prototype working (small/medium 100% arrival)
- 2026-09-30T17:11:08Z step 4 progress: OR reference (PP+SIPP+ordered execution) evaluated on 80 held-out episodes: arrival 100/99.3/98.2/95.7% (small/medium/large/xlarge, no malfunctions), 100/96.3/93.2/89.6% with malfunctions, 0 deadlocks; CPU shared with two sibling training jobs (load ~30)
- 2026-09-30T17:58:33Z step 4 progress: PPO (30 min, 8 workers) greedy arrival 60/35/14/7.5% small/medium/large/xlarge vs OR 100/99.3/98.2/95.7%; PPO avoids head-on deadlocks but gridlocks (trains wait at switches); fixed value warm-up bug and relaunched imitation
- 2026-09-30T19:33:17Z step 4 done: all baselines trained and evaluated (560 episodes); arrival small/medium/large/xlarge, no malfunctions: OR 100/99.3/98.2/95.7, best learned 63.0/35.0/16.5/16.0; every baseline trains+evaluates in <60 min (max 53.3, PPO-tree)
- 2026-09-30T19:33:17Z step 5 done: docs 04-06, for-leander, index, README with generated results tables and figures; notebooks 01-03 executed
