"""Where do the trains that did not arrive end up? Writes data/analysis/failure_modes.json.

For every policy, scenario and held-out seed (no malfunctions), classify each train at the end of
the episode as arrived, deadlocked (head-on core or queued behind one), stuck on the map (on the
map, not arrived, not deadlocked: waiting at the horizon), or never departed.

    python scripts/failure_modes.py --workers 8
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from evaluate import LEARNED, CKPT, make_policy, tree_builder  # noqa: E402

POLICIES = ["or", "ppo", "bc", "bc_ppo", "ppo_tree", "reactive_avoid", "shortest_path"]


def _job(args) -> dict:
    policy_name, scenario, seed = args
    import torch
    from flatland.envs.step_utils.states import TrainState

    from rl_flatland.deadlock import find_deadlocked
    from rl_flatland.graph import RailGraph
    from rl_flatland.scenarios import make_env

    torch.set_num_threads(1)
    obs_builder = None
    if policy_name in LEARNED:
        kind, sub, obs = LEARNED[policy_name]
        policy = make_policy(kind, str(CKPT / sub / "model.pt"), obs, policy_name)
        if obs == "tree":
            obs_builder = tree_builder()
    else:
        policy = make_policy(policy_name)
    env = make_env(scenario, seed, malfunctions=False, obs_builder=obs_builder)
    policy.reset(env)
    done = {"__all__": False}
    while not done["__all__"]:
        _, _, done, _ = env.step(policy.act(env))
        policy.observe(env)
    dl = find_deadlocked(env, getattr(policy, "graph", None) or RailGraph(env))
    counts = dict(arrived=0, deadlocked=0, stuck_on_map=0, never_departed=0)
    for a in env.agents:
        if a.state == TrainState.DONE:
            counts["arrived"] += 1
        elif a.state.is_off_map_state():
            counts["never_departed"] += 1
        elif a.handle in dl:
            counts["deadlocked"] += 1
        else:
            counts["stuck_on_map"] += 1
    return dict(policy=policy.name, scenario=scenario, seed=seed, n_agents=env.get_num_agents(), **counts)


def main() -> None:
    from rl_flatland.scenarios import SCENARIOS, TEST_SEEDS

    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    p.add_argument("--policies", nargs="+", default=POLICIES, help="re-run only these and merge into the existing file")
    args = p.parse_args()
    jobs = [(pol, sc, seed) for sc in args.scenarios for pol in args.policies for seed in TEST_SEEDS]
    jobs.sort(key=lambda j: (j[0] != "ppo_tree", -SCENARIOS[j[1]].n_agents))  # slow jobs first
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_job, jobs))
    out = ROOT / "data" / "analysis" / "failure_modes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and set(args.policies) != set(POLICIES):
        new_keys = {(r["policy"], r["scenario"], r["seed"]) for r in rows}
        old = [r for r in json.loads(out.read_text())["rows"] if (r["policy"], r["scenario"], r["seed"]) not in new_keys]
        rows = old + rows
    out.write_text(json.dumps({"split": "test (no malfunctions)", "wall_s": round(time.perf_counter() - t0, 1), "rows": rows}, indent=1))
    print(f"wrote {out}: {len(rows)} episodes in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
