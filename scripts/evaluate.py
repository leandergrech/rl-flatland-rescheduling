"""Evaluate baselines on the fixed held-out scenario/seed sets and write data/results/<policy>.json.

Examples:
    python scripts/evaluate.py --policy or
    python scripts/evaluate.py --policy ppo --checkpoint data/checkpoints/ppo/model.pt
    python scripts/evaluate.py --policy or --scenarios small medium --n-seeds 3
    python scripts/evaluate.py --stats          # scenario statistics and the scenario-set export
    python scripts/evaluate.py --table          # print the summary table from stored results
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from rl_flatland.evaluation import default_workers, evaluate, save_results, summary_table  # noqa: E402
from rl_flatland.scenarios import SCENARIOS, SEED_SPLITS, TEST_MALFUNCTION_SEEDS, TEST_SEEDS, TRAIN_SEEDS, make_env  # noqa: E402

RESULTS = ROOT / "data" / "results"
CKPT = ROOT / "data" / "checkpoints"


def make_policy(kind: str, checkpoint: str | None = None, obs: str = "compact", name: str | None = None):
    """Module-level factory so it pickles into worker processes."""
    if kind == "or":
        from rl_flatland.baselines.or_planner import PrioritizedPlannerPolicy

        return PrioritizedPlannerPolicy()
    if kind == "shortest_path":
        from rl_flatland.policies import ShortestPathPolicy

        return ShortestPathPolicy()
    if kind == "random":
        from rl_flatland.policies import RandomPolicy

        return RandomPolicy()
    if kind == "reactive_avoid":
        from rl_flatland.policies import ReactiveAvoidPolicy

        return ReactiveAvoidPolicy()
    if kind == "network":
        from rl_flatland.baselines.rl_policy import NetworkPolicy

        return NetworkPolicy(checkpoint, obs=obs, name=name or "network")
    raise ValueError(kind)


def tree_builder():
    from rl_flatland.observations import TreeObs

    return TreeObs().builder()


# name -> (factory kind, checkpoint subdir, observation)
LEARNED = {
    "ppo": ("network", "ppo", "compact"),
    "ppo_tree": ("network", "ppo_tree", "tree"),
    "bc": ("network", "bc", "compact"),
    "bc_ppo": ("network", "bc_ppo", "compact"),
}


def scenario_stats() -> dict:
    from rl_flatland.graph import RailGraph

    out = {}
    for name, sc in SCENARIOS.items():
        rows = []
        for seed in TEST_SEEDS:
            env = make_env(name, seed)
            g = RailGraph(env)
            sp, slack = [], []
            for a in env.agents:
                k = round(1 / float(a.speed_counter.max_speed))
                (r, c), d = a.initial_configuration
                dist = g.distance(a.handle, (r, c, d))
                sp.append(dist * k)
                slack.append(a.latest_arrival - a.earliest_departure - dist * k)
            rows.append(dict(seed=seed, T=int(env._max_episode_steps), rail_cells=int(g.is_rail.sum()), switch_cells=int(g.is_switch.sum()),
                             sp_time_median=float(np.median(sp)), slack_median=float(np.median(slack))))
        out[name] = dict(
            config=sc.to_dict(),
            T_min=min(r["T"] for r in rows),
            T_max=max(r["T"] for r in rows),
            rail_cells_mean=float(np.mean([r["rail_cells"] for r in rows])),
            switch_cells_mean=float(np.mean([r["switch_cells"] for r in rows])),
            sp_time_median=float(np.median([r["sp_time_median"] for r in rows])),
            slack_median=float(np.median([r["slack_median"] for r in rows])),
            per_seed=rows,
        )
    return out


def export_scenarios(path: Path) -> None:
    import flatland

    doc = {
        "flatland_rl_version": getattr(flatland, "__version__", "4.3.0"),
        "generator": "sparse_rail_generator + sparse_line_generator + Flatland 3 timetable_generator",
        "scenarios": {k: v.to_dict() for k, v in SCENARIOS.items()},
        "splits": {
            "train": {"seeds": f"{TRAIN_SEEDS.start}..{TRAIN_SEEDS.stop - 1}", "malfunctions": "50% of training episodes"},
            "test": {"seeds": TEST_SEEDS, "malfunctions": False},
            "test_malfunction": {"seeds": TEST_MALFUNCTION_SEEDS, "malfunctions": True},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))


def print_table(files) -> None:
    rows = []
    for f in files:
        rows += json.loads(Path(f).read_text())["rows"]
    table = summary_table(rows)
    print(f"{'policy':16s} {'scenario':8s} {'split':17s} {'n':>3s} {'arrival':>8s} {'norm_rew':>8s} {'deadlk':>6s} {'ms/step':>8s} {'setup_s':>7s}")
    for (pol, sc, split), s in table.items():
        print(f"{pol:16s} {sc:8s} {split:17s} {s['episodes']:3d} {s['arrival_rate']:8.3f} {s['normalized_reward']:8.3f} {s['deadlocked']:6.2f} {s['decision_ms_mean']:8.2f} {s['setup_s']:7.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["or", "shortest_path", "random", "reactive_avoid", *LEARNED])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    p.add_argument("--splits", nargs="+", default=list(SEED_SPLITS))
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--out", default=None)
    p.add_argument("--stats", action="store_true")
    p.add_argument("--table", action="store_true")
    args = p.parse_args()

    if args.stats:
        stats = scenario_stats()
        (ROOT / "data" / "scenarios").mkdir(parents=True, exist_ok=True)
        (ROOT / "data" / "scenarios" / "scenario_stats.json").write_text(json.dumps(stats, indent=1))
        export_scenarios(ROOT / "data" / "scenarios" / "scenarios.json")
        for k, v in stats.items():
            print(f"{k}: T {v['T_min']}-{v['T_max']}, rail cells {v['rail_cells_mean']:.0f}, switch cells {v['switch_cells_mean']:.0f}, "
                  f"median sp time {v['sp_time_median']:.0f}, median slack {v['slack_median']:.0f}")
        return
    if args.table:
        print_table(sorted(RESULTS.glob("*.json")))
        return
    if not args.policy:
        p.error("--policy is required")

    obs_builder_factory = None
    if args.policy in LEARNED:
        kind, sub, obs = LEARNED[args.policy]
        ckpt = args.checkpoint or str(CKPT / sub / "model.pt")
        factory = functools.partial(make_policy, kind, ckpt, obs, args.policy)
        if obs == "tree":
            obs_builder_factory = tree_builder
    else:
        factory = functools.partial(make_policy, args.policy)

    t0 = time.perf_counter()
    rows = evaluate(factory, args.scenarios, args.splits, args.n_seeds, workers=args.workers, obs_builder_factory=obs_builder_factory)
    wall = time.perf_counter() - t0
    out = Path(args.out) if args.out else RESULTS / f"{args.policy}.json"
    save_results(rows, out)
    meta = json.loads(out.read_text())
    meta["eval_wall_s"] = round(wall, 1)
    meta["workers"] = args.workers
    out.write_text(json.dumps(meta, indent=1))
    print(f"wrote {out} ({len(rows)} episodes, {wall:.0f} s with {args.workers} workers)")
    print_table([out])


if __name__ == "__main__":
    main()
