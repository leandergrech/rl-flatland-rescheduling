"""Train the learned baselines. Each run writes to data/checkpoints/<name>/.

    python scripts/train.py ppo                 # PPO, compact observation, ~30 min on 8 workers
    python scripts/train.py ppo_tree            # PPO, tree observation
    python scripts/train.py imitation           # OR demos -> behaviour cloning (bc) -> PPO fine-tune (bc_ppo)
    python scripts/train.py ppo --budget-s 60 --iterations 2   # smoke run

Budgets are wall-clock limits for the training loop; evaluation (scripts/evaluate.py) is separate
and takes a few minutes per policy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import torch  # noqa: E402

from rl_flatland.baselines.imitation import behaviour_cloning, collect_demos, load_demos, save_demos  # noqa: E402
from rl_flatland.baselines.ppo import PPOConfig, train_ppo  # noqa: E402
from rl_flatland.evaluation import default_workers  # noqa: E402
from rl_flatland.observations import make_obs  # noqa: E402

CKPT = ROOT / "data" / "checkpoints"
DEMOS = ROOT / "data" / "demos" / "or_demos.npz"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("baseline", choices=["ppo", "ppo_tree", "imitation"])
    p.add_argument("--budget-s", type=float, default=1800.0, help="wall-clock limit for the PPO loop")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--episodes-per-iter", type=int, default=16)
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--demo-episodes", type=int, default=60, help="OR episodes per scenario for imitation")
    p.add_argument("--bc-epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-suffix", default="")
    args = p.parse_args()
    torch.set_num_threads(2)

    t0 = time.perf_counter()
    if args.baseline in ("ppo", "ppo_tree"):
        obs = "compact" if args.baseline == "ppo" else "tree"
        cfg = PPOConfig(obs=obs, iterations=args.iterations, time_budget_s=args.budget_s, workers=args.workers,
                        episodes_per_iter=args.episodes_per_iter, seed=args.seed)
        out = CKPT / (args.baseline + args.out_suffix)
        train_ppo(cfg, out, log_name=args.baseline)
        (out / "wall_clock.json").write_text(json.dumps({"train_wall_s": round(time.perf_counter() - t0, 1), "workers": args.workers}))
        return

    # imitation: demos -> BC -> PPO fine-tune
    obs = "compact"
    rng = np.random.default_rng(args.seed)
    if DEMOS.exists() and not args.out_suffix:
        print(f"[imitation] reusing {DEMOS}")
        demos = load_demos(DEMOS)
    else:
        episodes = []
        for sc in ["small", "medium"]:
            seeds = rng.choice(1000, size=args.demo_episodes, replace=False)
            for k, seed in enumerate(seeds):
                episodes.append((sc, int(seed), bool(k % 2)))
        t = time.perf_counter()
        raw = collect_demos(episodes, obs, workers=args.workers)
        print(f"[imitation] {len(raw['a'])} labelled decisions from {len(episodes)} OR episodes in {time.perf_counter() - t:.0f} s; "
              f"label mix {np.bincount(raw['a'], minlength=3) / len(raw['a'])}; OR arrival {raw['episode_arrival'].mean():.3f}")
        DEMOS.parent.mkdir(parents=True, exist_ok=True)
        save_demos(raw, DEMOS)
        demos = load_demos(DEMOS)
    t_demo = time.perf_counter() - t0

    model, bc_log = behaviour_cloning(demos, make_obs(obs).dim, epochs=args.bc_epochs, seed=args.seed)
    bc_dir = CKPT / ("bc" + args.out_suffix)
    bc_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), bc_dir / "model.pt")
    (bc_dir / "bc_log.json").write_text(json.dumps(bc_log, indent=1))
    t_bc = time.perf_counter() - t0
    (bc_dir / "wall_clock.json").write_text(json.dumps({"demo_wall_s": round(t_demo, 1), "train_wall_s": round(t_bc, 1), "workers": args.workers}))

    cfg = PPOConfig(obs=obs, iterations=args.iterations, time_budget_s=args.budget_s, workers=args.workers,
                    episodes_per_iter=args.episodes_per_iter, seed=args.seed, lr=1e-4, ent_coef=0.003,
                    bc_coef_start=1.0, value_warmup_iters=3)
    out = CKPT / ("bc_ppo" + args.out_suffix)
    train_ppo(cfg, out, init_state=model.state_dict(), bc_data=demos, log_name="bc_ppo")
    (out / "wall_clock.json").write_text(json.dumps({"demo_wall_s": round(t_demo, 1), "bc_wall_s": round(t_bc - t_demo, 1),
                                                     "train_wall_s": round(time.perf_counter() - t0, 1), "workers": args.workers}))


if __name__ == "__main__":
    main()
