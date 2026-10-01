"""Train the TADA dispatcher with PPO. Writes data/tada/checkpoints/<name>/.

    python scripts/tada_train.py --name main                    # B=2, M=8, slack features, all actions; 120 x 8 episodes
    python scripts/tada_train.py --name B1 --budget 1
    python scripts/tada_train.py --name noyield --no-yield
    python scripts/tada_train.py --name M16 --M 16
    python scripts/tada_train.py --name shaping --shaping
    python scripts/tada_train.py --name tree --features slack+tree
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import torch  # noqa: E402

from rl_flatland.tada.env import DispatchConfig  # noqa: E402
from rl_flatland.tada.ppo import TadaPPOConfig, train  # noqa: E402
from rl_flatland.tada.window import WindowConfig  # noqa: E402


def dispatch_config(a) -> DispatchConfig:
    return DispatchConfig(
        window=WindowConfig(H=a.H, D=a.D, M=a.M, features=a.features, allow_yield=not a.no_yield),
        budget=a.budget, shaping=a.shaping, exact_action_masks=not a.structural_masks,
    )


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--budget", type=int, default=2)
    p.add_argument("--M", type=int, default=8)
    p.add_argument("--H", type=int, default=30)
    p.add_argument("--D", type=int, default=3)
    p.add_argument("--features", default="slack", choices=["slack", "slack+tree"])
    p.add_argument("--no-yield", action="store_true")
    p.add_argument("--shaping", action="store_true")
    p.add_argument("--structural-masks", action="store_true", help="skip speculative planning in action masks")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--scenario", default="medium")
    p.add_argument("--budget-s", type=float, default=3300.0, help="wall-clock cap (the brief allows under an hour)")
    p.add_argument("--iterations", type=int, default=120, help="PPO iterations; every run gets the same sample budget")
    p.add_argument("--episodes-per-iter", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    add_args(p)
    a = p.parse_args()
    torch.set_num_threads(2)
    dc = dispatch_config(a)
    cfg = TadaPPOConfig(dispatch=dc.to_dict(), scenario=a.scenario, malfunctions=True, time_budget_s=a.budget_s,
                        episodes_per_iter=a.episodes_per_iter, workers=a.workers, seed=a.seed, iterations=a.iterations)
    out = ROOT / "data" / "tada" / "checkpoints" / a.name
    t0 = time.perf_counter()
    train(cfg, out, log_name=a.name)
    print(f"trained {a.name} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
