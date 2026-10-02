"""Continuous Flatland: sweep the injection rate for the executor alone and for a trained dispatcher.

Each run is one medium map (map seed 5000) with a Poisson stream of trains, truncated at the step
budget. Writes data/tada/results/<out>.json.

    python scripts/tada_continuous.py --controllers executor --out continuous_executor
    python scripts/tada_continuous.py --controllers learned --name main --out continuous_learned
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")


def _job(args) -> dict:
    controller, ckpt_dir, rate, budget, arrival_seed, malf = args
    import numpy as np
    import torch

    from rl_flatland.tada.continuous import OnlineExecutor, continuous_metrics, make_continuous_env
    from rl_flatland.tada.env import DispatchConfig, DispatchEnv
    from rl_flatland.tada.executor import PROCEED
    from rl_flatland.tada.policy import act_at_point

    torch.set_num_threads(1)
    if controller == "executor":
        dc, net = DispatchConfig(), None
    else:
        from rl_flatland.tada.ppo import TadaPPOConfig, make_net

        cfg = TadaPPOConfig(**json.loads((Path(ckpt_dir) / "config.json").read_text()))
        net = make_net(cfg)
        net.load_state_dict(torch.load(Path(ckpt_dir) / "model.pt", map_location="cpu"))
        net.eval()
        dc = DispatchConfig.from_dict(cfg.dispatch)
    t0 = time.perf_counter()
    env = make_continuous_env(rate, budget=budget, arrival_seed=arrival_seed, malfunctions=malf)
    de = DispatchEnv(dc, executor_cls=OnlineExecutor)
    w, _ = de.reset("medium", arrival_seed, malf, env=env)
    while w is not None:
        if net is None:
            de.apply(int(np.argmax(w.choosable)), PROCEED)
        else:
            act_at_point(net, de, dc.budget, greedy=True)
        w, _ = de.advance()
    s = de.summary()
    m = continuous_metrics(env, budget, de.occupancy, de.terminated)
    m.update(controller=controller, rate=rate, budget=budget, arrival_seed=arrival_seed, malfunctions=malf,
             commits=s["commits"], deadlock_terminations=s["deadlock_terminations"], reservation_violations=s["reservation_violations"],
             rotation_flags=s["rotation_flags"], unplannable=int(de.ex.unplannable), wall_s=round(time.perf_counter() - t0, 1),
             decision_ms_per_step=1000 * (time.perf_counter() - t0) / max(1, m["steps"]))
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="main", help="checkpoint under data/tada/checkpoints for the learned dispatcher")
    p.add_argument("--rates", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--no-malfunctions", action="store_true")
    p.add_argument("--controllers", nargs="+", default=["executor", "learned"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="continuous")
    a = p.parse_args()
    ckpt = str(ROOT / "data" / "tada" / "checkpoints" / a.name)
    jobs = [(c, ckpt, r, a.budget, s, not a.no_malfunctions) for r in a.rates for c in a.controllers for s in range(a.seeds)]
    jobs.sort(key=lambda j: -j[2])  # longest first
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(_job, jobs))
    out = ROOT / "data" / "tada" / "results" / f"{a.out}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": a.name, "budget": a.budget, "wall_s": round(time.perf_counter() - t0, 1), "rows": rows}, indent=1))
    import numpy as np

    for c in a.controllers:
        for r in sorted(a.rates):
            rr = [x for x in rows if x["controller"] == c and x["rate"] == r]
            print(f"{c:9s} rate {r:.2f}: throughput/1000 {np.mean([x['throughput_per_1000'] for x in rr]):6.1f} "
                  f"delay {np.nanmean([x['mean_delay'] for x in rr]):6.1f} backlog {np.mean([x['backlog_off_map'] for x in rr]):6.1f} "
                  f"deadlocks/1000 {np.mean([x['deadlock_events_per_1000'] for x in rr]):.2f} window {np.mean([x['window_mean'] for x in rr]):.2f} "
                  f"ms/step {np.mean([x['decision_ms_per_step'] for x in rr]):.1f}")
    print(f"wrote {out} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
