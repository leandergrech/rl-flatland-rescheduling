"""Evaluate a trained dispatcher (greedy) on held-out seeds. Writes data/tada/results/<name>.json.

    python scripts/tada_evaluate.py --name main                          # medium, malfunctions off and on
    python scripts/tada_evaluate.py --name main --scenarios small large xlarge --out main_general
    python scripts/tada_evaluate.py --name executor                      # every clearance PROCEED (no policy)
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


def cpu_mhz() -> float:
    try:
        v = [float(l.split(":")[1]) for l in open("/proc/cpuinfo") if l.startswith("cpu MHz")]
        return sum(v) / len(v)
    except Exception:
        return float("nan")


def _job(args) -> dict:
    ckpt_dir, scenario, seed, malf = args
    import torch

    from rl_flatland.tada.env import DispatchConfig
    from rl_flatland.tada.ppo import TadaPPOConfig, make_net, run_episode

    torch.set_num_threads(1)
    if Path(ckpt_dir).name == "executor":
        return _executor_only(scenario, seed, malf)
    cfg = TadaPPOConfig(**json.loads((Path(ckpt_dir) / "config.json").read_text()))
    net = make_net(cfg)
    net.load_state_dict(torch.load(Path(ckpt_dir) / "model.pt", map_location="cpu"))
    net.eval()
    dc = DispatchConfig.from_dict(cfg.dispatch)
    _, _, _, _, _, s = run_episode(net, dc, scenario, seed, malf, cfg.gamma, True)
    s["cpu_mhz"] = cpu_mhz()
    return s


def _executor_only(scenario: str, seed: int, malf: bool) -> dict:
    import numpy as np

    from rl_flatland.tada.env import DispatchConfig, DispatchEnv
    from rl_flatland.tada.executor import PROCEED

    t0 = time.perf_counter()
    de = DispatchEnv(DispatchConfig())
    w, _ = de.reset(scenario, seed, malf)
    while w is not None:
        de.apply(int(np.argmax(w.choosable)), PROCEED)
        w, _ = de.advance()
    s = de.summary()
    s["wall_s"] = time.perf_counter() - t0
    s["decision_ms_per_step"] = 1000 * s["wall_s"] / max(1, s["steps"])
    s["policy_ms_per_step"] = 0.0
    s["cpu_mhz"] = cpu_mhz()
    return s


def main() -> None:
    from rl_flatland.scenarios import TEST_SEEDS

    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--scenarios", nargs="+", default=["medium"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    ckpt = ROOT / "data" / "tada" / "checkpoints" / a.name
    jobs = [(str(ckpt), sc, s, m) for sc in a.scenarios for m in (False, True) for s in TEST_SEEDS[: a.n_seeds]]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(_job, jobs))
    out = ROOT / "data" / "tada" / "results" / f"{a.out or a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": a.name, "eval_wall_s": round(time.perf_counter() - t0, 1), "rows": rows}, indent=1))
    import numpy as np

    for sc in a.scenarios:
        for m in (False, True):
            rr = [r for r in rows if r["scenario"] == sc and r["malfunctions"] == m]
            print(f"{a.name} {sc} malf={m}: arrival {100 * np.mean([r['arrival_rate'] for r in rr]):.1f}% "
                  f"nr {np.mean([r['normalized_reward'] for r in rr]):.4f} terminated {sum(r['terminated'] for r in rr)} "
                  f"truncated {sum(r['truncated'] for r in rr)} commits/ep {np.mean([r['commits'] for r in rr]):.1f} "
                  f"ms/step {np.mean([r['decision_ms_per_step'] for r in rr]):.1f}")
    print(f"wrote {out} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
