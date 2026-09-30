"""Step-1 acceptance checks for the TADA executor. Writes data/tada/verify.json.

1. PROCEED-only: on all 80 held-out episodes (4 scenarios x 10 seeds x malfunctions off/on) the
   executor, driven through the full dispatcher loop with every clearance PROCEED, must reproduce
   main's OR reference exactly (arrived, deadlocked, normalised reward, steps).
2. Random clearances: over 200 episodes (small and medium, half with malfunctions), a random
   policy picks a choosable train, a random legal action and a random legal partner, up to B=2
   per decision point. No episode may deadlock, violate a reservation or deviate from its plan.

    python scripts/tada_verify.py --workers 8
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


def proceed_only(args) -> dict:
    scenario, seed, malf = args
    import numpy as np

    from rl_flatland.tada.env import DispatchConfig, DispatchEnv
    from rl_flatland.tada.executor import PROCEED

    de = DispatchEnv(DispatchConfig())
    w, _ = de.reset(scenario, seed, malf)
    while w is not None:
        de.apply(int(np.argmax(w.choosable)), PROCEED)
        w, _ = de.advance()
    s = de.summary()
    s["occupancy"] = de.occupancy
    return s


def random_clearances(args) -> dict:
    scenario, seed, malf, rng_seed = args
    import numpy as np

    from rl_flatland.tada.env import DispatchConfig, DispatchEnv
    from rl_flatland.tada.executor import PROCEED, YIELD_TO

    rng = np.random.default_rng(rng_seed)
    de = DispatchEnv(DispatchConfig())
    w, _ = de.reset(scenario, seed, malf)
    while w is not None:
        chosen = set()
        while de.can_continue():
            cands = [i for i in range(len(w.trains)) if w.choosable[i] and i not in chosen]
            if not cands:
                break
            i = int(rng.choice(cands))
            chosen.add(i)
            mask, pm = de.action_mask(i)
            legal = [a for a in range(4) if mask[a]]
            a = int(rng.choice(legal)) if rng.random() < 0.5 else PROCEED
            j = int(rng.choice(np.flatnonzero(pm[YIELD_TO]))) if a == YIELD_TO else None
            de.apply(i, a, j)
            if rng.random() < 0.5:
                break
        w, _ = de.advance()
    return de.summary()


def main() -> None:
    from rl_flatland.scenarios import SCENARIOS, TEST_SEEDS

    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--random-episodes", type=int, default=200)
    args = p.parse_args()
    import subprocess

    rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"commit": rev}
    stored = {}
    for r in json.loads((ROOT / "data" / "results" / "or.json").read_text())["rows"]:
        stored[(r["scenario"], r["split"] == "test_malfunction", r["seed"])] = r
    jobs = [(sc, seed, malf) for sc in SCENARIOS for malf in (False, True) for seed in TEST_SEEDS]
    jobs.sort(key=lambda j: -SCENARIOS[j[0]].n_agents)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(proceed_only, jobs))
    mismatches = []
    for r in rows:
        ref = stored[(r["scenario"], r["malfunctions"], r["seed"])]
        for k in ("arrived", "deadlocked", "steps"):
            if r[k] != ref[k]:
                mismatches.append((r["scenario"], r["malfunctions"], r["seed"], k, r[k], ref[k]))
        if abs(r["normalized_reward"] - ref["normalized_reward"]) > 1e-9:
            mismatches.append((r["scenario"], r["malfunctions"], r["seed"], "normalized_reward", r["normalized_reward"], ref["normalized_reward"]))
    out["proceed_only"] = dict(episodes=len(rows), mismatches=mismatches, wall_s=round(time.perf_counter() - t0, 1),
                               rows=[{k: v for k, v in r.items() if k != "occupancy"} for r in rows])
    occ = {f"{r['scenario']}|{int(r['malfunctions'])}|{r['seed']}": r["occupancy"] for r in rows}
    print(f"PROCEED-only: {len(rows)} episodes, {len(mismatches)} mismatches against main's OR reference")

    rjobs = []
    for i in range(args.random_episodes):
        sc = "small" if i % 2 == 0 else "medium"
        rjobs.append((sc, 3000 + i, bool((i // 2) % 2), 10_000 + i))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rrows = list(ex.map(random_clearances, rjobs))
    bad = [r for r in rrows if r["deadlocked"] or r["terminated"] or r["reservation_violations"] or r["deviations"]]
    out["random_clearances"] = dict(
        episodes=len(rrows), bad_episodes=len(bad), commits=sum(r["commits"] for r in rrows), rejected=sum(r["rejected"] for r in rrows),
        wall_s=round(time.perf_counter() - t0, 1), rows=rrows)
    print(f"random clearances: {len(rrows)} episodes, {out['random_clearances']['commits']} committed edits, "
          f"{len(bad)} episodes with a deadlock, violation or deviation")
    d = ROOT / "data" / "tada"
    d.mkdir(parents=True, exist_ok=True)
    (d / "verify.json").write_text(json.dumps(out, indent=1))
    (d / "occupancy_proceed_only.json").write_text(json.dumps(occ))
    sys.exit(1 if (mismatches or bad) else 0)


if __name__ == "__main__":
    main()
