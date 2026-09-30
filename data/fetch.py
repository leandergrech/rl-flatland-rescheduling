"""Fetch or regenerate data that is too large to commit (everything lands in data/_large/, gitignored).

    python data/fetch.py --flatland-scenarios   # clone flatland-association/flatland-scenarios (MIT), which
                                                # holds the ECML 2026 and other competition-scale instances
    python data/fetch.py --demos 300            # regenerate a larger OR demonstration set (300 episodes per
                                                # scenario for small and medium) as data/_large/or_demos_300.npz

Committed in data/ (under 20 MB): scenarios/ (seeds, generator config, statistics), results/ (every
evaluation episode), checkpoints/ (trained weights, logs), demos/or_demos.npz (the OR decisions the
imitation baseline was trained on).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LARGE = ROOT / "data" / "_large"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flatland-scenarios", action="store_true")
    p.add_argument("--demos", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    LARGE.mkdir(parents=True, exist_ok=True)
    if args.flatland_scenarios:
        dst = LARGE / "flatland-scenarios"
        if dst.exists():
            print(f"{dst} exists")
        else:
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/flatland-association/flatland-scenarios.git", str(dst)], check=True)
    if args.demos:
        sys.path.insert(0, str(ROOT / "src"))
        import numpy as np

        from rl_flatland.baselines.imitation import collect_demos, save_demos

        rng = np.random.default_rng(1)
        eps = [(sc, int(s), bool(k % 2)) for sc in ["small", "medium"] for k, s in enumerate(rng.choice(1000, args.demos, replace=False))]
        demos = collect_demos(eps, workers=args.workers)
        out = LARGE / f"or_demos_{args.demos}.npz"
        save_demos(demos, out)
        print(f"wrote {out}: {len(demos['a'])} decisions")
    if not (args.flatland_scenarios or args.demos):
        p.print_help()


if __name__ == "__main__":
    main()
