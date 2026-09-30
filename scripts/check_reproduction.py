"""Compare freshly computed results against the stored ones in data/results.

Only deterministic metrics are compared (arrival rate, normalised reward, deadlocks, steps).
Wall-clock timings differ between machines and are reported, not checked.

    python scripts/check_reproduction.py .runs/repro
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["arrived", "normalized_reward", "deadlocked", "steps"]


def main() -> int:
    new_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / ".runs" / "repro")
    stored = {}
    for f in (ROOT / "data" / "results").glob("*.json"):
        for r in json.loads(f.read_text())["rows"]:
            stored[(r["policy"], r["scenario"], r["split"], r["seed"])] = r
    n, bad = 0, []
    for f in sorted(new_dir.glob("*.json")):
        for r in json.loads(f.read_text())["rows"]:
            key = (r["policy"], r["scenario"], r["split"], r["seed"])
            if key not in stored:
                bad.append(f"{key}: not in stored results")
                continue
            n += 1
            for k in KEYS:
                a, b = r[k], stored[key][k]
                if (abs(a - b) > 1e-6) if isinstance(a, float) else (a != b):
                    bad.append(f"{key} {k}: stored {b} reproduced {a}")
    print(f"checked {n} episodes against data/results")
    if bad:
        print("MISMATCHES:\n  " + "\n  ".join(bad[:50]))
        return 1
    print("all deterministic metrics match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
