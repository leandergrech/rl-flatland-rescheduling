"""Build the result tables (Markdown) and all figures (light and dark SVG) used in the docs.

    python scripts/make_report.py      # tables -> data/results/summary_tables.md and injected into docs/README,
                                       # figures -> docs/assets/figures/*-{light,dark}.svg
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rl_flatland.plotting import POLICY_LABEL, POLICY_ORDER, SCENARIO_ORDER, load_rows, plot_results, plot_training  # noqa: E402

ASSETS = ROOT / "docs" / "assets"


def cell(rows, pol, sc, split, key, fmt):
    v = np.array([r[key] for r in rows if r["policy"] == pol and r["scenario"] == sc and r["split"] == split], float)
    if not v.size:
        return "–"
    se = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
    return fmt(v.mean(), se)


def table(rows, key, split, fmt, policies) -> str:
    head = "| Policy | " + " | ".join(SCENARIO_ORDER) + " |\n|---|" + "---|" * len(SCENARIO_ORDER) + "\n"
    body = ""
    for p in policies:
        body += f"| {POLICY_LABEL.get(p, p)} | " + " | ".join(cell(rows, p, s, split, key, fmt) for s in SCENARIO_ORDER) + " |\n"
    return head + body


def main() -> None:
    rows = load_rows(ROOT / "data" / "results")
    pols = [p for p in POLICY_ORDER if any(r["policy"] == p for r in rows)]
    pct = lambda m, se: f"{100 * m:.1f} ± {100 * se:.1f}"
    num = lambda m, se: f"{m:.3f} ± {se:.3f}"
    dl = lambda m, se: f"{m:.1f}"
    ms = lambda m, se: f"{m:.2f}"
    out = []
    for split, title in [("test", "held-out seeds, no malfunctions"), ("test_malfunction", "same seeds with malfunctions")]:
        out.append(f"### Arrival rate (%), {title}\n\n" + table(rows, "arrival_rate", split, pct, pols))
        out.append(f"### Normalised reward, {title}\n\n" + table(rows, "normalized_reward", split, num, pols))
        out.append(f"### Deadlocked trains at episode end (mean per episode), {title}\n\n" + table(rows, "deadlocked", split, dl, pols))
    out.append("### Policy wall-clock per env step (ms, mean over episodes, no malfunctions)\n\n" + table(rows, "decision_ms_mean", "test", ms, pols))
    out.append("### One-off setup per episode (s; the OR planner's initial plan)\n\n" + table(rows, "setup_s", "test", lambda m, se: f"{m:.2f}", pols))
    ASSETS.mkdir(parents=True, exist_ok=True)
    tables = ROOT / "data" / "results" / "summary_tables.md"
    tables.write_text("\n\n".join(out) + "\n")

    from rl_flatland.figures import build_all

    paths = build_all(ASSETS / "figures")
    print(f"wrote {len(paths)} figure files")
    fm = ROOT / "data" / "analysis" / "failure_modes.json"
    if fm.exists():
        inject(ROOT / "docs" / "05-limitations.md", "FAILURE_MODES", failure_table(json.loads(fm.read_text())["rows"]))
    readme = readme_table(rows, pols)
    inject(ROOT / "docs" / "04-designs.md", "RESULTS", "\n\n".join(out))
    picture = ('<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/figures/results-arrival-dark.svg">\n'
               '  <img alt="Arrival rate by scenario and policy" src="docs/assets/figures/results-arrival-light.svg">\n</picture>\n\n')
    inject(ROOT / "README.md", "README_RESULTS", picture + readme)
    inject(ROOT / "docs" / "05-limitations.md", "README_RESULTS", readme)
    print(tables.read_text())
    print(readme)


def failure_table(rows) -> str:
    from rl_flatland.theme import POLICY_LABEL, POLICY_ORDER

    head = "| Policy | Scenario | Arrived | Stuck on the map | Never departed | Deadlocked |\n|---|---|---|---|---|---|\n"
    body = ""
    for p in POLICY_ORDER:
        for sc in SCENARIO_ORDER:
            rr = [r for r in rows if r["policy"] == p and r["scenario"] == sc]
            if not rr:
                continue
            n = sum(r["n_agents"] for r in rr)
            cells = [f"{100 * sum(r[k] for r in rr) / n:.1f}%" for k in ["arrived", "stuck_on_map", "never_departed", "deadlocked"]]
            body += f"| {POLICY_LABEL[p]} | {sc} | " + " | ".join(cells) + " |\n"
    return head + body


def wall_clock(policy: str) -> str:
    """Training + evaluation wall-clock in minutes, from the stored run metadata."""
    res = ROOT / "data" / "results" / f"{policy}.json"
    ev = json.loads(res.read_text()).get("eval_wall_s", 0.0) if res.exists() else 0.0
    ck = {"ppo": "ppo", "ppo_tree": "ppo_tree", "bc": "bc", "bc_ppo": "bc_ppo"}.get(policy)
    tr = 0.0
    if ck and (ROOT / "data" / "checkpoints" / ck / "wall_clock.json").exists():
        tr = json.loads((ROOT / "data" / "checkpoints" / ck / "wall_clock.json").read_text()).get("train_wall_s", 0.0)
    return f"{tr / 60:.1f} + {ev / 60:.1f}"


def readme_table(rows, pols) -> str:
    name = {"or_pp_sipp": "or"}
    head = ("| Policy | small 30×30, 10 trains | medium 50×50, 30 | large 80×80, 60 | xlarge 100×100, 100 | "
            "train + eval (min) |\n|---|---|---|---|---|---|\n")
    body = ""
    for p in pols:
        cells = []
        for sc in SCENARIO_ORDER:
            a = [r["arrival_rate"] for r in rows if r["policy"] == p and r["scenario"] == sc and r["split"] == "test"]
            b = [r["arrival_rate"] for r in rows if r["policy"] == p and r["scenario"] == sc and r["split"] == "test_malfunction"]
            cells.append(f"{100 * np.mean(a):.1f} / {100 * np.mean(b):.1f}" if a and b else "–")
        body += f"| {POLICY_LABEL.get(p, p)} | " + " | ".join(cells) + f" | {wall_clock(name.get(p, p))} |\n"
    note = ("\nArrival rate in % on the 10 held-out seeds per scenario, without / with malfunctions (same "
            "seeds, rate 1/1000 per train-step, 20 to 50 steps). Mean over episodes. Wall-clock on a "
            "ThinkPad i7-1260P (16 threads) with 8 worker processes, while two unrelated training jobs shared "
            "the CPU (1-minute load average median 23, range 11 to 44, logged in data/results/run_log/).\n")
    return head + body + note


def inject(path: Path, tag: str, text: str) -> None:
    s = path.read_text()
    a, b = f"<!-- {tag}:START -->", f"<!-- {tag}:END -->"
    if a in s and b in s:
        pre, rest = s.split(a, 1)
        _, post = rest.split(b, 1)
        path.write_text(pre + a + "\n" + text + "\n" + b + post)


if __name__ == "__main__":
    main()
