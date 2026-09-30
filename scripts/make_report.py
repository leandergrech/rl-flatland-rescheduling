"""Build the result tables (Markdown) and figures (PNG) used in the docs from data/results.

    python scripts/make_report.py            # writes docs/assets/*.png and data/results/summary_tables.md
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

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    plot_results(rows, "arrival_rate", "test", policies=pols, ax=axes[0], ylabel="arrival rate, no malfunctions")
    plot_results(rows, "arrival_rate", "test_malfunction", policies=pols, ax=axes[1], ylabel="arrival rate, malfunctions")
    axes[1].get_legend().remove()
    axes[0].legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    for ax in axes:
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(ASSETS / "arrival_rate.png", dpi=150)
    plt.close(fig)

    logs = {n: ROOT / "data" / "checkpoints" / n / "train_log.jsonl" for n in ["ppo", "bc_ppo", "ppo_tree"]}
    logs = {k: v for k, v in logs.items() if v.exists()}
    if logs:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        plot_training(logs, ax=ax)
        ax.set_ylim(0, 1.0)
        fig.tight_layout()
        fig.savefig(ASSETS / "training_curves.png", dpi=150)
        plt.close(fig)
    print(tables.read_text())


if __name__ == "__main__":
    main()
