"""Overview figures for the home page and the methods page, from the stored per-episode results.

    python scripts/make_overview_figures.py

Writes docs/assets/figures/overview-arrival-{light,dark}.svg (arrival rate of the planner, the learned
dispatcher on top of it, two policies learned from scratch and the reactive rule, by scenario, without
and with malfunctions), docs/assets/figures/overview-metrics-{light,dark}.svg (arrival rate against
normalised reward for every policy and scenario), and data/analysis/overview.json with every number
plotted. Inputs: data/results/<policy>.json (main's baselines) and data/tada/results/{main,main_general}.json
(the learned dispatcher; medium is its training scenario, the others are evaluated without retraining).
Means and standard errors are over the 10 held-out seeds (1000-1009) of each scenario.
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
matplotlib.rcParams["svg.hashsalt"] = "rl-flatland"  # stable element ids
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from rl_flatland import theme as th  # noqa: E402

SCEN = ["small", "medium", "large", "xlarge"]
FIG = ROOT / "docs" / "assets" / "figures"


def main_rows(policy: str):
    rows = json.loads((ROOT / "data" / "results" / f"{policy}.json").read_text())["rows"]
    return [dict(scenario=r["scenario"], malf=r["split"] == "test_malfunction", seed=r["seed"], arrival=r["arrival_rate"],
                 nr=r["normalized_reward"], deadlocked=r["deadlocked"]) for r in rows]


def tada_rows():
    out = []
    for f in ("main.json", "main_general.json"):
        for r in json.loads((ROOT / "data" / "tada" / "results" / f).read_text())["rows"]:
            out.append(dict(scenario=r["scenario"], malf=bool(r["malfunctions"]), seed=r["seed"], arrival=r["arrival_rate"],
                            nr=r["normalized_reward"], deadlocked=r["deadlocked"]))
    return out


def stats(rows, sc, malf, key):
    v = np.array([r[key] for r in rows if r["scenario"] == sc and r["malf"] == malf], float)
    return dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))), n=int(len(v)))


SERIES = [  # key, label, source, family
    ("or_pp_sipp", "OR planner (reproduced)", lambda: main_rows("or"), "planner"),
    ("tada", "Learned dispatcher on the planner (original)", tada_rows, "on planner"),
    ("ppo", "PPO, compact obs. (learned from scratch)", lambda: main_rows("ppo"), "from scratch"),
    ("ppo_tree", "PPO, tree obs. (learned from scratch)", lambda: main_rows("ppo_tree"), "from scratch"),
    ("reactive_avoid", "Reactive rule (no learning)", lambda: main_rows("reactive_avoid"), "rule"),
]


def collect():
    data = {}
    for key, label, src, fam in SERIES:
        rows = src()
        data[key] = dict(label=label, family=fam, cells={f"{sc}|{int(m)}": dict(arrival=stats(rows, sc, m, "arrival"), nr=stats(rows, sc, m, "nr"),
                                                                                  deadlocked=stats(rows, sc, m, "deadlocked"))
                                                          for sc in SCEN for m in (False, True)})
    return data


def series_style(key, mode):
    # the learned dispatcher is the planner plus a learned layer: the planner's colour, hatched
    if key == "tada":
        return dict(facecolor="none", edgecolor=th.color("or_pp_sipp", mode), hatch="////", linewidth=1.2)
    return dict(facecolor=th.color(key, mode), edgecolor="none")


def arrival_figure(data):
    def build(mode):
        t = th.TOKENS[mode]
        fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.2), sharex=True)
        n = len(SERIES)
        w = 0.82 / n
        x = np.arange(len(SCEN))
        for ax, malf, title in zip(axes, (False, True), ("No malfunctions", "Same seeds, malfunctions on (rate 1/1000 per train-step, 20–50 steps)")):
            for i, (key, label, _, _) in enumerate(SERIES):
                cells = [data[key]["cells"][f"{sc}|{int(malf)}"]["arrival"] for sc in SCEN]
                m = np.array([c["mean"] for c in cells]) * 100
                se = np.array([c["se"] for c in cells]) * 100
                xs = x - 0.41 + w * (i + 0.5)
                ax.bar(xs, m, width=w * 0.9, yerr=se, error_kw=dict(ecolor=t["ink2"], lw=0.9, capsize=0), **series_style(key, mode))
                for xi, mi, si in zip(xs, m, se):
                    # one decimal, as in the tables: rounding to integers would turn 89.1 against 89.6 into 89 against 90
                    ax.text(xi, mi + si + 2.0, f"{mi:.1f}", ha="center", va="bottom", rotation=90, fontsize=7.5, color=t["ink2"])
            ax.set_ylim(0, 128)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_ylabel("trains arrived (%)")
            ax.set_title(title)
            ax.grid(axis="x", visible=False)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([th.SCENARIO_LABEL[s].replace("\n", " · ") + " trains" for s in SCEN])
        handles = [Patch(label=label, **{k: v for k, v in series_style(key, mode).items() if k != "linewidth"}) for key, label, _, _ in SERIES]
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 1.0), ncol=2, frameon=False, handlelength=1.6)
        fig.tight_layout()
        return fig

    return build


def metrics_figure():
    pols = [p for p in th.POLICY_ORDER]
    rows = {p: main_rows({"or_pp_sipp": "or"}.get(p, p)) for p in pols}
    markers = {"small": "o", "medium": "s", "large": "^", "xlarge": "D"}

    def build(mode):
        t = th.TOKENS[mode]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for p in pols:
            for sc in SCEN:
                a = stats(rows[p], sc, False, "arrival")["mean"] * 100
                r = stats(rows[p], sc, False, "nr")["mean"]
                ax.scatter([a], [r], s=36, marker=markers[sc], color=th.color(p, mode), edgecolor=t["surface"], linewidth=0.8, zorder=3)
        ax.plot([0, 100], [0, 1], color=t["ink2"], lw=0.8, ls=(0, (3, 3)))
        ax.text(62, 0.58, "normalised reward = arrival share", fontsize=7.5, color=t["ink2"], rotation=24)
        ax.set_xlim(0, 102)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("trains arrived (%)")
        ax.set_ylabel("normalised reward")
        ax.set_title("The same episodes, two metrics (no malfunctions; each point: one policy on one scenario)")
        h1 = [Patch(color=th.color(p, mode), label=th.SHORT_LABEL[p]) for p in pols]
        h2 = [plt.Line2D([], [], marker=markers[s], color=t["ink2"], ls="none", label=s) for s in SCEN]
        fig.legend(handles=h1 + h2, loc="center left", bbox_to_anchor=(0.86, 0.5), frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0, 0.85, 1))
        return fig

    return build


def main() -> None:
    data = collect()
    th.save_both(arrival_figure(data), "overview-arrival", FIG)
    th.save_both(metrics_figure(), "overview-metrics", FIG)
    out = ROOT / "data" / "analysis" / "overview.json"
    out.write_text(json.dumps({"note": "means and standard errors over 10 held-out seeds per scenario; see scripts/make_overview_figures.py",
                               "series": data}, indent=1))
    for key, d in data.items():
        print(f"{d['label']:48s}", "  ".join(f"{sc[:2]} {100 * d['cells'][f'{sc}|0']['arrival']['mean']:5.1f}/{100 * d['cells'][f'{sc}|1']['arrival']['mean']:5.1f}" for sc in SCEN))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
