"""Figures for the documentation, each built in a light and a dark variant (see theme.py).

Every chart here has a table with the same numbers on the page that shows it. Literature numbers
are copied from the sources cited next to each figure in the docs; our own numbers are read from
data/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from rl_flatland import theme as th

ROOT = Path(__file__).resolve().parents[2]
EDGE = {0: (0.0, -0.5), 1: (0.5, 0.0), 2: (0.0, 0.5), 3: (-0.5, 0.0)}


# --------------------------------------------------------------------------------- data helpers
def load_rows() -> List[dict]:
    rows = []
    for f in sorted((ROOT / "data" / "results").glob("*.json")):
        rows += json.loads(f.read_text())["rows"]
    return rows


def _mean_se(rows, pol, sc, split, key):
    v = np.array([r[key] for r in rows if r["policy"] == pol and r["scenario"] == sc and r["split"] == split], float)
    if not v.size:
        return np.nan, 0.0
    return v.mean(), (v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)


def _policies(rows):
    present = {r["policy"] for r in rows}
    return [p for p in th.POLICY_ORDER if p in present]


def _legend_top(fig, handles, ncol=4, y=1.02):
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, handlelength=1.2, columnspacing=1.4)


def _policy_handles(pols, mode):
    return [Patch(facecolor=th.color(p, mode), label=th.POLICY_LABEL[p]) for p in pols]


def _grouped_bars(ax, rows, pols, key, split, mode, scale=1.0, err=True):
    t = th.TOKENS[mode]
    n = len(pols)
    width = 0.82 / n
    x0 = np.arange(len(th.SCENARIO_ORDER))
    for j, p in enumerate(pols):
        ms = [_mean_se(rows, p, s, split, key) for s in th.SCENARIO_ORDER]
        m = np.array([a for a, _ in ms]) * scale
        se = np.array([b for _, b in ms]) * scale
        x = x0 + (j - (n - 1) / 2) * width
        ax.bar(x, m, width=width, color=th.color(p, mode), edgecolor=t["surface"], linewidth=1.0, zorder=2)
        if err:
            ax.errorbar(x, m, yerr=se, fmt="none", ecolor=t["ink2"], elinewidth=0.9, capsize=0, zorder=3)
    ax.set_xticks(x0)
    ax.set_xticklabels([th.SCENARIO_LABEL[s] for s in th.SCENARIO_ORDER])
    ax.grid(axis="x", visible=False)


# --------------------------------------------------------------------------------- results
def results_arrival(rows) -> Callable[[str], plt.Figure]:
    def build(mode):
        pols = _policies(rows)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
        for ax, split, title in zip(axes, ["test", "test_malfunction"], ["No malfunctions", "Same seeds, malfunctions on"]):
            _grouped_bars(ax, rows, pols, "arrival_rate", split, mode, scale=100)
            ax.set_title(title)
            ax.set_ylim(0, 105)
        axes[0].set_ylabel("trains arrived (%)")
        _legend_top(fig, _policy_handles(pols, mode), ncol=4)
        fig.tight_layout()
        return fig

    return build


def results_metrics(rows) -> Callable[[str], plt.Figure]:
    def build(mode):
        pols = _policies(rows)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        _grouped_bars(axes[0], rows, pols, "normalized_reward", "test", mode)
        axes[0].set_ylim(0.5, 1.0)
        axes[0].set_title("Normalised reward")
        _grouped_bars(axes[1], rows, pols, "deadlocked", "test", mode)
        axes[1].set_title("Deadlocked trains per episode")
        _grouped_bars(axes[2], rows, pols, "decision_ms_mean", "test", mode, err=False)
        axes[2].set_yscale("log")
        axes[2].set_title("Policy time per env step (ms, log)")
        for ax in axes:
            ax.tick_params(axis="x", labelsize=7.5)
        _legend_top(fig, _policy_handles(pols, mode), ncol=4)
        fig.tight_layout()
        return fig

    return build


def training_curves() -> Callable[[str], plt.Figure]:
    logs = {n: ROOT / "data" / "checkpoints" / n / "train_log.jsonl" for n in ["ppo", "bc_ppo", "ppo_tree"]}

    def build(mode):
        t = th.TOKENS[mode]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3))
        handles = []
        for name, path in logs.items():
            if not path.exists():
                continue
            rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            x = np.array([r["wall_s"] for r in rows]) / 60
            for ax, key in zip(axes, ["arrival_rate", "deadlocked"]):
                y = np.array([r[key] for r in rows]) * 100
                k = min(5, len(y))
                ys = np.convolve(y, np.ones(k) / k, mode="valid")
                ax.plot(x[k - 1 :], ys, color=th.color(name, mode), lw=2)
            handles.append(Line2D([], [], color=th.color(name, mode), lw=2, label=th.POLICY_LABEL[name]))
        axes[0].set_title("Trains arrived in training episodes (%)")
        axes[1].set_title("Trains deadlocked in training episodes (%)")
        for ax in axes:
            ax.set_xlabel("training wall-clock (min)")
            ax.set_ylim(0, 100)
            ax.set_xlim(0, 32)
        axes[0].axvline(15, color=t["ink2"], lw=0.9, ls=(0, (3, 3)))
        axes[0].text(15.4, 92, "BC regulariser\nreaches 0", fontsize=8, color=t["ink2"], va="top")
        _legend_top(fig, handles, ncol=3)
        fig.tight_layout()
        return fig

    return build


def budgets() -> Callable[[str], plt.Figure]:
    def minutes(policy, ck):
        res = json.loads((ROOT / "data" / "results" / f"{policy}.json").read_text()).get("eval_wall_s", 0) / 60
        tr = 0.0
        if ck and (ROOT / "data" / "checkpoints" / ck / "wall_clock.json").exists():
            tr = json.loads((ROOT / "data" / "checkpoints" / ck / "wall_clock.json").read_text())["train_wall_s"] / 60
        return tr, res

    items = [("or_pp_sipp", "or", None), ("ppo", "ppo", "ppo"), ("bc", "bc", "bc"), ("bc_ppo", "bc_ppo", "bc_ppo"), ("ppo_tree", "ppo_tree", "ppo_tree")]

    def build(mode):
        t = th.TOKENS[mode]
        fig, ax = plt.subplots(figsize=(8.5, 2.9))
        for i, (pol, res, ck) in enumerate(items):
            tr, ev = minutes(res, ck)
            c = th.color(pol, mode)
            ax.barh(i, tr, color=c, edgecolor=t["surface"], height=0.62, zorder=2)
            ax.barh(i, ev, left=tr, color=c, alpha=0.45, hatch="////", edgecolor=t["surface"], height=0.62, zorder=2)
            ax.text(max(tr + ev, 60) + 0.8 if tr + ev > 45 else tr + ev + 0.8, i, f"{tr:.1f} + {ev:.1f} = {tr + ev:.1f} min", va="center", fontsize=8.5, color=t["ink"])
        ax.axvline(60, color=t["ink"], lw=1.2, ls=(0, (4, 3)), zorder=3)
        ax.text(60.8, -0.55, "1-hour ceiling", ha="left", va="center", fontsize=8.5, color=t["ink"])
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([th.POLICY_LABEL[p] for p, _, _ in items])
        ax.invert_yaxis()
        ax.set_xlim(0, 82)
        ax.set_xlabel("wall-clock minutes (8 worker processes, shared CPU)")
        ax.grid(axis="y", visible=False)
        ax.legend(handles=[Patch(facecolor=t["ink2"], label="training"), Patch(facecolor=t["ink2"], alpha=0.45, hatch="////", label="evaluation, 80 episodes")],
                  loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8)
        fig.tight_layout()
        return fig

    return build


# --------------------------------------------------------------------------------- literature
def or_components() -> Callable[[str], plt.Figure]:
    # Laurent et al. 2021 §4.1 (2020 winner, competition score) and Chen et al. 2023 (Flatland 3 winner, local benchmark)
    s2020 = [("PP + A*\n+ MCP", 282.6), ("+ SIPP", 285.4), ("+ LNS", 289.1), ("+ partial\nreplanning", 291.9), ("+ lazy\nplanning", 297.5)]
    s2021 = [("MAPF-LNS\n2020 system", 123.966), ("+ slack\npriorities", 124.227), ("+ delay-based\nneighbourhood", 124.432), ("+ periodic\nreplanning", 125.175)]

    def build(mode):
        t = th.TOKENS[mode]
        c = th.slot(0, mode)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [5, 4]})
        for ax, data, title in [(axes[0], s2020, "NeurIPS 2020 winner: score after each component"),
                                (axes[1], s2021, "Flatland 3 winner: score on its local benchmark")]:
            x = np.arange(len(data))
            y = [v for _, v in data]
            ax.plot(x, y, color=c, lw=2, marker="o", ms=7, markeredgecolor=t["surface"], markeredgewidth=1.5, zorder=3)
            for xi, yi in zip(x, y):
                ax.annotate(f"{yi:g}", (xi, yi), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, color=t["ink"])
            ax.set_xticks(x)
            ax.set_xticklabels([n for n, _ in data], fontsize=8)
            ax.set_title(title)
            pad = (max(y) - min(y)) * 0.25
            ax.set_ylim(min(y) - pad, max(y) + pad * 1.6)
            ax.grid(axis="x", visible=False)
        axes[0].text(0.02, 0.9, "best RL entry: 214.150 (off scale)", transform=axes[0].transAxes, fontsize=8, color=t["ink2"])
        fig.tight_layout()
        return fig

    return build


def rounds() -> Callable[[str], plt.Figure]:
    # score ratios: Laurent et al. 2021 Table 1; Chen et al. 2023; Jiang et al. 2023 Table 7; ECML 2026 results page
    ratio = [("NeurIPS 2020", 214.150 / 297.507), ("Flatland 3\n(at close)", 27.868 / 135.47), ("Flatland 3 stages\nTreeLSTM, post hoc", 125.3 / 141.0), ("ECML 2026", 9.4231 / 20.8429)]
    # arrival rates (%): same sources; ours from data/results
    arrival = [("NeurIPS 2020", 98.6, 78.5), ("Flatland 3 leaderboard", 88.0, 38.6), ("Flatland 3, TreeLSTM post hoc", 88.0, 66.4),
               ("This repo, small (10 trains)", 100.0, 63.0), ("This repo, xlarge (100 trains)", 95.7, 16.0)]

    def build(mode):
        t = th.TOKENS[mode]
        c_or, c_rl = th.slot(0, mode), th.slot(1, mode)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), gridspec_kw={"width_ratios": [4, 5]})
        ax = axes[0]
        x = np.arange(len(ratio))
        vals = [100 * v for _, v in ratio]
        ax.bar(x, vals, color=c_rl, width=0.6, edgecolor=t["surface"], zorder=2)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 3, f"{v:.0f}%", ha="center", fontsize=9, color=t["ink"])
        ax.axhline(100, color=t["ink"], lw=1.1, ls=(0, (4, 3)))
        ax.text(len(ratio) - 0.5, 102, "parity with best OR", ha="right", va="bottom", fontsize=8, color=t["ink"])
        ax.set_xticks(x)
        ax.set_xticklabels([n for n, _ in ratio], fontsize=8)
        ax.set_ylim(0, 118)
        ax.set_title("Best RL score as % of best OR score")
        ax.grid(axis="x", visible=False)
        ax = axes[1]
        y = np.arange(len(arrival))[::-1]
        for yi, (name, o, r) in zip(y, arrival):
            ax.plot([r, o], [yi, yi], color=t["axis"], lw=2.2, zorder=1, solid_capstyle="round")
            ax.scatter([o], [yi], s=70, color=c_or, edgecolor=t["surface"], linewidth=1.5, zorder=3)
            ax.scatter([r], [yi], s=70, color=c_rl, edgecolor=t["surface"], linewidth=1.5, zorder=3)
            ax.text(o + 2.5, yi, f"{o:g}", va="center", fontsize=8.5, color=t["ink"])
            ax.text(r - 2.5, yi, f"{r:g}", va="center", ha="right", fontsize=8.5, color=t["ink"])
        ax.set_yticks(y)
        ax.set_yticklabels([n for n, _, _ in arrival], fontsize=8.5)
        ax.set_xlim(0, 112)
        ax.set_xlabel("trains arrived (%)")
        ax.set_title("Trains arrived: best OR vs best RL")
        ax.grid(axis="y", visible=False)
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c_or, ms=8, label="best OR"),
                           Line2D([], [], marker="o", ls="", color=c_rl, ms=8, label="best RL")],
                  loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=8)
        fig.tight_layout()
        return fig

    return build


def openings() -> Callable[[str], plt.Figure]:
    # docs/06-open-questions.md effort estimates in weeks (2+ months drawn as 9 to 13 weeks, open-ended)
    items = [("1. Learned repair after malfunctions", 4, 6, True), ("2. Learned departure dispatcher", 3, 5, True),
             ("3. Learned PP priorities", 2, 3, True), ("4. DAgger from the planner", 3, 4, True),
             ("5. Shielded RL", 4, 6, False), ("6. Rail + air sequencing", 9, 13, False), ("7. Fast replanner at scale", 9, 13, False)]

    def build(mode):
        t = th.TOKENS[mode]
        c_a, c_b = th.slot(0, mode), th.slot(1, mode)
        fig, ax = plt.subplots(figsize=(8.5, 3.3))
        for i, (name, lo, hi, attach) in enumerate(items):
            c = c_a if attach else c_b
            ax.plot([lo, hi], [i, i], color=c, lw=9, solid_capstyle="round", zorder=2)
            label = f"{lo}–{hi} weeks" if i < 5 else "2–3 months" if i == 5 else "2+ months"
            ax.text(hi + 0.6, i, label, va="center", fontsize=8.5, color=t["ink"])
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([n for n, *_ in items], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 17)
        ax.set_xlabel("estimated effort (weeks), ranked by value per effort (top = first)")
        ax.grid(axis="y", visible=False)
        ax.legend(handles=[Line2D([], [], color=c_a, lw=6, label="adds learning to a planner"),
                           Line2D([], [], color=c_b, lw=6, label="learning replaces the planner or spans domains")],
                  loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8)
        fig.tight_layout()
        return fig

    return build


def scenario_stats() -> Callable[[str], plt.Figure]:
    stats = json.loads((ROOT / "data" / "scenarios" / "scenario_stats.json").read_text())

    def build(mode):
        t = th.TOKENS[mode]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 2.9))
        names = th.SCENARIO_ORDER
        y = np.arange(len(names))[::-1]
        ax = axes[0]
        for yi, n in zip(y, names):
            lo, hi = stats[n]["T_min"], stats[n]["T_max"]
            ax.plot([lo, hi], [yi, yi], color=t["accent"], lw=7, solid_capstyle="round")
            ax.text(hi + 40, yi, f"{lo:,}–{hi:,}", va="center", fontsize=8.5, color=t["ink"])
        ax.set_yticks(y)
        ax.set_yticklabels([th.SCENARIO_LABEL[n].replace("\n", ", ") + " trains" for n in names], fontsize=8.5)
        ax.set_xlim(0, 2500)
        ax.set_title("Episode horizon T over 10 held-out seeds (steps)")
        ax.grid(axis="y", visible=False)
        ax = axes[1]
        sp = [stats[n]["sp_time_median"] for n in names]
        sl = [stats[n]["slack_median"] for n in names]
        ax.barh(y, sp, color=t["ink2"], height=0.55, edgecolor=t["surface"], label="median shortest-path travel time")
        ax.barh(y, sl, left=sp, color=t["accent"], height=0.55, edgecolor=t["surface"], label="median slack LA − ED − travel time")
        for yi, a, b in zip(y, sp, sl):
            ax.text(a + b + 6, yi, f"{a:.0f} + {b:.0f}", va="center", fontsize=8.5, color=t["ink"])
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xlim(0, 520)
        ax.set_title("Timetable window of the median train (steps)")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, fontsize=7.8)
        fig.tight_layout()
        return fig

    return build


def failure_modes() -> Callable[[str], plt.Figure]:
    path = ROOT / "data" / "analysis" / "failure_modes.json"
    rows = json.loads(path.read_text())["rows"]
    cats = [("arrived", "arrived", th.STATUS["good"]), ("stuck_on_map", "stuck on the map at the horizon", th.STATUS["warning"]),
            ("never_departed", "never departed", th.STATUS["serious"]), ("deadlocked", "deadlocked", th.STATUS["critical"])]

    def build(mode):
        t = th.TOKENS[mode]
        pols = [p for p in th.POLICY_ORDER if any(r["policy"] == p for r in rows)]
        scen = [s for s in th.SCENARIO_ORDER if any(r["scenario"] == s for r in rows)]
        fig, axes = plt.subplots(1, len(scen), figsize=(12, 3.4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, s in zip(axes, scen):
            x = np.arange(len(pols))
            bottom = np.zeros(len(pols))
            for key, _, c in cats:
                v = []
                for p in pols:
                    rr = [r for r in rows if r["policy"] == p and r["scenario"] == s]
                    v.append(100 * sum(r[key] for r in rr) / max(1, sum(r["n_agents"] for r in rr)))
                v = np.array(v)
                ax.bar(x, v, bottom=bottom, color=c, width=0.72, edgecolor=t["surface"], linewidth=1.0, zorder=2)
                bottom += v
            ax.set_xticks(x)
            ax.set_xticklabels([th.SHORT_LABEL[p] for p in pols], rotation=55, ha="right", fontsize=7.8)
            ax.set_title(th.SCENARIO_LABEL[s].replace("\n", ", ") + " trains", fontsize=9.5)
            ax.set_ylim(0, 100)
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("share of trains at episode end (%)")
        _legend_top(fig, [Patch(facecolor=c, label=l) for _, l, c in cats], ncol=4)
        fig.tight_layout()
        return fig

    return build


# --------------------------------------------------------------------------------- maps
def _draw_rail(ax, env, mode, lw=1.6):
    t = th.TOKENS[mode]
    segs = set()
    for r in range(env.height):
        for c in range(env.width):
            for d in range(4):
                for e, ok in enumerate(env.rail.get_transitions(((r, c), d))):
                    if ok:
                        segs.add((r, c, (d + 2) % 4, e))
    for r, c, a, b in segs:
        ax.plot([c + EDGE[a][0], c, c + EDGE[b][0]], [r + EDGE[a][1], r, r + EDGE[b][1]], color=t["rail"], lw=lw, solid_capstyle="round", zorder=1)


def _frame(ax, env, mode, cells=None, margin=1.5):
    t = th.TOKENS[mode]
    if cells:
        rs, cs = zip(*cells)
        ax.set_xlim(min(cs) - margin, max(cs) + margin)
        ax.set_ylim(max(rs) + margin, min(rs) - margin)
    else:
        ax.set_xlim(-1, env.width)
        ax.set_ylim(env.height, -1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(t["grid"])


def map_decision_cells(scenario="small", seed=1000, frac=0.35) -> Callable[[str], plt.Figure]:
    """Real map with switch cells, the RL wrapper's decision cells, targets and an OR snapshot."""
    from rl_flatland.baselines.or_planner import PrioritizedPlannerPolicy
    from rl_flatland.graph import RailGraph
    from rl_flatland.scenarios import make_env

    env = make_env(scenario, seed)
    g = RailGraph(env)
    pol = PrioritizedPlannerPolicy()
    pol.reset(env)
    stop = int(frac * env._max_episode_steps)
    while env._elapsed_steps < stop:
        env.step(pol.act(env))
        pol.observe(env)
    rail = [(r, c) for r in range(env.height) for c in range(env.width) if g.is_rail[r, c]]
    switch = [(r, c) for r, c in rail if g.is_switch[r, c]]
    decision = sorted({(r, c) for r, c in rail for d in range(4) if any(env.rail.get_transitions(((r, c), d))) and g.is_decision_config((r, c, d))})
    targets = sorted({tuple(list(a.targets)[0][0]) for a in env.agents})
    trains = [a.current_configuration[0] for a in env.agents if a.state.is_on_map_state() and a.current_configuration]

    def build(mode):
        t = th.TOKENS[mode]
        fig, ax = plt.subplots(figsize=(7.4, 6.2))
        _draw_rail(ax, env, mode)
        for (r, c) in switch:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="none", edgecolor=t["ink2"], lw=0.9, zorder=2))
        dr, dc = zip(*decision)
        ax.scatter(dc, dr, s=16, color=t["accent"], zorder=3, label="decision cell (RL wrapper asks here)")
        tr, tc = zip(*targets)
        ax.scatter(tc, tr, s=70, marker="s", facecolor="none", edgecolor=th.slot(1, mode), lw=1.6, zorder=4, label="target station cell")
        if trains:
            rr, cc = zip(*trains)
            ax.scatter(cc, rr, s=58, color=t["ink"], edgecolor=t["surface"], lw=1.4, zorder=5, label=f"train (OR run, step {stop})")
        handles = [Line2D([], [], color=t["rail"], lw=2, label="track"),
                   Patch(facecolor="none", edgecolor=t["ink2"], label="switch or crossing cell")] + ax.get_legend_handles_labels()[0]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=8)
        _frame(ax, env, mode, rail)
        ax.set_title(f"{scenario} scenario, seed {seed}: {len(rail)} rail cells, {len(switch)} switch cells, {len(decision)} decision cells")
        fig.tight_layout()
        return fig

    return build


def deadlock_snapshot(scenario="medium", seed=1000) -> Callable[[str], plt.Figure]:
    """Uncoordinated shortest paths vs the OR reference on the same seed, at the same step."""
    from rl_flatland.baselines.or_planner import PrioritizedPlannerPolicy
    from rl_flatland.deadlock import find_deadlocked
    from rl_flatland.graph import RailGraph
    from rl_flatland.policies import ShortestPathPolicy
    from rl_flatland.scenarios import make_env

    def run(policy, until=None):
        env = make_env(scenario, seed)
        policy.reset(env)
        g = getattr(policy, "graph", None) or RailGraph(env)
        dl = set()
        while True:
            _, _, done, _ = env.step(policy.act(env))
            policy.observe(env)
            dl = find_deadlocked(env, g)
            if until is None and len(dl) >= 6:
                break
            if until is not None and env._elapsed_steps >= until:
                break
            if done["__all__"]:
                break
        return env, g, dl

    env_sp, g_sp, dl_sp = run(ShortestPathPolicy())
    step = env_sp._elapsed_steps
    env_or, g_or, dl_or = run(PrioritizedPlannerPolicy(), until=step)
    rail = [(r, c) for r in range(env_sp.height) for c in range(env_sp.width) if g_sp.is_rail[r, c]]

    def panel(ax, env, dl, mode, title):
        t = th.TOKENS[mode]
        _draw_rail(ax, env, mode, lw=1.3)
        on = [a for a in env.agents if a.state.is_on_map_state() and a.current_configuration]
        ok = [a.current_configuration[0] for a in on if a.handle not in dl]
        bad = [a.current_configuration[0] for a in on if a.handle in dl]
        if ok:
            r, c = zip(*ok)
            ax.scatter(c, r, s=40, color=t["ink"], edgecolor=t["surface"], lw=1.2, zorder=4)
        if bad:
            r, c = zip(*bad)
            ax.scatter(c, r, s=62, color=th.STATUS["critical"], edgecolor=t["surface"], lw=1.2, zorder=5, marker="X")
        arrived = sum(a.state.value == 6 for a in env.agents)
        _frame(ax, env, mode, rail)
        ax.set_title(f"{title}\n{arrived} arrived, {len(on)} on the map, {len(dl)} deadlocked", fontsize=9.5)

    def build(mode):
        t = th.TOKENS[mode]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
        panel(axes[0], env_sp, dl_sp, mode, f"Shortest path, no coordination, step {step}")
        panel(axes[1], env_or, dl_or, mode, f"OR reference, same seed, step {step}")
        handles = [Line2D([], [], marker="o", ls="", color=t["ink"], ms=7, label="train"),
                   Line2D([], [], marker="X", ls="", color=th.STATUS["critical"], ms=8, label="deadlocked train")]
        _legend_top(fig, handles, ncol=2, y=0.96)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return fig

    return build


def time_space() -> Callable[[str], plt.Figure]:
    """Schematic time-space diagrams of a 12-cell single-track corridor (position vs time)."""

    def build(mode):
        t = th.TOKENS[mode]
        cA, cB, cC = th.slot(0, mode), th.slot(1, mode), th.slot(2, mode)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True)
        T = np.arange(0, 13)
        # (a) conflicts
        ax = axes[0]
        ax.plot(T, T, color=cA, lw=2, marker="o", ms=3.5)
        ax.plot(T, 12 - T, color=cB, lw=2, marker="o", ms=3.5)
        T1 = np.arange(1, 13)
        ax.plot(T1, 13 - T1, color=cC, lw=2, marker="o", ms=3.5)
        ax.text(12.4, 12, "A, eastbound", va="center", fontsize=8, color=t["ink"])
        ax.text(12.4, -0.2, "B, westbound", va="center", fontsize=8, color=t["ink"])
        ax.text(12.4, 1.2, "C, westbound,\n1 step later", va="center", fontsize=8, color=t["ink"])
        ax.scatter([6], [6], s=260, facecolor="none", edgecolor=th.STATUS["critical"], lw=2, zorder=4)
        ax.annotate("vertex conflict:\nA and B in cell 6\nat step 6", (6, 6), xytext=(8.9, 5.6), fontsize=8, color=t["ink"],
                    arrowprops=dict(arrowstyle="-", color=t["ink2"], lw=0.8))
        ax.scatter([6.5], [6.5], s=110, marker="X", color=th.STATUS["critical"], zorder=4)
        ax.annotate("swap conflict: A and C exchange cells 6 and 7", (6.5, 6.5), xytext=(-0.3, 13.9), fontsize=8, color=t["ink"],
                    arrowprops=dict(arrowstyle="-", color=t["ink2"], lw=0.8))
        ax.set_xlim(-0.5, 17)
        ax.set_title("(a) What the planner must avoid")
        # (b) prioritized plan
        ax = axes[1]
        ax.plot(T, T, color=cA, lw=2)
        ax.plot([0, 13], [12, 12], color=cB, lw=2, ls=(0, (1, 2)))
        Tb = np.arange(13, 26)
        ax.plot(Tb, 12 - (Tb - 13), color=cB, lw=2)
        ax.text(0.3, 12.6, "B waits off the corridor", fontsize=8, color=t["ink"])
        ax.text(6.5, 3.5, "A planned first,\nits cell-times reserved", fontsize=8, color=t["ink"])
        ax.text(13.3, 9.5, "B planned around\nA's reservations", fontsize=8, color=t["ink"])
        ax.set_title("(b) Prioritized plan: A first, then B")
        # (c) ordered execution under a malfunction
        ax = axes[2]
        ax.plot(T, T, color=cA, lw=1.4, ls=(0, (3, 3)), alpha=0.8)
        ax.plot(Tb, 12 - (Tb - 13), color=cB, lw=1.4, ls=(0, (3, 3)), alpha=0.8)
        ta = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        pa = [0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        ax.plot(ta, pa, color=cA, lw=2.2)
        ax.axvspan(4, 10, ymin=0, ymax=1, color=th.STATUS["warning"], alpha=0.12, lw=0)
        ax.text(4.2, 0.4, "A broken down\n6 steps", fontsize=8, color=t["ink"])
        tb = np.arange(19, 32)
        ax.plot([0, 19], [12, 12], color=cB, lw=2, ls=(0, (1, 2)))
        ax.plot(tb, 12 - (tb - 19), color=cB, lw=2.2)
        ax.annotate("B enters only after A has\nleft cell 12: order kept,\nno deadlock", (19, 12), xytext=(20.5, 7.5), fontsize=8, color=t["ink"],
                    arrowprops=dict(arrowstyle="-", color=t["ink2"], lw=0.8))
        ax.plot([13, 15.5], [12, 9.5], color=th.STATUS["critical"], lw=1.6, ls=(0, (1, 1.5)))
        ax.scatter([15.5], [9.5], marker="X", s=90, color=th.STATUS["critical"], zorder=4)
        ax.text(9.2, 13.1, "entering at the planned time would\nmeet A head-on", fontsize=8, color=th.STATUS["critical"])
        ax.set_title("(c) Execution with a malfunction")
        ax.legend(handles=[Line2D([], [], color=t["ink2"], lw=1.4, ls=(0, (3, 3)), label="plan"),
                           Line2D([], [], color=t["ink2"], lw=2.2, label="what happens")], loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.5)
        for ax in axes:
            ax.set_xlabel("time step")
            ax.set_ylim(-0.8, 15.2)
        axes[0].set_ylabel("cell along the corridor (0 = west end)")
        fig.tight_layout()
        return fig

    return build


def study_plan() -> Callable[[str], plt.Figure]:
    """The two-week plan in docs/for-leander.md as a timeline."""
    tasks = [
        ("Environment", [(1, 1, "Problem, primer, notebook 01"), (2, 2, "State machine and MotionCheck")]),
        ("OR and MAPF", [(3, 3, "MAPF basics, read the planner"), (4, 4, "Li et al. 2021, ablate the ordering"), (5, 5, "Chen et al. 2023, compare priorities")]),
        ("Learning", [(6, 6, "JBR_HSE, 10-minute PPO run"), (7, 7, "TreeLSTM and MAMBA, tree vs compact"), (8, 8, "Imitation, BC error analysis")]),
        ("Context", [(9, 9, "ECML 2026, re-score with its rewards"), (10, 10, "ATC papers, pick the experiment")]),
        ("First experiment", [(11, 12, "Build and run"), (13, 13, "Robust MAPF, compare"), (14, 14, "One-page note")]),
    ]

    def build(mode):
        t = th.TOKENS[mode]
        rows = [(sec, i, a, b_, lab) for i, (sec, items) in enumerate(tasks) for a, b_, lab in items]
        fig, ax = plt.subplots(figsize=(9, 4.4))
        for y, (sec, i, a, b_, lab) in enumerate(rows):
            c = th.slot(i, mode)
            ax.barh(y, b_ - a + 1, left=a - 0.5, height=0.62, color=c, edgecolor=t["surface"], zorder=2)
            ax.text(b_ + 0.65, y, lab, va="center", fontsize=8.5, color=t["ink"])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"Day {a}" if a == b_ else f"Days {a}–{b_}" for _, _, a, b_, _ in rows], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xticks(range(1, 15))
        ax.set_xlim(0.4, 21.5)
        ax.set_xlabel("day")
        ax.grid(axis="y", visible=False)
        ax.legend(handles=[Patch(facecolor=th.slot(i, mode), label=sec) for i, (sec, _) in enumerate(tasks)],
                  loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=5, fontsize=8)
        fig.tight_layout()
        return fig

    return build


# --------------------------------------------------------------------------------- all
def build_all(out_dir: str | Path = ROOT / "docs" / "assets" / "figures") -> List[Path]:
    rows = load_rows()
    specs: Dict[str, Callable[[str], plt.Figure]] = {
        "results-arrival": results_arrival(rows),
        "results-metrics": results_metrics(rows),
        "training-curves": training_curves(),
        "budgets": budgets(),
        "or-components": or_components(),
        "rounds": rounds(),
        "openings": openings(),
        "scenario-stats": scenario_stats(),
        "time-space": time_space(),
        "map-decision-cells": map_decision_cells(),
        "deadlock-snapshot": deadlock_snapshot(),
        "study-plan": study_plan(),
    }
    if (ROOT / "data" / "analysis" / "failure_modes.json").exists():
        specs["failure-modes"] = failure_modes()
    out = []
    for name, builder in specs.items():
        out += th.save_both(builder, name, out_dir)
    return out
