"""Plots: rail network with trains, result bars per scenario, training curves.

Policies always get the same colour, in a fixed categorical order, so a figure that shows only
some policies does not repaint the others.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.step_utils.states import TrainState

from rl_flatland.theme import POLICY_LABEL, POLICY_ORDER, SERIES, STATUS, TOKENS

# same palette and order as the docs figures (light mode), see theme.py
PALETTE = SERIES["light"]
POLICY_COLOR = {p: PALETTE[i] for i, p in enumerate(POLICY_ORDER)}
POLICY_LABEL = {**POLICY_LABEL, "random": "Random"}
INK, INK_2, GRID, SURFACE = TOKENS["light"]["ink"], TOKENS["light"]["ink2"], TOKENS["light"]["grid"], TOKENS["light"]["surface"]
SCENARIO_ORDER = ["small", "medium", "large", "xlarge"]

EDGE = {0: (0.0, -0.5), 1: (0.5, 0.0), 2: (0.0, 0.5), 3: (-0.5, 0.0)}  # (dx, dy) to the edge in direction d


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_rail(env: RailEnv, ax=None, show_trains: bool = True, title: Optional[str] = None):
    """Draw every transition as a two-segment polyline through the cell centre; trains as dots."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    H, W = env.height, env.width
    segs = set()
    for r in range(H):
        for c in range(W):
            for d in range(4):
                trans = env.rail.get_transitions(((r, c), d))
                ein = (d + 2) % 4
                for e, ok in enumerate(trans):
                    if ok:
                        segs.add((r, c, ein, e))
    for r, c, ein, e in segs:
        x0, y0 = c + EDGE[ein][0], r + EDGE[ein][1]
        x1, y1 = c + EDGE[e][0], r + EDGE[e][1]
        ax.plot([x0, c, x1], [y0, r, y1], color=TOKENS["light"]["rail"], linewidth=1.0, solid_capstyle="round", zorder=1)
    targets = {tuple(list(a.targets)[0][0]) for a in env.agents if a.targets}
    if targets:
        tx, ty = zip(*[(c, r) for r, c in targets])
        ax.scatter(tx, ty, marker="s", s=28, facecolor="none", edgecolor=INK_2, linewidth=1.0, zorder=2, label="target cells")
    if show_trains:
        xs, ys, cs = [], [], []
        for a in env.agents:
            if a.state.is_on_map_state() and a.current_configuration is not None:
                (r, c), d = a.current_configuration
                xs.append(c)
                ys.append(r)
                cs.append(STATUS["critical"] if a.malfunction_handler.in_malfunction else INK)
        if xs:
            ax.scatter(xs, ys, s=40, c=cs, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        done = sum(a.state == TrainState.DONE for a in env.agents)
        on = len(xs)
        sub = f"t={env._elapsed_steps}/{env._max_episode_steps}: {on} on map, {done}/{env.get_num_agents()} arrived"
        ax.set_xlabel(sub, color=INK_2, fontsize=9)
    ax.set_xlim(-1, W)
    ax.set_ylim(H, -1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    return ax


def load_rows(results_dir: str | Path) -> List[dict]:
    rows = []
    for f in sorted(Path(results_dir).glob("*.json")):
        rows += json.loads(f.read_text())["rows"]
    return rows


def plot_results(
    rows: List[dict], metric: str = "arrival_rate", split: str = "test", policies: Optional[Sequence[str]] = None, ax=None,
    ylabel: Optional[str] = None,
):
    """Grouped bars: scenario on x, one bar per policy, mean over seeds with standard-error whiskers."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.6))
    present = {r["policy"] for r in rows if r["split"] == split}
    pols = [p for p in (policies or POLICY_ORDER) if p in present]
    scen = [s for s in SCENARIO_ORDER if any(r["scenario"] == s for r in rows)]
    width = 0.8 / max(1, len(pols))
    for j, p in enumerate(pols):
        means, ses = [], []
        for s in scen:
            v = np.array([r[metric] for r in rows if r["policy"] == p and r["scenario"] == s and r["split"] == split], float)
            means.append(v.mean() if v.size else np.nan)
            ses.append(v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)
        x = np.arange(len(scen)) + (j - (len(pols) - 1) / 2) * width
        ax.bar(x, means, width=width - 0.02, color=POLICY_COLOR.get(p, INK_2), edgecolor=SURFACE, linewidth=1.0,
               label=POLICY_LABEL.get(p, p), zorder=2)
        ax.errorbar(x, means, yerr=ses, fmt="none", ecolor=INK_2, elinewidth=1.0, capsize=0, zorder=3)
    ax.set_xticks(np.arange(len(scen)))
    ax.set_xticklabels(scen)
    ax.set_ylabel(ylabel or metric.replace("_", " "), color=INK_2, fontsize=9)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left", labelcolor=INK)
    return ax


def plot_training(logs: Dict[str, str | Path], key: str = "arrival_rate", x: str = "wall_s", ax=None):
    """Training curves (rolling mean over 5 iterations) for several runs' train_log.jsonl."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.4))
    for name, path in logs.items():
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        if not rows:
            continue
        xs = np.array([r[x] for r in rows], float)
        ys = np.array([r.get(key, np.nan) for r in rows], float)
        k = min(5, len(ys))
        smooth = np.convolve(ys, np.ones(k) / k, mode="valid")
        ax.plot(xs[k - 1 :] / (60.0 if x == "wall_s" else 1.0), smooth, color=POLICY_COLOR.get(name, INK_2), linewidth=2,
                label=POLICY_LABEL.get(name, name))
    ax.set_xlabel("training wall-clock (min)" if x == "wall_s" else x, color=INK_2, fontsize=9)
    ax.set_ylabel(key.replace("_", " ") + " (training episodes)", color=INK_2, fontsize=9)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    return ax
