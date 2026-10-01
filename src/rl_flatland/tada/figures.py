"""Figures for the TADA dispatcher page (light and dark SVGs, same theme as main's figures).

    python scripts/tada_report.py       # writes docs/assets/figures/tada-*.svg from data/tada/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from rl_flatland import theme as th

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "tada"


def _legend_top(fig, handles, ncol=4, y=1.02):
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, y), ncol=ncol, frameon=False, handlelength=1.6)


def occupancy(M: int = 8) -> Callable[[str], plt.Figure]:
    """Window occupancy over the episode under PROCEED-only, per scenario (median and 10-90% band)."""
    occ: Dict[str, List[int]] = json.loads((DATA / "occupancy_proceed_only.json").read_text())
    grid = np.linspace(0, 1, 101)

    def build(mode):
        t = th.TOKENS[mode]
        fig, ax = plt.subplots(figsize=(8.6, 3.2))
        handles = []
        for i, sc in enumerate(th.SCENARIO_ORDER):
            series = [np.asarray(v, float) for k, v in occ.items() if k.split("|")[0] == sc and len(v) > 1]
            if not series:
                continue
            ys = np.stack([np.interp(grid, np.linspace(0, 1, len(s)), s) for s in series])
            c = th.slot(i, mode)
            ax.fill_between(grid * 100, np.percentile(ys, 10, 0), np.percentile(ys, 90, 0), color=c, alpha=0.16, lw=0)
            ax.plot(grid * 100, np.median(ys, 0), color=c, lw=2)
            handles.append(Line2D([], [], color=c, lw=2, label=th.SCENARIO_LABEL[sc].replace("\n", " ") + " trains"))
        ax.axhline(M, color=t["ink2"], lw=0.9, ls=(0, (3, 3)))
        ax.text(99, M + 0.15, f"cap M = {M}", ha="right", va="bottom", fontsize=8, color=t["ink2"])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, M + 1)
        ax.set_xlabel("episode progress (% of steps taken)")
        ax.set_title("Trains in the context window, executor only (median, 10–90% band over 20 episodes)")
        _legend_top(fig, handles, ncol=4)
        fig.tight_layout()
        return fig

    return build


def training_curve(names: List[str], labels: Dict[str, str]) -> Callable[[str], plt.Figure]:
    """Arrival rate and normalised reward of training episodes against wall-clock, smoothed over 5 iterations."""

    def build(mode):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2))
        handles = []
        for i, name in enumerate(names):
            path = DATA / "checkpoints" / name / "train_log.jsonl"
            if not path.exists():
                continue
            rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            x = np.array([r["wall_s"] for r in rows]) / 60
            c = th.slot(i, mode)
            for ax, key, scale in zip(axes, ["arrival_rate", "normalized_reward"], [100, 1]):
                y = np.array([r[key] for r in rows]) * scale
                k = min(5, len(y))
                ax.plot(x[k - 1 :], np.convolve(y, np.ones(k) / k, mode="valid"), color=c, lw=2)
            handles.append(Line2D([], [], color=c, lw=2, label=labels.get(name, name)))
        axes[0].set_title("Trains arrived in training episodes (%)")
        axes[1].set_title("Normalised reward of training episodes")
        for ax in axes:
            ax.set_xlabel("training wall-clock (min)")
        _legend_top(fig, handles, ncol=min(4, max(1, len(handles))))
        fig.tight_layout()
        return fig

    return build


def grouped_bars(groups: List[str], series: Dict[str, List[float]], errs: Dict[str, List[float]], title: str, ylabel: str,
                 ylim=None, fmt="{:.1f}") -> Callable[[str], plt.Figure]:
    """Grouped bars (one group per x label, one bar per series) with value labels."""

    def build(mode):
        t = th.TOKENS[mode]
        fig, ax = plt.subplots(figsize=(8.6, 3.2))
        names = list(series)
        n = len(names)
        w = 0.8 / n
        x = np.arange(len(groups))
        handles = []
        for i, name in enumerate(names):
            c = th.slot(i, mode)
            xs = x - 0.4 + w * (i + 0.5)
            ys = np.asarray(series[name], float)
            ax.bar(xs, ys, width=w * 0.92, color=c, yerr=errs.get(name), error_kw=dict(ecolor=t["ink2"], lw=1, capsize=0))
            for xi, yi in zip(xs, ys):
                if np.isfinite(yi):
                    ax.text(xi, yi, fmt.format(yi), ha="center", va="bottom", fontsize=7.5, color=t["ink2"])
            handles.append(Line2D([], [], color=c, lw=6, label=name))
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(title)
        _legend_top(fig, handles, ncol=min(4, n))
        fig.tight_layout()
        return fig

    return build


def continuous_sweep(agg: Dict[tuple, dict], rates: List[float], controllers: List[str]) -> Callable[[str], plt.Figure]:
    """Throughput and mean delay against injection rate, one line per controller."""
    label = {"executor": "Executor only", "learned": "Learned dispatcher"}

    def build(mode):
        t = th.TOKENS[mode]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2))
        handles = []
        x = np.array(rates)
        axes[0].plot(x, 1000 * x, color=t["ink2"], lw=0.9, ls=(0, (3, 3)))
        axes[0].text(x[-1], 1000 * x[-1], " injected", fontsize=8, color=t["ink2"], va="center")
        for i, c in enumerate(controllers):
            col = th.slot(i, mode)
            ys = [agg.get((r, c), {}).get("throughput_per_1000", np.nan) for r in rates]
            ds = [agg.get((r, c), {}).get("mean_delay", np.nan) for r in rates]
            axes[0].plot(x, ys, color=col, lw=2, marker="o", ms=5)
            axes[1].plot(x, ds, color=col, lw=2, marker="o", ms=5)
            handles.append(Line2D([], [], color=col, lw=2, marker="o", ms=5, label=label.get(c, c)))
        axes[0].set_title("Throughput (arrivals per 1000 steps)")
        axes[1].set_title("Mean delay of arrived trains vs LA (steps)")
        for ax in axes:
            ax.set_xlabel("injection rate (trains per step)")
        _legend_top(fig, handles, ncol=2)
        fig.tight_layout()
        return fig

    return build
