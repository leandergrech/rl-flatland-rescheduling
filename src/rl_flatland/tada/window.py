"""The context window: which trains the dispatcher may act on this step, and what it sees of them.

Rules, all evaluated on the executor's re-timed schedule E (see executor.py):

(a) conflict: within the next H steps the train shares a cell with another train, i.e. it has a
    direct predecessor or successor in some cell's visiting order with both visits inside H;
(b) plan infeasible: its projected arrival is later than its latest arrival LA, or after T;
(c) approach: another train uses, within H, one of the cells up to and including its next D
    decision cells (facing switches and the cells before switches) plus the segment after them;
(d) departure: it is ready to depart and its first segment is occupied now or reserved within H.

Members are ranked by slack LA - (t + k * remaining shortest-path length), least first, and capped
at M. A member is *choosable* only if it stands at a decision instant (cell exit, or ready to
depart) and has at least one structurally legal non-default action.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np
from flatland.envs.step_utils.states import TrainState

from rl_flatland.graph import to_config
from rl_flatland.observations import _Occupancy, _scan_branch
from rl_flatland.tada.executor import HOLD, N_ACTIONS, PROCEED, REROUTE, YIELD_TO, TadaExecutor

TRAIN_FEATURES = 14
BRANCH_FEATURES = 3 * 6
GLOBAL_FEATURES = 6


@dataclass
class WindowConfig:
    H: int = 30
    D: int = 3
    M: int = 8
    features: str = "slack"  # "slack" (set i) or "slack+tree" (set ii)
    allow_yield: bool = True
    allow_hold: bool = True
    allow_reroute: bool = True

    @property
    def n_features(self) -> int:
        return TRAIN_FEATURES + (BRANCH_FEATURES if self.features == "slack+tree" else 0)


@dataclass
class Window:
    trains: List[int]
    feats: np.ndarray  # (M, F)
    mask: np.ndarray  # (M,) 1 for real members
    choosable: np.ndarray  # (M,) structural train-choice mask
    struct_actions: np.ndarray  # (M, N_ACTIONS) structural action legality
    partners_after: Dict[int, List[int]]  # window-slot -> window-slots it may yield to
    glob: np.ndarray
    reasons: Dict[int, Set[str]] = field(default_factory=dict)


def _partners(ex: TadaExecutor, E, now: int, H: int):
    """Direct neighbours in cell visiting orders with both visits inside [now, now+H]."""
    before: Dict[int, Set[int]] = defaultdict(set)
    after: Dict[int, Set[int]] = defaultdict(set)
    near_cells: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    horizon = now + H
    for cell, lst in ex.visits.items():
        start = ex._head(cell)
        prev = None
        for t, h, idx in lst[start:]:
            te = E.get((h, idx), now)  # in-progress visits count as now
            if te > horizon:
                break
            near_cells[cell].add(h)
            if prev is not None and prev != h:
                after[prev].add(h)
                before[h].add(prev)
            prev = h
    return before, after, near_cells


def build_window(ex: TadaExecutor, cfg: WindowConfig) -> Window:
    env = ex.env
    now = env._elapsed_steps
    T = float(env._max_episode_steps)
    E = ex.retime(now)
    before, after, near_cells = _partners(ex, E, now, cfg.H)
    occ_cells = {tuple(a.current_configuration[0]) for a in env.agents if a.state.is_on_map_state() and a.current_configuration}
    cand: Dict[int, Set[str]] = defaultdict(set)
    slack: Dict[int, float] = {}
    for h in ex.active():
        a = env.agents[h]
        p = ex.paths[h]
        j0 = ex.progress[h]
        info = ex.infos[h]
        cur = to_config(a.current_configuration) if a.state.is_on_map_state() else info.start
        dist = ex.graph.distance(h, cur)
        slack[h] = a.latest_arrival - (now + info.k * (dist if math.isfinite(dist) else env.width + env.height))
        if before[h] or after[h]:
            cand[h].add("a")
        arr = E.get((h, len(p) - 1))
        if arr is not None and (arr > a.latest_arrival or arr > T):
            cand[h].add("b")
        # (c) approach: cells up to the D-th decision cell ahead, plus the segment after it
        seen = 0
        j = max(j0, 0)
        stop = None
        while j < len(p):
            c = p[j][0]
            if ex.graph.is_decision_config(c):
                seen += 1
                if seen >= cfg.D and stop is None:
                    stop = j
            if stop is not None and j > stop and ex.graph.is_decision_config(c):
                break
            if len(near_cells.get((c[0], c[1]), ())) > (1 if h in near_cells.get((c[0], c[1]), ()) else 0):
                cand[h].add("c")
                break
            j += 1
        # (d) departure into an occupied or reserved first segment
        if a.state.is_off_map_state() and now >= a.earliest_departure:
            for jj in range(len(p)):
                c = p[jj][0]
                cell = (c[0], c[1])
                others = near_cells.get(cell, set()) - {h}
                if cell in occ_cells or others:
                    cand[h].add("d")
                    break
                if jj > 0 and ex.graph.is_decision_config(c):
                    break
    members = sorted(cand, key=lambda h: (slack[h], h))[: cfg.M]
    slot = {h: i for i, h in enumerate(members)}
    M, F = cfg.M, cfg.n_features
    feats = np.zeros((M, F), np.float32)
    mask = np.zeros(M, np.float32)
    choosable = np.zeros(M, np.float32)
    struct = np.zeros((M, N_ACTIONS), np.float32)
    partners_after: Dict[int, List[int]] = {}
    occ = _Occupancy(env) if cfg.features == "slack+tree" else None
    for i, h in enumerate(members):
        a = env.agents[h]
        p = ex.paths[h]
        j0 = ex.progress[h]
        info = ex.infos[h]
        mask[i] = 1
        off = a.state.is_off_map_state()
        cur = to_config(a.current_configuration) if not off else info.start
        dist = ex.graph.distance(h, cur)
        # steps until the next decision cell on the plan
        nd = 0.0
        for j in range(max(j0, 0) + (0 if off else 1), len(p)):
            if ex.graph.is_decision_config(p[j][0]):
                nd = E.get((h, j), now) - now
                break
        arr = E.get((h, len(p) - 1), now)
        nxt = j0 + 1
        lag = (E[(h, nxt)] - ex._base(h)[nxt]) if (h, nxt) in E else 0
        part = before[h] | after[h]
        mslack = min((slack.get(x, 0.0) for x in part), default=0.0)
        at_dec = ex.at_decision(h)
        feats[i, :TRAIN_FEATURES] = [
            np.clip(slack[h] / T, -1, 1),
            1.0 / info.k,
            min(nd / cfg.H, 2.0),
            (dist if math.isfinite(dist) else env.width + env.height) / (env.width + env.height),
            min(len(before[h]) / 5.0, 1.0),
            min(len(after[h]) / 5.0, 1.0),
            np.clip(mslack / T, -1, 1) if part else 0.0,
            min(a.malfunction_handler.malfunction_down_counter / 50.0, 1.0),
            float(arr <= a.latest_arrival),
            float(off),
            np.clip((arr - a.latest_arrival) / T, -1, 1),
            float(at_dec),
            min(lag / cfg.H, 2.0),
            float("b" in cand[h]),
        ]
        if occ is not None:
            branches = [s for s in ex.graph.successors(cur)] if not off else [cur]
            for b_i, s in enumerate(branches[:3]):
                sc = _scan_branch(ex.graph, occ, h, cur, s)
                d_s = ex.graph.distance(h, s)
                feats[i, TRAIN_FEATURES + 6 * b_i : TRAIN_FEATURES + 6 * b_i + 6] = [
                    1.0,
                    np.clip(((d_s if math.isfinite(d_s) else 99) - (dist if math.isfinite(dist) else 99)) / 5.0, -1, 1),
                    sc[1], sc[2], sc[3], sc[4],
                ]
        # structural action legality
        struct[i, PROCEED] = 1
        if at_dec:
            if cfg.allow_hold:
                struct[i, HOLD] = 1
            if cfg.allow_reroute and ex.reroute_forbidden(h, cfg.D) is not None:
                struct[i, REROUTE] = 1
            yl = [slot[x] for x in after[h] if x in slot]
            if cfg.allow_yield and yl:
                struct[i, YIELD_TO] = 1
                partners_after[i] = sorted(yl)
        choosable[i] = float(struct[i, 1:].sum() > 0)
    n = env.get_num_agents()
    glob = np.array([
        now / T,
        sum(a.state == TrainState.DONE for a in env.agents) / n,
        sum(a.state.is_on_map_state() for a in env.agents) / n,
        len(members) / cfg.M,
        sum(a.malfunction_handler.in_malfunction for a in env.agents) / n,
        len(cand) / n,
    ], np.float32)
    return Window(members, feats, mask, choosable, struct, partners_after, glob, {h: cand[h] for h in members})
