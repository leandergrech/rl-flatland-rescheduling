"""OR reference: prioritized planning with safe-interval search, executed in planned cell order.

This reproduces the core that the 2019 and 2020 Flatland winners share:

* **Prioritized planning (PP).** Trains are planned one at a time in a priority order. Each train
  gets the earliest-arrival path that avoids every cell-time already reserved by higher-priority
  trains (Silver 2005; Erdmann & Lozano-Perez 1987).
* **Safe-interval path planning (SIPP).** The single-train search runs over (configuration,
  safe interval) states instead of (configuration, time) states (Phillips & Likhachev 2011), as
  in the 2020 winner (Li et al. 2021).
* **Ordered execution.** At run time a train may enter a cell only after every train that was
  planned to visit that cell earlier has left it. This is the "minimum communication policy"
  used by the 2020 winner, and the "keep the same visiting order in every cell" rule from
  Andreica's 2019 winning entry. When the plan is collision-free, the rule makes execution
  deadlock-free even when malfunctions delay trains, because trains wait instead of diverging
  from the plan.
* **Priority restarts.** Several orderings are tried within a time budget and the plan that
  routes the most trains wins, a cheap stand-in for the 2020 winner's LNS.

Not reproduced here (see docs/04-designs.md): large neighbourhood search, partial replanning
after malfunctions, lazy planning, and the C++ implementation.

Conflict model, taken from flatland-rl 4.3.0's MotionCheck:
* a cell holds one train at a time (vertex constraint, on end-of-step occupancy);
* two trains may not swap cells in one step (edge constraint);
* a train may enter a cell in the same step its occupant leaves (close following is allowed).

A train with speed 1/k spends exactly k steps in each cell. It enters its first cell one step
after it becomes READY_TO_DEPART, and it disappears in the step it enters its target cell.
"""

from __future__ import annotations

import bisect
import heapq
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.step_utils.states import TrainState

from rl_flatland.graph import DO_NOTHING, MOVE_FORWARD, STOP_MOVING, Config, RailGraph, action_between, to_config
from rl_flatland.policies import Policy

INF = 10**9
Cell = Tuple[int, int]
Path = List[Tuple[Config, int]]  # (configuration, entry time)


class ReservationTable:
    """Per-cell reserved occupancy intervals [a, b] (inclusive) plus reserved moves (u, v, t)."""

    def __init__(self) -> None:
        self.intervals: Dict[Cell, List[Tuple[int, int]]] = defaultdict(list)
        self.moves: set = set()
        self._safe_cache: Dict[Cell, List[Tuple[int, int]]] = {}

    def reserve(self, cell: Cell, a: int, b: int) -> None:
        bisect.insort(self.intervals[cell], (a, b))
        self._safe_cache.pop(cell, None)

    def reserve_move(self, u: Cell, v: Cell, t: int) -> None:
        self.moves.add((u, v, t))

    def swap_blocked(self, u: Cell, v: Cell, t: int) -> bool:
        """Moving u -> v at time t is blocked if someone reserved v -> u at time t."""
        return (v, u, t) in self.moves

    def safe_intervals(self, cell: Cell) -> List[Tuple[int, int]]:
        s = self._safe_cache.get(cell)
        if s is not None:
            return s
        out = []
        t = 0
        for a, b in self.intervals.get(cell, ()):
            if a > t:
                out.append((t, a - 1))
            t = max(t, b + 1)
        if t < INF:
            out.append((t, INF))
        self._safe_cache[cell] = out
        return out

    def reserve_path(self, path: Path) -> None:
        for j in range(len(path) - 1):
            (cfg, t), (nxt, tn) = path[j], path[j + 1]
            self.reserve((cfg[0], cfg[1]), t, tn - 1)
            self.reserve_move((cfg[0], cfg[1]), (nxt[0], nxt[1]), tn)
        cfg, t = path[-1]
        self.reserve((cfg[0], cfg[1]), t, t)  # the target cell is held only in the arrival step


def sipp(
    graph: RailGraph,
    rt: ReservationTable,
    handle: int,
    start: Config,
    t_min: int,
    k: int,
    horizon: int,
    max_expansions: int = 200_000,
) -> Optional[Path]:
    """Earliest-arrival path for one train from off-map (entering ``start`` no earlier than t_min).

    States are (configuration, index of safe interval of its cell). ``g`` is the entry time.
    Heuristic: k * remaining cells, admissible because the train needs k steps per cell.
    """

    def h(cfg: Config) -> float:
        return k * graph.distance(handle, cfg)

    if not np.isfinite(graph.distance(handle, start)):
        return None
    heap: list = []
    best: Dict[Tuple[Config, int], int] = {}
    parent: Dict[Tuple[Config, int], Optional[Tuple[Config, int]]] = {}
    counter = 0
    start_cell = (start[0], start[1])
    for idx, (lo, hi) in enumerate(rt.safe_intervals(start_cell)):
        if hi < t_min:
            continue
        t = max(lo, t_min)
        if t + h(start) > horizon:
            break
        key = (start, idx)
        best[key] = t
        parent[key] = None
        heapq.heappush(heap, (t + h(start), t, counter, start, idx))
        counter += 1

    expansions = 0
    while heap:
        f, g, _, cfg, iv = heapq.heappop(heap)
        key = (cfg, iv)
        if best.get(key, INF) < g:
            continue
        if graph.distance(handle, cfg) == 0:
            path: Path = []
            node: Optional[Tuple[Config, int]] = key
            while node is not None:
                path.append((node[0], best[node]))
                node = parent[node]
            return path[::-1]
        expansions += 1
        if expansions > max_expansions:
            return None
        cell = (cfg[0], cfg[1])
        lo, hi = rt.safe_intervals(cell)[iv]
        leave_min = g + k
        leave_max = hi + 1 if hi < INF else INF
        if leave_min > leave_max:
            continue
        for s in graph.successors(cfg):
            hs = h(s)
            if not np.isfinite(hs):
                continue
            scell = (s[0], s[1])
            for j, (a, b) in enumerate(rt.safe_intervals(scell)):
                if b < leave_min:
                    continue
                if a > leave_max:
                    break
                t = max(leave_min, a)
                tmax = min(leave_max, b)
                while t <= tmax and rt.swap_blocked(cell, scell, t):
                    t += 1
                if t > tmax:
                    continue
                if t + hs > horizon:
                    break
                skey = (s, j)
                if t < best.get(skey, INF):
                    best[skey] = t
                    parent[skey] = key
                    heapq.heappush(heap, (t + hs, t, counter, s, j))
                    counter += 1
    return None


@dataclass
class AgentInfo:
    handle: int
    start: Config
    k: int  # steps per cell = 1 / max_speed
    earliest_departure: int
    latest_arrival: int
    sp_time: float  # shortest-path travel time in steps


def _agent_infos(env: RailEnv, graph: RailGraph) -> List[AgentInfo]:
    out = []
    for a in env.agents:
        start = to_config(a.initial_configuration)
        k = int(round(1.0 / float(a.speed_counter.max_speed)))
        out.append(
            AgentInfo(
                handle=a.handle,
                start=start,
                k=k,
                earliest_departure=int(a.earliest_departure),
                latest_arrival=int(a.latest_arrival),
                sp_time=k * graph.distance(a.handle, start),
            )
        )
    return out


ORDERINGS = {
    # fast trains first (Andreica 2019), then short trips
    "speed": lambda ai: (ai.k, ai.sp_time),
    # least timetable slack first
    "slack": lambda ai: (ai.latest_arrival - ai.earliest_departure - ai.sp_time, ai.k),
    # earliest scheduled departure first
    "departure": lambda ai: (ai.earliest_departure, ai.k),
    # shortest trips first
    "short": lambda ai: (ai.sp_time, ai.k),
}


def plan_all(
    graph: RailGraph, infos: List[AgentInfo], order: List[int], horizon: int, t0: int = 0
) -> Tuple[ReservationTable, Dict[int, Path]]:
    rt = ReservationTable()
    paths: Dict[int, Path] = {}
    by_handle = {ai.handle: ai for ai in infos}
    for h in order:
        ai = by_handle[h]
        if not np.isfinite(ai.sp_time):
            continue
        t_min = max(ai.earliest_departure, t0) + 1
        p = sipp(graph, rt, h, ai.start, t_min, ai.k, horizon)
        if p is not None:
            rt.reserve_path(p)
            paths[h] = p
    return rt, paths


def plan_quality(paths: Dict[int, Path], infos: List[AgentInfo]) -> Tuple[int, float, float]:
    """Higher is better: (trains routed, -total lateness, -total arrival time)."""
    late = 0.0
    arr = 0.0
    by_handle = {ai.handle: ai for ai in infos}
    for h, p in paths.items():
        t_arr = p[-1][1]
        arr += t_arr
        late += max(0, t_arr - by_handle[h].latest_arrival)
    return (len(paths), -late, -arr)


class PrioritizedPlannerPolicy(Policy):
    """PP + SIPP plan at reset, ordered (MCP-style) execution, retries for unrouted trains."""

    name = "or_pp_sipp"

    def __init__(
        self,
        orderings: Tuple[str, ...] = ("speed", "slack", "departure", "short"),
        n_random_orderings: int = 4,
        time_budget_s: Optional[float] = None,
        retry_every: int = 10,
        seed: int = 0,
    ):
        self.orderings = orderings
        self.n_random_orderings = n_random_orderings
        self.time_budget_s = time_budget_s
        self.retry_every = retry_every
        self.seed = seed

    # ------------------------------------------------------------------ planning
    def reset(self, env: RailEnv) -> None:
        t_start = time.perf_counter()
        self.env = env
        self.graph = RailGraph(env)
        self.horizon = int(env._max_episode_steps)
        self.infos = _agent_infos(env, self.graph)
        rng = np.random.default_rng(self.seed)
        candidates = [sorted(range(len(self.infos)), key=lambda i: ORDERINGS[o](self.infos[i])) for o in self.orderings]
        for _ in range(self.n_random_orderings):
            candidates.append(list(rng.permutation(len(self.infos))))
        best = None
        self.orderings_tried = 0
        for order in candidates:
            # Optional wall-clock cap (as a competition would impose). Off by default, because a cap
            # makes the plan, and so every result, depend on how busy the machine is.
            if self.time_budget_s is not None and self.orderings_tried > 0 and time.perf_counter() - t_start > self.time_budget_s:
                break
            rt, paths = plan_all(self.graph, self.infos, order, self.horizon)
            q = plan_quality(paths, self.infos)
            self.orderings_tried += 1
            if best is None or q > best[0]:
                best = (q, rt, paths)
            if q[0] == len(self.infos) and q[1] == 0:
                break
        _, self.rt, self.paths = best
        self._build_execution_state()
        self.setup_s = time.perf_counter() - t_start

    def _build_execution_state(self) -> None:
        # visits[cell] = sorted list of (planned entry time, handle, path index)
        self.visits: Dict[Cell, List[Tuple[int, int, int]]] = defaultdict(list)
        for h, p in self.paths.items():
            for idx, (cfg, t) in enumerate(p):
                self.visits[(cfg[0], cfg[1])].append((t, h, idx))
        for lst in self.visits.values():
            lst.sort()
        self.vptr: Dict[Cell, int] = defaultdict(int)
        self.progress: Dict[int, int] = {h: -1 for h in self.paths}
        self.deviations = 0

    def _try_plan_unrouted(self, now: int) -> None:
        for ai in self.infos:
            h = ai.handle
            if h in self.paths:
                continue
            agent = self.env.agents[h]
            if not agent.state.is_off_map_state() or not np.isfinite(ai.sp_time):
                continue
            t_min = max(ai.earliest_departure + 1, now + 2)
            p = sipp(self.graph, self.rt, h, ai.start, t_min, ai.k, self.horizon)
            if p is None:
                continue
            self.rt.reserve_path(p)
            self.paths[h] = p
            self.progress[h] = -1
            for idx, (cfg, t) in enumerate(p):
                bisect.insort(self.visits[(cfg[0], cfg[1])], (t, h, idx))

    # ------------------------------------------------------------------ execution
    def _completed(self, visit: Tuple[int, int, int]) -> bool:
        _, h, idx = visit
        if self.env.agents[h].state == TrainState.DONE:
            return True
        return self.progress[h] > idx

    def _head(self, cell: Cell) -> int:
        lst = self.visits[cell]
        ptr = self.vptr[cell]
        while ptr < len(lst) and self._completed(lst[ptr]):
            ptr += 1
        self.vptr[cell] = ptr
        return ptr

    def _will_move(self, h: int, now: int, memo: Dict[int, bool], stack: set) -> bool:
        """Can train h move into its next planned cell in this step?"""
        if h in memo:
            return memo[h]
        if h in stack:
            # h is waiting on a chain that leads back to h: a rotation of 3+ trains around a loop.
            # flatland's MotionCheck only blocks swaps and same-target conflicts, so the whole
            # cycle moves together. Every member already passed its own checks before recursing.
            return True
        stack.add(h)
        ok = self._will_move_inner(h, now, memo, stack)
        stack.discard(h)
        memo[h] = ok
        return ok

    def _will_move_inner(self, h: int, now: int, memo: Dict[int, bool], stack: set) -> bool:
        agent = self.env.agents[h]
        p = self.paths[h]
        nxt_idx = self.progress[h] + 1
        if nxt_idx >= len(p) or agent.state == TrainState.DONE:
            return False
        if agent.malfunction_handler.in_malfunction:
            return False
        nxt_cfg, t_planned = p[nxt_idx]
        if now + 1 < t_planned:  # never run ahead of the plan
            return False
        if agent.state.is_off_map_state():
            # after an off-map malfunction flatland keeps the train in MALFUNCTION_OFF_MAP and
            # places it directly on the map when it receives a movement action
            if agent.state not in (TrainState.READY_TO_DEPART, TrainState.MALFUNCTION_OFF_MAP):
                return False
            if now < agent.earliest_departure:
                return False
        elif not agent.speed_counter.is_cell_exit(agent.speed_counter.max_speed):
            return False
        cell = (nxt_cfg[0], nxt_cfg[1])
        ptr = self._head(cell)
        lst = self.visits[cell]
        my_visit = (t_planned, h, nxt_idx)
        pos = bisect.bisect_left(lst, my_visit)
        if pos == ptr:
            # every earlier visitor has left; the cell may still hold a train that visits later
            # only if that train ran ahead of plan, which _will_move forbids
            return True
        if pos == ptr + 1:
            # close following: the one earlier visitor is in the cell now and leaves in this step
            _, j, qidx = lst[ptr]
            if self.progress[j] == qidx and self.env.agents[j].state.is_on_map_state():
                return self._will_move(j, now, memo, stack)
        return False

    def act(self, env: RailEnv) -> Dict[int, int]:
        now = env._elapsed_steps
        if self.retry_every and now > 0 and now % self.retry_every == 0:
            self._try_plan_unrouted(now)
        memo: Dict[int, bool] = {}
        actions: Dict[int, int] = {}
        for a in env.agents:
            h = a.handle
            if a.state == TrainState.DONE:
                continue
            if h not in self.paths:
                actions[h] = DO_NOTHING
                continue
            p = self.paths[h]
            nxt_idx = self.progress[h] + 1
            if a.state.is_off_map_state():
                go = self._will_move(h, now, memo, set())
                actions[h] = MOVE_FORWARD if go else DO_NOTHING
                continue
            cur = to_config(a.current_configuration)
            nxt = p[nxt_idx][0]
            move = (action_between(cur, nxt))
            if not a.speed_counter.is_cell_exit(a.speed_counter.max_speed):
                actions[h] = move  # mid-cell: keep rolling, the exit step decides
            else:
                actions[h] = move if self._will_move(h, now, memo, set()) else STOP_MOVING
        return actions

    def observe(self, env: RailEnv) -> None:
        """Update plan progress after env.step(). Must be called once per step."""
        for h, p in self.paths.items():
            a = env.agents[h]
            if a.state == TrainState.DONE:
                self.progress[h] = len(p) - 1
                continue
            cfg = to_config(a.current_configuration)
            if cfg is None:
                continue
            nxt_idx = self.progress[h] + 1
            if nxt_idx < len(p) and (cfg[0], cfg[1]) == (p[nxt_idx][0][0], p[nxt_idx][0][1]):
                self.progress[h] = nxt_idx
            elif self.progress[h] >= 0 and (cfg[0], cfg[1]) != (p[self.progress[h]][0][0], p[self.progress[h]][0][1]):
                self.deviations += 1

    # ------------------------------------------------------------------ imitation labels
    def planned_next(self, h: int) -> Optional[Config]:
        """Next planned configuration for train h, or None if unrouted/finished."""
        if h not in self.paths:
            return None
        nxt_idx = self.progress[h] + 1
        p = self.paths[h]
        return p[nxt_idx][0] if nxt_idx < len(p) else None
