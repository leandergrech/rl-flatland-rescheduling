"""Planning primitives for the TADA-style dispatcher: an owned reservation table and SIPP from any state.

``TrainTable`` extends main's ``ReservationTable`` so every interval and move remembers which train
reserved it. That lets the executor lift one or two trains out of the table, plan them again, and
put everything back if the plan fails.

``sipp_from`` is main's safe-interval search generalised in two ways: a train may start on the map
(already inside a cell, able to leave no earlier than ``leave_min``), and a set of forbidden moves
can be passed (REROUTE forbids the old branch at the next facing switch).
"""

from __future__ import annotations

import bisect
import heapq
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from rl_flatland.baselines.or_planner import INF, Cell, Path, ReservationTable
from rl_flatland.graph import Config, RailGraph


class TrainTable(ReservationTable):
    """Reservation table whose entries are owned by trains and can be removed and restored."""

    def __init__(self) -> None:
        super().__init__()
        self.owned: Dict[int, List[tuple]] = defaultdict(list)

    def reserve_for(self, h: int, cell: Cell, a: int, b: int, tag: str = "f") -> None:
        self.reserve(cell, a, b)
        self.owned[h].append(("i", cell, a, b, tag))

    def move_for(self, h: int, u: Cell, v: Cell, t: int) -> None:
        self.reserve_move(u, v, t)
        self.owned[h].append(("m", u, v, t, "f"))

    def lift(self, h: int, tags: Tuple[str, ...] = ("f", "c")) -> List[tuple]:
        """Remove train h's entries with the given tags; return them for ``restore``."""
        keep, gone = [], []
        for e in self.owned.get(h, []):
            (gone if e[4] in tags else keep).append(e)
        for e in gone:
            if e[0] == "i":
                _, cell, a, b, _ = e
                self.intervals[cell].remove((a, b))
                self._safe_cache.pop(cell, None)
            else:
                self.moves.discard((e[1], e[2], e[3]))
        self.owned[h] = keep
        return [(h, e) for e in gone]

    def restore(self, lifted: List[tuple]) -> None:
        for h, e in lifted:
            if e[0] == "i":
                _, cell, a, b, tag = e
                self.reserve_for(h, cell, a, b, tag)
            else:
                self.move_for(h, e[1], e[2], e[3])

    def reserve_future(self, h: int, start: Config, path: Path, target_last: bool = True) -> None:
        """Reserve a planned future: moves from ``start`` into path[0], then each visit's interval."""
        prev = (start[0], start[1]) if start is not None else None
        for j, (cfg, t) in enumerate(path):
            cell = (cfg[0], cfg[1])
            if prev is not None:
                self.move_for(h, prev, cell, t)
            if j + 1 < len(path):
                self.reserve_for(h, cell, t, path[j + 1][1] - 1)
            else:
                self.reserve_for(h, cell, t, t)  # target cell: held only in the arrival step
            prev = cell


def sipp_from(
    graph: RailGraph,
    rt: ReservationTable,
    handle: int,
    start: Config,
    k: int,
    horizon: int,
    now: int,
    on_map: bool,
    leave_min: int,
    forbidden: FrozenSet[Tuple[Config, Config]] = frozenset(),
    max_expansions: int = 60_000,
) -> Optional[Path]:
    """Earliest-arrival timed path for one train, returned as future visits [(cfg, entry_time), ...].

    on_map=True: the train is in ``start`` now and can enter the next cell at ``leave_min`` or later;
    its start cell must be free from ``now`` until it leaves. on_map=False: the train is off the map
    and may enter ``start`` at ``leave_min`` or later (then the returned path begins with ``start``).
    """

    def h(cfg: Config) -> float:
        return k * graph.distance(handle, cfg)

    if not np.isfinite(h(start)):
        return None
    heap: list = []
    best: Dict[Tuple[Config, int], int] = {}
    parent: Dict[Tuple[Config, int], Optional[Tuple[Config, int]]] = {}
    counter = 0
    sc = (start[0], start[1])
    ivs = rt.safe_intervals(sc)
    if on_map:
        # the train already holds its cell: find the safe interval that contains `now`
        for idx, (lo, hi) in enumerate(ivs):
            if lo <= now <= hi:
                best[(start, idx)] = now
                parent[(start, idx)] = None
                heapq.heappush(heap, (now + h(start), now, counter, start, idx, True))
                counter += 1
                break
    else:
        for idx, (lo, hi) in enumerate(ivs):
            if hi < leave_min:
                continue
            t = max(lo, leave_min)
            if t + h(start) > horizon:
                break
            best[(start, idx)] = t
            parent[(start, idx)] = None
            heapq.heappush(heap, (t + h(start), t, counter, start, idx, False))
            counter += 1
    expansions = 0
    while heap:
        f, g, _, cfg, iv, is_start_node = heapq.heappop(heap)
        key = (cfg, iv)
        if best.get(key, INF) < g:
            continue
        if graph.distance(handle, cfg) == 0 and not (on_map and is_start_node):
            out: Path = []
            node: Optional[Tuple[Config, int]] = key
            while node is not None:
                out.append((node[0], best[node]))
                node = parent[node]
            out = out[::-1]
            return out[1:] if on_map else out
        expansions += 1
        if expansions > max_expansions:
            return None
        cell = (cfg[0], cfg[1])
        lo, hi = rt.safe_intervals(cell)[iv]
        leave_lo = leave_min if (on_map and is_start_node) else g + k
        leave_hi = hi + 1 if hi < INF else INF
        if leave_lo > leave_hi:
            continue
        for s in graph.successors(cfg):
            if (cfg, s) in forbidden:
                continue
            hs = h(s)
            if not np.isfinite(hs):
                continue
            scell = (s[0], s[1])
            for j, (a, b) in enumerate(rt.safe_intervals(scell)):
                if b < leave_lo:
                    continue
                if a > leave_hi:
                    break
                t = max(leave_lo, a)
                tmax = min(leave_hi, b)
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
                    heapq.heappush(heap, (t + hs, t, counter, s, j, False))
                    counter += 1
    return None
