"""The executor under the TADA dispatcher: main's OR reference plus editable plans.

It subclasses ``PrioritizedPlannerPolicy`` and changes nothing about how a plan is followed, so with
every clearance set to PROCEED it takes exactly main's code path. What it adds:

* **Re-timing.** ``retime(now)`` computes, for every future cell visit of every routed train, the
  earliest time consistent with (i) the visit's planned time (trains never run ahead of plan),
  (ii) where the train is now, including remaining dwell and malfunction, and (iii) the planned
  visiting order of every cell (a visitor enters only after the previous one has left). This is a
  longest-path computation over the ordering graph, solved by relaxation because main's plans
  can contain zero-delay rotation cycles.
* **A live reservation table.** ``table(E, now)`` reserves every train's current cell and its
  re-timed future. Plan edits are searched against it. Planning against the *original* planned
  times would be unsafe: a delayed train's stale intervals look free, and an edit could be ordered
  head-on against it.
* **Clearances.** HOLD, YIELD_TO and REROUTE lift one or two trains out of the table, plan them
  again with SIPP from where they stand, and commit only if every search succeeds. A failed edit
  leaves the old plan in force. After a commit every train's future is re-timed and the visit
  lists are rewritten, so the ordering stays a consistent, collision-free schedule and ordered
  execution stays deadlock-free.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from flatland.envs.rail_env import RailEnv
from flatland.envs.step_utils.states import TrainState

from rl_flatland.baselines.or_planner import Path, PrioritizedPlannerPolicy
from rl_flatland.graph import Config, to_config
from rl_flatland.tada.planning import TrainTable, sipp_from

PROCEED, HOLD, YIELD_TO, REROUTE = 0, 1, 2, 3
ACTION_NAMES = ["PROCEED", "HOLD", "YIELD_TO", "REROUTE"]
N_ACTIONS = 4


@dataclass(frozen=True)
class Clearance:
    train: int
    kind: int
    partner: Optional[int] = None


class TadaExecutor(PrioritizedPlannerPolicy):
    name = "tada_executor"

    def __init__(self, hold_steps: int = 5, **kw):
        super().__init__(**kw)
        self.hold_steps = hold_steps

    # ------------------------------------------------------------------ lifecycle
    def reset(self, env: RailEnv) -> None:
        super().reset(env)
        self.k = {ai.handle: ai.k for ai in self.infos}
        self.base: Dict[int, List[int]] = {h: [t for _, t in p] for h, p in self.paths.items()}
        self.version = 0  # bumps on every commit and every observed step
        self._E_cache: Tuple[int, int, Optional[dict]] = (-1, -1, None)
        self.stats = defaultdict(int)

    def observe(self, env: RailEnv) -> None:
        super().observe(env)
        self.version += 1

    def _base(self, h: int) -> List[int]:
        b = self.base.get(h)
        p = self.paths[h]
        if b is None or len(b) != len(p):  # trains added by main's retry of unrouted trains
            b = [t for _, t in p]
            self.base[h] = b
        return b

    # ------------------------------------------------------------------ timing
    def first_move_time(self, h: int, now: int) -> int:
        """Earliest step at which train h can enter its next cell (or its start cell, if off-map)."""
        a = self.env.agents[h]
        m = a.malfunction_handler.malfunction_down_counter
        if a.state.is_off_map_state():
            return max(now + 1, a.earliest_departure + 1, now + m + 1)
        sc = a.speed_counter
        v = float(sc.max_speed)
        n = max(1, math.ceil((1.0 - float(sc.distance)) / v - 1e-9))
        return now + n + m

    def active(self) -> List[int]:
        return [h for h in self.paths if self.env.agents[h].state != TrainState.DONE]

    def retime(self, now: int) -> Dict[Tuple[int, int], int]:
        """Earliest consistent entry time for every future visit (see module docstring)."""
        if self._E_cache[0] == now and self._E_cache[1] == self.version:
            return self._E_cache[2]
        keys: List[Tuple[int, int]] = []
        lbs: List[int] = []
        index: Dict[Tuple[int, int], int] = {}
        preds: List[List[Tuple[int, int]]] = []
        for h in self.active():
            p = self.paths[h]
            first = self.progress[h] + 1
            if first >= len(p):
                continue
            base = self._base(h)
            k = self.k[h]
            fm = self.first_move_time(h, now)
            for j in range(first, len(p)):
                i = len(keys)
                index[(h, j)] = i
                keys.append((h, j))
                lbs.append(max(base[j], fm) if j == first else base[j])
                preds.append([(i - 1, k)] if j > first else [])
        paths = self.paths
        for cell, lst in self.visits.items():
            start = self._head(cell)
            if len(lst) - start < 2:
                continue
            ph, pidx = lst[start][1], lst[start][2]
            for n in range(start + 1, len(lst)):
                _, h, idx = lst[n]
                i = index.get((h, idx))
                if i is not None:
                    if pidx == len(paths[ph]) - 1:
                        preds[i].append((index[(ph, pidx)], 1))
                    else:
                        preds[i].append((index[(ph, pidx + 1)], 0))
                ph, pidx = h, idx
        order = sorted(range(len(keys)), key=lbs.__getitem__)
        pos = [0] * len(keys)
        for r, i in enumerate(order):
            pos[i] = r
        # one pass suffices unless some ordering edge points backwards in planned time
        # (only zero-delay rotation cycles do); repeat until stable in that case
        backward = any(pos[pi] > pos[i] for i in range(len(keys)) for pi, _ in preds[i])
        E = list(lbs)
        for _ in range(200):
            changed = False
            for i in order:
                v = E[i]
                for pi, w in preds[i]:
                    c = E[pi] + w
                    if c > v:
                        v = c
                if v != E[i]:
                    E[i] = v
                    changed = True
            if not changed or not backward:
                break
        else:
            raise RuntimeError("retime did not converge: the visiting order has a positive cycle")
        out = dict(zip(keys, E))
        self._E_cache = (now, self.version, out)
        return out

    def exit_time(self, E, h: int, j: int, now: int) -> int:
        p = self.paths[h]
        if j + 1 < len(p):
            return E[(h, j + 1)]
        return (E[(h, j)] if j > self.progress[h] else now) + 1

    def table(self, E, now: int) -> TrainTable:
        """Reservation table of every active train's current cell and re-timed future (bulk-built)."""
        rt = TrainTable()
        iv = defaultdict(list)
        for h in self.active():
            p = self.paths[h]
            j0 = self.progress[h]
            owned = rt.owned[h]
            prev = None
            if j0 >= 0:
                c0 = p[j0][0]
                prev = (c0[0], c0[1])
                b = self.exit_time(E, h, j0, now) - 1
                iv[prev].append((now, b))
                owned.append(("i", prev, now, b, "c"))
            n = len(p)
            for j in range(j0 + 1, n):
                cfg = p[j][0]
                cell = (cfg[0], cfg[1])
                t = E[(h, j)]
                if prev is not None:
                    rt.moves.add((prev, cell, t))
                    owned.append(("m", prev, cell, t, "f"))
                b = E[(h, j + 1)] - 1 if j + 1 < n else t
                iv[cell].append((t, b))
                owned.append(("i", cell, t, b, "f"))
                prev = cell
        for cell, lst in iv.items():
            lst.sort()
            rt.intervals[cell] = lst
        return rt

    def live_table(self, now: int) -> TrainTable:
        """Cached table for the current step and plan version."""
        c = getattr(self, "_T_cache", None)
        if c is not None and c[0] == now and c[1] == self.version:
            return c[2]
        rt = self.table(self.retime(now), now)
        self._T_cache = (now, self.version, rt)
        return rt

    # ------------------------------------------------------------------ clearances
    def reroute_forbidden(self, h: int, max_decisions: int) -> Optional[FrozenSet[Tuple[Config, Config]]]:
        """The (switch, old branch) move to forbid for REROUTE, if a facing switch with a finite
        alternative lies within the train's next ``max_decisions`` decision cells."""
        p = self.paths.get(h)
        if p is None:
            return None
        j0 = max(self.progress[h], 0) if self.env.agents[h].state.is_on_map_state() else 0
        seen = 0
        for j in range(j0, len(p) - 1):
            cfg, nxt = p[j][0], p[j + 1][0]
            succ = [s for s in self.graph.successors(cfg) if math.isfinite(self.graph.distance(h, s))]
            if len(succ) > 1 and nxt in succ:
                return frozenset({(cfg, nxt)})
            if self.graph.is_decision_config(cfg):
                seen += 1
                if seen >= max_decisions:
                    break
        return None

    def at_decision(self, h: int) -> bool:
        a = self.env.agents[h]
        if a.state == TrainState.DONE or a.malfunction_handler.in_malfunction:
            return False
        if a.state.is_off_map_state():
            return a.state in (TrainState.READY_TO_DEPART, TrainState.MALFUNCTION_OFF_MAP) and self.env._elapsed_steps >= a.earliest_departure
        return a.speed_counter.is_cell_exit(a.speed_counter.max_speed)

    def _plan_one(self, h: int, rt: TrainTable, now: int, hold: int, forbidden) -> Optional[Path]:
        a = self.env.agents[h]
        info = self.infos[h]
        leave = self.first_move_time(h, now) + hold
        if a.state.is_off_map_state():
            return sipp_from(self.graph, rt, h, info.start, info.k, self.horizon, now, False, leave, forbidden)
        cur = to_config(a.current_configuration)
        return sipp_from(self.graph, rt, h, cur, info.k, self.horizon, now, True, leave, forbidden)

    def plan_clearance(self, c: Clearance, rt: TrainTable, now: int, max_decisions: int = 3) -> Optional[Dict[int, Path]]:
        """Speculatively plan a clearance against ``rt``. Returns {train: new future} or None.
        ``rt`` is left exactly as it was."""
        if c.kind == PROCEED:
            return {}
        if c.kind == HOLD:
            steps = [(c.train, self.hold_steps, frozenset())]
        elif c.kind == REROUTE:
            forb = self.reroute_forbidden(c.train, max_decisions)
            if forb is None:
                return None
            steps = [(c.train, 0, forb)]
        elif c.kind == YIELD_TO:
            if c.partner is None or c.partner == c.train or c.partner not in self.paths:
                return None
            steps = [(c.partner, 0, frozenset()), (c.train, 0, frozenset())]
        else:
            raise ValueError(c.kind)
        involved = [x for x, _, _ in steps]
        lifted = []
        for x in involved:
            lifted += rt.lift(x, tags=("f",))
        new: Dict[int, Path] = {}
        ok = True
        for x, hold, forb in steps:
            lifted += rt.lift(x, tags=("c",))
            path = self._plan_one(x, rt, now, hold, forb)
            if path is None:
                ok = False
                break
            a = self.env.agents[x]
            start = to_config(a.current_configuration) if a.state.is_on_map_state() else None
            if start is not None:
                # x holds its current cell until it leaves; the next train in `steps` must see that
                rt.reserve_for(x, (start[0], start[1]), now, path[0][1] - 1, "f")
            rt.reserve_future(x, start, path)
            new[x] = path
        for x in involved:
            rt.lift(x, tags=("f",))  # only the tentative futures; lifted entries come back below
        rt.restore(lifted)
        return new if ok else None

    def commit(self, new: Dict[int, Path], now: int) -> None:
        """Install new futures for the trains in ``new`` and re-time every plan."""
        if not new:
            return
        E = self.retime(now)
        self._write_times(E)
        for h, fut in new.items():
            p = self.paths[h]
            j0 = self.progress[h]
            for j in range(j0 + 1, len(p)):
                cfg, t = p[j]
                self.visits[(cfg[0], cfg[1])].remove((t, h, j))
            newp = p[: j0 + 1] + list(fut)
            self.paths[h] = newp
            self.base[h] = self._base_prefix(h, j0) + [t for _, t in fut]
            for j in range(j0 + 1, len(newp)):
                cfg, t = newp[j]
                bisect.insort(self.visits[(cfg[0], cfg[1])], (t, h, j))
        self.version += 1
        E2 = self.retime(now)
        self._write_times(E2)
        self.vptr = defaultdict(int)
        self.rt = self.table(E2, now)
        # the written times are E2 itself, so both caches stay valid for this step
        self._E_cache = (now, self.version, E2)
        self._T_cache = (now, self.version, self.table(E2, now))
        self.stats["commits"] += 1

    def _base_prefix(self, h: int, j0: int) -> List[int]:
        b = self.base.get(h) or [t for _, t in self.paths[h]]
        return list(b[: j0 + 1])

    def _write_times(self, E) -> None:
        """Rewrite every future visit's time to its re-timed value. The visiting order of every
        cell must be unchanged (re-timed entries are strictly increasing within a cell)."""
        for cell, lst in self.visits.items():
            before = [(h, j) for _, h, j in lst]
            new = sorted((E.get((h, j), t), h, j) for t, h, j in lst)
            assert [(h, j) for _, h, j in new] == before, f"re-timing reordered cell {cell}"
            self.visits[cell] = new
        for h in self.active():
            p = self.paths[h]
            for j in range(self.progress[h] + 1, len(p)):
                p[j] = (p[j][0], E[(h, j)])
        self.version += 1
