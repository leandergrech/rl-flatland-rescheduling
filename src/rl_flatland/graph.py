"""Rail-network helpers shared by the planner, the observation and the action wrapper.

Flatland describes a train's location as a *configuration* ``((row, col), heading)``, where heading
is 0=N, 1=E, 2=S, 3=W and is the direction the train travelled to enter the cell. Here we flatten
that to a tuple ``(row, col, heading)`` for hashing speed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from flatland.envs.rail_env import RailEnv

Config = Tuple[int, int, int]  # (row, col, heading)

MOVE = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

# Native flatland actions as plain ints (RailEnvActions is a fastenum and does not cast to int).
DO_NOTHING, MOVE_LEFT, MOVE_FORWARD, MOVE_RIGHT, STOP_MOVING = 0, 1, 2, 3, 4


def to_config(configuration) -> Optional[Config]:
    """Convert a flatland configuration ``((r, c), d)`` (or None) to ``(r, c, d)``."""
    if configuration is None or configuration[0] is None:
        return None
    (r, c), d = configuration
    return (int(r), int(c), int(d))


def action_between(cur: Config, nxt: Config) -> int:
    """Native action that moves a train from configuration ``cur`` to successor ``nxt``.

    The successor heading equals the direction of travel. Heading - 1 is a left turn and heading + 1
    a right turn. Straight moves and dead-end reversals use MOVE_FORWARD, which flatland also
    accepts on plain curves.
    """
    d0, d1 = cur[2], nxt[2]
    if d1 == (d0 - 1) % 4:
        return MOVE_LEFT
    if d1 == (d0 + 1) % 4:
        return MOVE_RIGHT
    return MOVE_FORWARD


class RailGraph:
    """Successor lists, switch flags and per-agent distance-to-target lookups for one env reset."""

    def __init__(self, env: RailEnv):
        self.env = env
        self.height, self.width = env.height, env.width
        rail = env.rail
        self._succ: Dict[Config, List[Config]] = {}
        # A cell is a "switch" if some heading has more than one exit, or if more than two headings
        # can enter it (merging switch or crossing). Trains only interact on such cells or on the
        # single-track segments between them.
        self.is_switch = np.zeros((self.height, self.width), dtype=bool)
        self.is_rail = np.zeros((self.height, self.width), dtype=bool)
        for r in range(self.height):
            for c in range(self.width):
                n_exits_total = 0
                entries = 0
                any_multi = False
                for d in range(4):
                    trans = rail.get_transitions(((r, c), d))
                    k = int(sum(1 for t in trans if t))
                    if k > 0:
                        entries += 1
                        n_exits_total += k
                    if k > 1:
                        any_multi = True
                if entries > 0:
                    self.is_rail[r, c] = True
                if any_multi or entries > 2:
                    self.is_switch[r, c] = True
        # Distance-to-target per agent: shape (n_agents, h, w, 4), inf where unreachable.
        self.dist = env.distance_map.get()

    def successors(self, cfg: Config) -> List[Config]:
        s = self._succ.get(cfg)
        if s is None:
            out = self.env.rail.get_successor_configurations(((cfg[0], cfg[1]), cfg[2]))
            s = [to_config(o) for o in out]
            self._succ[cfg] = s
        return s

    def distance(self, handle: int, cfg: Config) -> float:
        return float(self.dist[handle, cfg[0], cfg[1], cfg[2]])

    def ranked_successors(self, handle: int, cfg: Config) -> List[Tuple[Config, float]]:
        """Successors of ``cfg`` sorted by the agent's remaining distance to target."""
        out = [(s, self.distance(handle, s)) for s in self.successors(cfg)]
        out.sort(key=lambda x: x[1])
        return out

    def is_decision_config(self, cfg: Config) -> bool:
        """True where a train should decide rather than just continue.

        That is at a diverging switch (more than one successor) or on the cell right before a switch
        cell (the last place to stop before entering a shared resource).
        """
        succ = self.successors(cfg)
        if len(succ) > 1:
            return True
        return any(self.is_switch[s[0], s[1]] for s in succ)

    def shortest_path(self, handle: int, start: Config, max_len: int = 10_000) -> List[Config]:
        """Greedy walk along decreasing distance to the agent's target, starting at ``start``."""
        path = [start]
        cur = start
        for _ in range(max_len):
            if self.distance(handle, cur) == 0:
                break
            ranked = self.ranked_successors(handle, cur)
            if not ranked or not np.isfinite(ranked[0][1]):
                break
            cur = ranked[0][0]
            path.append(cur)
        return path


@lru_cache(maxsize=None)
def opposite(d: int) -> int:
    return (d + 2) % 4
