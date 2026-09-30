"""Deadlock detection.

A train is *deadlocked* when every cell it could move into is held by a train that is itself
deadlocked. The simplest case is two trains facing each other on a single-track segment: each
one's only successor cell holds the other. Flatland never resolves a swap, so such trains stay put
until the episode ends.

We first compute the greatest fixed point S: start from all on-map trains whose every successor
cell is occupied, then repeatedly drop any train that has a successor cell held by a train outside
S. What remains is a union of blocking cycles plus the queues behind them. A train stuck behind a
malfunctioning train is not in S, because the broken train will eventually move on.

Not every cycle in S is a deadlock. flatland-rl 4.3.0's MotionCheck blocks swaps (two trains
entering each other's cells) but lets a ring of three or more trains rotate in one step. So the
deadlock cores are head-on pairs (each train's cell is a successor of the other), and a train in S
is deadlocked if every successor cell holds a core member or another deadlocked train.
"""

from __future__ import annotations

from typing import Dict, Set, Tuple

from flatland.envs.rail_env import RailEnv
from flatland.envs.step_utils.states import TrainState

from rl_flatland.graph import Config, RailGraph, to_config


def find_deadlocked(env: RailEnv, graph: RailGraph) -> Set[int]:
    cell_holder: Dict[Tuple[int, int], int] = {}
    cfgs: Dict[int, Config] = {}
    for a in env.agents:
        if a.state.is_on_map_state() and a.state != TrainState.DONE:
            cfg = to_config(a.current_configuration)
            if cfg is None:
                continue
            cfgs[a.handle] = cfg
            cell_holder[(cfg[0], cfg[1])] = a.handle

    candidates: Set[int] = set()
    for h, cfg in cfgs.items():
        succ = graph.successors(cfg)
        if succ and all((s[0], s[1]) in cell_holder for s in succ):
            candidates.add(h)

    changed = True
    while changed:
        changed = False
        for h in list(candidates):
            for s in graph.successors(cfgs[h]):
                if cell_holder[(s[0], s[1])] not in candidates:
                    candidates.discard(h)
                    changed = True
                    break

    def succ_cells(h: int) -> Set[Tuple[int, int]]:
        return {(s[0], s[1]) for s in graph.successors(cfgs[h])}

    deadlocked: Set[int] = set()
    for h in candidates:
        cell = (cfgs[h][0], cfgs[h][1])
        for c in succ_cells(h):
            other = cell_holder[c]
            if cell in succ_cells(other):
                deadlocked.add(h)
                break
    changed = True
    while changed:
        changed = False
        for h in candidates - deadlocked:
            if all(cell_holder[c] in deadlocked for c in succ_cells(h)):
                deadlocked.add(h)
                changed = True
    return deadlocked
