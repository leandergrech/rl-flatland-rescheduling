"""Common policy interface plus two trivial references (random, shortest path)."""

from __future__ import annotations

from typing import Dict

import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.step_utils.states import TrainState

from rl_flatland.graph import MOVE_FORWARD, RailGraph, action_between, to_config


class Policy:
    """A joint policy: sees the whole RailEnv, returns one native action per train."""

    name = "policy"

    def reset(self, env: RailEnv) -> None:  # called after env.reset(), before the first step
        pass

    def act(self, env: RailEnv) -> Dict[int, int]:
        raise NotImplementedError

    def observe(self, env: RailEnv) -> None:  # called after every env.step()
        pass


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env: RailEnv) -> Dict[int, int]:
        return {i: int(self.rng.integers(0, 5)) for i in range(env.get_num_agents())}


class ShortestPathPolicy(Policy):
    """Every train departs as soon as allowed and follows its shortest path; no coordination.

    This is the "Shortest Path" reference of Mohanty et al. (2020): it shows how many trains arrive
    when nobody yields.
    """

    name = "shortest_path"

    def reset(self, env: RailEnv) -> None:
        self.graph = RailGraph(env)

    def act(self, env: RailEnv) -> Dict[int, int]:
        actions = {}
        for a in env.agents:
            if a.state == TrainState.DONE:
                continue
            if a.state.is_off_map_state():
                actions[a.handle] = MOVE_FORWARD
                continue
            cfg = to_config(a.current_configuration)
            ranked = self.graph.ranked_successors(a.handle, cfg)
            if ranked and np.isfinite(ranked[0][1]):
                actions[a.handle] = (action_between(cfg, ranked[0][0]))
            else:
                actions[a.handle] = MOVE_FORWARD
        return actions
