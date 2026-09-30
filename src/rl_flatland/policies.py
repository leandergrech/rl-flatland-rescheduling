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


class ReactiveAvoidPolicy(Policy):
    """Rule-based reference on the RL wrapper's decisions and compact features.

    At each decision point: take the shortest-path branch if no opposing train is within the
    30-cell look-ahead, else the alternative branch if that one is clear, else wait. Off-map
    trains depart only if the look-ahead from their start cell is clear. It shows how far the
    compact observation gets you without learning, like the flatland-baselines deadlock-avoidance
    heuristic but on our features.
    """

    name = "reactive_avoid"

    def reset(self, env: RailEnv) -> None:
        from rl_flatland.env import DecisionEnv

        self.helper = DecisionEnv("compact")
        self.helper.attach(env)
        self.graph = self.helper.graph
        self.setup_s = 0.0

    def act(self, env: RailEnv) -> Dict[int, int]:
        from rl_flatland.env import GO_ALT, GO_BEST, WAIT

        h = self.helper
        meta = {}
        if h.decision_agents:
            obs, masks = h.observations(), h.action_masks()
            for i in h.decision_agents:
                o = obs[i]
                # compact layout: own 0-11; best branch 12-19 and alternative 20-27, each
                # [extra distance, exists, dist to opposing, n opposing, dist same-dir, first cell occupied, opposing broken, segment length]
                best_clear = o[13] > 0 and o[14] >= 1.0 and o[17] < 0.5
                alt_clear = masks[i][2] > 0 and o[21] > 0 and o[22] >= 1.0 and o[25] < 0.5
                meta[i] = GO_BEST if best_clear else (GO_ALT if alt_clear else WAIT)
        return {i: h.native_action(i, meta.get(i)) for i in range(h.n)}

    def observe(self, env: RailEnv) -> None:
        self.helper._refresh()
