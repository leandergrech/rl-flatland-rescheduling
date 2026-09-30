"""Multi-agent RL wrapper around flatland's RailEnv.

Design choices (discussed in docs/04-designs.md):

* **Decisions only where they matter.** A train asks the policy for an action only when it is
  ready to depart, or when it stands at the exit of a cell that is either a facing switch or the
  last cell before a switch. Everywhere else it keeps rolling along its only successor. This
  "decision-cell masking" was used by nearly every RL entry in 2020 (Laurent et al. 2021, §3).
* **Three meta-actions.** 0 = WAIT (stay off the map, or STOP_MOVING), 1 = GO along the branch
  with the shortest remaining distance, 2 = GO along the alternative branch. Action 2 is masked
  when there is only one branch. The wrapper maps each meta-action to the native action.
* **Shaped reward per train (JBR_HSE, Laurent et al. 2021, §4.2):**
  ``0.01 * (distance decrease) - 5 * [becomes deadlocked] + 10 * [arrives]``.
  The environment's own normalised reward is still computed for evaluation.
* **Semi-MDP bookkeeping is left to the trainer.** ``step`` returns the reward every train earned
  in this env step. The trainer accumulates it between that train's decisions with discount
  gamma^dt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.step_utils.states import TrainState

from rl_flatland.deadlock import find_deadlocked
from rl_flatland.graph import DO_NOTHING, MOVE_FORWARD, STOP_MOVING, Config, RailGraph, action_between, to_config
from rl_flatland.metrics import count_arrived, normalized_reward
from rl_flatland.observations import branch_options, make_obs
from rl_flatland.scenarios import make_env

WAIT, GO_BEST, GO_ALT = 0, 1, 2
N_META_ACTIONS = 3


@dataclass
class Shaping:
    progress: float = 0.01
    deadlock: float = -5.0
    arrival: float = 10.0


class DecisionEnv:
    """Train-level decision interface. One instance can be reset on many scenarios and seeds."""

    def __init__(self, obs: str = "compact", shaping: Shaping = Shaping()):
        self.encoder = make_obs(obs)
        self.shaping = shaping
        self.env: Optional[RailEnv] = None

    # ------------------------------------------------------------------ episode control
    def reset(self, scenario: str, seed: int, malfunctions: bool = False) -> None:
        self.env = make_env(scenario, seed, malfunctions=malfunctions, obs_builder=self.encoder.builder())
        self.attach(self.env)

    def attach(self, env: RailEnv) -> None:
        """Use an already-reset RailEnv (for evaluation through the Policy interface)."""
        self.env = env
        self.graph = RailGraph(env)
        self.n = env.get_num_agents()
        self.dist_prev = np.array([self._distance(i) for i in range(self.n)], dtype=float)
        self.deadlocked: Set[int] = set()
        self.cumulative_env_reward = {i: 0.0 for i in range(self.n)}
        self.finished = np.zeros(self.n, dtype=bool)  # arrived or deadlocked
        self._refresh()

    def _cfg(self, i: int) -> Config:
        a = self.env.agents[i]
        c = to_config(a.current_configuration)
        return c if c is not None else to_config(a.initial_configuration)

    def _distance(self, i: int) -> float:
        a = self.env.agents[i]
        if a.state == TrainState.DONE:
            return 0.0
        d = self.graph.distance(i, self._cfg(i))
        return d if np.isfinite(d) else float(self.env.width + self.env.height)

    # ------------------------------------------------------------------ decisions
    def _needs_decision(self, i: int) -> bool:
        a = self.env.agents[i]
        if self.finished[i] or a.state == TrainState.DONE:
            return False
        if a.malfunction_handler.in_malfunction:
            return False
        t = self.env._elapsed_steps
        if a.state.is_off_map_state():
            return t >= a.earliest_departure and a.state in (TrainState.READY_TO_DEPART, TrainState.MALFUNCTION_OFF_MAP)
        if not a.speed_counter.is_cell_exit(a.speed_counter.max_speed):
            return False
        return self.graph.is_decision_config(self._cfg(i))

    def _refresh(self) -> None:
        self.decision_agents: List[int] = [i for i in range(self.n) if self._needs_decision(i)]
        self.encoder.prepare(self.env, self.graph)

    def observations(self) -> Dict[int, np.ndarray]:
        return {i: self.encoder.get(i, self._cfg(i)) for i in self.decision_agents}

    def action_masks(self) -> Dict[int, np.ndarray]:
        out = {}
        for i in self.decision_agents:
            a = self.env.agents[i]
            has_alt = (not a.state.is_off_map_state()) and len(branch_options(self.graph, i, self._cfg(i))) > 1
            out[i] = np.array([1.0, 1.0, float(has_alt)], dtype=np.float32)
        return out

    def native_action(self, i: int, meta: Optional[int]) -> int:
        """Native flatland action for train i given its meta-action (None = not deciding)."""
        a = self.env.agents[i]
        if a.state == TrainState.DONE:
            return DO_NOTHING
        if a.state.is_off_map_state():
            return MOVE_FORWARD if meta in (GO_BEST, GO_ALT) else DO_NOTHING
        cfg = self._cfg(i)
        opts = branch_options(self.graph, i, cfg)
        if meta is None:
            succ = self.graph.successors(cfg)
            if not succ:
                return MOVE_FORWARD
            target = opts[0][0] if opts else succ[0]
            return action_between(cfg, target)
        if meta == WAIT:
            return STOP_MOVING
        if not opts:
            return MOVE_FORWARD
        j = 1 if (meta == GO_ALT and len(opts) > 1) else 0
        return action_between(cfg, opts[j][0])

    # ------------------------------------------------------------------ stepping
    def step(self, meta_actions: Dict[int, int]) -> Tuple[np.ndarray, Dict[int, str], bool]:
        """Advance one env step.

        Returns (per-train shaped reward for this step, {train: 'arrived'|'deadlocked'} for trains
        that finished in this step, episode_over).
        """
        env = self.env
        actions = {}
        for i in range(self.n):
            actions[i] = self.native_action(i, meta_actions.get(i))
        _, env_rewards, dones, _ = env.step(actions)
        for i, r in env_rewards.items():
            self.cumulative_env_reward[i] += float(r)

        r = np.zeros(self.n, dtype=np.float32)
        events: Dict[int, str] = {}
        dist_now = np.array([self._distance(i) for i in range(self.n)], dtype=float)
        progress = np.where(self.finished, 0.0, self.dist_prev - dist_now)
        r += self.shaping.progress * progress.astype(np.float32)
        self.dist_prev = dist_now
        for i in range(self.n):
            if not self.finished[i] and env.agents[i].state == TrainState.DONE:
                r[i] += self.shaping.arrival
                self.finished[i] = True
                events[i] = "arrived"
        dl = find_deadlocked(env, self.graph)
        for i in dl - self.deadlocked:
            if not self.finished[i]:
                r[i] += self.shaping.deadlock
                self.finished[i] = True
                events[i] = "deadlocked"
        self.deadlocked |= dl
        over = bool(dones["__all__"]) or bool(self.finished.all())
        self._refresh()
        return r, events, over

    # ------------------------------------------------------------------ metrics
    def arrival_rate(self) -> float:
        return count_arrived(self.env) / self.n

    def normalized_reward(self) -> float:
        return normalized_reward(self.env, self.cumulative_env_reward)
