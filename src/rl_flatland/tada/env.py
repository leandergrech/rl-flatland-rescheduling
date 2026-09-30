"""The dispatcher's environment: the OR executor steps the world, the policy edits plans at decision points.

One *decision point* is an env step at which the context window holds at least one choosable
train. There the policy issues up to B clearances, chosen one at a time (train, action, partner,
continue?). Every clearance is planned against the live reservation table and committed only if
the plan exists, so no applied clearance can create a head-on or a swap. Then every train follows
its (possibly edited) plan through the env step, and the loop runs on to the next decision point.

Episode ends:
* **termination** (terminal value 0): a deadlock (``rl_flatland.deadlock.find_deadlocked``) or a
  reservation violation (two trains in one cell; asserted by flatland, logged here). With a correct
  executor neither should ever happen.
* **truncation** (bootstrap): the horizon T.

Reward per env step is Flatland 3's ``DefaultRewards`` summed over trains, scaled by
100 / (N * T), so an episode's return is 100 x (normalised reward - 1) up to the per-train cap.
Optional shaping adds the change in total slack of the window members over the step, same scale.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from flatland.envs.step_utils.states import TrainState

from rl_flatland.deadlock import find_deadlocked
from rl_flatland.metrics import count_arrived, normalized_reward
from rl_flatland.scenarios import make_env
from rl_flatland.tada.executor import HOLD, N_ACTIONS, PROCEED, REROUTE, YIELD_TO, Clearance, TadaExecutor
from rl_flatland.tada.window import Window, WindowConfig, build_window


@dataclass
class DispatchConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    budget: int = 2  # B: clearances per decision point
    shaping: bool = False
    hold_steps: int = 5
    exact_action_masks: bool = True
    reward_scale: float = 100.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "DispatchConfig":
        d = dict(d)
        d["window"] = WindowConfig(**d["window"])
        return DispatchConfig(**d)


@dataclass
class StepResult:
    reward: float  # discounted sum over the env steps advanced
    steps: int  # env steps advanced
    terminated: bool
    truncated: bool


class DispatchEnv:
    def __init__(self, cfg: DispatchConfig = DispatchConfig(), gamma: float = 0.99, executor_cls=TadaExecutor):
        self.cfg = cfg
        self.gamma = gamma
        self.executor_cls = executor_cls

    # ------------------------------------------------------------------ episode
    def reset(self, scenario: str, seed: int, malfunctions: bool, env=None) -> Tuple[Optional[Window], StepResult]:
        self.env = env if env is not None else make_env(scenario, seed, malfunctions=malfunctions)
        self.scenario, self.seed, self.malfunctions = scenario, seed, malfunctions
        self.ex = self.executor_cls(hold_steps=self.cfg.hold_steps)
        self.ex.reset(self.env)
        self.cum = {i: 0.0 for i in range(self.env.get_num_agents())}
        self.stats = defaultdict(float)
        self.occupancy: List[int] = []
        self.done = False
        self.terminated = False
        self.truncated = False
        self.decision_ms: List[float] = []
        self.window: Optional[Window] = None
        self._clr_this_point = 0
        self._at_point = False
        return self.advance()

    def _norm(self) -> float:
        return 1.0 / (self.env.get_num_agents() * self.env._max_episode_steps)

    def _window_slack(self, trains) -> float:
        env, ex = self.env, self.ex
        now = env._elapsed_steps
        s = 0.0
        for h in trains:
            a = env.agents[h]
            if a.state == TrainState.DONE:
                continue
            info = ex.infos[h]
            cur = tuple(a.current_configuration[0]) + (a.current_configuration[1],) if a.state.is_on_map_state() else info.start
            d = ex.graph.distance(h, cur)
            if np.isfinite(d):
                s += a.latest_arrival - (now + info.k * d)
        return s

    def advance(self) -> Tuple[Optional[Window], StepResult]:
        """Step with default actions until a decision point or the end of the episode."""
        total, disc, steps = 0.0, 1.0, 0
        env, ex = self.env, self.ex
        if getattr(self, "_at_point", False) and not self.done:
            # the policy has acted at this decision point: take the env step before looking again
            self._at_point = False
            total += self._env_step()
            disc *= self.gamma
            steps += 1
        while True:
            if not self.done:
                w = build_window(ex, self.cfg.window)
                self.occupancy.append(len(w.trains))
                self.stats["window_sum"] += len(w.trains)
                self.stats["window_steps"] += 1
                if w.choosable.sum() > 0 and self.cfg.budget > 0:
                    self.window = w
                    self._clr_this_point = 0
                    self._point_members = list(w.trains)
                    self._at_point = True
                    return w, StepResult(total, steps, False, False)
            else:
                return None, StepResult(total, steps, self.terminated, self.truncated)
            r = self._env_step()
            total += disc * r
            disc *= self.gamma
            steps += 1

    def _env_step(self) -> float:
        env, ex = self.env, self.ex
        members = getattr(self, "_point_members", [])
        s0 = self._window_slack(members) if self.cfg.shaping else 0.0
        t0 = time.perf_counter()
        actions = ex.act(env)
        self.decision_ms.append(1000 * (time.perf_counter() - t0))
        try:
            _, rewards, dones, _ = env.step(actions)
        except AssertionError:
            self.stats["reservation_violations"] += 1
            self.done = self.terminated = True
            return 0.0
        ex.observe(env)
        r = 0.0
        for i, v in rewards.items():
            self.cum[i] += float(v)
            r += float(v)
        r *= self.cfg.reward_scale * self._norm()
        if self.cfg.shaping:
            r += self.cfg.reward_scale * self._norm() * (self._window_slack(members) - s0)
        if find_deadlocked(env, ex.graph):
            self.stats["deadlock_terminations"] += 1
            self.done = self.terminated = True
        elif dones["__all__"]:
            self.done = self.truncated = True
        self._point_members = []
        return r

    # ------------------------------------------------------------------ clearances
    def action_mask(self, slot: int) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """Legal actions for the train in window slot ``slot`` and, for YIELD_TO, legal partners.
        With exact_action_masks every non-default action is planned speculatively first."""
        w = self.window
        h = w.trains[slot]
        mask = w.struct_actions[slot].copy()
        partner_mask = np.zeros(len(w.mask), np.float32)
        if not self.cfg.exact_action_masks:
            for j in w.partners_after.get(slot, []):
                partner_mask[j] = 1
            return mask, {YIELD_TO: partner_mask}
        now = self.env._elapsed_steps
        rt = self.ex.live_table(now)
        for kind in (HOLD, REROUTE):
            if mask[kind] and self.ex.plan_clearance(Clearance(h, kind), rt, now, self.cfg.window.D) is None:
                mask[kind] = 0
        if mask[YIELD_TO]:
            for j in w.partners_after.get(slot, []):
                if self.ex.plan_clearance(Clearance(h, YIELD_TO, w.trains[j]), rt, now, self.cfg.window.D) is not None:
                    partner_mask[j] = 1
            if partner_mask.sum() == 0:
                mask[YIELD_TO] = 0
        return mask, {YIELD_TO: partner_mask}

    def apply(self, slot: int, kind: int, partner_slot: Optional[int] = None) -> bool:
        """Plan and commit one clearance. Returns True if committed (PROCEED always 'succeeds')."""
        w = self.window
        h = w.trains[slot]
        self._clr_this_point += 1
        self.stats[f"clr_{kind}"] += 1
        if kind == PROCEED:
            return True
        partner = w.trains[partner_slot] if (kind == YIELD_TO and partner_slot is not None) else None
        now = self.env._elapsed_steps
        rt = self.ex.live_table(now)
        new = self.ex.plan_clearance(Clearance(h, kind, partner), rt, now, self.cfg.window.D)
        if new is None:
            self.stats["rejected"] += 1
            return False
        self.ex.commit(new, now)
        self.stats["commits"] += 1
        return True

    def can_continue(self) -> bool:
        return self._clr_this_point < self.cfg.budget

    # ------------------------------------------------------------------ metrics
    def summary(self) -> dict:
        env = self.env
        n = env.get_num_agents()
        arrived = count_arrived(env)
        return dict(
            scenario=self.scenario,
            seed=self.seed,
            malfunctions=self.malfunctions,
            n_agents=n,
            steps=int(env._elapsed_steps),
            max_steps=int(env._max_episode_steps),
            arrived=arrived,
            arrival_rate=arrived / n,
            normalized_reward=normalized_reward(env, self.cum),
            deadlocked=len(find_deadlocked(env, self.ex.graph)),
            terminated=bool(self.terminated),
            truncated=bool(self.truncated),
            deadlock_terminations=int(self.stats["deadlock_terminations"]),
            reservation_violations=int(self.stats["reservation_violations"]),
            deviations=int(self.ex.deviations),
            commits=int(self.stats["commits"]),
            rejected=int(self.stats["rejected"]),
            clearances={k: int(v) for k, v in self.stats.items() if k.startswith("clr_")},
            window_mean=self.stats["window_sum"] / max(1, self.stats["window_steps"]),
            executor_ms_mean=float(np.mean(self.decision_ms)) if self.decision_ms else 0.0,
        )
