"""Episode metrics shared by every baseline.

* ``arrival_rate``: fraction of trains in state DONE when the episode ends (Flatland's
  "% done" / "arrival ratio").
* ``normalized_reward``: the official Flatland 3 normalisation of the environment's own
  ``DefaultRewards``, ``1 + sum_i max(R_i, -T) / (N * T)``, computed by
  ``env.rewards.normalize``. 1.0 means every train arrived on time.
* ``deadlocked``: number of trains in a deadlock (see deadlock.py) at the end of the episode.
* ``decision_ms_mean`` / ``decision_ms_max``: wall-clock milliseconds the policy spent choosing the
  joint action per environment step, excluding the environment's own step time.
* ``setup_s``: one-off policy time before the first step (for example the OR planner's initial
  plan), in seconds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.step_utils.states import TrainState


@dataclass
class EpisodeResult:
    scenario: str
    split: str
    seed: int
    policy: str
    n_agents: int
    max_steps: int
    steps: int
    arrived: int
    arrival_rate: float
    normalized_reward: float
    deadlocked: int
    n_malfunctions: int
    setup_s: float
    decision_ms_mean: float
    decision_ms_max: float
    env_ms_mean: float

    def to_dict(self) -> dict:
        return asdict(self)


def normalized_reward(env: RailEnv, cumulative: Dict[int, float]) -> float:
    rewards: List[float] = [float(cumulative.get(i, 0.0)) for i in range(env.get_num_agents())]
    return float(env.rewards.normalize(*rewards, num_agents=env.get_num_agents(), max_episode_steps=env._max_episode_steps))


def count_arrived(env: RailEnv) -> int:
    return int(sum(a.state == TrainState.DONE for a in env.agents))


def summarize(results: List[EpisodeResult]) -> dict:
    """Mean and standard error over episodes, for the result tables."""
    if not results:
        return {}
    keys = ["arrival_rate", "normalized_reward", "deadlocked", "decision_ms_mean", "setup_s"]
    out = {"episodes": len(results)}
    for k in keys:
        v = np.array([getattr(r, k) for r in results], dtype=float)
        out[k] = float(v.mean())
        out[k + "_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return out
