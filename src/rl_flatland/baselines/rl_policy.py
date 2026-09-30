"""Wrap a trained ActorCritic as a Policy so the shared evaluation harness can run it."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from flatland.envs.rail_env import RailEnv

from rl_flatland.baselines.ppo import ActorCritic
from rl_flatland.env import DecisionEnv
from rl_flatland.policies import Policy


class NetworkPolicy(Policy):
    """Greedy (argmax over unmasked meta-actions) execution of a trained actor."""

    def __init__(self, model_path: str | Path, obs: str = "compact", hidden: int = 128, name: str = "ppo", greedy: bool = True):
        self.name = name
        self.obs = obs
        self.greedy = greedy
        self.helper = DecisionEnv(obs)
        self.model = ActorCritic(self.helper.encoder.dim, hidden)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.rng = torch.Generator().manual_seed(0)

    def reset(self, env: RailEnv) -> None:
        self.helper.attach(env)
        self.graph = self.helper.graph
        self.setup_s = 0.0

    def act(self, env: RailEnv) -> Dict[int, int]:
        h = self.helper
        agents = h.decision_agents
        meta: Dict[int, int] = {}
        if agents:
            obs, masks = h.observations(), h.action_masks()
            x = torch.as_tensor(np.stack([obs[i] for i in agents]))
            m = torch.as_tensor(np.stack([masks[i] for i in agents]))
            with torch.no_grad():
                logits, _ = self.model(x, m)
                if self.greedy:
                    a = logits.argmax(-1)
                else:
                    a = torch.multinomial(torch.softmax(logits, -1), 1, generator=self.rng).squeeze(-1)
            meta = {i: int(a[k]) for k, i in enumerate(agents)}
        return {i: h.native_action(i, meta.get(i)) for i in range(h.n)}

    def observe(self, env: RailEnv) -> None:
        self.helper._refresh()
