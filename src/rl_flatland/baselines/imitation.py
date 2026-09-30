"""Imitation-then-RL baseline: behaviour cloning on OR rollouts, then PPO fine-tuning.

1. **Demonstrations.** Run the OR planner (PrioritizedPlannerPolicy) on training seeds. At every
   step, each train that the RL wrapper would ask for a decision gets a label in the same 3-action
   space: WAIT if the planner holds it (off-map or STOP_MOVING), GO_BEST / GO_ALT depending on
   which branch the planner's next configuration lies on. The env is driven by the planner's own
   actions (plain behaviour cloning, no DAgger). Off-map WAIT labels are kept with probability
   ``keep_offmap_wait`` to limit their share of the data.
2. **Behaviour cloning.** Cross-entropy on masked logits.
3. **PPO fine-tuning** from the cloned weights, with the BC loss on the demonstrations added and
   annealed to zero by half-way (``PPOConfig.bc_coef_start``), so the policy starts from the
   planner's behaviour but may leave it where the shaped reward says so.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from rl_flatland.baselines.or_planner import PrioritizedPlannerPolicy
from rl_flatland.baselines.ppo import ActorCritic
from rl_flatland.env import GO_ALT, GO_BEST, N_META_ACTIONS, WAIT, DecisionEnv
from rl_flatland.graph import STOP_MOVING, DO_NOTHING
from rl_flatland.observations import branch_options
from rl_flatland.scenarios import make_env


def _demo_episode(args) -> dict:
    scenario, seed, malfunctions, obs_name, keep_offmap_wait = args
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed + 7919)
    helper = DecisionEnv(obs_name)
    env = make_env(scenario, seed, malfunctions=malfunctions, obs_builder=helper.encoder.builder())
    planner = PrioritizedPlannerPolicy()
    planner.reset(env)
    helper.attach(env)
    X, M, A = [], [], []
    done = {"__all__": False}
    while not done["__all__"]:
        actions = planner.act(env)
        agents = helper.decision_agents
        if agents:
            obs, masks = helper.observations(), helper.action_masks()
            for i in agents:
                a = env.agents[i]
                native = actions.get(i, DO_NOTHING)
                if a.state.is_off_map_state():
                    label = WAIT if native == DO_NOTHING else GO_BEST
                    if label == WAIT and rng.random() > keep_offmap_wait:
                        continue
                elif native == STOP_MOVING:
                    label = WAIT
                else:
                    nxt = planner.planned_next(i)
                    opts = branch_options(helper.graph, i, helper._cfg(i))
                    label = GO_ALT if (len(opts) > 1 and nxt == opts[1][0]) else GO_BEST
                X.append(obs[i])
                M.append(masks[i])
                A.append(label)
        _, _, done, _ = env.step(actions)
        planner.observe(env)
        helper._refresh()
    arrived = sum(a.state.value == 6 for a in env.agents) / env.get_num_agents()
    return dict(
        obs=np.array(X, np.float32).reshape(-1, helper.encoder.dim),
        mask=np.array(M, np.float32).reshape(-1, N_META_ACTIONS),
        a=np.array(A, np.int64),
        scenario=scenario,
        seed=seed,
        arrival_rate=arrived,
    )


def collect_demos(
    episodes: List[Tuple[str, int, bool]],
    obs_name: str = "compact",
    keep_offmap_wait: float = 0.2,
    workers: int = 8,
) -> dict:
    jobs = [(sc, seed, malf, obs_name, keep_offmap_wait) for sc, seed, malf in episodes]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_demo_episode, jobs))
    return dict(
        obs=np.concatenate([r["obs"] for r in res]),
        mask=np.concatenate([r["mask"] for r in res]),
        a=np.concatenate([r["a"] for r in res]),
        episode_scenario=np.array([r["scenario"] for r in res]),
        episode_seed=np.array([r["seed"] for r in res]),
        episode_arrival=np.array([r["arrival_rate"] for r in res]),
        episode_len=np.array([len(r["a"]) for r in res]),
    )


def save_demos(demos: dict, path: str | Path) -> None:
    """float16 observations, compressed. Kept small enough to commit (<20 MB for the repo)."""
    d = dict(demos)
    d["obs"] = d["obs"].astype(np.float16)
    d["mask"] = d["mask"].astype(np.uint8)
    d["a"] = d["a"].astype(np.uint8)
    np.savez_compressed(path, **d)


def load_demos(path: str | Path) -> dict:
    z = np.load(path)
    return dict(obs=z["obs"].astype(np.float32), mask=z["mask"].astype(np.float32), a=z["a"].astype(np.int64))


def behaviour_cloning(
    demos: dict, obs_dim: int, hidden: int = 128, epochs: int = 15, lr: float = 1e-3, batch: int = 1024, seed: int = 0
) -> Tuple[ActorCritic, List[dict]]:
    torch.manual_seed(seed)
    model = ActorCritic(obs_dim, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.as_tensor(demos["obs"])
    M = torch.as_tensor(demos["mask"])
    A = torch.as_tensor(demos["a"])
    n = len(A)
    idx = torch.randperm(n)
    n_val = max(1, n // 10)
    val, tr = idx[:n_val], idx[n_val:]
    log = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        perm = tr[torch.randperm(len(tr))]
        losses = []
        for s in range(0, len(perm), batch):
            j = perm[s : s + batch]
            logits, _ = model(X[j], M[j])
            loss = F.cross_entropy(logits, A[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        with torch.no_grad():
            vl, _ = model(X[val], M[val])
            val_loss = float(F.cross_entropy(vl, A[val]))
            val_acc = float((vl.argmax(-1) == A[val]).float().mean())
        log.append(dict(epoch=ep, train_loss=float(np.mean(losses)), val_loss=val_loss, val_acc=val_acc, wall_s=round(time.perf_counter() - t0, 1)))
        print(f"[bc] epoch={ep} train={log[-1]['train_loss']:.4f} val={val_loss:.4f} acc={val_acc:.3f}", flush=True)
    return model, log
