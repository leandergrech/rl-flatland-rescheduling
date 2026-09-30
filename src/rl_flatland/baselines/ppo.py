"""Parameter-shared PPO for train-level decisions, sized for a laptop CPU.

One small actor-critic is shared by every train. Each train is its own trajectory in a semi-MDP:
it acts only at decision points (see env.DecisionEnv), and the time between two of its decisions,
``tau``, varies. Rewards earned between decisions are discounted inside the transition, and GAE
uses ``gamma**tau`` and ``(gamma*lambda)**tau``:

    delta_j = R_j + gamma**tau_j * V(s_{j+1}) * (1 - done_j) - V(s_j)
    A_j     = delta_j + (gamma*lambda)**tau_j * (1 - done_j) * A_{j+1}

Rollouts run in worker processes (one episode per job, one torch thread each). Why not
Stable-Baselines3: SB3 assumes one agent per env step with a fixed step length. Here the number
of acting trains changes every step and each train's step length is variable.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_flatland.env import N_META_ACTIONS, DecisionEnv
from rl_flatland.observations import make_obs


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128, n_actions: int = N_META_ACTIONS):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        logits = self.pi(h).masked_fill(mask < 0.5, -1e9)
        return logits, self.v(h).squeeze(-1)


@dataclass
class PPOConfig:
    obs: str = "compact"
    hidden: int = 128
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 2048
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    episodes_per_iter: int = 16
    iterations: int = 60
    time_budget_s: float = 1800.0
    workers: int = 8
    # scenario -> sampling weight for training episodes
    train_mix: Dict[str, float] = field(default_factory=lambda: {"small": 0.5, "medium": 0.5})
    malfunction_prob: float = 0.5
    seed: int = 0
    # imitation regulariser (used by the imitation-then-PPO baseline)
    bc_coef_start: float = 0.0
    # iterations at the start that train only the value head (policy frozen)
    value_warmup_iters: int = 0


def _collect_episode(args) -> dict:
    """Worker: roll out one training episode with the given weights; return flat GAE batch."""
    state_dict, cfg_d, scenario, seed, malfunctions = args
    torch.set_num_threads(1)
    cfg = PPOConfig(**cfg_d)
    denv = DecisionEnv(cfg.obs)
    denv.reset(scenario, seed, malfunctions)
    model = ActorCritic(denv.encoder.dim, cfg.hidden)
    model.load_state_dict(state_dict)
    model.eval()
    rng = torch.Generator().manual_seed(seed)

    open_tr: Dict[int, dict] = {}
    seqs: Dict[int, List[dict]] = defaultdict(list)
    t = 0
    while True:
        agents = denv.decision_agents
        actions: Dict[int, int] = {}
        for i in agents:
            if i in open_tr:
                tr = open_tr.pop(i)
                tr["tau"], tr["done"] = t - tr["t0"], 0.0
                seqs[i].append(tr)
        if agents:
            obs = denv.observations()
            masks = denv.action_masks()
            x = torch.as_tensor(np.stack([obs[i] for i in agents]))
            m = torch.as_tensor(np.stack([masks[i] for i in agents]))
            with torch.no_grad():
                logits, v = model(x, m)
                probs = torch.softmax(logits, -1)
                a = torch.multinomial(probs, 1, generator=rng).squeeze(-1)
                logp = torch.log(probs.gather(-1, a[:, None]).squeeze(-1) + 1e-12)
            for k, i in enumerate(agents):
                actions[i] = int(a[k])
                open_tr[i] = dict(obs=obs[i], mask=masks[i], a=int(a[k]), logp=float(logp[k]), v=float(v[k]), t0=t, R=0.0)
        r, events, over = denv.step(actions)
        for i, tr in open_tr.items():
            tr["R"] += (cfg.gamma ** (t - tr["t0"])) * float(r[i])
        t += 1
        for i in events:
            if i in open_tr:
                tr = open_tr.pop(i)
                tr["tau"], tr["done"] = t - tr["t0"], 1.0
                seqs[i].append(tr)
        if over:
            for i, tr in open_tr.items():
                tr["tau"], tr["done"] = t - tr["t0"], 1.0
                seqs[i].append(tr)
            break

    obs_l, mask_l, a_l, logp_l, adv_l, ret_l = [], [], [], [], [], []
    for i, seq in seqs.items():
        adv_next, v_next = 0.0, 0.0
        advs = [0.0] * len(seq)
        for j in reversed(range(len(seq))):
            tr = seq[j]
            g = cfg.gamma ** tr["tau"]
            gl = (cfg.gamma * cfg.lam) ** tr["tau"]
            nonterm = 1.0 - tr["done"]
            delta = tr["R"] + g * v_next * nonterm - tr["v"]
            adv_next = delta + gl * nonterm * adv_next
            advs[j] = adv_next
            v_next = tr["v"]
        for tr, adv in zip(seq, advs):
            obs_l.append(tr["obs"])
            mask_l.append(tr["mask"])
            a_l.append(tr["a"])
            logp_l.append(tr["logp"])
            adv_l.append(adv)
            ret_l.append(adv + tr["v"])
    n = denv.n
    return dict(
        obs=np.array(obs_l, np.float32).reshape(-1, denv.encoder.dim),
        mask=np.array(mask_l, np.float32).reshape(-1, N_META_ACTIONS),
        a=np.array(a_l, np.int64),
        logp=np.array(logp_l, np.float32),
        adv=np.array(adv_l, np.float32),
        ret=np.array(ret_l, np.float32),
        stats=dict(
            scenario=scenario,
            arrival_rate=denv.arrival_rate(),
            deadlocked=len(denv.deadlocked) / n,
            steps=t,
            decisions=len(a_l),
        ),
    )


def ppo_update(
    model, opt, batch: dict, cfg: PPOConfig, bc_data: Optional[dict] = None, bc_coef: float = 0.0, value_only: bool = False
) -> dict:
    obs = torch.as_tensor(batch["obs"])
    mask = torch.as_tensor(batch["mask"])
    a = torch.as_tensor(batch["a"])
    old_logp = torch.as_tensor(batch["logp"])
    adv = torch.as_tensor(batch["adv"])
    ret = torch.as_tensor(batch["ret"])
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(a)
    stats = defaultdict(list)
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for s in range(0, n, cfg.minibatch):
            idx = perm[s : s + cfg.minibatch]
            if value_only:
                # warm-up: fit the value head on a frozen trunk, so the (e.g. cloned) policy is untouched
                with torch.no_grad():
                    h = model.body(obs[idx])
                vf = F.mse_loss(model.v(h).squeeze(-1), ret[idx])
                opt.zero_grad()
                (cfg.vf_coef * vf).backward()
                opt.step()
                stats["vf"].append(float(vf.detach()))
                continue
            logits, v = model(obs[idx], mask[idx])
            logp_all = F.log_softmax(logits, -1)
            logp = logp_all.gather(-1, a[idx][:, None]).squeeze(-1)
            ratio = torch.exp(logp - old_logp[idx])
            pg = -torch.min(ratio * adv[idx], torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[idx]).mean()
            vf = F.mse_loss(v, ret[idx])
            p = torch.exp(logp_all)
            ent = -(p * torch.where(mask[idx] > 0.5, logp_all, torch.zeros_like(logp_all))).sum(-1).mean()
            loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent
            if bc_data is not None and bc_coef > 0:
                j = torch.randint(0, len(bc_data["a"]), (min(cfg.minibatch, len(bc_data["a"])),))
                bl, _ = model(bc_data["obs"][j], bc_data["mask"][j])
                bc = F.cross_entropy(bl, bc_data["a"][j])
                loss = loss + bc_coef * bc
                stats["bc"].append(float(bc.detach()))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            stats["pg"].append(float(pg.detach()))
            stats["vf"].append(float(vf.detach()))
            stats["ent"].append(float(ent.detach()))
            stats["clipfrac"].append(float(((ratio - 1).abs() > cfg.clip).float().mean()))
    return {k: float(np.mean(v)) for k, v in stats.items()}


def train_ppo(
    cfg: PPOConfig,
    out_dir: str | Path,
    init_state: Optional[dict] = None,
    bc_data: Optional[dict] = None,
    log_name: str = "ppo",
) -> ActorCritic:
    """Train and save ``model.pt``, ``config.json`` and ``train_log.jsonl`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    obs_dim = make_obs(cfg.obs).dim
    model = ActorCritic(obs_dim, cfg.hidden)
    if init_state is not None:
        model.load_state_dict(init_state)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    if bc_data is not None:
        bc_data = {k: torch.as_tensor(v) for k, v in bc_data.items()}
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    log_f = open(out_dir / "train_log.jsonl", "w")
    scen = list(cfg.train_mix)
    weights = np.array([cfg.train_mix[s] for s in scen], dtype=float)
    weights /= weights.sum()
    t_start = time.perf_counter()
    seed_counter = 0
    total_decisions = 0
    total_steps = 0
    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        for it in range(cfg.iterations):
            if time.perf_counter() - t_start > cfg.time_budget_s:
                break
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            jobs = []
            for _ in range(cfg.episodes_per_iter):
                sc = scen[rng.choice(len(scen), p=weights)]
                seed = int(rng.integers(0, 1000))  # TRAIN_SEEDS
                seed_counter += 1
                jobs.append((state, asdict(cfg), sc, seed, bool(rng.random() < cfg.malfunction_prob)))
            results = list(pool.map(_collect_episode, jobs))
            batch = {k: np.concatenate([r[k] for r in results]) for k in ["obs", "mask", "a", "logp", "adv", "ret"]}
            total_decisions += len(batch["a"])
            total_steps += sum(r["stats"]["steps"] for r in results)
            # progress = the larger of iteration and wall-clock fraction (runs are usually time-capped)
            frac = max(it / max(1, cfg.iterations - 1), (time.perf_counter() - t_start) / cfg.time_budget_s)
            bc_coef = cfg.bc_coef_start * max(0.0, 1.0 - 2.0 * frac)  # anneal to 0 by half-way
            upd = ppo_update(model, opt, batch, cfg, bc_data, bc_coef, value_only=it < cfg.value_warmup_iters)
            row = dict(
                iter=it,
                wall_s=round(time.perf_counter() - t_start, 1),
                env_steps=total_steps,
                decisions=total_decisions,
                arrival_rate=float(np.mean([r["stats"]["arrival_rate"] for r in results])),
                deadlocked=float(np.mean([r["stats"]["deadlocked"] for r in results])),
                bc_coef=bc_coef,
                **upd,
            )
            for sc in scen:
                vals = [r["stats"]["arrival_rate"] for r in results if r["stats"]["scenario"] == sc]
                if vals:
                    row[f"arrival_{sc}"] = float(np.mean(vals))
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()
            print(f"[{log_name}] it={it} t={row['wall_s']}s arr={row['arrival_rate']:.3f} dl={row['deadlocked']:.3f} "
                  f"ent={row.get('ent', 0):.3f} dec={len(batch['a'])}", flush=True)
    log_f.close()
    torch.save(model.state_dict(), out_dir / "model.pt")
    return model
