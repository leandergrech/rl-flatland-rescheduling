"""PPO for the TADA dispatcher: one agent, decisions at decision points, semi-MDP returns.

Transition j runs from decision point j to decision point j+1 (tau_j env steps). Its reward is the
discounted sum of per-step rewards in between. Termination (deadlock or reservation violation)
cuts the bootstrap; truncation at the horizon bootstraps from the value of the final state:

    delta_j = R_j + gamma^tau_j * V(s_{j+1}) * (1 - terminated) - V(s_j)
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from rl_flatland.tada.env import DispatchConfig, DispatchEnv
from rl_flatland.tada.policy import DispatcherNet, act_at_point, evaluate_records
from rl_flatland.tada.window import GLOBAL_FEATURES, build_window


@dataclass
class TadaPPOConfig:
    dispatch: dict = field(default_factory=lambda: DispatchConfig().to_dict())
    scenario: str = "medium"
    malfunctions: bool = True
    d_model: int = 64
    layers: int = 1
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 512
    # The joint entropy sums up to 4 heads per clearance and B clearances, and per-decision advantages
    # are small (most decisions barely matter), so 0.01 let the bonus outweigh the policy gradient:
    # entropy rose from 1.55 to 2.2 and plan edits tripled with no reward gain (checkpoints/main_ent01_stopped).
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    episodes_per_iter: int = 8
    iterations: int = 400
    time_budget_s: float = 1800.0
    workers: int = 8
    seed: int = 0
    save_every: int = 20  # also keep model_it{n}.pt, so shorter ablations can be compared at equal samples


def cpu_mhz() -> float:
    """Mean current clock over all cores (logged because this machine throttles between 400 MHz and 2.2 GHz)."""
    try:
        v = [float(l.split(":")[1]) for l in open("/proc/cpuinfo") if l.startswith("cpu MHz")]
        return round(sum(v) / len(v), 0)
    except Exception:
        return float("nan")


def make_net(cfg: TadaPPOConfig) -> DispatcherNet:
    dc = DispatchConfig.from_dict(cfg.dispatch)
    return DispatcherNet(dc.window.n_features, GLOBAL_FEATURES, d=cfg.d_model, layers=cfg.layers)


def run_episode(net: DispatcherNet, dc: DispatchConfig, scenario: str, seed: int, malfunctions: bool, gamma: float,
                greedy: bool, gen: Optional[torch.Generator] = None, max_steps: Optional[int] = None, env=None,
                executor_cls=None, return_env: bool = False):
    """Roll out one episode; returns (records, rewards, taus, final_value, terminated, summary)."""
    de = DispatchEnv(dc, gamma=gamma) if executor_cls is None else DispatchEnv(dc, gamma=gamma, executor_cls=executor_cls)
    t0 = time.perf_counter()
    w, res = de.reset(scenario, seed, malfunctions, env=env)
    records, rewards, taus = [], [], []
    policy_s = 0.0
    while w is not None:
        t1 = time.perf_counter()
        rec = act_at_point(net, de, dc.budget, greedy=greedy, gen=gen)
        policy_s += time.perf_counter() - t1
        w, res = de.advance()
        records.append(rec)
        rewards.append(res.reward)
        taus.append(res.steps)
        if max_steps is not None and de.env._elapsed_steps >= max_steps:
            break
    final_value = 0.0
    if not de.terminated and not greedy:
        fw = build_window(de.ex, dc.window)
        with torch.no_grad():
            g, _ = net.encode(torch.as_tensor(fw.feats)[None], torch.as_tensor(fw.mask)[None], torch.as_tensor(fw.glob)[None],
                              torch.zeros(1, len(fw.mask)), torch.tensor([0.0]))
            final_value = float(net.value(g))
    s = de.summary()
    s["wall_s"] = time.perf_counter() - t0
    s["decision_ms_per_step"] = 1000 * s["wall_s"] / max(1, s["steps"])  # everything except nothing: full dispatcher + env
    s["policy_ms_per_step"] = 1000 * policy_s / max(1, s["steps"])
    s["decision_points"] = len(records)
    if return_env:
        return records, rewards, taus, final_value, bool(de.terminated), s, de
    return records, rewards, taus, final_value, bool(de.terminated), s


def _worker(args) -> dict:
    state, cfg_d, seed = args
    torch.set_num_threads(1)
    cfg = TadaPPOConfig(**cfg_d)
    dc = DispatchConfig.from_dict(cfg.dispatch)
    net = make_net(cfg)
    net.load_state_dict(state)
    net.eval()
    gen = torch.Generator().manual_seed(seed)
    recs, rews, taus, vfin, term, summ = run_episode(net, dc, cfg.scenario, seed, cfg.malfunctions, cfg.gamma, False, gen)
    n = len(recs)
    adv = np.zeros(n, np.float32)
    nxt_v, nxt_a = vfin * (0.0 if term else 1.0), 0.0
    for j in reversed(range(n)):
        g = cfg.gamma ** taus[j]
        gl = (cfg.gamma * cfg.lam) ** taus[j]
        delta = rews[j] + g * nxt_v - recs[j].value
        nxt_a = delta + gl * nxt_a
        adv[j] = nxt_a
        nxt_v = recs[j].value
    ret = adv + np.array([r.value for r in recs], np.float32)
    batch = {k: np.stack([getattr(r, k) for r in recs]) for k in ["feats", "mask", "glob", "valid", "choose_mask", "train", "amask", "action", "pmask", "partner", "has_cont", "cont"]} if n else {}
    if n:
        batch["logp"] = np.array([r.logp for r in recs], np.float32)
        batch["adv"], batch["ret"] = adv, ret
    return {"batch": batch, "summary": summ}


def ppo_update(net, opt, batch: dict, cfg: TadaPPOConfig, budget: int) -> dict:
    t = {k: torch.as_tensor(v) for k, v in batch.items()}
    adv = (t["adv"] - t["adv"].mean()) / (t["adv"].std() + 1e-8)
    n = len(adv)
    stats = defaultdict(list)
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for s in range(0, n, cfg.minibatch):
            idx = perm[s : s + cfg.minibatch]
            mb = {k: v[idx] for k, v in t.items()}
            logp, ent, value = evaluate_records(net, mb, budget)
            ratio = torch.exp(logp - mb["logp"])
            a = adv[idx]
            pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
            vf = F.mse_loss(value, mb["ret"])
            loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            stats["pg"].append(float(pg.detach()))
            stats["vf"].append(float(vf.detach()))
            stats["ent"].append(float(ent.mean().detach()))
            stats["clipfrac"].append(float(((ratio - 1).abs() > cfg.clip).float().mean()))
    return {k: float(np.mean(v)) for k, v in stats.items()}


def train(cfg: TadaPPOConfig, out_dir: str | Path, log_name: str = "tada") -> DispatcherNet:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    net = make_net(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    budget = DispatchConfig.from_dict(cfg.dispatch).budget
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    log = open(out_dir / "train_log.jsonl", "w")
    t0 = time.perf_counter()
    steps = 0
    done_iters = 0
    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        for it in range(cfg.iterations):
            if time.perf_counter() - t0 > cfg.time_budget_s:
                break
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            seeds = [int(rng.integers(0, 1000)) for _ in range(cfg.episodes_per_iter)]
            res = list(pool.map(_worker, [(state, asdict(cfg), s) for s in seeds]))
            parts = [r["batch"] for r in res if r["batch"]]
            if not parts:
                continue
            batch = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            upd = ppo_update(net, opt, batch, cfg, budget)
            summ = [r["summary"] for r in res]
            steps += sum(s["steps"] for s in summ)
            clr = defaultdict(int)
            for s in summ:
                for k, v in s["clearances"].items():
                    clr[k] += v
            row = dict(iter=it, wall_s=round(time.perf_counter() - t0, 1), env_steps=steps, decisions=len(batch["logp"]),
                       arrival_rate=float(np.mean([s["arrival_rate"] for s in summ])),
                       normalized_reward=float(np.mean([s["normalized_reward"] for s in summ])),
                       terminations=int(sum(s["terminated"] for s in summ)), commits=int(sum(s["commits"] for s in summ)),
                       rejected=int(sum(s["rejected"] for s in summ)), clearances=dict(clr), cpu_mhz=cpu_mhz(), **upd)
            done_iters += 1
            if cfg.save_every and done_iters % cfg.save_every == 0:
                torch.save(net.state_dict(), out_dir / f"model_it{done_iters}.pt")
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"[{log_name}] it={it} t={row['wall_s']}s arr={row['arrival_rate']:.3f} nr={row['normalized_reward']:.4f} "
                  f"term={row['terminations']} commits={row['commits']} ent={row.get('ent', 0):.3f}", flush=True)
    log.close()
    torch.save(net.state_dict(), out_dir / "model.pt")
    (out_dir / "wall_clock.json").write_text(json.dumps({"train_wall_s": round(time.perf_counter() - t0, 1), "workers": cfg.workers,
                                                         "iterations_done": done_iters, "iterations_planned": cfg.iterations, "env_steps": steps}))
    return net
