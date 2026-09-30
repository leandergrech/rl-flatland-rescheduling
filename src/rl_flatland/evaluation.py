"""Run policies on the fixed scenario/seed sets and collect EpisodeResult rows."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from rl_flatland.deadlock import find_deadlocked
from rl_flatland.graph import RailGraph
from rl_flatland.metrics import EpisodeResult, count_arrived, normalized_reward, summarize
from rl_flatland.policies import Policy
from rl_flatland.scenarios import SCENARIOS, SEED_SPLITS, make_env

PolicyFactory = Callable[[], Policy]


def run_episode(
    policy: Policy,
    scenario: str,
    seed: int,
    malfunctions: bool,
    split: str = "",
    obs_builder=None,
    max_steps: Optional[int] = None,
) -> EpisodeResult:
    """Roll out ``policy`` for one full episode (to all-done or max_episode_steps)."""
    env = make_env(scenario, seed, malfunctions=malfunctions, obs_builder=obs_builder)
    t0 = time.perf_counter()
    policy.reset(env)
    setup_s = getattr(policy, "setup_s", time.perf_counter() - t0)
    cumulative: Dict[int, float] = {i: 0.0 for i in range(env.get_num_agents())}
    decision_ms: List[float] = []
    env_ms: List[float] = []
    T = env._max_episode_steps if max_steps is None else min(max_steps, env._max_episode_steps)
    steps = 0
    done = {"__all__": False}
    while not done["__all__"] and steps < T:
        t1 = time.perf_counter()
        actions = policy.act(env)
        t2 = time.perf_counter()
        _, rewards, done, _ = env.step(actions)
        t3 = time.perf_counter()
        policy.observe(env)
        for i, r in rewards.items():
            cumulative[i] += float(r)
        decision_ms.append(1000.0 * (t2 - t1))
        env_ms.append(1000.0 * (t3 - t2))
        steps += 1
    graph = getattr(policy, "graph", None) or RailGraph(env)
    arrived = count_arrived(env)
    n = env.get_num_agents()
    return EpisodeResult(
        scenario=scenario,
        split=split,
        seed=seed,
        policy=policy.name,
        n_agents=n,
        max_steps=int(env._max_episode_steps),
        steps=steps,
        arrived=arrived,
        arrival_rate=arrived / n,
        normalized_reward=normalized_reward(env, cumulative),
        deadlocked=len(find_deadlocked(env, graph)),
        n_malfunctions=int(sum(a.malfunction_handler.num_malfunctions for a in env.agents)),
        setup_s=float(setup_s),
        decision_ms_mean=float(np.mean(decision_ms)) if decision_ms else 0.0,
        decision_ms_max=float(np.max(decision_ms)) if decision_ms else 0.0,
        env_ms_mean=float(np.mean(env_ms)) if env_ms else 0.0,
    )


def _job(args) -> dict:
    factory, scenario, split, seed, malfunctions, obs_builder_factory = args
    import torch

    torch.set_num_threads(1)
    policy = factory()
    obs_builder = obs_builder_factory() if obs_builder_factory else None
    return run_episode(policy, scenario, seed, malfunctions, split=split, obs_builder=obs_builder).to_dict()


def evaluate(
    factory: PolicyFactory,
    scenarios: Iterable[str] = tuple(SCENARIOS),
    splits: Iterable[str] = tuple(SEED_SPLITS),
    n_seeds: Optional[int] = None,
    workers: int = 1,
    obs_builder_factory: Optional[Callable] = None,
) -> List[dict]:
    """Evaluate a policy factory on every (scenario, split, seed). Returns result dicts.

    ``factory`` and ``obs_builder_factory`` must be picklable (module-level callables or
    functools.partial) when ``workers > 1``.
    """
    jobs = []
    for sc in scenarios:
        for split in splits:
            seeds, malf = SEED_SPLITS[split]
            for seed in list(seeds)[: n_seeds or len(seeds)]:
                jobs.append((factory, sc, split, seed, malf, obs_builder_factory))
    # largest scenarios first so the pool stays busy
    jobs.sort(key=lambda j: -SCENARIOS[j[1]].n_agents)
    if workers <= 1:
        return [_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_job, jobs))


def summary_table(rows: List[dict]) -> Dict[Tuple[str, str, str], dict]:
    """Group rows by (policy, scenario, split) and summarize."""
    groups: Dict[Tuple[str, str, str], List[EpisodeResult]] = {}
    for r in rows:
        groups.setdefault((r["policy"], r["scenario"], r["split"]), []).append(EpisodeResult(**r))
    return {k: summarize(v) for k, v in sorted(groups.items())}


def save_results(rows: List[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"rows": rows, "summary": {"|".join(k): v for k, v in summary_table(rows).items()}}, f, indent=1)


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) // 2))
