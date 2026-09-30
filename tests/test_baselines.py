"""Smoke tests: each baseline runs a few steps / iterations end to end."""

from dataclasses import asdict

import numpy as np
import torch

from rl_flatland.baselines.imitation import behaviour_cloning, collect_demos
from rl_flatland.baselines.or_planner import PrioritizedPlannerPolicy, ReservationTable
from rl_flatland.baselines.ppo import ActorCritic, PPOConfig, _collect_episode, ppo_update
from rl_flatland.evaluation import run_episode
from rl_flatland.observations import make_obs
from rl_flatland.policies import ReactiveAvoidPolicy, ShortestPathPolicy


def test_reservation_table_safe_intervals():
    rt = ReservationTable()
    rt.reserve((0, 0), 5, 9)
    rt.reserve((0, 0), 20, 20)
    assert rt.safe_intervals((0, 0))[:2] == [(0, 4), (10, 19)]
    rt.reserve_move((0, 0), (0, 1), 10)
    assert rt.swap_blocked((0, 1), (0, 0), 10)
    assert not rt.swap_blocked((0, 0), (0, 1), 10)


def test_or_planner_full_episode_small():
    """The OR reference routes every train on a small held-out seed without deadlocks."""
    pol = PrioritizedPlannerPolicy()
    r = run_episode(pol, "small", 1000, malfunctions=False)
    assert r.arrival_rate == 1.0
    assert r.deadlocked == 0
    assert pol.deviations == 0


def test_or_planner_with_malfunctions_is_deadlock_free():
    pol = PrioritizedPlannerPolicy()
    r = run_episode(pol, "small", 2000, malfunctions=True)
    assert r.deadlocked == 0 and pol.deviations == 0


def test_heuristic_policies_run():
    for pol in [ShortestPathPolicy(), ReactiveAvoidPolicy()]:
        r = run_episode(pol, "small", 1001, malfunctions=False, max_steps=30)
        assert r.steps == 30


def test_ppo_collect_and_update():
    cfg = PPOConfig(minibatch=256, epochs=1)
    model = ActorCritic(make_obs("compact").dim)
    batch = _collect_episode((model.state_dict(), asdict(cfg), "small", 11, False))
    assert len(batch["a"]) > 0 and np.all(np.isfinite(batch["adv"]))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    stats = ppo_update(model, opt, batch, cfg)
    assert np.isfinite(stats["pg"]) and np.isfinite(stats["vf"])


def test_imitation_demos_and_bc():
    demos = collect_demos([("small", 12, False)], workers=1)
    assert len(demos["a"]) > 0 and set(np.unique(demos["a"])) <= {0, 1, 2}
    model, log = behaviour_cloning(demos, make_obs("compact").dim, epochs=1, batch=256)
    assert np.isfinite(log[-1]["val_loss"])


def test_network_policy_runs(tmp_path):
    from rl_flatland.baselines.rl_policy import NetworkPolicy

    model = ActorCritic(make_obs("compact").dim)
    torch.save(model.state_dict(), tmp_path / "m.pt")
    r = run_episode(NetworkPolicy(tmp_path / "m.pt"), "small", 1000, malfunctions=False, max_steps=30)
    assert r.steps == 30


def test_value_warmup_leaves_policy_unchanged():
    cfg = PPOConfig(minibatch=256, epochs=1)
    model = ActorCritic(make_obs("compact").dim)
    batch = _collect_episode((model.state_dict(), asdict(cfg), "small", 13, False))
    x = torch.as_tensor(batch["obs"][:64])
    m = torch.as_tensor(batch["mask"][:64])
    before, _ = model(x, m)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ppo_update(model, opt, batch, cfg, value_only=True)
    after, _ = model(x, m)
    assert torch.allclose(before, after)
