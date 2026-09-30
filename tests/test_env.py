"""Sanity tests for the scenarios, the rail graph and the flatland semantics the planner relies on."""

import numpy as np
import pytest
from flatland.envs.step_utils.states import TrainState

from rl_flatland.deadlock import find_deadlocked
from rl_flatland.env import DecisionEnv, N_META_ACTIONS
from rl_flatland.graph import MOVE_FORWARD, RailGraph, to_config
from rl_flatland.observations import CompactObs, TreeObs
from rl_flatland.scenarios import SCENARIOS, TEST_MALFUNCTION_SEEDS, TEST_SEEDS, make_env


def test_scenarios_are_mid_size():
    for sc in SCENARIOS.values():
        assert 30 <= sc.width <= 100 and 30 <= sc.height <= 100
        assert 10 <= sc.n_agents <= 100
    assert not set(TEST_SEEDS) & set(TEST_MALFUNCTION_SEEDS)
    assert max(TEST_SEEDS + TEST_MALFUNCTION_SEEDS) >= 1000  # disjoint from training seeds 0..999


def test_make_env_is_deterministic():
    a, b = make_env("small", 1000), make_env("small", 1000)
    assert a._max_episode_steps == b._max_episode_steps
    assert [x.initial_configuration for x in a.agents] == [x.initial_configuration for x in b.agents]
    assert [x.earliest_departure for x in a.agents] == [x.earliest_departure for x in b.agents]
    assert np.array_equal(a.rail.grid, b.rail.grid)


def test_graph_distances_and_successors():
    env = make_env("small", 1000)
    g = RailGraph(env)
    assert g.is_rail.sum() > 0 and g.is_switch.sum() > 0
    for a in env.agents:
        start = to_config(a.initial_configuration)
        d = g.distance(a.handle, start)
        assert np.isfinite(d) and d > 0
        path = g.shortest_path(a.handle, start)
        assert g.distance(a.handle, path[-1]) == 0
        assert len(path) == int(d) + 1


@pytest.mark.parametrize("k", [1, 3])
def test_train_spends_k_steps_per_cell(k):
    """A train at speed 1/k enters its first cell one step after READY_TO_DEPART and moves every k steps."""
    from flatland.envs.line_generators import sparse_line_generator
    from flatland.envs.rail_env import RailEnv
    from flatland.envs.rail_generators import sparse_rail_generator

    env = RailEnv(width=30, height=30, number_of_agents=1,
                  rail_generator=sparse_rail_generator(max_num_cities=2, max_rails_between_cities=2, max_rail_pairs_in_city=2),
                  line_generator=sparse_line_generator({1.0 / k: 1.0}), random_seed=3)
    env.reset(random_seed=3)
    agent = env.agents[0]
    cells = []
    for _ in range(agent.earliest_departure + 1 + 3 * k):
        env.step({0: MOVE_FORWARD})
        cells.append(to_config(agent.current_configuration))
    on_map = [c for c in cells if c is not None]
    # run lengths of identical cells are exactly k (except possibly the last, incomplete run)
    runs, cur, n = [], on_map[0], 0
    for c in on_map:
        if c[:2] == cur[:2]:
            n += 1
        else:
            runs.append(n)
            cur, n = c, 1
    assert runs and all(r == k for r in runs)


def test_deadlock_detector_finds_head_on_pair():
    """Drive every train along its shortest path with no coordination until someone deadlocks."""
    from rl_flatland.policies import ShortestPathPolicy

    env = make_env("small", 1000)
    pol = ShortestPathPolicy()
    pol.reset(env)
    found = set()
    for _ in range(env._max_episode_steps):
        _, _, done, _ = env.step(pol.act(env))
        found = find_deadlocked(env, pol.graph)
        if found or done["__all__"]:
            break
    assert found, "uncoordinated shortest-path runs on this seed are known to deadlock"
    cells = {to_config(env.agents[h].current_configuration)[:2] for h in found}
    assert len(cells) == len(found)


@pytest.mark.parametrize("obs", ["compact", "tree"])
def test_decision_env_step_and_observations(obs):
    denv = DecisionEnv(obs)
    denv.reset("small", 5, malfunctions=True)
    dim = denv.encoder.dim
    steps = 0
    over = False
    while not over and steps < 60:
        o, m = denv.observations(), denv.action_masks()
        for i in denv.decision_agents:
            assert o[i].shape == (dim,) and np.all(np.isfinite(o[i]))
            assert m[i].shape == (N_META_ACTIONS,) and m[i][0] == 1 and m[i][1] == 1
        r, events, over = denv.step({i: 1 for i in denv.decision_agents})
        assert r.shape == (denv.n,)
        steps += 1
    assert steps > 0


def test_observation_dims():
    assert CompactObs.dim == 28
    assert TreeObs().dim == 12 * 21 + 8


def test_normalized_reward_bounds():
    from rl_flatland.evaluation import run_episode
    from rl_flatland.policies import RandomPolicy

    r = run_episode(RandomPolicy(), "small", 1000, malfunctions=False, max_steps=40)
    assert 0.0 <= r.normalized_reward <= 1.0 + 1e-9
    assert 0.0 <= r.arrival_rate <= 1.0
