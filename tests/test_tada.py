"""Tests for the TADA dispatcher (feat/tada-dispatcher).

Fast by default. Set RUN_SLOW=1 for the full step-1 acceptance checks (80 PROCEED-only episodes
against main's OR reference); ``scripts/tada_verify.py`` runs those plus 200 random-clearance
episodes and stores the outcome in data/tada/verify.json.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from rl_flatland.tada.env import DispatchConfig, DispatchEnv
from rl_flatland.tada.executor import HOLD, PROCEED, REROUTE, YIELD_TO, Clearance
from rl_flatland.tada.window import WindowConfig

ROOT = Path(__file__).resolve().parents[1]
SLOW = os.environ.get("RUN_SLOW") == "1"


def _stored_or():
    rows = json.loads((ROOT / "data" / "results" / "or.json").read_text())["rows"]
    return {(r["scenario"], r["split"] == "test_malfunction", r["seed"]): r for r in rows}


def _run(de, policy):
    w, _ = de.reset(*policy["episode"])
    while w is not None:
        policy["act"](de, w)
        w, _ = de.advance()
    return de.summary()


def _proceed(de, w):
    de.apply(int(np.argmax(w.choosable)), PROCEED)


def _random(rng):
    def act(de, w):
        chosen = set()
        while de.can_continue():
            cands = [i for i in range(len(w.trains)) if w.choosable[i] and i not in chosen]
            if not cands:
                break
            i = int(rng.choice(cands))
            chosen.add(i)
            mask, pm = de.action_mask(i)
            a = int(rng.choice([k for k in range(4) if mask[k]]))
            j = int(rng.choice(np.flatnonzero(pm[YIELD_TO]))) if a == YIELD_TO else None
            de.apply(i, a, j)
            if rng.random() < 0.5:
                break

    return act


EPISODES = [("small", 1000, False), ("small", 1001, True), ("medium", 1003, True)]
if SLOW:
    from rl_flatland.scenarios import SCENARIOS, TEST_SEEDS

    EPISODES = [(sc, s, m) for sc in SCENARIOS for m in (False, True) for s in TEST_SEEDS]


@pytest.mark.parametrize("episode", EPISODES)
def test_proceed_only_reproduces_main(episode):
    ref = _stored_or()[(episode[0], episode[2], episode[1])]
    s = _run(DispatchEnv(DispatchConfig()), {"episode": episode, "act": _proceed})
    assert (s["arrived"], s["deadlocked"], s["steps"]) == (ref["arrived"], ref["deadlocked"], ref["steps"])
    assert abs(s["normalized_reward"] - ref["normalized_reward"]) < 1e-9
    assert s["commits"] == 0 and not s["terminated"]


@pytest.mark.parametrize("seed", [3000, 3001, 3002, 3003])
def test_random_clearances_never_deadlock(seed):
    rng = np.random.default_rng(seed)
    s = _run(DispatchEnv(DispatchConfig()), {"episode": ("small", seed, seed % 2 == 1), "act": _random(rng)})
    assert s["commits"] > 0
    assert s["deadlocked"] == 0 and not s["terminated"] and s["reservation_violations"] == 0 and s["deviations"] == 0


def test_speculative_planning_leaves_table_intact_and_conflict_free():
    de = DispatchEnv(DispatchConfig())
    w, _ = de.reset("medium", 42, True)
    checked = 0
    while w is not None and checked < 60:
        now = de.env._elapsed_steps
        rt = de.ex.live_table(now)
        snap = ({c: list(v) for c, v in rt.intervals.items() if v}, set(rt.moves))
        for slot, h in enumerate(w.trains):
            for kind in (HOLD, REROUTE, YIELD_TO):
                for p in [w.trains[j] for j in w.partners_after.get(slot, [])] or [None]:
                    de.ex.plan_clearance(Clearance(h, kind, p), rt, now)
                    checked += 1
        assert snap == ({c: list(v) for c, v in rt.intervals.items() if v}, set(rt.moves))
        for c, v in rt.intervals.items():
            assert all(b1 < a2 for (_, b1), (a2, _) in zip(v, v[1:])), c
        de.apply(int(np.argmax(w.choosable)), PROCEED)
        w, _ = de.advance()
    assert checked > 0


@pytest.mark.parametrize("M", [4, 8])
def test_window_invariants(M):
    rng = np.random.default_rng(5)
    de = DispatchEnv(DispatchConfig(window=WindowConfig(M=M)))
    w, _ = de.reset("medium", 7, True)
    seen = 0
    while w is not None and seen < 150:
        assert len(w.trains) <= M and int(w.mask.sum()) == len(w.trains)
        for i in range(len(w.trains)):
            assert w.struct_actions[i, PROCEED] == 1
            if w.choosable[i]:
                assert w.struct_actions[i, 1:].sum() >= 1  # a legal non-default action exists
                assert de.ex.at_decision(w.trains[i])
        _random(rng)(de, w)
        w, _ = de.advance()
        seen += 1
    assert max(de.occupancy) <= M


def test_policy_smoke_200_steps():
    import torch

    from rl_flatland.tada.ppo import TadaPPOConfig, _worker, make_net, ppo_update, run_episode

    cfg = TadaPPOConfig()
    net = make_net(cfg)
    dc = DispatchConfig.from_dict(cfg.dispatch)
    recs, rews, taus, vfin, term, s = run_episode(net, dc, "medium", 7, True, 0.99, False, torch.Generator().manual_seed(0), max_steps=200)
    assert s["steps"] >= 200 and len(recs) > 0 and not term
    from dataclasses import asdict

    cfg.scenario = "small"
    out = _worker((net.state_dict(), asdict(cfg), 3))
    stats = ppo_update(net, torch.optim.Adam(net.parameters(), 3e-4), out["batch"], cfg, dc.budget)
    assert all(np.isfinite(v) for v in stats.values())


def test_empty_window_value_is_finite():
    import torch

    from rl_flatland.tada.policy import DispatcherNet

    net = DispatcherNet(14, 6).eval()
    with torch.no_grad():
        g, _ = net.encode(torch.zeros(1, 8, 14), torch.zeros(1, 8), torch.zeros(1, 6), torch.zeros(1, 8), torch.tensor([0.0]))
    assert torch.isfinite(net.value(g)).all()
