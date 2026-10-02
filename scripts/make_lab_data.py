"""Export maps and recorded episodes for the Scheduling Lab (docs/javascripts/flatland-core.js).

For every map (scenario, held-out seed, malfunctions off/on) this writes
docs/assets/lab/maps/<id>.json with

* the rail grid (flatland's 16-bit transition codes), the trains (start, heading, target, speed,
  earliest departure, latest arrival) and the horizon;
* the malfunction draws: every non-zero ``num_broken_steps`` flatland's malfunction generator
  returned, as (step, train, duration). The draws come from the env's random stream once per
  train per step whatever the trains do, so they are the same for every policy (checked here);
* the successor order of every configuration with more than one successor, as Python iterates it
  (the planner breaks search ties by insertion order, so the JavaScript port needs the same order);
* the four random priority orderings main's planner tries after its four rule-based ones;
* every recorded policy's actions, run-length encoded per train, its outcome, and for the learned
  dispatcher the window members and every clearance it issued.

data/lab_fixture/<id>.json holds what the Node check compares against (scripts/check_lab.mjs):
per-step train states, distance maps and main's OR plan. docs/assets/lab/index.json lists the maps.

    python scripts/make_lab_data.py                      # small and medium, seeds 1000-1009, both splits
    python scripts/make_lab_data.py --scenarios small --seeds 1000 1001 --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

OUT = ROOT / "docs" / "assets" / "lab"
FIX = ROOT / "data" / "lab_fixture"
POLICIES = ["or", "tada", "ppo", "shortest_path", "reactive_avoid"]
LABELS = {
    "or": "OR planner (PP + SIPP + ordered execution)",
    "tada": "TADA dispatcher on the planner (learned)",
    "ppo": "PPO, compact observation (main's best learned)",
    "shortest_path": "Shortest path, no coordination",
    "reactive_avoid": "Reactive rule",
}


# ------------------------------------------------------------------ recording hooks
class Recorder:
    """Collects malfunction draws and per-step actions/states for one episode."""

    def __init__(self):
        self.env = None
        self.draws = []
        self._step = None
        self._idx = 0
        self.actions = []  # per step: list of int actions (len n)
        self.states = []  # per step after env.step: per train [state, r, c, d, dist_num, malf]

    def draw_hook(self, orig):
        def wrapped(gen, rng):
            n = orig(gen, rng)
            s = self.env._elapsed_steps
            if s != self._step:
                self._step, self._idx = s, 0
            if n:
                self.draws.append([int(s), self._idx, int(n)])
            self._idx += 1
            return n

        return wrapped

    def snapshot(self):
        out = []
        for a in self.env.agents:
            cfg = a.current_configuration
            sc = a.speed_counter
            k = int(round(1 / float(sc.max_speed)))
            dist_units = int(round(float(sc.distance) * k))
            spd = 0 if float(sc.speed) == 0 else 1
            if cfg is None or cfg[0] is None:
                out.append([int(a.state.value), -1, -1, -1, dist_units, spd, a.malfunction_handler.malfunction_down_counter])
            else:
                (r, c), d = cfg
                out.append([int(a.state.value), int(r), int(c), int(d), dist_units, spd, a.malfunction_handler.malfunction_down_counter])
        return out


REC = Recorder()


def _install_draw_hook():
    import flatland.envs.step_utils.malfunction_handler as mh

    if not getattr(mh, "_lab_hooked", False):
        mh.get_number_of_steps_to_break = REC.draw_hook(mh.get_number_of_steps_to_break)
        mh._lab_hooked = True


def _wrap_env_step(env):
    orig = env.step

    def step(actions):
        n = env.get_num_agents()
        REC.actions.append([int(getattr(actions.get(i, 0), "value", actions.get(i, 0))) for i in range(n)])
        out = orig(actions)
        REC.states.append(REC.snapshot())
        return out

    env.step = step


def rle(actions_per_step, n):
    """Per train: [[step, action], ...] whenever the action changes (step 1-based, initial 0)."""
    out = []
    for i in range(n):
        prev, lst = 0, []
        for s, acts in enumerate(actions_per_step, start=1):
            a = acts[i]
            if a != prev:
                lst.append([s, a])
                prev = a
        out.append(lst)
    return out


def state_changes(states, n):
    """Per train: [[step, state, r, c, d, dist, spd, malf], ...] whenever anything changes."""
    out = []
    for i in range(n):
        prev, lst = None, []
        for s, st in enumerate(states, start=1):
            row = st[i]
            if row != prev:
                lst.append([s] + row)
                prev = row
        out.append(lst)
    return out


# ------------------------------------------------------------------ one policy, one map
def run_policy(kind: str, scenario: str, seed: int, malf: bool) -> dict:
    import numpy as np
    import torch

    from rl_flatland.deadlock import find_deadlocked
    from rl_flatland.graph import RailGraph
    from rl_flatland.metrics import count_arrived, normalized_reward
    from rl_flatland.scenarios import make_env

    torch.set_num_threads(1)
    _install_draw_hook()
    REC.__init__()
    extra = {}
    if kind == "tada":
        import rl_flatland.tada.env as tenv
        from rl_flatland.tada.env import DispatchConfig, DispatchEnv
        from rl_flatland.tada.policy import act_at_point
        from rl_flatland.tada.ppo import TadaPPOConfig, make_net

        ck = ROOT / "data" / "tada" / "checkpoints" / "main"
        cfg = TadaPPOConfig(**json.loads((ck / "config.json").read_text()))
        net = make_net(cfg)
        net.load_state_dict(torch.load(ck / "model.pt", map_location="cpu"))
        net.eval()
        dc = DispatchConfig.from_dict(cfg.dispatch)
        windows, clearances = [], []
        orig_bw = tenv.build_window

        def bw(ex, wcfg):
            w = orig_bw(ex, wcfg)
            windows.append([int(ex.env._elapsed_steps), [int(h) for h in w.trains], [int(h) for h, c in zip(w.trains, w.choosable) if c]])
            return w

        tenv.build_window = bw
        orig_apply = DispatchEnv.apply

        def apply(self, slot, kind_, partner_slot=None):
            h = int(self.window.trains[slot])
            p = int(self.window.trains[partner_slot]) if partner_slot is not None else None
            before = self.stats["commits"]
            ok = orig_apply(self, slot, kind_, partner_slot)
            if kind_ != 0:
                clearances.append([int(self.env._elapsed_steps), h, int(kind_), p, int(self.stats["commits"] > before)])
            return ok

        DispatchEnv.apply = apply
        env = make_env(scenario, seed, malfunctions=malf)
        REC.env = env
        _wrap_env_step(env)
        de = DispatchEnv(dc)
        w, _ = de.reset(scenario, seed, malf, env=env)
        while w is not None:
            act_at_point(net, de, dc.budget, greedy=True)
            w, _ = de.advance()
        s = de.summary()
        tenv.build_window = orig_bw
        DispatchEnv.apply = orig_apply
        summary = dict(arrived=s["arrived"], n_agents=s["n_agents"], normalized_reward=s["normalized_reward"], deadlocked=s["deadlocked"],
                       steps=s["steps"], terminated=s["terminated"], commits=s["commits"])
        # window: keep only steps where membership changes
        wl, prev = [], None
        for st, members, choos in windows:
            if (members, choos) != prev:
                wl.append([st, members, choos])
                prev = (members, choos)
        extra = dict(window=wl, clearances=clearances)
    else:
        sys.path.insert(0, str(ROOT / "scripts"))
        from evaluate import LEARNED, make_policy  # noqa: E402

        if kind in LEARNED:
            fk, sub, obs = LEARNED[kind]
            policy = make_policy(fk, str(ROOT / "data" / "checkpoints" / sub / "model.pt"), obs, kind)
        else:
            policy = make_policy(kind)
        obs_builder = None
        if kind in LEARNED and LEARNED[kind][2] == "tree":
            from evaluate import tree_builder

            obs_builder = tree_builder()
        env = make_env(scenario, seed, malfunctions=malf, obs_builder=obs_builder)
        REC.env = env
        _wrap_env_step(env)
        policy.reset(env)
        cum = {i: 0.0 for i in range(env.get_num_agents())}
        done = {"__all__": False}
        while not done["__all__"]:
            acts = policy.act(env)
            _, rew, done, _ = env.step(acts)
            policy.observe(env)
            for i, r in rew.items():
                cum[i] += float(r)
        graph = getattr(policy, "graph", None) or RailGraph(env)
        summary = dict(arrived=count_arrived(env), n_agents=env.get_num_agents(), normalized_reward=normalized_reward(env, cum),
                       deadlocked=len(find_deadlocked(env, graph)), steps=int(env._elapsed_steps))
        if kind == "or":
            extra["plan"] = {str(h): [[list(cfg), int(t)] for cfg, t in p] for h, p in policy.paths.items()}
            extra["initial_plan_count"] = len(policy.paths)
    n = env.get_num_agents()
    return dict(kind=kind, summary=summary, actions=rle(REC.actions, n), draws=REC.draws, states=state_changes(REC.states, n), extra=extra)


def draws_to_horizon(scenario: str, seed: int, malf: bool) -> list:
    """Every malfunction draw up to the horizon: no train ever departs, so the env runs to T."""
    from rl_flatland.scenarios import make_env

    _install_draw_hook()
    REC.__init__()
    env = make_env(scenario, seed, malfunctions=malf)
    REC.env = env
    done = {"__all__": False}
    while not done["__all__"]:
        _, _, done, _ = env.step({})
    return REC.draws


# ------------------------------------------------------------------ the map itself
def map_static(scenario: str, seed: int, malf: bool) -> dict:
    import numpy as np

    from rl_flatland.graph import RailGraph
    from rl_flatland.scenarios import make_env

    env = make_env(scenario, seed, malfunctions=malf)
    g = RailGraph(env)
    H, W = env.height, env.width
    grid = [int(x) for x in env.rail.grid.reshape(-1)]
    succ_order = {}
    for r in range(H):
        for c in range(W):
            if not grid[r * W + c]:
                continue
            for d in range(4):
                s = g.successors((r, c, d))
                if len(s) > 1:
                    succ_order[f"{r},{c},{d}"] = [list(x) for x in s]
    agents = []
    for a in env.agents:
        (r, c), d = a.initial_configuration
        k = int(round(1 / float(a.speed_counter.max_speed)))
        targets = sorted([int(t[0][0]), int(t[0][1]), int(t[1])] for t in a.targets)  # arrival needs cell AND heading
        agents.append(dict(start=[int(r), int(c), int(d)], targets=targets, k=k, ed=int(a.earliest_departure), la=int(a.latest_arrival)))
    init = [[int(a.state.value), a.malfunction_handler.malfunction_down_counter, int(round(float(a.speed_counter.distance) * agents[i]["k"]))]
            for i, a in enumerate(env.agents)]
    rng = np.random.default_rng(0)  # PrioritizedPlannerPolicy(seed=0) draws its random orderings like this
    orderings = [[int(x) for x in rng.permutation(len(env.agents))] for _ in range(4)]
    dist = g.dist  # (n, H, W, 4), inf where unreachable
    dist_list = [[int(v) if np.isfinite(v) else -1 for v in dist[i].reshape(-1)] for i in range(len(env.agents))]
    return dict(H=H, W=W, T=int(env._max_episode_steps), grid=grid, succ_order=succ_order, agents=agents, init=init,
                orderings=orderings, dist=dist_list)


def job(args) -> dict:
    scenario, seed, malf = args
    t0 = time.perf_counter()
    st = map_static(scenario, seed, malf)
    eps = {}
    draws = draws_to_horizon(scenario, seed, malf)
    for kind in POLICIES:
        r = run_policy(kind, scenario, seed, malf)
        last = max([d[0] for d in r["draws"]], default=0)
        if r["draws"] != [d for d in draws if d[0] <= last]:
            raise AssertionError(f"malfunction draws of {kind} differ from the no-op run on {scenario} {seed} {malf}")
        eps[kind] = r
    return dict(scenario=scenario, seed=seed, malf=malf, static=st, draws=draws, eps=eps, wall=time.perf_counter() - t0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+", default=["small", "medium"])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1000, 1010)))
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    jobs = [(sc, s, m) for sc in a.scenarios for s in a.seeds for m in (False, True)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "maps").mkdir(exist_ok=True)
    FIX.mkdir(parents=True, exist_ok=True)
    idx_path = OUT / "index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {"maps": []}
    known = {m["id"]: m for m in index["maps"]}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(job, jobs):
            mid = f"{r['scenario']}-{r['seed']}-{'m' if r['malf'] else 'n'}"
            st = r["static"]
            # draws are recorded up to the longest episode; the env ends at T at the latest
            doc = dict(id=mid, scenario=r["scenario"], seed=r["seed"], malfunctions=r["malf"], H=st["H"], W=st["W"], T=st["T"],
                       grid=st["grid"], agents=st["agents"], init=st["init"], orderings=st["orderings"],
                       malf=r["draws"], labels=LABELS,
                       episodes={k: dict(actions=e["actions"], summary=e["summary"], **({"window": e["extra"]["window"], "clearances": e["extra"]["clearances"]} if k == "tada" else {}))
                                 for k, e in r["eps"].items()})
            (OUT / "maps" / f"{mid}.json").write_text(json.dumps(doc, separators=(",", ":")))
            fix = dict(id=mid, dist=st["dist"], succ_order=st["succ_order"], plan=r["eps"]["or"]["extra"].get("plan"),
                       states={k: e["states"] for k, e in r["eps"].items()}, summaries={k: e["summary"] for k, e in r["eps"].items()})
            (FIX / f"{mid}.json").write_text(json.dumps(fix, separators=(",", ":")))
            known[mid] = dict(id=mid, scenario=r["scenario"], seed=r["seed"], malfunctions=r["malf"], n_agents=len(st["agents"]), T=st["T"],
                              H=st["H"], W=st["W"], n_malf=len(r["draws"]),
                              outcomes={k: e["summary"] for k, e in r["eps"].items()})
            print(f"{mid}: {r['wall']:.0f}s " + " ".join(f"{k}={e['summary']['arrived']}/{e['summary']['n_agents']}" for k, e in r["eps"].items()), flush=True)
    order = {"small": 0, "medium": 1, "large": 2, "xlarge": 3}
    index["maps"] = sorted(known.values(), key=lambda m: (order.get(m["scenario"], 9), m["seed"], m["malfunctions"]))
    index["labels"] = LABELS
    idx_path.write_text(json.dumps(index, indent=1))
    print(f"wrote {len(jobs)} maps in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
