"""Continuous Flatland: trains keep arriving on a fixed map, and nothing ends except the step budget.

``make_continuous_env`` builds a mid-size map with a pool of trains whose earliest departures are
the arrival times of a Poisson process with the given rate. A train therefore "appears" at its
injection time and leaves the grid when it arrives. The pool size is fixed across rates, so the map
and every train's start, target and speed are identical for every rate; only the injection times
change. Each train's latest arrival is its injection time plus Flatland 3's travel allowance,
ceil(1.3 * tau_i + 0.2 * mean tau). The episode is truncated at the step budget.

``OnlineExecutor`` is the OR executor without foresight: a train is planned only once it has
appeared, against the live (re-timed) reservation table, and committed through the same path as a
clearance. With every clearance PROCEED it is the plan-and-repair baseline for this setting.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.step_utils.states import TrainState
from flatland.envs.timetable_utils import Timetable

from rl_flatland.baselines.or_planner import _agent_infos
from rl_flatland.graph import RailGraph
from rl_flatland.scenarios import SCENARIOS
from rl_flatland.tada.executor import TadaExecutor
from rl_flatland.tada.planning import TrainTable, sipp_from

POOL = 400


def make_continuous_env(rate: float, budget: int = 2000, scenario: str = "medium", map_seed: int = 5000, arrival_seed: int = 0,
                        malfunctions: bool = True, pool: int = POOL) -> RailEnv:
    sc = SCENARIOS[scenario]
    rng = np.random.default_rng(arrival_seed)
    arrivals = np.floor(np.cumsum(rng.exponential(1.0 / rate, size=pool))).astype(int)

    def timetable(agents, distance_map, hints, np_random):
        sp = distance_map.get_shortest_paths()
        tau = np.array([(len(sp[a.handle]) if sp[a.handle] is not None else 0) / float(a.speed_counter.max_speed) for a in agents])
        mean_tau = float(tau.mean())
        eds, las = [], []
        for a in agents:
            ed = int(arrivals[a.handle])
            la = ed + int(math.ceil(1.3 * tau[a.handle] + 0.2 * mean_tau))
            eds.append([ed, None])
            las.append([None, la])
        return Timetable(earliest_departures=eds, latest_arrivals=las, max_episode_steps=budget)

    malf = ParamMalfunctionGen(MalfunctionParameters(sc.malfunction_rate, sc.malfunction_min_duration, sc.malfunction_max_duration)) if malfunctions else None
    env = RailEnv(
        width=sc.width, height=sc.height, number_of_agents=pool,
        rail_generator=sparse_rail_generator(max_num_cities=sc.n_cities, grid_mode=False,
                                             max_rails_between_cities=sc.max_rails_between_cities, max_rail_pairs_in_city=sc.max_rail_pairs_in_city),
        line_generator=sparse_line_generator(sc.speed_ratio_map), timetable_generator=timetable,
        malfunction_generator=malf, obs_builder_object=DummyObservationBuilder(), random_seed=map_seed,
    )
    env.reset(random_seed=map_seed)
    return env


class OnlineExecutor(TadaExecutor):
    """Plans each train when it appears (earliest departure reached), against the live table."""

    name = "online_executor"

    def reset(self, env: RailEnv) -> None:
        import time

        t0 = time.perf_counter()
        self.env = env
        self.graph = RailGraph(env)
        self.horizon = 10**7  # the env truncates; planning must never give up on a late train
        self.infos = _agent_infos(env, self.graph)
        self.paths = {}
        self.rt = TrainTable()
        self.orderings_tried = 0
        self._build_execution_state()
        self.k = {ai.handle: ai.k for ai in self.infos}
        self.base = {}
        self.version = 0
        self._E_cache = (-1, -1, None)
        self.stats = defaultdict(int)
        self.unplannable = 0
        self.inject(0)
        self.setup_s = time.perf_counter() - t0

    def _try_plan_unrouted(self, now: int) -> None:  # replaced by inject()
        return

    def inject(self, now: int) -> None:
        env = self.env
        new_ids = [ai.handle for ai in self.infos
                   if ai.handle not in self.paths and env.agents[ai.handle].state.is_off_map_state()
                   and env.agents[ai.handle].earliest_departure <= now + 1 and math.isfinite(ai.sp_time)]
        if not new_ids:
            return
        new_ids.sort(key=lambda h: (env.agents[h].latest_arrival - now - self.infos[h].sp_time, h))
        rt = self.live_table(now) if self.paths else TrainTable()
        planned = {}
        for h in new_ids:
            p = sipp_from(self.graph, rt, h, self.infos[h].start, self.k[h], self.horizon, now, False, self.first_move_time(h, now))
            if p is None:
                self.unplannable += 1
                continue
            rt.reserve_future(h, None, p)
            planned[h] = p
        for h in planned:
            self.paths[h] = []
            self.progress[h] = -1
            self.base[h] = []
        self.commit(planned, now)

    def act(self, env: RailEnv):
        self.inject(env._elapsed_steps)
        return super().act(env)


def continuous_metrics(env: RailEnv, budget: int, window_occupancy: Optional[List[int]] = None, terminated: bool = False) -> dict:
    injected = [a for a in env.agents if a.earliest_departure < budget]
    arrived = [a for a in injected if a.state == TrainState.DONE and a.arrival_time is not None]
    delays = [max(0, a.arrival_time - a.latest_arrival) for a in arrived]
    steps = max(1, env._elapsed_steps)
    return dict(
        steps=int(env._elapsed_steps),
        injected=len(injected),
        arrived=len(arrived),
        throughput_per_1000=1000.0 * len(arrived) / steps,
        mean_delay=float(np.mean(delays)) if delays else float("nan"),
        on_time_share=float(np.mean([d == 0 for d in delays])) if delays else float("nan"),
        backlog_off_map=sum(1 for a in injected if a.state.is_off_map_state()),
        on_map_at_end=sum(1 for a in injected if a.state.is_on_map_state()),
        deadlock_events_per_1000=(1000.0 / steps) if terminated else 0.0,
        terminated=bool(terminated),
        window_mean=float(np.mean(window_occupancy)) if window_occupancy else float("nan"),
    )
