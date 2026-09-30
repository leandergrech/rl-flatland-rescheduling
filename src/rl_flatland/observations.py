"""Per-train observations for the RL baselines.

Two encoders, both returning a flat float32 vector for one train at a decision point:

* ``CompactObs`` (default, 28 features). Own state, timetable slack and, for each of the two
  branches a train can take (the shortest-path branch and the alternative, if any), a short look
  ahead along the path: how far to the nearest opposing train, how many opposing trains, how far
  to the train ahead, how long the single-track stretch is. The question a dispatcher asks at a
  switch is "if I go this way now, will I meet someone coming the other way?", and these features
  answer it directly.
* ``TreeObs`` (260 features). flatland's own ``TreeObsForRailEnv`` (depth 2, shortest-path
  predictor over 20 steps, 12 features per node), flattened and normalised the way flatland's
  ``FlattenedNormalizedTreeObsForRailEnv`` and the NeurIPS 2020 starter kits do it, plus 8 of the
  own features above. Branches are indexed Left/Forward/Right/Back, not best/alternative. The
  flattening is re-implemented here (``_flatten_tree``) because flatland.ml imports ray.

``CompactObs`` needs no observation builder inside the env. ``TreeObs`` must be installed as the
env's observation builder (see ``TreeObs.builder``), because the tree is computed in ``env.step``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.step_utils.states import TrainState

from rl_flatland.graph import Config, RailGraph, to_config

LOOKAHEAD = 30  # cells scanned along each branch


def branch_options(graph: RailGraph, handle: int, cfg: Config) -> List[Tuple[Config, float]]:
    """Up to two successors of ``cfg`` with a finite distance to target, best first."""
    ranked = [(s, d) for s, d in graph.ranked_successors(handle, cfg) if np.isfinite(d)]
    return ranked[:2]


class _Occupancy:
    """Cell -> (handle, configuration) for trains on the map, rebuilt once per env step."""

    def __init__(self, env: RailEnv):
        self.at: Dict[Tuple[int, int], Tuple[int, Config]] = {}
        for a in env.agents:
            if a.state.is_on_map_state():
                c = to_config(a.current_configuration)
                if c is not None:
                    self.at[(c[0], c[1])] = (a.handle, c)


def _own_features(env: RailEnv, graph: RailGraph, handle: int, cfg: Config, n_done: int, n_on_map: int) -> np.ndarray:
    a = env.agents[handle]
    T = float(env._max_episode_steps)
    t = float(env._elapsed_steps)
    k = 1.0 / float(a.speed_counter.max_speed)
    d = graph.distance(handle, cfg)
    d = d if np.isfinite(d) else float(env.width + env.height)
    s = a.state
    onehot = [
        float(s.is_off_map_state()),
        float(s == TrainState.MOVING),
        float(s == TrainState.STOPPED),
        float(s.is_malfunction_state()),
    ]
    slack = (a.latest_arrival - t - k * d) / T
    n = env.get_num_agents()
    return np.array(
        onehot
        + [
            float(a.speed_counter.max_speed),
            min(a.malfunction_handler.malfunction_down_counter / 50.0, 1.0),
            d / float(env.width + env.height),
            float(np.clip(slack, -1.0, 1.0)),
            t / T,
            n_done / n,
            n_on_map / n,
            float(len(graph.successors(cfg)) > 1),
        ],
        dtype=np.float32,
    )


def _scan_branch(graph: RailGraph, occ: _Occupancy, handle: int, prev: Config, first: Config) -> np.ndarray:
    """Walk up to LOOKAHEAD cells along the shortest path starting with ``first``."""
    L = float(LOOKAHEAD)
    dist_opp, n_opp, dist_same, opp_malf, seg_len = L, 0, L, 0.0, L
    first_occupied = float((first[0], first[1]) in occ.at)
    for step, (cell, prev_cell, is_switch, _) in enumerate(graph.lookahead(handle, prev, first, LOOKAHEAD), start=1):
        if seg_len == L and is_switch and step > 1:
            seg_len = float(step - 1)
        hit = occ.at.get(cell)
        if hit is not None and hit[0] != handle:
            other, ocfg = hit
            if any((s[0], s[1]) == prev_cell for s in graph.successors(ocfg)):
                n_opp += 1
                if dist_opp == L:
                    dist_opp = float(step)
                    opp_malf = float(graph.env.agents[other].malfunction_handler.in_malfunction)
            elif dist_same == L:
                dist_same = float(step)
    return np.array(
        [1.0, dist_opp / L, min(n_opp / 5.0, 1.0), dist_same / L, first_occupied, opp_malf, seg_len / L],
        dtype=np.float32,
    )


class CompactObs:
    """28-dimensional hand-built observation (see module docstring)."""

    name = "compact"
    dim = 12 + 2 * 8

    def builder(self):
        return None  # computed on demand, no flatland observation builder needed

    def prepare(self, env: RailEnv, graph: RailGraph) -> None:
        """Call once per env step before ``get`` for any train."""
        self.env, self.graph = env, graph
        self.occ = _Occupancy(env)
        self.n_done = sum(a.state == TrainState.DONE for a in env.agents)
        self.n_on_map = sum(a.state.is_on_map_state() for a in env.agents)

    def get(self, handle: int, cfg: Config) -> np.ndarray:
        """``cfg`` is the train's current configuration, or its start configuration if off-map."""
        env, graph = self.env, self.graph
        own = _own_features(env, graph, handle, cfg, self.n_done, self.n_on_map)
        opts = branch_options(graph, handle, cfg)
        d_best = opts[0][1] if opts else 0.0
        feats = [own]
        off_map = env.agents[handle].state.is_off_map_state()
        for j in range(2):
            if j < len(opts):
                s, d = opts[j]
                if off_map:
                    # an off-map train enters cfg itself; scan from there
                    scan = _scan_branch(graph, self.occ, handle, cfg, cfg) if j == 0 else np.zeros(7, np.float32)
                    extra = 0.0
                else:
                    scan = _scan_branch(graph, self.occ, handle, cfg, s)
                    extra = (d - d_best) / float(env.width + env.height)
                feats.append(np.concatenate([[extra], scan]).astype(np.float32))
            else:
                feats.append(np.zeros(8, np.float32))
        return np.concatenate(feats)


def _node_groups(node) -> Tuple[List[float], List[float], List[float]]:
    data = [node.dist_own_target_encountered, node.dist_other_target_encountered, node.dist_other_agent_encountered,
            node.dist_potential_conflict, node.dist_unusable_switch, node.dist_to_next_branch]
    agent = [node.num_agents_same_direction, node.num_agents_opposite_direction, node.num_agents_malfunctioning,
             node.speed_min_fractional, node.num_agents_ready_to_depart]
    return data, [node.dist_min_to_target], agent


def _flatten_tree(node, depth: int, max_depth: int, out: Tuple[list, list, list]) -> None:
    """Pre-order (node, L, F, R, B) flattening into three feature groups, padding missing subtrees."""
    if node == -np.inf or node is None:
        n = (4 ** (max_depth - depth + 1) - 1) // 3
        out[0].extend([-np.inf] * 6 * n)
        out[1].extend([-np.inf] * n)
        out[2].extend([-np.inf] * 5 * n)
        return
    d, dist, ag = _node_groups(node)
    out[0].extend(d)
    out[1].extend(dist)
    out[2].extend(ag)
    if depth == max_depth:
        return
    for c in TreeObsForRailEnv.tree_explored_actions_char:
        _flatten_tree(node.childs.get(c, -np.inf) if node.childs else -np.inf, depth + 1, max_depth, out)


def _norm_clip(x: np.ndarray, fixed_radius: float = 0, normalize_to_range: bool = False) -> np.ndarray:
    """Normalisation from the NeurIPS 2020 starter kit (as in flatland's FlattenedNormalizedTreeObs)."""
    if fixed_radius > 0:
        max_obs = fixed_radius
    else:
        finite = x[(x >= 0) & (x < 1000)]
        max_obs = max(1.0, float(finite.max()) if finite.size else 0.0) + 1
    min_obs = 0.0
    if normalize_to_range:
        pos = x[x >= 0]
        min_obs = float(pos.min()) if pos.size else np.inf
    if min_obs > max_obs:
        min_obs = max_obs
    if max_obs == min_obs:
        return np.clip(x / max_obs, -1, 1)
    return np.clip((x - min_obs) / abs(max_obs - min_obs), -1, 1)


class TreeObs:
    """flatland's depth-2 tree observation, flattened and normalised, plus own features."""

    name = "tree"

    def __init__(self, max_depth: int = 2, predictor_depth: int = 20, radius: int = 10):
        self.max_depth = max_depth
        self.predictor_depth = predictor_depth
        self.radius = radius
        n_nodes = sum(4**i for i in range(max_depth + 1))
        self.dim = 12 * n_nodes + 8

    def builder(self):
        return TreeObsForRailEnv(max_depth=self.max_depth, predictor=ShortestPathPredictorForRailEnv(max_depth=self.predictor_depth))

    def prepare(self, env: RailEnv, graph: RailGraph) -> None:
        self.env, self.graph = env, graph
        self.n_done = sum(a.state == TrainState.DONE for a in env.agents)
        self.n_on_map = sum(a.state.is_on_map_state() for a in env.agents)

    def encode_tree(self, node) -> np.ndarray:
        if node is None or node == -np.inf:
            return np.zeros(self.dim - 8, np.float32)
        out: Tuple[list, list, list] = ([], [], [])
        _flatten_tree(node, 0, self.max_depth, out)
        data = _norm_clip(np.array(out[0], dtype=float), fixed_radius=self.radius)
        dist = _norm_clip(np.array(out[1], dtype=float), normalize_to_range=True)
        agent = np.clip(np.array(out[2], dtype=float), -1, 1)
        v = np.concatenate([data, dist, agent])
        return np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    def get(self, handle: int, cfg: Config) -> np.ndarray:
        tree = self.encode_tree(self.env.obs_dict.get(handle))
        own = _own_features(self.env, self.graph, handle, cfg, self.n_done, self.n_on_map)[:8]
        return np.concatenate([tree, own]).astype(np.float32)


def make_obs(name: str):
    if name == "compact":
        return CompactObs()
    if name == "tree":
        return TreeObs()
    raise ValueError(name)
