"""Fixed mid-size Flatland scenarios and seed splits.

Every scenario is a ``sparse_rail_generator`` city network with a Flatland-3-style mixed speed
profile. A scenario plus a seed fully determines the rail network, the lines (start, target,
speed), the timetable and, if enabled, the malfunction sequence, so results are reproducible
across machines that run the same flatland-rl version (4.3.0 here).

Scope: 30x30 to 100x100 cells and 10 to 100 trains, small enough that every baseline trains and
evaluates on a laptop CPU inside one hour. The NeurIPS 2020 evaluation went to 314x314 cells and
6,256 trains; see docs/05-limitations.md for what changes at that scale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from flatland.core.env_observation_builder import DummyObservationBuilder, ObservationBuilder
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator

# Flatland 3 round-2 style mix: a quarter of the trains at each of 1, 1/2, 1/3 and 1/4 cells/step.
MIXED_SPEEDS: Dict[float, float] = {1.0: 0.25, 1.0 / 2.0: 0.25, 1.0 / 3.0: 0.25, 1.0 / 4.0: 0.25}


@dataclass(frozen=True)
class Scenario:
    """Generator configuration for one scenario family."""

    name: str
    width: int
    height: int
    n_agents: int
    n_cities: int
    max_rails_between_cities: int = 2
    max_rail_pairs_in_city: int = 2
    speed_ratio_map: Dict[float, float] = field(default_factory=lambda: dict(MIXED_SPEEDS))
    # Malfunctions (only used when ``malfunctions=True`` is passed to make_env).
    # Per-train, per-step malfunction probability is 1 - exp(-rate) in flatland-rl 4.3.0.
    malfunction_rate: float = 1.0 / 1000.0
    malfunction_min_duration: int = 20
    malfunction_max_duration: int = 50

    def to_dict(self) -> dict:
        d = asdict(self)
        d["speed_ratio_map"] = {str(k): v for k, v in self.speed_ratio_map.items()}
        return d


SCENARIOS: Dict[str, Scenario] = {
    "small": Scenario("small", 30, 30, 10, 2),
    "medium": Scenario("medium", 50, 50, 30, 4),
    "large": Scenario("large", 80, 80, 60, 6),
    "xlarge": Scenario("xlarge", 100, 100, 100, 8),
}

# Seed splits. Training code may use any seed in TRAIN_SEEDS; evaluation uses the held-out sets.
TRAIN_SEEDS = range(0, 1000)
TEST_SEEDS: List[int] = list(range(1000, 1010))
TEST_MALFUNCTION_SEEDS: List[int] = list(range(2000, 2010))

SEED_SPLITS = {
    "test": (TEST_SEEDS, False),
    "test_malfunction": (TEST_MALFUNCTION_SEEDS, True),
}


def make_env(
    scenario: str | Scenario,
    seed: int,
    malfunctions: bool = False,
    obs_builder: Optional[ObservationBuilder] = None,
    rewards=None,
) -> RailEnv:
    """Build and reset a RailEnv for ``scenario`` and ``seed``.

    The returned env has already been reset with ``random_seed=seed``, so ``env.agents``,
    ``env.rail`` and the timetable are populated. The initial observations are available as
    ``env.obs_dict``. ``rewards`` swaps the scoring function (default: flatland's DefaultRewards,
    i.e. Flatland 3 scoring; e.g. ``flatland.envs.rewards.ECML2026Rewards()``).
    """
    sc = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    malfunction_generator = None
    if malfunctions:
        malfunction_generator = ParamMalfunctionGen(
            MalfunctionParameters(
                malfunction_rate=sc.malfunction_rate,
                min_duration=sc.malfunction_min_duration,
                max_duration=sc.malfunction_max_duration,
            )
        )
    # RailEnv's default builder (GlobalObsForRailEnv) builds full-grid tensors for every train on
    # every step. Nothing here reads them, so use a dummy unless a builder is asked for. It uses no
    # randomness, so dynamics and results are identical either way; only env.step gets faster.
    kwargs = {"obs_builder_object": obs_builder if obs_builder is not None else DummyObservationBuilder()}
    if rewards is not None:
        kwargs["rewards"] = rewards
    env = RailEnv(
        width=sc.width,
        height=sc.height,
        number_of_agents=sc.n_agents,
        rail_generator=sparse_rail_generator(
            max_num_cities=sc.n_cities,
            grid_mode=False,
            max_rails_between_cities=sc.max_rails_between_cities,
            max_rail_pairs_in_city=sc.max_rail_pairs_in_city,
        ),
        line_generator=sparse_line_generator(sc.speed_ratio_map),
        malfunction_generator=malfunction_generator,
        random_seed=seed,
        **kwargs,
    )
    env.reset(random_seed=seed)
    return env
