"""Colour and type tokens for figures, matching the site theme ("Departure board").

The categorical palette was validated with scripts/validate_palette.py on the site's own surfaces
(light #f8f6f1, dark #0f1115). Adjacent-pair colour-blind separation is dE 16.3 (light) and 13.0
(dark), against a target of 8; normal-vision separation is at least 19.3. Slots are assigned in this
fixed order and never cycled. Four light-mode slots are below 3:1 contrast on the paper surface, so
every chart ships next to a table with the same numbers.

Figures are saved in both modes; pages show them with mkdocs-material's #only-light / #only-dark.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
from matplotlib import font_manager

# Inter is registered explicitly when installed (CFF .otf files are not always in matplotlib's cache).
for _f in sorted(Path("/usr/share/fonts/opentype/inter").glob("Inter-*.otf")) if Path("/usr/share/fonts/opentype/inter").exists() else []:
    try:
        font_manager.fontManager.addfont(str(_f))
    except Exception:
        pass

MODES = ("light", "dark")

TOKENS: Dict[str, Dict[str, str]] = {
    "light": dict(surface="#f8f6f1", ink="#161614", ink2="#5c5a54", grid="#e4e0d6", axis="#b9b4a8", rail="#8f8a7e", accent="#9b5a00"),
    "dark": dict(surface="#0f1115", ink="#ece9e0", ink2="#a39f95", grid="#23262d", axis="#4a4d55", rail="#6b6e76", accent="#ffc93c"),
}

# slot order: amber, magenta, green, violet, orange, blue, aqua
SERIES: Dict[str, List[str]] = {
    "light": ["#eda100", "#e87ba4", "#008300", "#4a3aa7", "#eb6834", "#2a78d6", "#1baf7a"],
    "dark": ["#c98500", "#d55181", "#008300", "#9085e9", "#d95926", "#3987e5", "#199e70"],
}

# status colours are fixed (never themed) and always paired with a text label
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

POLICY_ORDER = ["or_pp_sipp", "ppo", "bc", "bc_ppo", "ppo_tree", "reactive_avoid", "shortest_path"]
POLICY_LABEL = {
    "or_pp_sipp": "OR: PP+SIPP+ordered execution",
    "ppo": "PPO, compact obs",
    "bc": "BC from OR",
    "bc_ppo": "BC then PPO",
    "ppo_tree": "PPO, tree obs",
    "reactive_avoid": "Reactive rule",
    "shortest_path": "Shortest path, no coordination",
}
SHORT_LABEL = {
    "or_pp_sipp": "OR",
    "ppo": "PPO",
    "bc": "BC",
    "bc_ppo": "BC→PPO",
    "ppo_tree": "PPO tree",
    "reactive_avoid": "Reactive",
    "shortest_path": "Shortest",
}
SCENARIO_ORDER = ["small", "medium", "large", "xlarge"]
SCENARIO_LABEL = {"small": "small\n30×30, 10", "medium": "medium\n50×50, 30", "large": "large\n80×80, 60", "xlarge": "xlarge\n100×100, 100"}


def color(policy: str, mode: str) -> str:
    return SERIES[mode][POLICY_ORDER.index(policy)]


def slot(i: int, mode: str) -> str:
    return SERIES[mode][i]


@contextmanager
def style(mode: str):
    t = TOKENS[mode]
    rc = {
        "font.family": ["Inter", "DejaVu Sans"],
        "font.size": 9.5,
        "svg.fonttype": "path",
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "axes.edgecolor": t["axis"],
        "axes.labelcolor": t["ink2"],
        "axes.titlecolor": t["ink"],
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": t["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": t["ink2"],
        "ytick.color": t["ink2"],
        "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink2"],
        "text.color": t["ink"],
        "legend.frameon": False,
        "legend.labelcolor": t["ink"],
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    }
    with plt.rc_context(rc):
        yield t


def save_both(builder: Callable[[str], "plt.Figure"], name: str, out_dir: str | Path) -> List[Path]:
    """Build the figure once per mode and save ``{name}-light.svg`` and ``{name}-dark.svg``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for mode in MODES:
        with style(mode):
            fig = builder(mode)
            p = out_dir / f"{name}-{mode}.svg"
            fig.savefig(p, bbox_inches="tight", pad_inches=0.12, metadata={"Date": None})
            plt.close(fig)
            paths.append(p)
    return paths


def markdown(name: str, alt: str, rel: str = "assets/figures") -> str:
    """Markdown for a light/dark image pair."""
    return f"![{alt}]({rel}/{name}-light.svg#only-light)\n![{alt}]({rel}/{name}-dark.svg#only-dark)"
