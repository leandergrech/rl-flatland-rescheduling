"""The chart palette must keep passing the colour checks, and the themed figure pipeline must run."""

import importlib.util
from pathlib import Path

from rl_flatland import theme as th

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validate_palette", ROOT / "scripts" / "validate_palette.py")
vp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vp)


def test_palette_passes_checks_in_both_modes():
    for mode in th.MODES:
        res = vp.validate(th.SERIES[mode], mode, th.TOKENS[mode]["surface"], "adjacent")
        assert res["ok"], res["report"]
        assert res["min_cvd"] >= 8.0 and res["min_normal"] >= 15.0


def test_validator_matches_reference_numbers():
    ref = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    res = vp.validate(ref, "light", "#fcfcfb")
    assert round(res["min_cvd"], 1) == 9.1 and round(res["min_normal"], 1) == 19.6


def test_figure_builds_in_both_modes(tmp_path):
    from rl_flatland.figures import or_components

    paths = th.save_both(or_components(), "or-components", tmp_path)
    assert [p.name for p in paths] == ["or-components-light.svg", "or-components-dark.svg"]
    assert all(p.stat().st_size > 1000 for p in paths)
