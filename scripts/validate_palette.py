"""Validate a categorical chart palette with computed (not eyeballed) colour checks.

Python port of the checks used for this site's charts:

2. Lightness band: OKLCH L in [0.43, 0.77] (light mode) or [0.48, 0.67] (dark mode).
3. Chroma floor: OKLCH C >= 0.10, otherwise a hue reads as grey.
4. CVD separation: OKLab Delta E x100 between the compared pairs under protanopia and
   deuteranopia (Machado, Oliveira & Fernandes 2009, severity 1.0): target >= 8, floor >= 6.
4b. Normal-vision floor: worst unsimulated Delta E >= 15 (hard gate).
5. Contrast vs surface: WCAG ratio >= 3:1, otherwise the chart needs a relief channel (labels or
   a table view).

Adjacent pairs are compared by default (bars, stacks, lines); use --pairs all for charts where
any two colours can sit side by side.

    python scripts/validate_palette.py "#c98500,#3987e5,..." --mode dark --surface "#0f1115"
"""

from __future__ import annotations

import argparse
import math
import sys
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR, CVD_TARGET, CVD_FLOOR, NORMAL_FLOOR, CONTRAST_MIN = 0.10, 8.0, 6.0, 15.0, 3.0
MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}


def _srgb(h: str) -> List[float]:
    h = h.strip().lstrip("#")
    return [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]


def _lin(h: str) -> List[float]:
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in _srgb(h)]


def contrast(a: str, b: str) -> float:
    def lum(h):
        r, g, bl = _lin(h)
        return 0.2126 * r + 0.7152 * g + 0.0722 * bl

    hi, lo = sorted([lum(a), lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _oklab_lin(rgb: Sequence[float]) -> Tuple[float, float, float]:
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


def oklch(h: str) -> Tuple[float, float]:
    L, a, b = _oklab_lin(_lin(h))
    return L, math.hypot(a, b)


def _simulate(h: str, kind: str) -> List[float]:
    r, g, b = _lin(h)
    M = MACHADO[kind]
    return [min(1.0, max(0.0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3)]


def delta_e(a: str, b: str, kind: str | None = None) -> float:
    pa = _oklab_lin(_simulate(a, kind) if kind else _lin(a))
    pb = _oklab_lin(_simulate(b, kind) if kind else _lin(b))
    return 100 * math.dist(pa, pb)


def validate(palette: Sequence[str], mode: str = "light", surface: str = "#fcfcfb", pairs: str = "adjacent") -> Dict:
    lo, hi = BAND[mode]
    idx = list(combinations(range(len(palette)), 2)) if pairs == "all" else [(i, i + 1) for i in range(len(palette) - 1)]
    off = [(c, round(oklch(c)[0], 3)) for c in palette if not lo <= oklch(c)[0] <= hi]
    grey = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    cvd = min(((delta_e(palette[i], palette[j], k), k, palette[i], palette[j]) for k in ("protan", "deutan") for i, j in idx), default=(99, "", "", ""))
    normal = min(((delta_e(palette[i], palette[j]), palette[i], palette[j]) for i, j in idx), default=(99, "", ""))
    low = [(c, round(contrast(c, surface), 2)) for c in palette if contrast(c, surface) < CONTRAST_MIN]
    cvd_state = "PASS" if cvd[0] >= CVD_TARGET else ("WARN" if cvd[0] >= CVD_FLOOR else "FAIL")
    report = {
        "lightness band": ("PASS" if not off else "FAIL", off or f"all in L {lo}-{hi}"),
        "chroma floor": ("PASS" if not grey else "FAIL", grey or f"all >= {CHROMA_FLOOR}"),
        "CVD separation": (cvd_state, f"worst {cvd[2]}<->{cvd[3]} dE {cvd[0]:.1f} ({cvd[1]})"),
        "normal-vision floor": ("PASS" if normal[0] >= NORMAL_FLOOR else "FAIL", f"worst {normal[1]}<->{normal[2]} dE {normal[0]:.1f}"),
        "contrast vs surface": ("PASS" if not low else "WARN", low or f"all >= {CONTRAST_MIN}:1"),
    }
    ok = all(v[0] != "FAIL" for v in report.values())
    return {"ok": ok, "report": report, "min_cvd": cvd[0], "min_normal": normal[0]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("palette")
    p.add_argument("--mode", choices=["light", "dark"], default="light")
    p.add_argument("--surface", default=None)
    p.add_argument("--pairs", choices=["adjacent", "all"], default="adjacent")
    a = p.parse_args()
    pal = [c.strip() for c in a.palette.split(",") if c.strip()]
    surface = a.surface or ("#fcfcfb" if a.mode == "light" else "#1a1a19")
    res = validate(pal, a.mode, surface, a.pairs)
    print(f"Palette ({a.mode}, surface {surface}, {a.pairs} pairs): {len(pal)} slots")
    for k, (state, detail) in res["report"].items():
        print(f"  [{state}] {k:22s} {detail}")
    print("  ->", "ALL CHECKS PASS" if res["ok"] else "FAILED")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
