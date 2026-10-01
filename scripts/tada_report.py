"""Tables and figures for docs/08-tada-dispatcher.md, built only from files under data/tada/ (plus main's
stored PPO baseline). Missing inputs leave their section untouched.

    python scripts/tada_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_flatland import theme as th  # noqa: E402
from rl_flatland.tada import figures as tf  # noqa: E402

DATA = ROOT / "data" / "tada"
DOC = ROOT / "docs" / "08-tada-dispatcher.md"
FIG = ROOT / "docs" / "assets" / "figures"
GH = "https://github.com/leandergrech/rl-flatland-rescheduling/blob/feat/tada-dispatcher/"

ABLATIONS = [  # name, what changes against main (B=2, M=8, slack features, all actions, no shaping)
    ("main", "B = 2, M = 8, set (i), all actions"),
    ("B1", "B = 1"),
    ("B4", "B = 4"),
    ("noyield", "YIELD_TO disabled"),
    ("M4", "M = 4"),
    ("M16", "M = 16"),
    ("shaping", "dense slack shaping on"),
    ("tree", "feature set (ii): + 3-branch summary"),
]


def load(rel: str):
    p = DATA / rel
    return json.loads(p.read_text()) if p.exists() else None


def link(rel: str) -> str:
    return f"[{Path(rel).name}]({GH}data/tada/{rel})"


def mse(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


def pct(rows, key="arrival_rate"):
    m, s = mse([r[key] for r in rows])
    return f"{100 * m:.1f} ± {100 * s:.1f}"


def num(rows, key, fmt="{:.3f}"):
    return fmt.format(np.mean([r[key] for r in rows]))


def pick(rows, scenario, malf):
    return [r for r in rows if r["scenario"] == scenario and bool(r["malfunctions"]) == malf]


def inject(tag: str, text: str) -> None:
    s = DOC.read_text()
    a, b = f"<!-- {tag}:START -->", f"<!-- {tag}:END -->"
    if a in s and b in s:
        pre, rest = s.split(a, 1)
        _, post = rest.split(b, 1)
        DOC.write_text(pre + a + "\n" + text.strip("\n") + "\n" + b + post)


def figure(name: str, builder, alt: str) -> str:
    th.save_both(builder, name, FIG)
    return th.markdown(name, alt)


# ---------------------------------------------------------------------------------------------- step 1
def verify() -> None:
    v = load("verify.json")
    if not v or "random_clearances" not in v:
        return
    po, rc = v["proceed_only"], v["random_clearances"]
    ref = {(r["scenario"], r["split"] == "test_malfunction", r["seed"]): r
           for r in json.loads((ROOT / "data" / "results" / "or.json").read_text())["rows"]}
    lines = ["| Scenario | Malfunctions | Episodes | Arrived (executor) | Arrived (main's OR) | Identical (arrived, deadlocked, steps, normalised reward) | Terminated | Rotation flags |",
             "|---|---|---|---|---|---|---|---|"]
    for sc in th.SCENARIO_ORDER:
        for m in (False, True):
            rr = pick(po["rows"], sc, m)
            if not rr:
                continue
            same = sum((r["arrived"], r["deadlocked"], r["steps"]) == (ref[(sc, m, r["seed"])]["arrived"], ref[(sc, m, r["seed"])]["deadlocked"], ref[(sc, m, r["seed"])]["steps"])
                       and abs(r["normalized_reward"] - ref[(sc, m, r["seed"])]["normalized_reward"]) < 1e-9 for r in rr)
            lines.append(f"| {sc} | {'on' if m else 'off'} | {len(rr)} | {sum(r['arrived'] for r in rr)} | "
                         f"{sum(ref[(sc, m, r['seed'])]['arrived'] for r in rr)} | {same}/{len(rr)} | {sum(r['terminated'] for r in rr)} | "
                         f"{sum(r.get('rotation_flags', 0) for r in rr)} |")
    rrows = rc["rows"]
    txt = f"""With every clearance forced to PROCEED, the dispatch loop (window, masks, re-timing, termination
checks) must leave main's OR reference untouched. Over all 80 held-out episodes it matches the stored
results ([or.json]({GH}data/results/or.json)) in arrived trains, deadlocked trains, episode length
and normalised reward to 1e-9: **{po['episodes'] - len({(m[0], m[1], m[2]) for m in po['mismatches']})} of
{po['episodes']} episodes identical, {len(po['mismatches'])} mismatching fields** ({link('verify.json')},
commit `{v.get('commit', '?')[:7]}`).

{chr(10).join(lines)}

"Rotation flags" counts steps at which main's `find_deadlocked` flagged a ring of trains that the
plan rotates through a block of switches in one step (see [What did not work](#what-did-not-work)).

The second check drives the dispatcher with a random policy: at every decision point it picks
random choosable trains and, half the time, a random legal clearance (up to B = 2), on
{rc['episodes']} episodes (small and medium, half with malfunctions, seeds 3000 to {3000 + rc['episodes'] - 1}).
That committed **{rc['commits']:,} plan edits** ({np.mean([r['commits'] for r in rrows]):.0f} per episode) with
**{rc['bad_episodes']} episodes** showing a deadlock, a reservation violation or a train leaving its
plan. Arrival under random edits was {100 * np.mean([r['arrival_rate'] for r in rrows if r['scenario'] == 'small']):.1f}%
on small and {100 * np.mean([r['arrival_rate'] for r in rrows if r['scenario'] == 'medium']):.1f}% on medium: random edits are safe
but not free."""
    inject("TADA_VERIFY", txt)


# ---------------------------------------------------------------------------------------------- step 2
def occupancy() -> None:
    occ = load("occupancy_proceed_only.json")
    if not occ:
        return
    md = figure("tada-occupancy", tf.occupancy(8), "Window occupancy over the episode under the executor alone")
    rows = ["| Scenario | Mean trains in window | Steps with the window full (M = 8) | Steps with an empty window |", "|---|---|---|---|"]
    for sc in th.SCENARIO_ORDER:
        v = np.concatenate([np.asarray(x) for k, x in occ.items() if k.split("|")[:2] == [sc, "0"]])
        rows.append(f"| {sc} | {v.mean():.2f} | {100 * np.mean(v >= 8):.0f}% | {100 * np.mean(v == 0):.0f}% |")
    txt = f"""{md}

*Executor alone (every clearance PROCEED), the 10 malfunction-free held-out episodes per scenario,
one sample per env step. Source: {link('occupancy_proceed_only.json')}, written by `scripts/tada_verify.py`.*

{chr(10).join(rows)}

The window never exceeds M by construction, and every train in it has a legal action (both are
asserted in `tests/test_tada.py`). On medium and larger maps the window is full for much of the
episode, so ranking by slack is doing real selection there."""
    inject("TADA_OCCUPANCY", txt)


# ---------------------------------------------------------------------------------------------- step 3
def _ppo_baseline():
    s = json.loads((ROOT / "data" / "results" / "ppo.json").read_text())
    return [r for r in s["rows"] if r["scenario"] == "medium"]


def medium() -> None:
    ex, le = load("results/executor.json"), load("results/main.json")
    if not ex or not le:
        return
    ppo = _ppo_baseline()
    head = ["| Controller | Arrival, no malf. (%) | Arrival, malf. (%) | Norm. reward, no malf. | Norm. reward, malf. | Terminations | Truncations | Wall-clock per step (ms) |",
            "|---|---|---|---|---|---|---|---|"]

    def row(name, rows, timing=True):
        a, b = pick(rows, "medium", False), pick(rows, "medium", True)
        term = sum(r.get("terminated", 0) for r in a + b) if timing else "–"
        trunc = sum(r.get("truncated", 0) for r in a + b) if timing else "–"
        ms = f"{np.mean([r['decision_ms_per_step'] for r in a + b]):.1f}" if timing else "–"
        return f"| {name} | {pct(a)} | {pct(b)} | {num(a, 'normalized_reward')} | {num(b, 'normalized_reward')} | {term} | {trunc} | {ms} |"

    ppo_rows = [dict(r, malfunctions=r["split"] == "test_malfunction") for r in ppo]
    lines = head + [row("Executor only (main's OR, every clearance PROCEED)", ex["rows"]),
                    row("Learned dispatcher (TADA on rails)", le["rows"]),
                    row("main's best learned baseline (PPO, compact obs.)", ppo_rows, timing=False)]
    cfg = json.loads((DATA / "checkpoints" / "main" / "config.json").read_text())
    wc = json.loads((DATA / "checkpoints" / "main" / "wall_clock.json").read_text())
    log = [json.loads(l) for l in (DATA / "checkpoints" / "main" / "train_log.jsonl").read_text().splitlines() if l.strip()]
    mhz = [r["cpu_mhz"] for r in log if "cpu_mhz" in r]
    bars = figure("tada-medium", tf.grouped_bars(
        ["no malfunctions", "malfunctions"],
        {"Executor only": [100 * np.mean([r["arrival_rate"] for r in pick(ex["rows"], "medium", m)]) for m in (False, True)],
         "Learned dispatcher": [100 * np.mean([r["arrival_rate"] for r in pick(le["rows"], "medium", m)]) for m in (False, True)],
         "main's PPO": [100 * np.mean([r["arrival_rate"] for r in pick(ppo_rows, "medium", m)]) for m in (False, True)]},
        {}, "Trains arrived on medium, 10 held-out seeds (%)", "arrived (%)", ylim=(0, 105)), "Arrival on medium by controller")
    curve = figure("tada-training", tf.training_curve(["main"], {"main": "Learned dispatcher (B = 2, M = 8)"}), "Training curve of the dispatcher")
    clr = {}
    for r in le["rows"]:
        for k, v in r["clearances"].items():
            clr[k] = clr.get(k, 0) + v
    names = {"clr_0": "PROCEED", "clr_1": "HOLD", "clr_2": "YIELD_TO", "clr_3": "REROUTE"}
    tot = sum(clr.values()) or 1
    mix = ", ".join(f"{names[k]} {100 * clr.get(k, 0) / tot:.1f}%" for k in sorted(names))
    txt = f"""Medium (50×50, 30 trains), 10 held-out seeds (1000 to 1009), each with malfunctions off and on,
greedy policy. Arrival is mean ± standard error over seeds. Terminations and truncations are summed
over the 20 episodes. Wall-clock per step covers everything: window, masks, policy, plan edits and
the env step.

{chr(10).join(lines)}

Sources: {link('results/executor.json')}, {link('results/main.json')} (`scripts/tada_evaluate.py`),
main's PPO from [ppo.json]({GH}data/results/ppo.json).

{bars}

Training: {wc.get('iterations_done', len(log))} PPO iterations of {cfg['episodes_per_iter']} medium episodes with malfunctions
({wc.get('env_steps', 0):,} env steps) in {wc['train_wall_s'] / 60:.0f} minutes on {cfg['workers']} worker processes,
CPU clock {min(mhz):.0f} to {max(mhz):.0f} MHz across iterations (median {np.median(mhz):.0f}).
Log: {link('checkpoints/main/train_log.jsonl')}. Clearances issued by the greedy policy on the 20 evaluation
episodes: {mix}.

{curve}"""
    inject("TADA_MEDIUM", txt)


# ---------------------------------------------------------------------------------------------- step 4
def ablations() -> None:
    have = [(n, d) for n, d in ABLATIONS if load(f"results/{n}.json")]
    if len(have) < 2:
        return
    lines = ["| Run | Change | Arrival, no malf. (%) | Arrival, malf. (%) | Norm. reward, malf. | Terminations | Truncations | Edits per episode | Wall-clock per step (ms) | Iterations done | Training (min) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    series_a, series_b = [], []
    for n, d in have:
        rows = load(f"results/{n}.json")["rows"]
        a, b = pick(rows, "medium", False), pick(rows, "medium", True)
        wc = json.loads((DATA / "checkpoints" / n / "wall_clock.json").read_text())
        lines.append(f"| `{n}` | {d} | {pct(a)} | {pct(b)} | {num(b, 'normalized_reward')} | {sum(r['terminated'] for r in a + b)} | "
                     f"{sum(r['truncated'] for r in a + b)} | {np.mean([r['commits'] for r in a + b]):.1f} | "
                     f"{np.mean([r['decision_ms_per_step'] for r in a + b]):.1f} | {wc.get('iterations_done', '?')}/{wc.get('iterations_planned', '?')} | "
                     f"{wc['train_wall_s'] / 60:.0f} |")
        series_a.append(100 * np.mean([r["arrival_rate"] for r in a]))
        series_b.append(100 * np.mean([r["arrival_rate"] for r in b]))
    bars = figure("tada-ablations", tf.grouped_bars([n for n, _ in have], {"no malfunctions": series_a, "malfunctions": series_b}, {},
                                                    "Ablations: trains arrived on medium (%)", "arrived (%)", ylim=(0, 105)),
                  "Ablation results on medium")
    txt = f"""Every run uses the same sample budget (up to 120 iterations of 8 medium episodes with malfunctions)
with a 55-minute wall-clock cap, and is evaluated like the main run (10 seeds × malfunctions off/on,
greedy). "Iterations done" shows where the cap cut a run short.

{chr(10).join(lines)}

Sources: `data/tada/results/<run>.json` and `data/tada/checkpoints/<run>/` for each run.

{bars}"""
    inject("TADA_ABLATIONS", txt)


# ---------------------------------------------------------------------------------------------- step 5
def general() -> None:
    g, v = load("results/main_general.json"), load("verify.json")
    if not g or not v:
        return
    lines = ["| Scenario | Executor arrival, no malf. (%) | Learned, no malf. (%) | Executor arrival, malf. (%) | Learned, malf. (%) | Learned terminations | Learned truncations | Wall-clock per step (ms) |",
             "|---|---|---|---|---|---|---|---|"]
    for sc in ["small", "large", "xlarge"]:
        ea, eb = pick(v["proceed_only"]["rows"], sc, False), pick(v["proceed_only"]["rows"], sc, True)
        la, lb = pick(g["rows"], sc, False), pick(g["rows"], sc, True)
        if not la:
            continue
        lines.append(f"| {sc} | {pct(ea)} | {pct(la)} | {pct(eb)} | {pct(lb)} | {sum(r['terminated'] for r in la + lb)} | "
                     f"{sum(r['truncated'] for r in la + lb)} | {np.mean([r['decision_ms_per_step'] for r in la + lb]):.1f} |")
    txt = f"""The policy trained on medium, evaluated without retraining on the other three scenarios (10
held-out seeds each, malfunctions off and on). The executor columns are the PROCEED-only runs from
step 1.

{chr(10).join(lines)}

Sources: {link('results/main_general.json')}, {link('verify.json')}."""
    inject("TADA_GENERAL", txt)


# ---------------------------------------------------------------------------------------------- step 6
def continuous() -> None:
    c = load("results/continuous.json")
    if not c:
        return
    rows = c["rows"]
    ctrls = [x for x in ["executor", "learned"] if any(r["controller"] == x for r in rows)]
    rates = sorted({r["rate"] for r in rows})
    lines = ["| Rate (trains/step) | Controller | Injected | Throughput (arrivals/1000 steps) | Mean delay vs LA (steps) | On time (%) | Waiting off-map at end | Deadlock terminations | Mean window occupancy | Wall-clock per step (ms) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    agg = {}
    for r in rates:
        for ctl in ctrls:
            rr = [x for x in rows if x["rate"] == r and x["controller"] == ctl]
            if not rr:
                continue
            m = {k: float(np.nanmean([x[k] for x in rr])) for k in ["injected", "throughput_per_1000", "mean_delay", "on_time_share", "backlog_off_map", "window_mean", "decision_ms_per_step"]}
            m["terms"] = sum(x["terminated"] for x in rr)
            agg[(r, ctl)] = m
            lines.append(f"| {r:g} | {ctl} | {m['injected']:.0f} | {m['throughput_per_1000']:.1f} | {m['mean_delay']:.1f} | {100 * m['on_time_share']:.0f} | "
                         f"{m['backlog_off_map']:.0f} | {m['terms']}/{len(rr)} | {m['window_mean']:.2f} | {m['decision_ms_per_step']:.0f} |")
    fig = figure("tada-continuous", tf.continuous_sweep(agg, rates, ctrls), "Throughput and delay against injection rate")
    seeds = sorted({r["arrival_seed"] for r in rows})
    txt = f"""One medium map (seed 5000), a pool of 400 trains whose earliest departures follow a Poisson process at
the given rate, latest arrival = injection + ceil(1.3 τ + 0.2 τ̄) (Flatland 3's allowance), malfunctions
on, truncated at {c['budget']} steps; {len(seeds)} arrival seeds per rate. Both controllers plan a train only once
it appears (`OnlineExecutor`). Delay is measured on trains that arrived, so at high rates it understates
the delay of the backlog, which is reported separately.

{chr(10).join(lines)}

Source: {link('results/continuous.json')} (`scripts/tada_continuous.py`).

{fig}"""
    inject("TADA_CONTINUOUS", txt)


if __name__ == "__main__":
    for f in (verify, occupancy, medium, ablations, general, continuous):
        f()
        print("done", f.__name__)
