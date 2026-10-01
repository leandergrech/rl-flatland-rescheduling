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
    ("main_it60", "main run at 60 iterations: B = 2, M = 8, set (i), all actions"),
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
    n_same = po['episodes'] - len({(m[0], m[1], m[2]) for m in po['mismatches']})
    txt = f"""With every clearance forced to PROCEED, the dispatch loop (window, masks, re-timing, termination
checks) must leave main's OR reference untouched. Over all 80 held-out episodes it is compared with
the stored results ([or.json]({GH}data/results/or.json)) on arrived trains, deadlocked trains,
episode length and normalised reward (to 1e-9).
**{n_same} of {po['episodes']} episodes are identical, with {len(po['mismatches'])} mismatching fields**
({link('verify.json')}, commit `{v.get('commit', '?')[:7]}`).

{chr(10).join(lines)}

"Rotation flags" counts steps at which main's `find_deadlocked` flagged a ring of trains that the
plan rotates through a block of switches in one step (see [What did not work](#what-did-not-work)).

The second check drives the dispatcher with a random policy: at every decision point it picks
random choosable trains and, half the time, a random legal clearance (up to B = 2), on
{rc['episodes']} episodes (small and medium, half with malfunctions, seeds 3000 to {3000 + rc['episodes'] - 1}).
That committed **{rc['commits']:,} plan edits** ({np.mean([r['commits'] for r in rrows]):.0f} per episode).
**{rc['bad_episodes']} episodes** showed a deadlock, a reservation violation or a train leaving its plan.
Arrival under random edits was {100 * np.mean([r['arrival_rate'] for r in rrows if r['scenario'] == 'small']):.1f}% on small and
{100 * np.mean([r['arrival_rate'] for r in rrows if r['scenario'] == 'medium']):.1f}% on medium, against
{100 * np.mean([r['arrival_rate'] for r in po['rows'] if r['scenario'] == 'small']):.1f}% and {100 * np.mean([r['arrival_rate'] for r in po['rows'] if r['scenario'] == 'medium']):.1f}% for the
executor alone on the held-out seeds (different seeds, so only indicative): random edits are safe but not free."""
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
    ref = load("results/executor_train_seeds.json")
    by_seed = {r["seed"]: r for r in ref["rows"]} if ref else None
    curve = figure("tada-training", tf.training_curve(
        ["main", "main_ent01_stopped"],
        {"main": "main run (entropy coef. 0.001)", "main_ent01_stopped": "first run, entropy coef. 0.01 (stopped)"},
        by_seed, "executor alone on the same maps"), "Training curves of the dispatcher")
    clr = {}
    for r in le["rows"]:
        for k, v in r["clearances"].items():
            clr[k] = clr.get(k, 0) + v
    names = {"clr_0": "PROCEED", "clr_1": "HOLD", "clr_2": "YIELD_TO", "clr_3": "REROUTE"}
    tot = sum(clr.values()) or 1
    mix = ", ".join(f"{names[k]} {100 * clr.get(k, 0) / tot:.1f}%" for k in sorted(names))
    # paired, per held-out seed
    exk = {(r["seed"], r["malfunctions"]): r for r in ex["rows"]}
    paired = []
    for m in (False, True):
        lr = pick(le["rows"], "medium", m)
        da = [r["arrived"] - exk[(r["seed"], m)]["arrived"] for r in lr]
        dn = np.array([r["normalized_reward"] - exk[(r["seed"], m)]["normalized_reward"] for r in lr])
        paired.append(f"{'with' if m else 'without'} malfunctions, {sum(x > 0 for x in da)} seeds gain a train and "
                      f"{sum(x < 0 for x in da)} lose one (net {sum(da):+d} of {sum(r['n_agents'] for r in lr)}), normalised reward "
                      f"{dn.mean():+.4f} ± {dn.std(ddof=1) / np.sqrt(len(dn)):.4f}")
    # paired, per training iteration (the executor on the same 8 maps)
    ref = load("results/executor_train_seeds.json")
    train_txt = ""
    if ref:
        by = {r["seed"]: r for r in ref["rows"]}
        its = tf.train_seeds(len(log))
        da = np.array([r["arrival_rate"] - np.mean([by[x]["arrival_rate"] for x in it]) for r, it in zip(log, its)])
        dn = np.array([r["normalized_reward"] - np.mean([by[x]["normalized_reward"] for x in it]) for r, it in zip(log, its)])

        def seg(lo, hi):
            a_, n_ = da[lo:hi], dn[lo:hi]
            return (f"{100 * a_.mean():+.2f} ± {100 * a_.std(ddof=1) / np.sqrt(len(a_)):.2f} points of arrival and "
                    f"{n_.mean():+.4f} ± {n_.std(ddof=1) / np.sqrt(len(n_)):.4f} normalised reward")
        train_txt = f"""
On the training maps the comparison has more headroom: they are harder than the held-out seeds (the
executor delivers {100 * np.mean([r['arrival_rate'] for r in ref['rows']]):.1f}% on the {len(ref['rows'])} distinct maps the run drew,
[executor_train_seeds.json]({GH}data/tada/results/executor_train_seeds.json)). Paired with the executor on the
same eight maps per iteration, the stochastic training policy scored {seg(0, len(log))} over all
{len(log)} iterations ({int((da > 0).sum())} iterations ahead on arrival). Split by phase: iterations 1–20 {seg(0, 20)};
iterations 61–{len(log)} {seg(60, len(log))}. The arrival gain is present from the first iterations, so it
comes from issuing edits at all, not from learning which ones; what training changed is the delay those
edits cost, which shrank to about zero."""
    txt = f"""Medium (50×50, 30 trains), 10 held-out seeds (1000 to 1009), each with malfunctions off and on,
greedy policy. Arrival is mean ± standard error over seeds. Terminations and truncations are summed
over the 20 episodes. Wall-clock per step covers everything: window, masks, policy, plan edits and
the env step.

{chr(10).join(lines)}

Sources: {link('results/executor.json')}, {link('results/main.json')} (`scripts/tada_evaluate.py`),
main's PPO from [ppo.json]({GH}data/results/ppo.json).

Paired by seed against the executor: {paired[0]}; {paired[1]}. On held-out medium the learned layer
is indistinguishable from the plan it sits on.
{train_txt}

{bars}

Training: {wc.get('iterations_done', len(log))} PPO iterations of {cfg['episodes_per_iter']} medium episodes with malfunctions
({wc.get('env_steps', 0):,} env steps) in {wc['train_wall_s'] / 60:.0f} minutes on {cfg['workers']} worker processes,
CPU clock {min(mhz):.0f} to {max(mhz):.0f} MHz across iterations (median {np.median(mhz):.0f}).
Log: {link('checkpoints/main/train_log.jsonl')}. Clearances issued by the greedy policy on the 20 evaluation
episodes: {mix}.

{curve}"""
    inject("TADA_MEDIUM", txt)


# ---------------------------------------------------------------------------------------------- step 4
def _wall(n: str) -> dict:
    if n == "main_it60":  # the main run's checkpoint at 60 iterations: wall-clock read off its log
        log = [json.loads(l) for l in (DATA / "checkpoints" / "main" / "train_log.jsonl").read_text().splitlines() if l.strip()]
        return {"iterations_done": 60, "iterations_planned": 60, "train_wall_s": log[59]["wall_s"]}
    return json.loads((DATA / "checkpoints" / n / "wall_clock.json").read_text())


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
        wc = _wall(n)
        lines.append(f"| `{n}` | {d} | {pct(a)} | {pct(b)} | {num(b, 'normalized_reward')} | {sum(r['terminated'] for r in a + b)} | "
                     f"{sum(r['truncated'] for r in a + b)} | {np.mean([r['commits'] for r in a + b]):.1f} | "
                     f"{np.mean([r['decision_ms_per_step'] for r in a + b]):.1f} | {wc.get('iterations_done', '?')}/{wc.get('iterations_planned', '?')} | "
                     f"{wc['train_wall_s'] / 60:.0f} |")
        series_a.append(100 * np.mean([r["arrival_rate"] for r in a]))
        series_b.append(100 * np.mean([r["arrival_rate"] for r in b]))
    bars = figure("tada-ablations", tf.grouped_bars([n for n, _ in have], {"no malfunctions": series_a, "malfunctions": series_b}, {},
                                                    "Ablations: trains arrived on medium (%)", "arrived (%)", ylim=(0, 105)),
                  "Ablation results on medium")
    txt = f"""Every ablation trains for 60 iterations of 8 medium episodes with malfunctions (half the main run's
budget, so the grid fits), with a 55-minute wall-clock cap. The baseline row is the main run's own
checkpoint after 60 iterations, so every row has seen the same number of episodes. All rows are
evaluated like the main run (10 seeds × malfunctions off/on, greedy). "Iterations done" shows where
the cap cut a run short.

{chr(10).join(lines)}

Sources: `data/tada/results/<run>.json` and `data/tada/checkpoints/<run>/` for each run.

{bars}"""
    inject("TADA_ABLATIONS", txt)


# ---------------------------------------------------------------------------------------------- step 5
def general() -> None:
    g, v = load("results/main_general.json"), load("verify.json")
    if not g or not v:
        return
    lines = ["| Scenario | Malfunctions | Executor arrival (%) | Learned arrival (%) | Trains gained / lost (paired seeds) | Norm. reward, executor | Norm. reward, learned | Edits per episode | Terminations | Truncations | Wall-clock per step (ms) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    exk = {(r["scenario"], r["seed"], r["malfunctions"]): r for r in v["proceed_only"]["rows"]}
    for sc in ["small", "large", "xlarge"]:
        for m in (False, True):
            e, l = pick(v["proceed_only"]["rows"], sc, m), pick(g["rows"], sc, m)
            if not l:
                continue
            d = [r["arrived"] - exk[(sc, r["seed"], m)]["arrived"] for r in l]
            lines.append(f"| {sc} | {'on' if m else 'off'} | {pct(e)} | {pct(l)} | +{sum(x for x in d if x > 0)} / -{-sum(x for x in d if x < 0)} | "
                         f"{num(e, 'normalized_reward', '{:.4f}')} | {num(l, 'normalized_reward', '{:.4f}')} | {np.mean([r['commits'] for r in l]):.0f} | "
                         f"{sum(r['terminated'] for r in l)} | {sum(r['truncated'] for r in l)} | {np.mean([r['decision_ms_per_step'] for r in l]):.0f} |")
    txt = f"""The policy trained on medium, evaluated without retraining on the other three scenarios (10
held-out seeds each, malfunctions off and on). The executor columns are the PROCEED-only runs from
step 1.

{chr(10).join(lines)}

Sources: {link('results/main_general.json')}, {link('verify.json')}.

Without malfunctions the learned layer changes nothing that matters on any map: it edits plans
but never gains or loses a train. With malfunctions it gains a few trains on the larger maps, where
the executor loses the most, at unchanged normalised reward: the trains it saves arrive late. The
edit count grows with map size and malfunctions, which is where the window is full most of the time.
Wall-clock per step here mixes runs at 400 MHz and 2.5 GHz (the evaluation shared the CPU with
ablation training), so compare it only within a row."""
    inject("TADA_GENERAL", txt)


# ---------------------------------------------------------------------------------------------- step 6
def continuous() -> None:
    parts = [load(f"results/{n}.json") for n in ("continuous_executor", "continuous_learned")]
    parts = [x for x in parts if x]
    if not parts:
        return
    c = parts[0]
    rows = [r for x in parts for r in x["rows"]]
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

    def first(ctl, cond):
        r = [x for x in rates if (x, ctl) in agg and cond(agg[(x, ctl)])]
        return f"{r[0]:g}" if r else "beyond the sweep"

    reading = []
    for ctl in ctrls:
        cap = max(agg[(x, ctl)]["throughput_per_1000"] for x in rates if (x, ctl) in agg)
        reading.append(f"{'the executor' if ctl == 'executor' else 'the learned dispatcher'} drops below half of its trains on time at rate "
                       f"{first(ctl, lambda m: m['on_time_share'] < 0.5)}, starts queueing more than 10 trains off-map at "
                       f"{first(ctl, lambda m: m['backlog_off_map'] > 10)}, and peaks at {cap:.0f} arrivals per 1000 steps")
    edits = {r: np.mean([x["commits"] for x in rows if x["controller"] == "learned" and x["rate"] == r]) for r in rates} if "learned" in ctrls else {}
    deadlocks = sum(x["terminated"] for x in rows)
    reading_txt = (f"Neither controller ever deadlocked ({deadlocks} terminations in {len(rows)} runs): both break by running out of capacity, "
                   f"not by locking up. {reading[0][0].upper() + reading[0][1:]}" + (f"; {reading[1]}." if len(reading) > 1 else ".") +
                   (f" The learned layer issues {edits[rates[-1]]:.0f} plan edits per run at rate {rates[-1]:g} and none below "
                    f"{min([r for r in rates if edits[r] > 0], default=rates[-1]):g}, without moving throughput or delay beyond the seed-to-seed "
                    f"spread; it does cost wall-clock, because at high rates the window is always full and every candidate edit is a "
                    f"SIPP search through a dense table." if edits else ""))
    seeds = sorted({r["arrival_seed"] for r in rows})
    txt = f"""One medium map (seed 5000), a pool of 400 trains whose earliest departures follow a Poisson process at
the given rate, latest arrival = injection + ceil(1.3 τ + 0.2 τ̄) (Flatland 3's allowance), malfunctions
on, truncated at {c['budget']} steps; {len(seeds)} arrival seeds per rate. Both controllers plan a train only once
it appears (`OnlineExecutor`). Delay is measured on trains that arrived, so at high rates it understates
the delay of the backlog, which is reported separately.

{chr(10).join(lines)}

Sources: {link('results/continuous_executor.json')}, {link('results/continuous_learned.json')} (`scripts/tada_continuous.py`).

{fig}

{reading_txt}"""
    inject("TADA_CONTINUOUS", txt)


# ---------------------------------------------------------------------------------------------- candid
def failed() -> None:
    v, occ = load("verify.json"), load("occupancy_proceed_only.json")
    stop = DATA / "checkpoints" / "main_ent01_stopped" / "train_log.jsonl"
    mainlog = DATA / "checkpoints" / "main" / "train_log.jsonl"
    if not (v and occ and stop.exists() and mainlog.exists()):
        return
    po = v["proceed_only"]["rows"]
    flagged = [r for r in po if r.get("rotation_flags", 0) > 0]
    st = [json.loads(l) for l in stop.read_text().splitlines() if l.strip()]
    ml = [json.loads(l) for l in mainlog.read_text().splitlines() if l.strip()]
    wc = json.loads((DATA / "checkpoints" / "main" / "wall_clock.json").read_text())
    dec_per_step = sum(r["decisions"] for r in ml) / max(1, ml[-1]["env_steps"])
    full = {sc: 100 * np.mean(np.concatenate([np.asarray(x) for k, x in occ.items() if k.split("|")[:2] == [sc, "0"]]) >= 8) for sc in th.SCENARIO_ORDER}
    orr = json.loads((ROOT / "data" / "results" / "or.json").read_text())["summary"]
    gap = {sc: 100 * (orr[f"or_pp_sipp|{sc}|test"]["arrival_rate"] - orr[f"or_pp_sipp|{sc}|test_malfunction"]["arrival_rate"]) for sc in th.SCENARIO_ORDER}
    mhz = [r["cpu_mhz"] for r in ml]
    txt = f"""**Main's deadlock detector flags rotations the plan relies on.** `find_deadlocked` treats two trains
that sit in each other's successor cells as a head-on pair. In a 2×2 block of switches, a ring of
four trains can be in exactly that position while the plan rotates them one cell each in a single
step, which flatland allows. With termination on every flag, {len(flagged)} of the 80 executor-only
episodes ({', '.join(sorted({r['scenario'] for r in flagged}))}) would have ended at their first flag, and step 1 could
not pass. The dispatcher now follows each flagged train's planned next cell and terminates only if
that chain closes in a swap or reaches a train with no plan; the {sum(r['rotation_flags'] for r in po)} flagged steps
are logged instead. Main's detector is unchanged and still right where it was used: at the end of
an episode.

**Early termination was rewarded.** Flatland charges a train for missing its target only at the
horizon, so an episode cut short by a termination skipped every penalty. The first episode that hit
one (a false rotation flag, before the fix above) ended early with most trains still out and a
perfect normalised reward, a better score than running it out. Termination now charges every
unfinished train its horizon penalty, as if it stayed put until T.

**A YIELD_TO bug that only the re-timing caught.** YIELD_TO replans two trains in sequence. The
first was lifted out of the table, current cell included, and only its new future put back, so the
second could be routed head-on through the first train's cell. Nothing failed at plan time; the
next re-timing found a positive cycle in the visiting order and raised. The executor now reserves the
first train's current cell while the second is planned, and the 200 random-clearance episodes in
step 1 exercise exactly this path.

**The entropy bonus outweighed the signal.** With coefficient 0.01 on the summed entropy of the
autoregressive heads, entropy rose from {st[0]['ent']:.2f} to {st[-1]['ent']:.2f} in {len(st)} iterations and plan
edits per episode from {st[0]['commits'] / 8:.0f} to {st[-1]['commits'] / 8:.0f}, while reward did not improve. That run was
stopped and kept ([log]({GH}data/tada/checkpoints/main_ent01_stopped/train_log.jsonl)); the main run uses 0.001.

**The window is rarely selective, and decisions are everywhere.** Rules (a) to (d) admit almost
every train that is near another one, so with M = 8 the window is full on {full['small']:.0f}%, {full['medium']:.0f}%,
{full['large']:.0f}% and {full['xlarge']:.0f}% of steps on small to xlarge. A decision point occurs on
{100 * min(dec_per_step, 1):.0f}% of env steps in training ({sum(r['decisions'] for r in ml):,} decisions in {ml[-1]['env_steps']:,} steps), and almost all
of them change nothing. PPO has to find the few edits that matter among hundreds of no-ops per
episode, with the reward arriving at the horizon. TADA's window was narrower because its rule picked
the aircraft that mattered next; the rail rules here are a safety filter more than a relevance filter.

**Medium had no room to improve.** The brief chose medium as the scenario with the largest
malfunction gap. On the current paired seeds the executor loses {gap['small']:.1f}, {gap['medium']:.1f}, {gap['large']:.1f} and
{gap['xlarge']:.1f} points of arrival to malfunctions on small, medium, large and xlarge
([or.json]({GH}data/results/or.json)), so medium has the smallest non-zero gap, and the executor
already delivers 98–99% of trains on its held-out seeds. A learned layer can at most recover a
handful of trains there.

**The machine.** The CPU switched between 400 MHz and about 2.5 GHz throughout (main run: {min(mhz):.0f}
to {max(mhz):.0f} MHz per iteration). The 55-minute cap cut the main run at {wc['iterations_done']} of
{wc['iterations_planned']} iterations, and wall-clock figures on this page are only comparable within a run."""
    inject("TADA_FAILED", txt)


if __name__ == "__main__":
    for f in (verify, occupancy, medium, ablations, general, continuous, failed):
        f()
        print("done", f.__name__)
