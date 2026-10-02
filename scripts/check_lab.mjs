// Check the Scheduling Lab's JavaScript port of flatland against flatland-rl itself.
//
//   node scripts/check_lab.mjs                 # every map in docs/assets/lab/maps that has a fixture
//   node scripts/check_lab.mjs small-1000-m    # one map
//
// For each map (fixtures from `python scripts/make_lab_data.py`, in data/lab_fixture/):
//   1. distance maps: every train's distance-to-target from every configuration equals flatland's;
//   2. successor order: Python's iteration order for every configuration with a choice;
//   3. replay: each recorded policy's actions, run through FL.RailEnv, give every train the same state,
//      position, heading, in-cell distance, speed and malfunction counter as flatland at every step, and
//      the same arrivals and normalised reward;
//   4. live planner: FL.Planner plans the same paths as main's PrioritizedPlannerPolicy and, executed on
//      FL.RailEnv, reproduces the recorded OR episode step for step;
//   5. dispatcher: FL.TadaExecutor (tada-core.js) rebuilds the learned dispatcher's window at every step and
//      re-applies its recorded clearances; the windows, the clearance outcomes and the episode must match.
// Exit code 1 on any mismatch.
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const FL = require(join(ROOT, "docs/javascripts/flatland-core.js"));
require(join(ROOT, "docs/javascripts/tada-core.js"));
const MAPS = join(ROOT, "docs/assets/lab/maps");
const FIX = join(ROOT, "data/lab_fixture");

const ids = process.argv.slice(2).length
  ? process.argv.slice(2)
  : readdirSync(MAPS).filter((f) => f.endsWith(".json")).map((f) => f.slice(0, -5)).filter((id) => existsSync(join(FIX, id + ".json")));

function expandStates(changes, n, T) {
  // per train change lists [[step, state, r, c, d, dist, spd, malf], ...] -> per step arrays
  const out = [];
  const cur = new Array(n).fill(null), ptr = new Array(n).fill(0);
  for (let s = 1; s <= T; s++) {
    const row = [];
    for (let i = 0; i < n; i++) {
      const l = changes[i];
      while (ptr[i] < l.length && l[ptr[i]][0] <= s) cur[i] = l[ptr[i]++].slice(1);
      row.push(cur[i]);
    }
    out.push(row);
  }
  return out;
}

function compareStep(env, expected, step, label) {
  for (const a of env.agents) {
    const e = expected[a.handle];
    if (!e) continue;
    const got = [a.state, a.cfg ? a.cfg[0] : -1, a.cfg ? a.cfg[1] : -1, a.cfg ? a.cfg[2] : -1, a.dist, a.speed, a.malf];
    if (got.some((v, k) => v !== e[k])) return `${label}: step ${step} train ${a.handle}: js ${JSON.stringify(got)} python ${JSON.stringify(e)}`;
  }
  return null;
}

let failures = 0, checks = 0;
const t0 = Date.now();
for (const id of ids) {
  const map = JSON.parse(readFileSync(join(MAPS, id + ".json"), "utf8"));
  const fix = JSON.parse(readFileSync(join(FIX, id + ".json"), "utf8"));
  const n = map.agents.length;
  const errs = [];
  // 1. distances
  const env0 = new FL.RailEnv(map);
  const g = env0.graph;
  for (let h = 0; h < n; h++) {
    const py = fix.dist[h];
    const js = g.dist[h];
    for (let i = 0; i < py.length; i++) {
      const v = isFinite(js[i]) ? js[i] : -1;
      if (v !== py[i]) {
        errs.push(`distance: train ${h} index ${i}: js ${v} python ${py[i]}`);
        break;
      }
    }
  }
  // 2. successor order
  for (const [k, lst] of Object.entries(fix.succ_order || {})) {
    const [r, c, d] = k.split(",").map(Number);
    const js = g.successors([r, c, d]);
    if (JSON.stringify(js) !== JSON.stringify(lst)) errs.push(`successors of ${k}: js ${JSON.stringify(js)} python ${JSON.stringify(lst)}`);
  }
  // 3. replays
  for (const [kind, ep] of Object.entries(map.episodes)) {
    const env = new FL.RailEnv(map);
    const next = FL.decodeActions(ep.actions, n);
    const exp = expandStates(fix.states[kind], n, env.T);
    let err = null;
    while (!env.done) {
      env.step(next(env.elapsed + 1));
      err = compareStep(env, exp[env.elapsed - 1], env.elapsed, `replay ${kind}`);
      if (err) break;
    }
    if (!err) {
      const s = ep.summary;
      const nr = env.normalizedReward();
      if (env.arrived() !== s.arrived || env.elapsed !== s.steps || Math.abs(nr - s.normalized_reward) > 1e-9)
        err = `replay ${kind}: outcome js ${env.arrived()}/${env.elapsed}/${nr} python ${s.arrived}/${s.steps}/${s.normalized_reward}`;
    }
    if (err) errs.push(err);
  }
  // 4. live planner
  {
    const env = new FL.RailEnv(map);
    const pl = new FL.Planner();
    pl.reset(env);
    const pyPlan = fix.plan;
    const jsPlan = {};
    for (const [h, p] of pl.paths) jsPlan[h] = p.map(([cfg, t]) => [cfg, t]);
    const same = Object.keys(pyPlan).length === Object.keys(jsPlan).length && Object.entries(pyPlan).every(([h, p]) => JSON.stringify(p) === JSON.stringify(jsPlan[h]));
    if (!same) {
      const diff = Object.keys(pyPlan).find((h) => JSON.stringify(pyPlan[h]) !== JSON.stringify(jsPlan[h]));
      errs.push(`plan differs (${Object.keys(jsPlan).length} vs ${Object.keys(pyPlan).length} routed; first train ${diff}; js chose ${pl.chosen})`);
    }
    const exp = expandStates(fix.states.or, n, env.T);
    let err = null;
    while (!env.done) {
      const act = pl.act(env);
      env.step(act);
      pl.observe(env);
      err = compareStep(env, exp[env.elapsed - 1], env.elapsed, "live planner");
      if (err) break;
    }
    const s = map.episodes.or.summary;
    if (!err && (env.arrived() !== s.arrived || Math.abs(env.normalizedReward() - s.normalized_reward) > 1e-9))
      err = `live planner outcome js ${env.arrived()} ${env.normalizedReward()} python ${s.arrived} ${s.normalized_reward}`;
    if (err) errs.push(err);
  }
  // 5. the learned dispatcher: the JS executor rebuilds the window every step and re-applies the recorded
  //    clearances (committed ones must be plannable, rejected ones not); the episode must match flatland's
  if (map.episodes.tada) {
    const ep = map.episodes.tada;
    const env = new FL.RailEnv(map);
    const ex = new FL.TadaExecutor();
    ex.reset(env);
    const exp = expandStates(fix.states.tada, n, env.T);
    const byStep = new Map();
    for (const c of ep.clearances) {
      if (!byStep.has(c[0])) byStep.set(c[0], []);
      byStep.get(c[0]).push(c);
    }
    const win = new Map();
    let wcur = null, wi = 0;
    let err = null;
    try {
      while (!env.done) {
        const now = env.elapsed;
        while (wi < ep.window.length && ep.window[wi][0] <= now) wcur = ep.window[wi++];
        const w = FL.buildWindow(ex);
        const choos = w.rows.filter((r) => r.choosable).map((r) => r.train);
        if (wcur && (JSON.stringify(w.members) !== JSON.stringify(wcur[1]) || JSON.stringify(choos) !== JSON.stringify(wcur[2]))) {
          err = `tada window at step ${now}: js ${JSON.stringify([w.members, choos])} python ${JSON.stringify(wcur.slice(1))}`;
          break;
        }
        for (const [, h, kind, partner, committed] of byStep.get(now) || []) {
          const plan = ex.planClearance({ train: h, kind, partner }, ex.liveTable(now), now, 3);
          if (!!plan !== !!committed) {
            err = `tada clearance at step ${now} (train ${h}, ${FL.ACTION_NAMES[kind]}): js ${plan ? "planned" : "rejected"}, python ${committed ? "committed" : "rejected"}`;
            break;
          }
          if (plan) ex.commit(plan, now);
        }
        if (err) break;
        env.step(ex.act(env));
        ex.observe(env);
        err = compareStep(env, exp[env.elapsed - 1], env.elapsed, "tada clearances");
        if (err) break;
      }
    } catch (e) {
      err = "tada: " + e.message;
    }
    if (!err && (env.arrived() !== ep.summary.arrived || Math.abs(env.normalizedReward() - ep.summary.normalized_reward) > 1e-9))
      err = `tada outcome js ${env.arrived()} ${env.normalizedReward()} python ${ep.summary.arrived} ${ep.summary.normalized_reward}`;
    if (err) errs.push(err);
  }
  checks++;
  if (errs.length) {
    failures++;
    console.log(`FAIL ${id}`);
    for (const e of errs.slice(0, 6)) console.log("   " + e);
  } else console.log(`ok   ${id}  (${Object.keys(map.episodes).length} replays, distances, successors, live planner, dispatcher clearances)`);
}
console.log(`${checks - failures}/${checks} maps match flatland-rl in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
process.exit(failures ? 1 : 0);
