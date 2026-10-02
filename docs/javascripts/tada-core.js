/* tada-core: the TADA dispatcher's executor and context window, ported from src/rl_flatland/tada/.
 *
 *   FL.TadaExecutor   main's planner plus re-timing, a live reservation table, and the HOLD / YIELD_TO /
 *                     REROUTE clearances, each a speculative SIPP search that leaves the table unchanged
 *                     unless it is committed (tada/executor.py, tada/planning.py)
 *   FL.buildWindow    the context window: rules (a)-(d), least slack first, capped at M (tada/window.py;
 *                     membership, reasons, choosable trains and structurally legal actions; not the features)
 *   FL.tadaDeadlock   termination test: find_deadlocked minus rotations the plan schedules (tada/env.py)
 *
 * Needs flatland-core.js. scripts/check_lab.mjs re-applies the learned dispatcher's recorded clearances
 * through this file and checks the episode against flatland step for step.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory(require("./flatland-core.js"));
  else factory(root.FL);
})(typeof self !== "undefined" ? self : this, function (FL) {
  "use strict";
  const { ST, onMap, offMap, key3, cellKey, INF, Planner, ReservationTable, Heap, insort, cmpTriple } = FL;
  const PROCEED = 0, HOLD = 1, YIELD_TO = 2, REROUTE = 3;
  const ACTION_NAMES = ["PROCEED", "HOLD", "YIELD_TO", "REROUTE"];
  const cellOf = (cfg) => cellKey(cfg[0], cfg[1]);
  const eqCfg = (a, b) => a[0] === b[0] && a[1] === b[1] && a[2] === b[2];

  // ------------------------------------------------------------------ TrainTable (planning.TrainTable)
  class TrainTable extends ReservationTable {
    reserveFor(h, cell, a, b, tag = "f") {
      this.reserve(cell, a, b);
      if (!this.owned.has(h)) this.owned.set(h, []);
      this.owned.get(h).push(["i", cell, a, b, tag]);
    }
    moveFor(h, u, v, t) {
      this.reserveMove(u, v, t);
      if (!this.owned.has(h)) this.owned.set(h, []);
      this.owned.get(h).push(["m", u, v, t, "f"]);
    }
    lift(h, tags) {
      const keep = [], gone = [];
      for (const e of this.owned.get(h) || []) (tags.includes(e[4]) ? gone : keep).push(e);
      for (const e of gone) {
        if (e[0] === "i") {
          const l = this.intervals.get(e[1]);
          const i = l.findIndex((x) => x[0] === e[2] && x[1] === e[3]);
          l.splice(i, 1);
          this._safe.delete(e[1]);
        } else this.moves.delete(e[1] + "|" + e[2] + "|" + e[3]);
      }
      this.owned.set(h, keep);
      return gone.map((e) => [h, e]);
    }
    restore(lifted) {
      for (const [h, e] of lifted) {
        if (e[0] === "i") this.reserveFor(h, e[1], e[2], e[3], e[4]);
        else this.moveFor(h, e[1], e[2], e[3]);
      }
    }
    reserveFuture(h, start, path) {
      let prev = start ? cellOf(start) : null;
      path.forEach(([cfg, t], j) => {
        const cell = cellOf(cfg);
        if (prev !== null) this.moveFor(h, prev, cell, t);
        if (j + 1 < path.length) this.reserveFor(h, cell, t, path[j + 1][1] - 1);
        else this.reserveFor(h, cell, t, t);
        prev = cell;
      });
    }
  }

  // ------------------------------------------------------------------ sipp_from (planning.sipp_from)
  function sippFrom(graph, rt, h, start, k, horizon, now, onMapStart, leaveMin, forbidden = null, maxExp = 60000) {
    const hf = (cfg) => k * graph.distance(h, cfg);
    if (!isFinite(hf(start))) return null;
    const heap = new Heap(), best = new Map(), parent = new Map();
    let counter = 0;
    const sk = (cfg, iv) => key3(cfg[0], cfg[1], cfg[2]) * 1024 + iv;
    const ivs = rt.safeIntervals(cellOf(start));
    if (onMapStart) {
      for (let idx = 0; idx < ivs.length; idx++) {
        const [lo, hi] = ivs[idx];
        if (lo <= now && now <= hi) {
          best.set(sk(start, idx), now);
          parent.set(sk(start, idx), null);
          heap.push([now + hf(start), now, counter++, start, idx, true]);
          break;
        }
      }
    } else {
      for (let idx = 0; idx < ivs.length; idx++) {
        const [lo, hi] = ivs[idx];
        if (hi < leaveMin) continue;
        const t = Math.max(lo, leaveMin);
        if (t + hf(start) > horizon) break;
        best.set(sk(start, idx), t);
        parent.set(sk(start, idx), null);
        heap.push([t + hf(start), t, counter++, start, idx, false]);
      }
    }
    let exp = 0;
    while (heap.size) {
      const [, g, , cfg, iv, isStart] = heap.pop();
      const key = sk(cfg, iv);
      if ((best.has(key) ? best.get(key) : INF) < g) continue;
      if (graph.distance(h, cfg) === 0 && !(onMapStart && isStart)) {
        const out = [];
        let node = [cfg, iv];
        while (node) {
          const kk = sk(node[0], node[1]);
          out.push([node[0], best.get(kk)]);
          node = parent.get(kk);
        }
        out.reverse();
        return onMapStart ? out.slice(1) : out;
      }
      if (++exp > maxExp) return null;
      const cell = cellOf(cfg);
      const hi = rt.safeIntervals(cell)[iv][1];
      const leaveLo = onMapStart && isStart ? leaveMin : g + k;
      const leaveHi = hi < INF ? hi + 1 : INF;
      if (leaveLo > leaveHi) continue;
      for (const s of graph.successors(cfg)) {
        if (forbidden && forbidden.some(([u, v]) => eqCfg(u, cfg) && eqCfg(v, s))) continue;
        const hs = hf(s);
        if (!isFinite(hs)) continue;
        const scell = cellOf(s);
        const sivs = rt.safeIntervals(scell);
        for (let j = 0; j < sivs.length; j++) {
          const [a, b] = sivs[j];
          if (b < leaveLo) continue;
          if (a > leaveHi) break;
          let t = Math.max(leaveLo, a);
          const tmax = Math.min(leaveHi, b);
          while (t <= tmax && rt.swapBlocked(cell, scell, t)) t++;
          if (t > tmax) continue;
          if (t + hs > horizon) break;
          const skey = sk(s, j);
          if (t < (best.has(skey) ? best.get(skey) : INF)) {
            best.set(skey, t);
            parent.set(skey, [cfg, iv]);
            heap.push([t + hs, t, counter++, s, j, false]);
          }
        }
      }
    }
    return null;
  }

  // ------------------------------------------------------------------ the executor (executor.TadaExecutor)
  class TadaExecutor extends Planner {
    constructor(opts = {}) {
      super(opts);
      this.holdSteps = opts.holdSteps === undefined ? 5 : opts.holdSteps;
    }
    reset(env) {
      super.reset(env);
      this.base = new Map();
      for (const [h, p] of this.paths) this.base.set(h, p.map((x) => x[1]));
      this.version = 0;
      this._E = null;
      this._T = null;
      this.commits = 0;
    }
    observe(env) {
      super.observe(env);
      this.version++;
    }
    _base(h) {
      let b = this.base.get(h);
      const p = this.paths.get(h);
      if (!b || b.length !== p.length) {
        b = p.map((x) => x[1]);
        this.base.set(h, b);
      }
      return b;
    }
    firstMoveTime(h, now) {
      const a = this.env.agents[h], m = a.malf;
      if (offMap(a.state)) return Math.max(now + 1, a.ed + 1, now + m + 1);
      const n = Math.max(1, a.k - a.dist);
      return now + n + m;
    }
    active() {
      return [...this.paths.keys()].filter((h) => this.env.agents[h].state !== ST.DONE);
    }
    // re-timing: longest path over train-path (+k) and cell-order (+0 / +1 after a target visit) edges
    retime(now) {
      if (this._E && this._E.now === now && this._E.v === this.version) return this._E.E;
      const keys = [], lbs = [], index = new Map(), preds = [];
      const ik = (h, j) => h * 100000 + j;
      for (const h of this.active()) {
        const p = this.paths.get(h), first = this.progress.get(h) + 1;
        if (first >= p.length) continue;
        const base = this._base(h), k = this.env.agents[h].k, fm = this.firstMoveTime(h, now);
        for (let j = first; j < p.length; j++) {
          const i = keys.length;
          index.set(ik(h, j), i);
          keys.push([h, j]);
          lbs.push(j === first ? Math.max(base[j], fm) : base[j]);
          preds.push(j > first ? [[i - 1, k]] : []);
        }
      }
      for (const [cell, lst] of this.visits) {
        const start = this._head(cell);
        if (lst.length - start < 2) continue;
        let ph = lst[start][1], pidx = lst[start][2];
        for (let n = start + 1; n < lst.length; n++) {
          const [, h, idx] = lst[n];
          const i = index.get(ik(h, idx));
          if (i !== undefined) {
            if (pidx === this.paths.get(ph).length - 1) preds[i].push([index.get(ik(ph, pidx)), 1]);
            else preds[i].push([index.get(ik(ph, pidx + 1)), 0]);
          }
          ph = h;
          pidx = idx;
        }
      }
      const order = [...keys.keys()].sort((x, y) => lbs[x] - lbs[y]);
      const pos = new Array(keys.length);
      order.forEach((i, r) => (pos[i] = r));
      let backward = false;
      for (let i = 0; i < keys.length && !backward; i++) for (const [pi] of preds[i]) if (pos[pi] > pos[i]) backward = true;
      const E = lbs.slice();
      let it = 0;
      for (; it < 200; it++) {
        let changed = false;
        for (const i of order) {
          let v = E[i];
          for (const [pi, w] of preds[i]) if (E[pi] + w > v) v = E[pi] + w;
          if (v !== E[i]) {
            E[i] = v;
            changed = true;
          }
        }
        if (!changed || !backward) break;
      }
      if (it === 200) throw new Error("retime did not converge: the visiting order has a positive cycle");
      const out = new Map();
      keys.forEach(([h, j], i) => out.set(ik(h, j), E[i]));
      out.get2 = (h, j) => out.get(ik(h, j));
      this._E = { now, v: this.version, E: out };
      return out;
    }
    exitTime(E, h, j, now) {
      const p = this.paths.get(h);
      if (j + 1 < p.length) return E.get2(h, j + 1);
      return (j > this.progress.get(h) ? E.get2(h, j) : now) + 1;
    }
    table(E, now) {
      const rt = new TrainTable(), iv = new Map();
      const add = (cell, a, b) => {
        if (!iv.has(cell)) iv.set(cell, []);
        iv.get(cell).push([a, b]);
      };
      for (const h of this.active()) {
        const p = this.paths.get(h), j0 = this.progress.get(h);
        const owned = [];
        rt.owned.set(h, owned);
        let prev = null;
        if (j0 >= 0) {
          prev = cellOf(p[j0][0]);
          const b = this.exitTime(E, h, j0, now) - 1;
          add(prev, now, b);
          owned.push(["i", prev, now, b, "c"]);
        }
        for (let j = j0 + 1; j < p.length; j++) {
          const cell = cellOf(p[j][0]), t = E.get2(h, j);
          if (prev !== null) {
            rt.reserveMove(prev, cell, t);
            owned.push(["m", prev, cell, t, "f"]);
          }
          const b = j + 1 < p.length ? E.get2(h, j + 1) - 1 : t;
          add(cell, t, b);
          owned.push(["i", cell, t, b, "f"]);
          prev = cell;
        }
      }
      for (const [cell, lst] of iv) rt.intervals.set(cell, lst.sort((x, y) => x[0] - y[0] || x[1] - y[1]));
      return rt;
    }
    liveTable(now) {
      if (this._T && this._T.now === now && this._T.v === this.version) return this._T.rt;
      const rt = this.table(this.retime(now), now);
      this._T = { now, v: this.version, rt };
      return rt;
    }
    rerouteForbidden(h, maxDec) {
      const p = this.paths.get(h);
      if (!p) return null;
      const a = this.env.agents[h];
      const j0 = onMap(a.state) ? Math.max(this.progress.get(h), 0) : 0;
      let seen = 0;
      for (let j = j0; j < p.length - 1; j++) {
        const cfg = p[j][0], nxt = p[j + 1][0];
        const succ = this.graph.successors(cfg).filter((s) => isFinite(this.graph.distance(h, s)));
        if (succ.length > 1 && succ.some((s) => eqCfg(s, nxt))) return [[cfg, nxt]];
        if (this.graph.isDecision(cfg)) {
          seen++;
          if (seen >= maxDec) break;
        }
      }
      return null;
    }
    atDecision(h) {
      const a = this.env.agents[h];
      if (a.state === ST.DONE || a.malf > 0) return false;
      if (offMap(a.state)) return (a.state === ST.READY_TO_DEPART || a.state === ST.MALFUNCTION_OFF_MAP) && this.env.elapsed >= a.ed;
      return this.env.isCellExit(a, 1);
    }
    _planOne(h, rt, now, hold, forbidden) {
      const a = this.env.agents[h], info = this.infos[h];
      const leave = this.firstMoveTime(h, now) + hold;
      if (offMap(a.state)) return sippFrom(this.graph, rt, h, info.start, info.k, this.horizon, now, false, leave, forbidden);
      return sippFrom(this.graph, rt, h, a.cfg, info.k, this.horizon, now, true, leave, forbidden);
    }
    // speculative: returns Map(train -> new future) or null; rt is left as it was
    planClearance(c, rt, now, maxDec = 3) {
      if (c.kind === PROCEED) return new Map();
      let steps;
      if (c.kind === HOLD) steps = [[c.train, this.holdSteps, null]];
      else if (c.kind === REROUTE) {
        const forb = this.rerouteForbidden(c.train, maxDec);
        if (!forb) return null;
        steps = [[c.train, 0, forb]];
      } else if (c.kind === YIELD_TO) {
        if (c.partner === null || c.partner === undefined || c.partner === c.train || !this.paths.has(c.partner)) return null;
        steps = [[c.partner, 0, null], [c.train, 0, null]];
      }
      const involved = steps.map((x) => x[0]);
      let lifted = [];
      for (const x of involved) lifted = lifted.concat(rt.lift(x, ["f"]));
      const out = new Map();
      let ok = true;
      for (const [x, hold, forb] of steps) {
        lifted = lifted.concat(rt.lift(x, ["c"]));
        const path = this._planOne(x, rt, now, hold, forb);
        if (!path) {
          ok = false;
          break;
        }
        const a = this.env.agents[x];
        const start = onMap(a.state) ? a.cfg : null;
        if (start) rt.reserveFor(x, cellOf(start), now, path[0][1] - 1, "f");
        rt.reserveFuture(x, start, path);
        out.set(x, path);
      }
      for (const x of involved) rt.lift(x, ["f"]);
      rt.restore(lifted);
      return ok ? out : null;
    }
    commit(nw, now) {
      if (!nw.size) return;
      const E = this.retime(now);
      this._writeTimes(E);
      for (const [h, fut] of nw) {
        const p = this.paths.get(h), j0 = this.progress.get(h);
        for (let j = j0 + 1; j < p.length; j++) {
          const [cfg, t] = p[j];
          const l = this.visits.get(cellOf(cfg));
          l.splice(l.findIndex((v) => v[0] === t && v[1] === h && v[2] === j), 1);
        }
        const np = p.slice(0, j0 + 1).concat(fut);
        this.paths.set(h, np);
        const b = this.base.get(h) || p.map((x) => x[1]);
        this.base.set(h, b.slice(0, j0 + 1).concat(fut.map((x) => x[1])));
        for (let j = j0 + 1; j < np.length; j++) {
          const [cfg, t] = np[j];
          const c = cellOf(cfg);
          if (!this.visits.has(c)) this.visits.set(c, []);
          insort(this.visits.get(c), [t, h, j], cmpTriple);
        }
      }
      this.version++;
      const E2 = this.retime(now);
      this._writeTimes(E2);
      this.vptr = new Map();
      this.rt = this.table(E2, now);
      this._E = { now, v: this.version, E: E2 };
      this._T = { now, v: this.version, rt: this.table(E2, now) };
      this.commits++;
    }
    _writeTimes(E) {
      for (const [cell, lst] of this.visits) {
        const nl = lst.map(([t, h, j]) => {
          const v = E.get2(h, j);
          return [v === undefined ? t : v, h, j];
        });
        nl.sort(cmpTriple);
        for (let i = 0; i < nl.length; i++)
          if (nl[i][1] !== lst[i][1] || nl[i][2] !== lst[i][2]) throw new Error("re-timing reordered cell " + cell);
        this.visits.set(cell, nl);
      }
      for (const h of this.active()) {
        const p = this.paths.get(h);
        for (let j = this.progress.get(h) + 1; j < p.length; j++) p[j] = [p[j][0], E.get2(h, j)];
      }
      this.version++;
    }
  }

  // ------------------------------------------------------------------ the context window (window.build_window)
  function partners(ex, E, now, H) {
    const before = new Map(), after = new Map(), near = new Map();
    const add = (m, k, v) => {
      if (!m.has(k)) m.set(k, new Set());
      m.get(k).add(v);
    };
    const horizon = now + H;
    for (const [cell, lst] of ex.visits) {
      const start = ex._head(cell);
      let prev = null;
      for (let i = start; i < lst.length; i++) {
        const [, h, idx] = lst[i];
        const te = E.get2(h, idx);
        if ((te === undefined ? now : te) > horizon) break;
        add(near, cell, h);
        if (prev !== null && prev !== h) {
          add(after, prev, h);
          add(before, h, prev);
        }
        prev = h;
      }
    }
    return { before, after, near };
  }

  function buildWindow(ex, cfg = {}) {
    const H = cfg.H || 30, D = cfg.D || 3, M = cfg.M || 8;
    const allowYield = cfg.allowYield !== false, allowHold = cfg.allowHold !== false, allowReroute = cfg.allowReroute !== false;
    const env = ex.env, now = env.elapsed, T = env.T;
    const E = ex.retime(now);
    const { before, after, near } = partners(ex, E, now, H);
    const occ = new Set(env.agents.filter((a) => onMap(a.state) && a.cfg).map((a) => cellOf(a.cfg)));
    const cand = new Map(), slack = new Map();
    const reasons = (h) => {
      if (!cand.has(h)) cand.set(h, new Set());
      return cand.get(h);
    };
    for (const h of ex.active()) {
      const a = env.agents[h], p = ex.paths.get(h), j0 = ex.progress.get(h), info = ex.infos[h];
      const cur = onMap(a.state) ? a.cfg : info.start;
      const dist = ex.graph.distance(h, cur);
      slack.set(h, a.la - (now + info.k * (isFinite(dist) ? dist : env.rail.W + env.rail.H)));
      if ((before.get(h) && before.get(h).size) || (after.get(h) && after.get(h).size)) reasons(h).add("a");
      const arr = E.get2(h, p.length - 1);
      if (arr !== undefined && (arr > a.la || arr > T)) reasons(h).add("b");
      let seen = 0, stop = null;
      for (let j = Math.max(j0, 0); j < p.length; j++) {
        const c = p[j][0];
        if (ex.graph.isDecision(c)) {
          seen++;
          if (seen >= D && stop === null) stop = j;
        }
        if (stop !== null && j > stop && ex.graph.isDecision(c)) break;
        const nc = near.get(cellOf(c));
        if (nc && nc.size > (nc.has(h) ? 1 : 0)) {
          reasons(h).add("c");
          break;
        }
      }
      if (offMap(a.state) && now >= a.ed)
        for (let jj = 0; jj < p.length; jj++) {
          const c = p[jj][0], cell = cellOf(c);
          const others = near.get(cell) ? [...near.get(cell)].filter((x) => x !== h) : [];
          if (occ.has(cell) || others.length) {
            reasons(h).add("d");
            break;
          }
          if (jj > 0 && ex.graph.isDecision(c)) break;
        }
    }
    const members = [...cand.keys()].sort((x, y) => slack.get(x) - slack.get(y) || x - y).slice(0, M);
    const slot = new Map(members.map((h, i) => [h, i]));
    const rows = members.map((h) => {
      const atDec = ex.atDecision(h);
      const legal = [true, false, false, false];
      let partnersAfter = [];
      if (atDec) {
        if (allowHold) legal[HOLD] = true;
        if (allowReroute && ex.rerouteForbidden(h, D)) legal[REROUTE] = true;
        partnersAfter = after.get(h) ? [...after.get(h)].filter((x) => slot.has(x)).sort((x, y) => slot.get(x) - slot.get(y)) : [];
        if (allowYield && partnersAfter.length) legal[YIELD_TO] = true;
      }
      return { train: h, slack: slack.get(h), reasons: [...cand.get(h)], atDecision: atDec, legal, partnersAfter, choosable: legal.slice(1).some(Boolean) };
    });
    return { now, members, rows, E };
  }

  // ------------------------------------------------------------------ termination test (tada/env.py _deadlock)
  function tadaDeadlock(env, ex) {
    const dl = FL.findDeadlocked(env);
    if (!dl.size) return { deadlock: false, flagged: dl, rotation: false };
    const holder = new Map();
    for (const h of dl) holder.set(cellOf(env.agents[h].cfg), h);
    const nxt = new Map();
    for (const h of dl) {
      const p = ex.paths.get(h), j = ex.progress.has(h) ? ex.progress.get(h) : -1;
      if (!p || j + 1 >= p.length) return { deadlock: true, flagged: dl, rotation: false };
      const c = cellOf(p[j + 1][0]);
      nxt.set(h, holder.has(c) ? holder.get(c) : null);
    }
    for (const h of dl) {
      const seen = [h];
      while (nxt.get(seen[seen.length - 1]) !== null && !seen.includes(nxt.get(seen[seen.length - 1]))) seen.push(nxt.get(seen[seen.length - 1]));
      const last = nxt.get(seen[seen.length - 1]);
      if (last !== null && seen.length - seen.indexOf(last) < 3) return { deadlock: true, flagged: dl, rotation: false };
    }
    return { deadlock: false, flagged: dl, rotation: true };
  }

  Object.assign(FL, { PROCEED, HOLD, YIELD_TO, REROUTE, ACTION_NAMES, TrainTable, sippFrom, TadaExecutor, buildWindow, tadaDeadlock });
  return FL;
});
