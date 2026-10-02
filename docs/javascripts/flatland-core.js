/* flatland-core: a JavaScript port of flatland-rl 4.3.0's RailEnv dynamics and of this repo's OR planner.
 *
 *   FL.RailEnv      one episode on an exported map: transitions, actions, speed counters, the train state
 *                   machine, MotionCheck, malfunctions (replayed from the draws Python recorded) and the
 *                   Flatland 3 DefaultRewards with the benchmark's normalisation
 *   FL.Graph        configurations (row, col, heading), successors, switch cells, distance-to-target maps
 *   FL.Planner      main's OR reference: prioritized planning with safe-interval path planning (SIPP) and
 *                   ordered execution (src/rl_flatland/baselines/or_planner.py, line for line)
 *   FL.findDeadlocked   src/rl_flatland/deadlock.py
 *
 * Every function mirrors the Python it names. scripts/check_lab.mjs replays the recorded episodes of
 * docs/assets/lab/maps/*.json through this file and compares every train's state, position, speed and
 * malfunction counter with flatland-rl at every step, and runs the planner live against Python's plan.
 * Works in the browser (window.FL) and in Node (module.exports). No dependencies.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.FL = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ------------------------------------------------------------------ constants
  const N = 0, E = 1, S = 2, W = 3;
  const DR = [-1, 0, 1, 0], DC = [0, 1, 0, -1];
  const DO_NOTHING = 0, MOVE_LEFT = 1, MOVE_FORWARD = 2, MOVE_RIGHT = 3, STOP_MOVING = 4;
  const ST = { WAITING: 0, READY_TO_DEPART: 1, MALFUNCTION_OFF_MAP: 2, MOVING: 3, STOPPED: 4, MALFUNCTION: 5, DONE: 6 };
  const STATE_NAMES = ["waiting", "ready to depart", "broken down before departure", "moving", "stopped", "broken down", "arrived"];
  const onMap = (s) => s === ST.MOVING || s === ST.STOPPED || s === ST.MALFUNCTION;
  const offMap = (s) => s === ST.WAITING || s === ST.READY_TO_DEPART || s === ST.MALFUNCTION_OFF_MAP;
  const isMoving = (a) => a === MOVE_LEFT || a === MOVE_FORWARD || a === MOVE_RIGHT;
  const INF = 1e9;
  const key3 = (r, c, d) => (r * 4096 + c) * 4 + d; // numeric key for a configuration
  const cellKey = (r, c) => r * 4096 + c;

  // ------------------------------------------------------------------ the rail grid (GridTransitionMap)
  class Rail {
    constructor(H, W, grid) {
      this.H = H;
      this.W = W;
      this.grid = Uint16Array.from(grid);
    }
    code(r, c) {
      return r < 0 || c < 0 || r >= this.H || c >= this.W ? 0 : this.grid[r * this.W + c];
    }
    // Grid4Transitions.get_transitions: the four exits (N, E, S, W) for a train heading `d` in cell (r, c)
    trans(r, c, d) {
      const bits = (this.code(r, c) >> ((3 - d) * 4)) & 0xf;
      return [(bits >> 3) & 1, (bits >> 2) & 1, (bits >> 1) & 1, bits & 1];
    }
    inBounds(r, c) {
      return r >= 0 && c >= 0 && r < this.H && c < this.W;
    }
    validConfig(r, c, d) {
      if (!this.inBounds(r, c)) return false;
      const t = this.trans(r, c, d);
      return t[0] + t[1] + t[2] + t[3] > 0;
    }
    // RailGridTransitionMap._check_action_new: [new direction, transition valid, preprocessed action, action valid]
    checkActionNew(action, r, c, d) {
      const pt = this.trans(r, c, d);
      const n = pt[0] + pt[1] + pt[2] + pt[3];
      if (n === 1) {
        const nd = pt.indexOf(1);
        if (action === MOVE_LEFT && nd !== (d + 3) % 4) return [nd, false, MOVE_FORWARD, true];
        if (action === MOVE_RIGHT && nd !== (d + 1) % 4) return [nd, false, MOVE_FORWARD, true];
        return [nd, true, action, true];
      }
      if (action === MOVE_LEFT) {
        const nd = (d + 3) % 4;
        if (pt[nd]) return [nd, true, MOVE_LEFT, true];
        if (pt[d]) return [d, false, MOVE_FORWARD, true];
      } else if (action === MOVE_RIGHT) {
        const nd = (d + 1) % 4;
        if (pt[nd]) return [nd, true, MOVE_RIGHT, true];
        if (pt[d]) return [d, false, MOVE_FORWARD, true];
      } else if (pt[d]) {
        return [d, true, action, true];
      }
      return [d, false, STOP_MOVING, false];
    }
    // apply_action_independent: [[r, c, d], straight] or null
    applyAction(action, r, c, d) {
      const [nd, , , actionValid] = this.checkActionNew(action, r, c, d);
      const nr = r + DR[nd], nc = c + DC[nd];
      if (actionValid && this.validConfig(nr, nc, nd)) return [[nr, nc, nd], nd % 2 === d % 2];
      return null;
    }
    // get_successor_configurations: an ordered set over MOVE_LEFT, MOVE_FORWARD, MOVE_RIGHT
    successors(r, c, d) {
      const out = [];
      for (const a of [MOVE_LEFT, MOVE_FORWARD, MOVE_RIGHT]) {
        const t = this.applyAction(a, r, c, d);
        if (t && this.validConfig(t[0][0], t[0][1], t[0][2]) && !out.some((x) => x[0] === t[0][0] && x[1] === t[0][1] && x[2] === t[0][2])) out.push(t[0]);
      }
      return out;
    }
    // get_predecessor_configurations
    predecessors(r, c, d) {
      const pr = r + DR[(d + 2) % 4], pc = c + DC[(d + 2) % 4];
      if (!this.inBounds(pr, pc)) return [];
      const out = [];
      for (let o = 0; o < 4; o++) if (this.trans(pr, pc, o)[d]) out.push([pr, pc, o]);
      return out;
    }
  }

  // ------------------------------------------------------------------ graph: successors, switches, distances
  class Graph {
    constructor(rail, agents) {
      this.rail = rail;
      this.H = rail.H;
      this.W = rail.W;
      this._succ = new Map();
      this.isSwitch = new Uint8Array(rail.H * rail.W);
      this.isRail = new Uint8Array(rail.H * rail.W);
      for (let r = 0; r < rail.H; r++)
        for (let c = 0; c < rail.W; c++) {
          let entries = 0, multi = false;
          for (let d = 0; d < 4; d++) {
            const t = rail.trans(r, c, d), k = t[0] + t[1] + t[2] + t[3];
            if (k > 0) entries++;
            if (k > 1) multi = true;
          }
          if (entries > 0) this.isRail[r * rail.W + c] = 1;
          if (multi || entries > 2) this.isSwitch[r * rail.W + c] = 1;
        }
      this.dist = agents.map((a) => this._distanceMap(a.targets));
    }
    successors(cfg) {
      const k = key3(cfg[0], cfg[1], cfg[2]);
      let s = this._succ.get(k);
      if (!s) {
        s = this.rail.successors(cfg[0], cfg[1], cfg[2]);
        this._succ.set(k, s);
      }
      return s;
    }
    // DistanceMapWalker: a backward BFS from every target configuration, minimum over targets
    _distanceMap(targets) {
      const { H, W } = this;
      const dist = new Float64Array(H * W * 4).fill(Infinity);
      const idx = (r, c, d) => (r * W + c) * 4 + d;
      for (const t of targets) {
        const seen = new Uint8Array(H * W * 4);
        dist[idx(t[0], t[1], t[2])] = 0;
        seen[idx(t[0], t[1], t[2])] = 1;
        let frontier = [[t[0], t[1], t[2], 0]];
        // BFS by levels; distances are the minimum over all target configurations
        while (frontier.length) {
          const next = [];
          for (const [r, c, d, k] of frontier) {
            for (const p of this.rail.predecessors(r, c, d)) {
              const i = idx(p[0], p[1], p[2]);
              if (k + 1 < dist[i]) dist[i] = k + 1;
              if (!seen[i]) {
                seen[i] = 1;
                next.push([p[0], p[1], p[2], k + 1]);
              }
            }
          }
          frontier = next;
        }
      }
      return dist;
    }
    distance(h, cfg) {
      return this.dist[h][(cfg[0] * this.W + cfg[1]) * 4 + cfg[2]];
    }
    isSwitchCell(r, c) {
      return this.isSwitch[r * this.W + c] === 1;
    }
    // a diverging switch, or the last cell before a switch cell
    isDecision(cfg) {
      const s = this.successors(cfg);
      if (s.length > 1) return true;
      return s.some((x) => this.isSwitchCell(x[0], x[1]));
    }
    // greedy walk along decreasing distance
    shortestPath(h, start, maxLen = 10000) {
      const path = [start];
      let cur = start;
      for (let i = 0; i < maxLen; i++) {
        if (this.distance(h, cur) === 0) break;
        let best = null, bd = Infinity;
        for (const s of this.successors(cur)) {
          const d = this.distance(h, s);
          if (d < bd) (bd = d), (best = s);
        }
        if (!best || !isFinite(bd)) break;
        cur = best;
        path.push(cur);
      }
      return path;
    }
  }

  // ------------------------------------------------------------------ CPython set iteration order
  // MotionCheck iterates a Python set of train handles, and the order changes the outcome when several trains
  // are stopped at once. This replays CPython 3.12's set insertion (open addressing, 9 linear probes, then
  // perturbation; resize to 4x used when 3/5 full) for small non-negative ints, whose hash is the int itself.
  function pySetOrder(seq) {
    let mask = 7, table = new Array(8).fill(null), fill = 0;
    const clean = (tab, m, key) => {
      let i = key & m, perturb = key;
      for (;;) {
        if (tab[i] === null) return void (tab[i] = key);
        if (i + 9 <= m) for (let j = 1; j <= 9; j++) if (tab[i + j] === null) return void (tab[i + j] = key);
        perturb = Math.floor(perturb / 32);
        i = (i * 5 + 1 + perturb) & m;
      }
    };
    for (const key of seq) {
      let i = key & mask, perturb = key, slot = -1, present = false;
      outer: for (;;) {
        const probes = i + 9 <= mask ? 9 : 0;
        for (let j = 0; j <= probes; j++) {
          const e = table[i + j];
          if (e === null) {
            slot = i + j;
            break outer;
          }
          if (e === key) {
            present = true;
            break outer;
          }
        }
        perturb = Math.floor(perturb / 32);
        i = (i * 5 + 1 + perturb) & mask;
      }
      if (present) continue;
      table[slot] = key;
      fill++;
      if (fill * 5 >= mask * 3) {
        let size = 8;
        while (size <= fill * 4) size <<= 1;
        const nt = new Array(size).fill(null);
        for (const k of table) if (k !== null) clean(nt, size - 1, k);
        table = nt;
        mask = size - 1;
      }
    }
    return table.filter((k) => k !== null);
  }

  // ------------------------------------------------------------------ MotionCheck (agent_chains.MotionCheck)
  // Resources are cells; a train off the map holds the dummy resource "x<i>".
  function motionCheck(moves) {
    // moves: array (in handle order) of [i, r1, r2] with r1 = current resource or null, r2 = wanted resource or null
    const agents = new Map(), reverse = new Map(), stopped = new Set(), deadlocked = [];
    const res = (r, i) => (r === null ? "x" + i : r);
    for (const [i, a, b] of moves) {
      const r1 = res(a, i), r2 = res(b, i);
      agents.set(i, [r1, r2]);
      if (!reverse.has(r2)) reverse.set(r2, [i]);
      else reverse.get(r2).push(i);
    }
    let conflicts = [];
    for (const [i, [pos, target]] of agents) {
      if (reverse.has(pos))
        for (const a2 of reverse.get(pos)) {
          if (i >= a2) continue;
          const [p2, t2] = agents.get(a2);
          if (pos === t2 && target === p2) {
            stopped.add(i);
            stopped.add(a2);
            deadlocked.push(i, a2);
          }
        }
      for (const a2 of reverse.get(target)) {
        if (i >= a2) continue;
        conflicts.push([i, a2]);
      }
      if (pos === target) stopped.add(i);
    }
    const stopAndUpdate = (v, vpos, vtarget) => {
      agents.set(v, [vpos, vpos]);
      stopped.add(v);
      if (reverse.has(vpos)) {
        for (const o of reverse.get(vpos)) if (o !== v) conflicts.push(v < o ? [v, o] : [o, v]);
        reverse.get(vpos).push(v);
      } else reverse.set(vpos, [v]);
      const lst = reverse.get(vtarget);
      lst.splice(lst.indexOf(v), 1);
    };
    for (const a of pySetOrder(deadlocked)) {
      const [ap, at] = agents.get(a);
      agents.set(a, [ap, ap]);
      stopAndUpdate(a, ap, at);
    }
    let q = 0;
    while (q < conflicts.length) {
      const [u, v] = conflicts[q];
      const [up, ut] = agents.get(u), [vp, vt] = agents.get(v);
      if (vp === vt) stopAndUpdate(u, up, ut);
      else if (ut === vt) stopAndUpdate(v, vp, vt);
      q++;
    }
    return stopped;
  }

  // ------------------------------------------------------------------ the environment
  class RailEnv {
    /* map: an exported lab map (docs/assets/lab/maps/<id>.json).
     * opts.malfunctions: replay the recorded malfunction draws (default: as recorded for this map)
     * opts.extraMalfunctions: array of [step, train, duration] added on top (the Lab's "break a train") */
    constructor(map, opts = {}) {
      this.map = map;
      this.rail = new Rail(map.H, map.W, map.grid);
      this.T = map.T;
      this.n = map.agents.length;
      this.graph = new Graph(this.rail, map.agents);
      this.opts = opts;
      this.reset();
    }
    reset() {
      const m = this.map;
      this.elapsed = 0;
      this.done = false;
      this.draws = new Map();
      const use = this.opts.malfunctions !== false;
      for (const [s, i, d] of use ? m.malf || [] : []) this.draws.set(s * 100000 + i, d);
      for (const [s, i, d] of this.opts.extraMalfunctions || []) this.draws.set(s * 100000 + i, Math.max(d, this.draws.get(s * 100000 + i) || 0));
      this.agents = m.agents.map((a, i) => ({
        handle: i,
        start: a.start,
        targets: a.targets,
        targetSet: new Set(a.targets.map((t) => key3(t[0], t[1], t[2]))),
        k: a.k,
        ed: a.ed,
        la: a.la,
        state: m.init ? m.init[i][0] : ST.WAITING,
        prevState: null,
        cfg: null,
        oldCfg: null,
        speed: 1, // in units of max speed: 0 or 1
        dist: 0, // distance travelled in the current cell, in units of max speed (0 .. k-1)
        malf: m.init ? m.init[i][1] : 0,
        nMalf: 0,
        arrival: null,
        reward: 0,
      }));
      this.rewards = new Array(this.n).fill(0);
      this.lastActions = new Array(this.n).fill(DO_NOTHING);
      this.events = [];
    }
    isCellExit(a, su) {
      return a.dist + Math.min(su, 1) >= a.k;
    }
    travelTimeSP(a) {
      // EnvAgent.get_travel_time_on_shortest_path: ceil(len(shortest path) / speed), path length = distance + 1
      const cfg = offMap(a.state) ? a.start : a.cfg;
      if (!cfg) return 0;
      if (a.targetSet.has(key3(cfg[0], cfg[1], cfg[2]))) return a.k;
      const d = this.graph.distance(a.handle, cfg);
      return isFinite(d) ? (d + 1) * a.k : 0;
    }
    step(actionDict) {
      this.elapsed += 1;
      if (this.done) throw new Error("Episode is done");
      const t = this.elapsed;
      // effects generator: one malfunction draw per train per step
      for (const a of this.agents) {
        const d = this.draws.get(t * 100000 + a.handle);
        if (d && a.malf === 0) {
          a.malf = d;
          a.nMalf++;
          this.events.push({ t, type: "malfunction", train: a.handle, steps: d });
        }
      }
      const tmp = [];
      const moves = [];
      for (const a of this.agents) {
        const cur = a.cfg || a.start;
        a.oldCfg = a.cfg;
        let action = actionDict[a.handle];
        if (action === undefined || action === null) action = DO_NOTHING;
        const tr = this.rail.applyAction(action, cur[0], cur[1], cur[2]);
        const actionValid = tr !== null;
        const stop = action === STOP_MOVING;
        const inMalf = a.malf > 0;
        const mv = isMoving(action);
        const edReached = a.ed <= t;
        const state = a.state;
        let ns = a.speed;
        if (!actionValid) ns = 0;
        else if ((state === ST.STOPPED || state === ST.MALFUNCTION) && mv) ns += 1;
        else if (action === MOVE_FORWARD && tr[1]) ns += 1;
        else if (stop) ns -= 1;
        ns = Math.max(0, Math.min(1, ns));
        let ncfg = null;
        if (state === ST.READY_TO_DEPART && mv && actionValid) ncfg = a.start;
        else if (state === ST.MALFUNCTION_OFF_MAP && !inMalf && edReached && actionValid && (mv || stop)) ncfg = a.start;
        else if (onMap(state)) {
          ncfg = cur;
          let canMove = state === ST.MOVING && !((stop || !actionValid) && ns === 0);
          canMove = canMove || (state === ST.MALFUNCTION && !inMalf && mv && actionValid);
          canMove = canMove || (state === ST.STOPPED && mv && actionValid);
          canMove = canMove && !inMalf;
          if (this.isCellExit(a, ns) && canMove) ncfg = tr[0];
        }
        const r1 = a.cfg ? cellKey(a.cfg[0], a.cfg[1]) : null;
        const r2 = ncfg ? cellKey(ncfg[0], ncfg[1]) : null;
        tmp.push({ inMalf, edReached, stop, mv, actionValid, ns, ncfg, speed: a.speed });
        moves.push([a.handle, r1, r2]);
        this.lastActions[a.handle] = action;
      }
      const stopped = motionCheck(moves);
      let allDone = true;
      for (const a of this.agents) {
        const d = tmp[a.handle];
        const motionOk = !stopped.has(a.handle);
        const allowed = d.actionValid && ((onMap(a.state) && !this.isCellExit(a, d.ns)) || motionOk);
        // state machine
        const s = a.state;
        let next;
        if (s === ST.WAITING) next = d.inMalf ? ST.MALFUNCTION_OFF_MAP : d.edReached ? ST.READY_TO_DEPART : ST.WAITING;
        else if (s === ST.READY_TO_DEPART) next = d.inMalf ? ST.MALFUNCTION_OFF_MAP : d.mv && allowed ? ST.MOVING : ST.READY_TO_DEPART;
        else if (s === ST.MALFUNCTION_OFF_MAP) {
          if (!d.inMalf) {
            if (d.edReached) next = d.mv && allowed ? ST.MOVING : d.stop && allowed ? ST.STOPPED : ST.READY_TO_DEPART;
            else next = ST.WAITING;
          } else next = ST.MALFUNCTION_OFF_MAP;
        } else if (s === ST.MOVING) {
          if (d.inMalf) next = ST.MALFUNCTION;
          else if ((d.stop && d.ns === 0) || !allowed) next = ST.STOPPED;
          else next = ST.MOVING;
        } else if (s === ST.STOPPED) next = d.inMalf ? ST.MALFUNCTION : d.mv && allowed ? ST.MOVING : ST.STOPPED;
        else if (s === ST.MALFUNCTION) next = !d.inMalf ? (d.mv && allowed ? ST.MOVING : ST.STOPPED) : ST.MALFUNCTION;
        else next = ST.DONE;
        a.prevState = s;
        a.state = next;
        if (a.state === ST.MOVING) {
          a.cfg = d.ncfg;
          if (!(a.prevState === ST.READY_TO_DEPART || a.prevState === ST.MALFUNCTION_OFF_MAP)) {
            a.speed = d.ns;
            a.dist += a.speed;
            while (a.dist >= a.k) a.dist -= a.k;
          }
          if (a.targetSet.has(key3(a.cfg[0], a.cfg[1], a.cfg[2]))) {
            a.prevState = a.state;
            a.state = ST.DONE;
          }
        } else if (a.prevState === ST.MALFUNCTION_OFF_MAP && a.state === ST.STOPPED) a.cfg = a.start;
        if (onMap(a.state) && a.state !== ST.MOVING) a.speed = 0;
        // done
        if (a.state === ST.DONE && a.arrival === null) {
          a.arrival = t;
          a.cfg = null;
          this.events.push({ t, type: "arrival", train: a.handle, late: Math.max(0, t - a.la) });
          const r = Math.min(a.la - a.arrival, 0);
          a.reward += r;
          this.rewards[a.handle] += r;
        }
        allDone = allDone && a.state === ST.DONE;
        if (a.malf > 0) a.malf -= 1;
      }
      if (allDone || t >= this.T) {
        for (const a of this.agents) {
          if (a.state === ST.DONE) continue;
          let r = 0;
          if (offMap(a.state)) r = -this.travelTimeSP(a);
          else if (onMap(a.state)) r = Math.min(0, a.la - t - this.travelTimeSP(a));
          a.reward += r;
          this.rewards[a.handle] += r;
        }
        this.done = true;
      }
      return this.done;
    }
    // DefaultRewards.normalize
    normalizedReward() {
      let s = 0;
      for (const a of this.agents) s += Math.max(a.reward, -this.T);
      return s / (this.T * this.n) + 1;
    }
    arrived() {
      return this.agents.filter((a) => a.state === ST.DONE).length;
    }
    snapshot() {
      return this.agents.map((a) => ({ state: a.state, cfg: a.cfg ? a.cfg.slice() : null, dist: a.dist, speed: a.speed, malf: a.malf }));
    }
  }

  // ------------------------------------------------------------------ deadlock detection (rl_flatland.deadlock)
  function findDeadlocked(env) {
    const g = env.graph;
    const holder = new Map(), cfgs = new Map();
    for (const a of env.agents)
      if (onMap(a.state) && a.state !== ST.DONE && a.cfg) {
        cfgs.set(a.handle, a.cfg);
        holder.set(cellKey(a.cfg[0], a.cfg[1]), a.handle);
      }
    const cand = new Set();
    for (const [h, cfg] of cfgs) {
      const s = g.successors(cfg);
      if (s.length && s.every((x) => holder.has(cellKey(x[0], x[1])))) cand.add(h);
    }
    let changed = true;
    while (changed) {
      changed = false;
      for (const h of [...cand])
        for (const x of g.successors(cfgs.get(h)))
          if (!cand.has(holder.get(cellKey(x[0], x[1])))) {
            cand.delete(h);
            changed = true;
            break;
          }
    }
    const succCells = (h) => new Set(g.successors(cfgs.get(h)).map((x) => cellKey(x[0], x[1])));
    const dl = new Set();
    for (const h of cand) {
      const cell = cellKey(cfgs.get(h)[0], cfgs.get(h)[1]);
      for (const c of succCells(h)) {
        if (succCells(holder.get(c)).has(cell)) {
          dl.add(h);
          break;
        }
      }
    }
    changed = true;
    while (changed) {
      changed = false;
      for (const h of cand)
        if (!dl.has(h) && [...succCells(h)].every((c) => dl.has(holder.get(c)))) {
          dl.add(h);
          changed = true;
        }
    }
    return dl;
  }

  // ------------------------------------------------------------------ binary heap on [f, g, counter, ...]
  class Heap {
    constructor() {
      this.a = [];
    }
    get size() {
      return this.a.length;
    }
    static less(x, y) {
      if (x[0] !== y[0]) return x[0] < y[0];
      if (x[1] !== y[1]) return x[1] < y[1];
      return x[2] < y[2];
    }
    push(x) {
      const a = this.a;
      a.push(x);
      let i = a.length - 1;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (!Heap.less(a[i], a[p])) break;
        [a[i], a[p]] = [a[p], a[i]];
        i = p;
      }
    }
    pop() {
      const a = this.a, top = a[0], last = a.pop();
      if (a.length) {
        a[0] = last;
        let i = 0;
        for (;;) {
          const l = 2 * i + 1, r = l + 1;
          let m = i;
          if (l < a.length && Heap.less(a[l], a[m])) m = l;
          if (r < a.length && Heap.less(a[r], a[m])) m = r;
          if (m === i) break;
          [a[i], a[m]] = [a[m], a[i]];
          i = m;
        }
      }
      return top;
    }
  }

  // ------------------------------------------------------------------ reservation table + SIPP (or_planner.py)
  function insort(lst, item, cmp) {
    let lo = 0, hi = lst.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cmp(lst[mid], item) <= 0) lo = mid + 1;
      else hi = mid;
    }
    lst.splice(lo, 0, item);
  }
  const cmpPair = (x, y) => (x[0] !== y[0] ? x[0] - y[0] : x[1] - y[1]);
  const cmpTriple = (x, y) => (x[0] !== y[0] ? x[0] - y[0] : x[1] !== y[1] ? x[1] - y[1] : x[2] - y[2]);

  class ReservationTable {
    constructor() {
      this.intervals = new Map(); // cell -> sorted [[a, b], ...]
      this.moves = new Set(); // "u|v|t"
      this._safe = new Map();
      this.owned = new Map(); // train -> [[kind, ...], ...] (only the TADA executor uses it)
    }
    reserve(cell, a, b) {
      let l = this.intervals.get(cell);
      if (!l) this.intervals.set(cell, (l = []));
      insort(l, [a, b], cmpPair);
      this._safe.delete(cell);
    }
    reserveMove(u, v, t) {
      this.moves.add(u + "|" + v + "|" + t);
    }
    swapBlocked(u, v, t) {
      return this.moves.has(v + "|" + u + "|" + t);
    }
    safeIntervals(cell) {
      let s = this._safe.get(cell);
      if (s) return s;
      s = [];
      let t = 0;
      for (const [a, b] of this.intervals.get(cell) || []) {
        if (a > t) s.push([t, a - 1]);
        t = Math.max(t, b + 1);
      }
      if (t < INF) s.push([t, INF]);
      this._safe.set(cell, s);
      return s;
    }
    reservePath(path) {
      for (let j = 0; j < path.length - 1; j++) {
        const [cfg, t] = path[j], [nxt, tn] = path[j + 1];
        this.reserve(cellKey(cfg[0], cfg[1]), t, tn - 1);
        this.reserveMove(cellKey(cfg[0], cfg[1]), cellKey(nxt[0], nxt[1]), tn);
      }
      const [cfg, t] = path[path.length - 1];
      this.reserve(cellKey(cfg[0], cfg[1]), t, t);
    }
  }

  // sipp: earliest-arrival path from off the map. Returns [[cfg, entry time], ...] or null.
  function sipp(graph, rt, h, start, tMin, k, horizon, maxExp = 200000) {
    const hf = (cfg) => k * graph.distance(h, cfg);
    if (!isFinite(graph.distance(h, start))) return null;
    const heap = new Heap(), best = new Map(), parent = new Map();
    let counter = 0;
    const sk = (cfg, iv) => key3(cfg[0], cfg[1], cfg[2]) * 1024 + iv;
    const startCell = cellKey(start[0], start[1]);
    const ivs0 = rt.safeIntervals(startCell);
    for (let idx = 0; idx < ivs0.length; idx++) {
      const [lo, hi] = ivs0[idx];
      if (hi < tMin) continue;
      const t = Math.max(lo, tMin);
      if (t + hf(start) > horizon) break;
      const key = sk(start, idx);
      best.set(key, t);
      parent.set(key, null);
      heap.push([t + hf(start), t, counter++, start, idx]);
    }
    let exp = 0;
    while (heap.size) {
      const [, g, , cfg, iv] = heap.pop();
      const key = sk(cfg, iv);
      if ((best.has(key) ? best.get(key) : INF) < g) continue;
      if (graph.distance(h, cfg) === 0) {
        const path = [];
        let node = [cfg, iv];
        while (node) {
          const kk = sk(node[0], node[1]);
          path.push([node[0], best.get(kk)]);
          node = parent.get(kk);
        }
        return path.reverse();
      }
      if (++exp > maxExp) return null;
      const cell = cellKey(cfg[0], cfg[1]);
      const hi = rt.safeIntervals(cell)[iv][1];
      const leaveMin = g + k, leaveMax = hi < INF ? hi + 1 : INF;
      if (leaveMin > leaveMax) continue;
      for (const s of graph.successors(cfg)) {
        const hs = hf(s);
        if (!isFinite(hs)) continue;
        const scell = cellKey(s[0], s[1]);
        const ivs = rt.safeIntervals(scell);
        for (let j = 0; j < ivs.length; j++) {
          const [a, b] = ivs[j];
          if (b < leaveMin) continue;
          if (a > leaveMax) break;
          let t = Math.max(leaveMin, a);
          const tmax = Math.min(leaveMax, b);
          while (t <= tmax && rt.swapBlocked(cell, scell, t)) t++;
          if (t > tmax) continue;
          if (t + hs > horizon) break;
          const skey = sk(s, j);
          if (t < (best.has(skey) ? best.get(skey) : INF)) {
            best.set(skey, t);
            parent.set(skey, [cfg, iv]);
            heap.push([t + hs, t, counter++, s, j]);
          }
        }
      }
    }
    return null;
  }

  function cmpTuple(x, y) {
    for (let i = 0; i < x.length; i++) {
      if (x[i] < y[i]) return -1;
      if (x[i] > y[i]) return 1;
    }
    return 0;
  }

  const ORDERINGS = {
    speed: (ai) => [ai.k, ai.spTime],
    slack: (ai) => [ai.la - ai.ed - ai.spTime, ai.k],
    departure: (ai) => [ai.ed, ai.k],
    short: (ai) => [ai.spTime, ai.k],
  };

  function agentInfos(env) {
    return env.agents.map((a) => ({ handle: a.handle, start: a.start, k: a.k, ed: a.ed, la: a.la, spTime: a.k * env.graph.distance(a.handle, a.start) }));
  }

  function planAll(graph, infos, order, horizon, t0 = 0) {
    const rt = new ReservationTable(), paths = new Map();
    for (const h of order) {
      const ai = infos[h];
      if (!isFinite(ai.spTime)) continue;
      const p = sipp(graph, rt, h, ai.start, Math.max(ai.ed, t0) + 1, ai.k, horizon);
      if (p) {
        rt.reservePath(p);
        paths.set(h, p);
      }
    }
    return [rt, paths];
  }

  function planQuality(paths, infos) {
    let late = 0, arr = 0;
    for (const [h, p] of paths) {
      const ta = p[p.length - 1][1];
      arr += ta;
      late += Math.max(0, ta - infos[h].la);
    }
    return [paths.size, -late, -arr];
  }

  const actionBetween = (cur, nxt) => (nxt[2] === (cur[2] + 3) % 4 ? MOVE_LEFT : nxt[2] === (cur[2] + 1) % 4 ? MOVE_RIGHT : MOVE_FORWARD);

  /* Planner: PrioritizedPlannerPolicy. opts.orderings: names of the rule-based orderings (default all four);
   * opts.randomOrderings: permutations to try after them (the map's exported numpy permutations by default);
   * opts.order: plan exactly this priority order instead (the Lab's "drag the priority list"). */
  class Planner {
    constructor(opts = {}) {
      this.opts = opts;
      this.retryEvery = opts.retryEvery === undefined ? 10 : opts.retryEvery;
    }
    reset(env) {
      const t0 = Date.now();
      this.env = env;
      this.graph = env.graph;
      this.horizon = env.T;
      this.infos = agentInfos(env);
      const n = this.infos.length;
      const cands = [];
      if (this.opts.order) cands.push({ name: "custom", order: this.opts.order });
      else {
        for (const o of this.opts.orderings || ["speed", "slack", "departure", "short"]) {
          const idx = [...Array(n).keys()];
          idx.sort((i, j) => cmpTuple(ORDERINGS[o](this.infos[i]), ORDERINGS[o](this.infos[j])));
          cands.push({ name: o, order: idx });
        }
        for (const [i, perm] of (this.opts.randomOrderings || env.map.orderings || []).entries()) cands.push({ name: "random " + (i + 1), order: perm });
      }
      let best = null;
      this.tried = [];
      for (const c of cands) {
        const [rt, paths] = planAll(this.graph, this.infos, c.order, this.horizon);
        const q = planQuality(paths, this.infos);
        this.tried.push({ name: c.name, routed: q[0], late: -q[1], arrival: -q[2] });
        if (!best || cmpTuple(q, best.q) > 0) best = { q, rt, paths, name: c.name, order: c.order };
        if (q[0] === n && q[1] === 0) break;
      }
      this.rt = best.rt;
      this.paths = best.paths;
      this.chosen = best.name;
      this.order = best.order;
      this._buildExecution();
      this.setupMs = Date.now() - t0;
    }
    _buildExecution() {
      this.visits = new Map(); // cell -> sorted [[t, h, idx], ...]
      for (const [h, p] of this.paths)
        p.forEach(([cfg, t], idx) => {
          const c = cellKey(cfg[0], cfg[1]);
          if (!this.visits.has(c)) this.visits.set(c, []);
          this.visits.get(c).push([t, h, idx]);
        });
      for (const l of this.visits.values()) l.sort(cmpTriple);
      this.vptr = new Map();
      this.progress = new Map();
      for (const h of this.paths.keys()) this.progress.set(h, -1);
      this.deviations = 0;
    }
    _completed(v) {
      const [, h, idx] = v;
      if (this.env.agents[h].state === ST.DONE) return true;
      return this.progress.get(h) > idx;
    }
    _head(cell) {
      const lst = this.visits.get(cell) || [];
      let p = this.vptr.get(cell) || 0;
      while (p < lst.length && this._completed(lst[p])) p++;
      this.vptr.set(cell, p);
      return p;
    }
    _tryPlanUnrouted(now) {
      for (const ai of this.infos) {
        const h = ai.handle;
        if (this.paths.has(h)) continue;
        const a = this.env.agents[h];
        if (!offMap(a.state) || !isFinite(ai.spTime)) continue;
        const p = sipp(this.graph, this.rt, h, ai.start, Math.max(ai.ed + 1, now + 2), ai.k, this.horizon);
        if (!p) continue;
        this.rt.reservePath(p);
        this.paths.set(h, p);
        this.progress.set(h, -1);
        p.forEach(([cfg, t], idx) => {
          const c = cellKey(cfg[0], cfg[1]);
          if (!this.visits.has(c)) this.visits.set(c, []);
          insort(this.visits.get(c), [t, h, idx], cmpTriple);
        });
        this.env.events && this.env.events.push({ t: now, type: "replanned", train: h });
      }
    }
    _willMove(h, now, memo, stack) {
      if (memo.has(h)) return memo.get(h);
      if (stack.has(h)) return true; // a rotation of 3+ trains moves together
      stack.add(h);
      const ok = this._willMoveInner(h, now, memo, stack);
      stack.delete(h);
      memo.set(h, ok);
      return ok;
    }
    _willMoveInner(h, now, memo, stack) {
      const a = this.env.agents[h], p = this.paths.get(h), nxt = this.progress.get(h) + 1;
      if (nxt >= p.length || a.state === ST.DONE) return false;
      if (a.malf > 0) return false;
      const [ncfg, tp] = p[nxt];
      if (now + 1 < tp) return false;
      if (offMap(a.state)) {
        if (a.state !== ST.READY_TO_DEPART && a.state !== ST.MALFUNCTION_OFF_MAP) return false;
        if (now < a.ed) return false;
      } else if (!this.env.isCellExit(a, 1)) return false;
      const cell = cellKey(ncfg[0], ncfg[1]);
      const ptr = this._head(cell), lst = this.visits.get(cell);
      const mine = [tp, h, nxt];
      let lo = 0, hi = lst.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (cmpTriple(lst[mid], mine) < 0) lo = mid + 1;
        else hi = mid;
      }
      if (lo === ptr) return true;
      if (lo === ptr + 1) {
        const [, j, qidx] = lst[ptr];
        if (this.progress.get(j) === qidx && onMap(this.env.agents[j].state)) return this._willMove(j, now, memo, stack);
      }
      return false;
    }
    act(env) {
      const now = env.elapsed;
      if (this.retryEvery && now > 0 && now % this.retryEvery === 0) this._tryPlanUnrouted(now);
      const memo = new Map(), actions = {};
      for (const a of env.agents) {
        const h = a.handle;
        if (a.state === ST.DONE) continue;
        if (!this.paths.has(h)) {
          actions[h] = DO_NOTHING;
          continue;
        }
        const p = this.paths.get(h), nxt = this.progress.get(h) + 1;
        if (offMap(a.state)) {
          actions[h] = this._willMove(h, now, memo, new Set()) ? MOVE_FORWARD : DO_NOTHING;
          continue;
        }
        const move = actionBetween(a.cfg, p[nxt][0]);
        if (!env.isCellExit(a, 1)) actions[h] = move;
        else actions[h] = this._willMove(h, now, memo, new Set()) ? move : STOP_MOVING;
      }
      return actions;
    }
    observe(env) {
      for (const [h, p] of this.paths) {
        const a = env.agents[h];
        if (a.state === ST.DONE) {
          this.progress.set(h, p.length - 1);
          continue;
        }
        const cfg = a.cfg;
        if (!cfg) continue;
        const nxt = this.progress.get(h) + 1;
        if (nxt < p.length && cfg[0] === p[nxt][0][0] && cfg[1] === p[nxt][0][1]) this.progress.set(h, nxt);
        else if (this.progress.get(h) >= 0) {
          const c = p[this.progress.get(h)][0];
          if (cfg[0] !== c[0] || cfg[1] !== c[1]) this.deviations++;
        }
      }
    }
    plannedNext(h) {
      const p = this.paths.get(h);
      if (!p) return null;
      const nxt = this.progress.get(h) + 1;
      return nxt < p.length ? p[nxt][0] : null;
    }
  }

  // ------------------------------------------------------------------ replay of a recorded episode
  /* actions: per train [[step, action], ...] (run-length encoded). Returns a function step -> action dict. */
  function decodeActions(rle, n) {
    const cur = new Array(n).fill(DO_NOTHING), ptr = new Array(n).fill(0);
    return (step) => {
      for (let i = 0; i < n; i++) {
        const l = rle[i];
        while (ptr[i] < l.length && l[ptr[i]][0] <= step) {
          cur[i] = l[ptr[i]][1];
          ptr[i]++;
        }
      }
      const d = {};
      for (let i = 0; i < n; i++) d[i] = cur[i];
      return d;
    };
  }

  return {
    N, E, S, W, DR, DC, DO_NOTHING, MOVE_LEFT, MOVE_FORWARD, MOVE_RIGHT, STOP_MOVING, ST, STATE_NAMES, onMap, offMap, key3, cellKey, INF,
    Rail, Graph, RailEnv, motionCheck, pySetOrder, findDeadlocked, Heap, ReservationTable, sipp, planAll, planQuality, agentInfos, ORDERINGS, Planner,
    actionBetween, decodeActions, cmpTuple, insort, cmpTriple,
  };
});
