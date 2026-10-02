/* Interactive figures for the "How it works" chapters. Each <div class="fl-widget" data-widget="NAME"> becomes a small
 * app running the same JavaScript port of flatland as the Scheduling Lab (flatland-core.js), on hand-built layouts.
 *   cells     the cell types and their 16-bit transition codes
 *   drive     drive one train yourself: actions, configurations, decision cells
 *   speeds    speed classes, the train state machine, a breakdown
 *   deadlock  two trains, one passing loop: run them into a head-on deadlock, or let one wait
 *   sipp      prioritized planning with safe intervals on a time-distance diagram; change the priority order
 *   mcp       execute that plan under a breakdown: ordered execution against "go at the planned time"
 *   reward    flatland's per-train reward and the benchmark's normalised score */
(function () {
  "use strict";
  const FL = window.FL, D = window.FLDraw;
  if (!FL || !D) return;
  const ST = FL.ST;

  // ------------------------------------------------------------------ helpers
  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html !== undefined) e.innerHTML = html;
    return e;
  }
  function controls(root) {
    const c = el("div", "fl-controls");
    root.appendChild(c);
    return c;
  }
  function button(parent, text, onClick, cls = "fl-btn") {
    const b = el("button", cls, text);
    b.type = "button";
    b.addEventListener("click", onClick);
    parent.appendChild(b);
    return b;
  }
  function chips(parent, items, value, onPick) {
    const out = {};
    for (const [v, label] of items) {
      out[v] = button(parent, label, () => {
        for (const [k, b] of Object.entries(out)) b.setAttribute("aria-pressed", String(k === String(v)));
        onPick(v);
      }, "fl-chip");
      out[v].setAttribute("aria-pressed", String(String(v) === String(value)));
    }
    return out;
  }
  function slider(parent, label, min, max, step, value, fmt, onInput) {
    const wrap = el("label", "fl-ctl");
    wrap.appendChild(el("span", "fl-ctl__label", label));
    const inp = document.createElement("input");
    Object.assign(inp, { type: "range", min, max, step, value });
    const out = el("output", null, fmt(+value));
    inp.addEventListener("input", () => {
      out.textContent = fmt(+inp.value);
      onInput(+inp.value);
    });
    wrap.append(inp, out);
    parent.appendChild(wrap);
    return { input: inp, set: (v) => ((inp.value = v), (out.textContent = fmt(+v))) };
  }
  function note(root, html) {
    const n = el("p", "fl-note", html);
    root.appendChild(n);
    return n;
  }
  function onTheme(fn) {
    new MutationObserver(fn).observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
    let rz = null;
    window.addEventListener("resize", () => {
      clearTimeout(rz);
      rz = setTimeout(fn, 150);
    });
  }
  const DIRN = ["north", "east", "south", "west"], ARROW = ["↑", "→", "↓", "←"];
  const ACTN = ["DO_NOTHING", "MOVE_LEFT", "MOVE_FORWARD", "MOVE_RIGHT", "STOP_MOVING"];

  // ------------------------------------------------------------------ toy layouts
  /* a straight corridor on row 3 from column 1 to W-2, with passing loops on row 2 between main-line columns a and b */
  function corridor(W, loops) {
    const H = 5, row = 3, main = [];
    for (let c = 1; c <= W - 2; c++) main.push([row, c]);
    const tracks = [main];
    for (const [a, b] of loops) {
      const line = [
        [row, a],
        [row, a + 1],
        [row - 1, a + 1],
      ];
      for (let c = a + 2; c <= b - 1; c++) line.push([row - 1, c]);
      line.push([row - 1, b], [row, b], [row, b + 1]);
      tracks.push(line);
    }
    return { H, W, grid: D.buildRail(H, W, tracks) };
  }
  function toyMap(rail, agents, T) {
    return { H: rail.H, W: rail.W, grid: rail.grid, agents, T, malf: [], orderings: [], episodes: {} };
  }
  /* shortest configuration path from start to target that passes through cell `via` (BFS over configurations) */
  function pathVia(graph, start, via, targets) {
    const bfs = (src, goal) => {
      const key = (c) => c.join(",");
      const prev = new Map([[key(src), null]]);
      let q = [src];
      while (q.length) {
        const nq = [];
        for (const c of q) {
          if (goal(c)) {
            const out = [c];
            let k = prev.get(key(c));
            while (k) {
              out.push(k);
              k = prev.get(key(k));
            }
            return out.reverse();
          }
          for (const s of graph.successors(c))
            if (!prev.has(key(s))) {
              prev.set(key(s), c);
              nq.push(s);
            }
        }
        q = nq;
      }
      return null;
    };
    const a = bfs(start, (c) => c[0] === via[0] && c[1] === via[1]);
    if (!a) return null;
    const b = bfs(a[a.length - 1], (c) => targets.some((t) => t[0] === c[0] && t[1] === c[1] && t[2] === c[2]));
    return b ? a.concat(b.slice(1)) : null;
  }
  /* follow a configuration path: the action that takes a train from where it is to the next configuration */
  function followAction(a, path, hold) {
    if (a.state === ST.DONE) return FL.DO_NOTHING;
    if (FL.offMap(a.state)) return hold ? FL.DO_NOTHING : FL.MOVE_FORWARD;
    const i = path.findIndex((c) => c[0] === a.cfg[0] && c[1] === a.cfg[1]);
    if (i < 0 || i + 1 >= path.length) return FL.MOVE_FORWARD;
    if (hold) return FL.STOP_MOVING;
    return FL.actionBetween(a.cfg, path[i + 1]);
  }

  function drawToy(cv, env, opts = {}) {
    const col = D.colors(), ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv._w, cv._h);
    const view = D.fitView(env.rail, cv._w, cv._h, 10);
    D.drawRail(ctx, env.rail, env.graph, view, col);
    env.agents.forEach((a, i) => {
      if (opts.stations !== false && a.state !== ST.DONE) {
        const t = a.targets[0];
        D.drawStation(ctx, view, t[0], t[1], opts.colors ? opts.colors[i] : col.muted, view.s >= 18 ? "→" + (opts.names ? opts.names[i] : i) : undefined);
      }
    });
    env.agents.forEach((a, i) => {
      if (!a.cfg) return;
      D.drawTrain(ctx, view, a.cfg, opts.colors ? opts.colors[i] : D.stateColor(a.state, col), {
        label: opts.names ? opts.names[i] : i,
        labelColor: col.fg,
        ring: a.state === ST.MALFUNCTION ? col.bad : opts.ring && opts.ring(i) ? col.accent : null,
        cross: opts.cross && opts.cross.has(i) ? col.bad : null,
      });
    });
    return view;
  }

  // ------------------------------------------------------------------ 1. cells
  const CELL_TYPES = [
    ["straight", "Straight", [[1, 1], [3, 3]]],
    ["curve", "Curve", [[1, 0], [2, 3]]],
    ["switch", "Simple switch", [[1, 1], [3, 3], [1, 0], [2, 3]]],
    ["crossing", "Diamond crossing", [[1, 1], [3, 3], [0, 0], [2, 2]]],
    ["slip", "Single slip", [[1, 1], [3, 3], [0, 0], [2, 2], [1, 0], [2, 3]]],
    ["dslip", "Double slip", [[1, 1], [3, 3], [0, 0], [2, 2], [1, 0], [2, 3], [3, 2], [0, 1]]],
    ["sym", "Symmetric switch", [[0, 1], [0, 3], [3, 2], [1, 2]]],
    ["dead", "Dead end", [[1, 3], [3, 3]]],
  ];
  const codeOf = (pairs) => pairs.reduce((c, [h, x]) => c | (1 << ((3 - h) * 4 + (3 - x))), 0);
  function wCells(root) {
    let type = "switch", heading = 1;
    const c = controls(root);
    chips(c, CELL_TYPES.map(([k, l]) => [k, l]), type, (v) => ((type = v), draw()));
    const c2 = controls(root);
    c2.appendChild(el("span", "fl-ctl__label", "Train heading"));
    chips(c2, [0, 1, 2, 3].map((d) => [d, ARROW[d] + " " + DIRN[d]]), heading, (v) => ((heading = +v), draw()));
    const row = el("div", "fl-row2");
    root.appendChild(row);
    const left = el("div");
    row.appendChild(left);
    const cv = D.makeCanvas(left, 230);
    const right = el("div", "fl-readout");
    row.appendChild(right);
    function draw() {
      const pairs = CELL_TYPES.find((x) => x[0] === type)[2];
      const code = codeOf(pairs);
      const rail = new FL.Rail(1, 1, [code]);
      const col = D.colors(), ctx = cv.getContext("2d");
      const w = cv._w, h = cv._h, s = Math.min(h - 40, w - 40);
      const x0 = (w - s) / 2, y0 = (h - s) / 2;
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = col.grid;
      ctx.lineWidth = 1;
      ctx.strokeRect(x0, y0, s, s);
      ctx.save();
      ctx.lineCap = "round";
      ctx.strokeStyle = col.rail;
      ctx.lineWidth = s * 0.06;
      ctx.beginPath();
      for (const [a, b] of D.cellPieces(rail, 0, 0)) {
        const [ax, ay] = D.SIDE[a], [bx, by] = D.SIDE[b];
        ctx.moveTo(x0 + ax * s, y0 + ay * s);
        if (a === b) ctx.lineTo(x0 + (ax * 0.5 + 0.25) * s, y0 + (ay * 0.5 + 0.25) * s);
        else if ((a + 2) % 4 === b) ctx.lineTo(x0 + bx * s, y0 + by * s);
        else ctx.quadraticCurveTo(x0 + 0.5 * s, y0 + 0.5 * s, x0 + bx * s, y0 + by * s);
      }
      ctx.stroke();
      // the train entering with `heading` comes in through side (heading + 2) % 4
      const t = rail.trans(0, 0, heading), any = t.some(Boolean);
      const ent = D.SIDE[(heading + 2) % 4];
      const arrow = (fx, fy, tx, ty, color, wdt) => {
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = wdt;
        ctx.beginPath();
        ctx.moveTo(fx, fy);
        ctx.lineTo(tx, ty);
        ctx.stroke();
        const a = Math.atan2(ty - fy, tx - fx), L = 10;
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        ctx.lineTo(tx - L * Math.cos(a - 0.4), ty - L * Math.sin(a - 0.4));
        ctx.lineTo(tx - L * Math.cos(a + 0.4), ty - L * Math.sin(a + 0.4));
        ctx.closePath();
        ctx.fill();
      };
      const ex0 = x0 + ent[0] * s, ey0 = y0 + ent[1] * s;
      const out = [ex0 - [0, 1, 0, -1][(heading + 2) % 4] * 0, ey0];
      arrow(x0 + (ent[0] - 0.5) * s * 1.3 + 0.5 * s, y0 + (ent[1] - 0.5) * s * 1.3 + 0.5 * s, ex0, ey0, any ? col.accent : col.muted, 3);
      for (let nd = 0; nd < 4; nd++)
        if (t[nd]) {
          const sd = D.SIDE[nd];
          arrow(x0 + 0.5 * s, y0 + 0.5 * s, x0 + (sd[0] - 0.5) * s * 1.18 + 0.5 * s, y0 + (sd[1] - 0.5) * s * 1.18 + 0.5 * s, col.ok, 2.5);
        }
      ctx.restore();
      // the code
      const bits = code.toString(2).padStart(16, "0");
      let html = `<p><b>16-bit code ${code}</b> (binary, one group of four exits per heading, in the order N E S W):</p><p class="fl-bits">`;
      for (let hd = 0; hd < 4; hd++) {
        html += `<span title="heading ${DIRN[hd]}">${ARROW[hd]}&nbsp;`;
        for (let x = 0; x < 4; x++) {
          const bit = bits[hd * 4 + x];
          html += `<i class="${hd === heading && bit === "1" ? "on" : ""}">${bit}</i>`;
        }
        html += "</span>&nbsp;&nbsp; ";
      }
      html += "</p>";
      const exits = [0, 1, 2, 3].filter((d) => t[d]).map((d) => DIRN[d]);
      html += exits.length
        ? `<p>A train heading <b>${DIRN[heading]}</b> (entering from the ${DIRN[(heading + 2) % 4]} side) may leave <b>${exits.join(" or ")}</b>${exits.length > 1 ? ": a <b>facing switch</b>, the only place a train chooses its route" : ""}.</p>`
        : `<p>No train can be in this cell heading <b>${DIRN[heading]}</b>: flatland treats that configuration as invalid.</p>`;
      right.innerHTML = html;
    }
    draw();
    onTheme(() => (cv.fit(), draw()));
    note(root, "Each cell stores which exits are open for each of the four headings a train can have inside it. Flatland's generated maps only use a handful of patterns, all of which are here.");
  }

  // ------------------------------------------------------------------ 2. drive
  function wDrive(root) {
    const rail = corridor(24, [[8, 15]]);
    let k = 1;
    let env, last = "";
    const mk = () => {
      env = new FL.RailEnv(toyMap(rail, [{ start: [3, 1, 1], targets: [[3, 22, 1]], k, ed: 0, la: 60 }], 80));
      last = "The train is ready to depart. Any move action puts it on its start cell.";
    };
    mk();
    const c = controls(root);
    c.appendChild(el("span", "fl-ctl__label", "Speed"));
    chips(c, [[1, "1 cell per step"], [2, "1/2"], [3, "1/3"]], k, (v) => ((k = +v), mk(), draw()));
    const pad = controls(root);
    const act = (a) => {
      if (env.done) return;
      const ag = env.agents[0];
      const cur = ag.cfg || ag.start;
      const chk = env.rail.checkActionNew(a, cur[0], cur[1], cur[2]);
      env.step({ 0: a });
      const corrected = chk[2] !== a && a !== FL.DO_NOTHING && a !== FL.STOP_MOVING;
      last = `${ACTN[a]}${corrected ? ` → flatland reads it as ${ACTN[chk[2]]}${chk[3] ? "" : " (invalid here)"}` : ""}.`;
      draw();
    };
    button(pad, "← MOVE_LEFT", () => act(FL.MOVE_LEFT));
    button(pad, "↑ MOVE_FORWARD", () => act(FL.MOVE_FORWARD));
    button(pad, "→ MOVE_RIGHT", () => act(FL.MOVE_RIGHT));
    button(pad, "■ STOP_MOVING", () => act(FL.STOP_MOVING));
    button(pad, "· DO_NOTHING", () => act(FL.DO_NOTHING));
    button(pad, "Reset", () => (mk(), draw()));
    const cv = D.makeCanvas(root, 170);
    const out = el("div", "fl-readout");
    root.appendChild(out);
    function draw() {
      drawToy(cv, env, { colors: [D.colors().accent] });
      const a = env.agents[0], g = env.graph;
      const cfg = a.cfg;
      let html = `<p><b>t = ${env.elapsed}</b> · state <b>${FL.STATE_NAMES[a.state]}</b>`;
      if (cfg) {
        const succ = g.successors(cfg);
        html += ` · configuration <b>(row ${cfg[0]}, col ${cfg[1]}, heading ${DIRN[cfg[2]]})</b> · ${a.dist}/${a.k} of the way through this cell · ${g.distance(0, cfg)} cells to the target</p>`;
        html += `<p>Next cells: ${succ.map((s) => `(${s[0]}, ${s[1]}) ${DIRN[s[2]]}`).join(" or ")}. ${g.isDecision(cfg) ? "<b>Decision cell</b>: a facing switch is here or next, so this is where the train must decide." : "Plain track: only one way on."}</p>`;
      } else html += a.state === ST.DONE ? ` at t = ${a.arrival} (due by ${a.la}).</p>` : ".</p>";
      html += `<p class="fl-status">${last}</p>`;
      out.innerHTML = html;
    }
    draw();
    onTheme(() => (cv.fit(), draw()));
    note(root, "MOVE_LEFT at the switch takes the loop; elsewhere flatland quietly turns a LEFT or RIGHT into FORWARD. At slower speeds a train needs several steps per cell and only the last one (the cell exit) can change where it goes.");
  }

  // ------------------------------------------------------------------ 3. speeds, states, breakdowns
  const SM = [
    [ST.WAITING, "waiting", 0, 0],
    [ST.READY_TO_DEPART, "ready to depart", 1, 0],
    [ST.MALFUNCTION_OFF_MAP, "broken down before departure", 1, 1],
    [ST.MOVING, "moving", 2, 0],
    [ST.STOPPED, "stopped", 2, 1],
    [ST.MALFUNCTION, "broken down", 3, 1],
    [ST.DONE, "arrived", 3, 0],
  ];
  function wSpeeds(root) {
    const tracks = [1, 3, 5, 7].map((r) => {
      const l = [];
      for (let c = 1; c <= 22; c++) l.push([r, c]);
      return l;
    });
    const rail = { H: 9, W: 24, grid: D.buildRail(9, 24, tracks) };
    const agents = [1, 2, 3, 4].map((k, i) => ({ start: [1 + 2 * i, 1, 1], targets: [[1 + 2 * i, 22, 1]], k, ed: 2 * i, la: 30 + 15 * i }));
    let brk = [], sel = 1, env, timer = null;
    const mk = () => (env = new FL.RailEnv(Object.assign(toyMap(rail, agents, 120), {}), { extraMalfunctions: brk }));
    mk();
    const c = controls(root);
    const play = button(c, "Play", () => toggle(), "fl-btn fl-btn--on");
    button(c, "Step", () => step());
    button(c, "Reset", () => ((brk = []), mk(), stop(), draw()));
    const c2 = controls(root);
    c2.appendChild(el("span", "fl-ctl__label", "Train"));
    chips(c2, agents.map((a, i) => [i, `${i} · speed 1/${a.k}`]), sel, (v) => ((sel = +v), draw()));
    button(c2, "Break it for 8 steps", () => {
      brk.push([env.elapsed + 1, sel, 8]);
      // re-run to now with the new breakdown, deterministic
      const t = env.elapsed;
      mk();
      while (env.elapsed < t && !env.done) env.step(policy());
      draw();
    });
    const row = el("div", "fl-row2");
    root.appendChild(row);
    const l = el("div");
    row.appendChild(l);
    const cv = D.makeCanvas(l, 200);
    const r = el("div");
    row.appendChild(r);
    const sm = D.makeCanvas(r, 200);
    const out = el("div", "fl-readout");
    root.appendChild(out);
    const policy = () => Object.fromEntries(env.agents.map((a) => [a.handle, FL.MOVE_FORWARD]));
    function step() {
      if (env.done) return stop();
      env.step(policy());
      draw();
    }
    function toggle() {
      if (timer) stop();
      else {
        if (env.done) mk();
        timer = setInterval(step, 300);
        play.textContent = "Pause";
      }
    }
    function stop() {
      clearInterval(timer);
      timer = null;
      play.textContent = "Play";
    }
    function draw() {
      drawToy(cv, env, { ring: (i) => i === sel });
      // state machine of the selected train
      const col = D.colors(), ctx = sm.getContext("2d"), w = sm._w, h = sm._h;
      ctx.clearRect(0, 0, w, h);
      const a = env.agents[sel];
      const bw = (w - 30) / 4, bh = 40;
      const pos = (cx, cy) => [10 + cx * (bw + 3.3), 20 + cy * (bh + 40)];
      ctx.font = D.FONT_SMALL;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (const [s, name, cx, cy] of SM) {
        const [x, y] = pos(cx, cy);
        const on = a.state === s, was = a.prevState === s && !on;
        ctx.fillStyle = on ? (s === ST.MALFUNCTION || s === ST.MALFUNCTION_OFF_MAP ? col.bad : col.accent) : col.panel;
        ctx.strokeStyle = was ? col.accent : col.muted;
        ctx.lineWidth = was ? 2 : 1;
        ctx.beginPath();
        ctx.rect(x, y, bw, bh);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = on ? "#15171c" : col.fg;
        const words = name.split(" ");
        if (words.length > 2) {
          ctx.fillText(words.slice(0, 2).join(" "), x + bw / 2, y + bh / 2 - 7);
          ctx.fillText(words.slice(2).join(" "), x + bw / 2, y + bh / 2 + 7);
        } else ctx.fillText(name, x + bw / 2, y + bh / 2);
      }
      ctx.fillStyle = col.muted;
      ctx.textAlign = "left";
      ctx.fillText(`train ${sel}: now ${FL.STATE_NAMES[a.state]}${a.prevState !== null ? ", was " + FL.STATE_NAMES[a.prevState] : ""}`, 10, h - 10);
      // table
      let html = `<p><b>t = ${env.elapsed}</b></p><table><thead><tr><th>Train</th><th>Speed</th><th>Departs ≥</th><th>Due by</th><th>State</th><th>In this cell</th><th>Breakdown left</th><th>Arrived</th></tr></thead><tbody>`;
      for (const x of env.agents)
        html += `<tr><td>${x.handle}</td><td>1/${x.k}</td><td>${x.ed}</td><td>${x.la}</td><td>${FL.STATE_NAMES[x.state]}</td><td>${x.cfg ? `${x.dist}/${x.k}` : "–"}</td><td>${x.malf || "–"}</td><td>${x.arrival !== null ? x.arrival + (x.arrival > x.la ? ` (${x.arrival - x.la} late)` : " (on time)") : "–"}</td></tr>`;
      out.innerHTML = html + "</tbody></table>";
    }
    draw();
    onTheme(() => (cv.fit(), sm.fit(), draw()));
    note(root, "Every train is always told MOVE_FORWARD here. A train of speed 1/k spends k steps in each cell; it can only leave at the last one. A breakdown freezes it wherever it is, on or off the map, and it resumes when the counter reaches zero.");
  }

  // ------------------------------------------------------------------ 4. deadlock
  function wDeadlock(root) {
    const rail = corridor(24, [[8, 15]]);
    const agents = [
      { start: [3, 1, 1], targets: [[3, 22, 1]], k: 1, ed: 0, la: 40 },
      { start: [3, 22, 3], targets: [[3, 1, 3]], k: 1, ed: 0, la: 40 },
    ];
    const MODES = [
      ["run", "run straight through"],
      ["loop", "take the loop"],
      ["wait", "take the loop and wait"],
    ];
    let mode = ["run", "run"], env, timer = null, paths, msg = "";
    const names = ["A", "B"];
    function mk() {
      env = new FL.RailEnv(toyMap(rail, agents, 60));
      const g = env.graph;
      paths = agents.map((a, i) => (mode[i] === "run" ? g.shortestPath(i, a.start) : pathVia(g, a.start, [2, 11], a.targets)));
      msg = "";
    }
    mk();
    const ctl = [0, 1].map((i) => {
      const c = controls(root);
      c.appendChild(el("span", "fl-ctl__label", `Train ${names[i]} (${i === 0 ? "eastbound" : "westbound"})`));
      chips(c, MODES, mode[i], (v) => ((mode[i] = v), reset()));
      return c;
    });
    const c = controls(root);
    const play = button(c, "Play", () => toggle(), "fl-btn fl-btn--on");
    button(c, "Step", () => step());
    button(c, "Reset", () => reset());
    const cv = D.makeCanvas(root, 150);
    const status = el("div", "fl-status");
    root.appendChild(status);
    function policy() {
      const out = {};
      env.agents.forEach((a, i) => {
        const other = env.agents[1 - i];
        let hold = false;
        if (mode[i] === "wait" && a.cfg && a.cfg[0] === 2) {
          // wait at the far end of the loop until the other train has passed the loop
          const end = i === 0 ? 15 : 9;
          const passed = other.state === ST.DONE || (other.cfg && (i === 0 ? other.cfg[1] < 9 : other.cfg[1] > 15)) || (FL.offMap(other.state) && other.state !== ST.WAITING && false);
          if (a.cfg[1] === end && !passed) hold = true;
        }
        out[i] = followAction(a, paths[i], hold);
      });
      return out;
    }
    function step() {
      if (env.done) return stop();
      env.step(policy());
      draw();
      if (env.done || FL.findDeadlocked(env).size) stop();
    }
    function toggle() {
      if (timer) stop();
      else {
        if (env.done) mk();
        timer = setInterval(step, 220);
        play.textContent = "Pause";
      }
    }
    function stop() {
      clearInterval(timer);
      timer = null;
      play.textContent = "Play";
    }
    function reset() {
      stop();
      mk();
      draw();
    }
    function draw() {
      const dl = FL.findDeadlocked(env);
      const col = D.colors();
      drawToy(cv, env, { names, colors: [col.series[0], col.series[5]], cross: dl });
      const [A, B] = env.agents;
      if (dl.size) {
        status.className = "fl-status bad";
        status.textContent = `t = ${env.elapsed}: deadlock. A and B face each other on single track. Moving would be a swap, which flatland's MotionCheck never allows, and neither can reverse: they stay here until the episode ends.`;
      } else if (A.state === ST.DONE && B.state === ST.DONE) {
        status.className = "fl-status good";
        status.textContent = `Both arrived: A at t = ${A.arrival}, B at t = ${B.arrival} (due by 40).`;
      } else {
        status.className = "fl-status";
        status.textContent = `t = ${env.elapsed}: A ${FL.STATE_NAMES[A.state]}, B ${FL.STATE_NAMES[B.state]}.`;
      }
    }
    draw();
    onTheme(() => (cv.fit(), draw()));
    note(root, "Try both trains on “run straight through”, then one on “take the loop and wait”. Taking the loop without waiting is not enough when the timing is wrong: the decision that avoids a deadlock is a decision about order, made before the trains meet.");
  }

  // ------------------------------------------------------------------ 5 and 6. the planning corridor
  const PL_RAIL = () => corridor(30, [[7, 13], [17, 23]]);
  const PL_AGENTS = [
    { start: [3, 1, 1], targets: [[3, 28, 1]], k: 1, ed: 0, la: 40, name: "A" },
    { start: [3, 28, 3], targets: [[3, 1, 3]], k: 1, ed: 2, la: 42, name: "B" },
    { start: [3, 1, 1], targets: [[3, 28, 1]], k: 2, ed: 6, la: 80, name: "C" },
    { start: [3, 28, 3], targets: [[3, 1, 3]], k: 1, ed: 12, la: 55, name: "D" },
  ];
  function plMap() {
    return toyMap(PL_RAIL(), PL_AGENTS.map((a) => ({ start: a.start, targets: a.targets, k: a.k, ed: a.ed, la: a.la })), 120);
  }
  /* time-distance diagram: x = time, y = column along the corridor (west at the top), loop cells dashed */
  function drawTD(cv, lines, opts) {
    const col = D.colors(), ctx = cv.getContext("2d"), w = cv._w, h = cv._h;
    ctx.clearRect(0, 0, w, h);
    const T = opts.T, left = 46, right = 12, top = 10, bottom = 24;
    const X = (t) => left + (t / T) * (w - left - right), Y = (c) => top + ((c - 1) / 27) * (h - top - bottom);
    ctx.font = D.FONT_SMALL;
    ctx.strokeStyle = col.grid;
    ctx.fillStyle = col.muted;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let t = 0; t <= T; t += 10) {
      ctx.beginPath();
      ctx.moveTo(X(t), top);
      ctx.lineTo(X(t), h - bottom);
      ctx.stroke();
      ctx.fillText(String(t), X(t), h - bottom + 5);
    }
    // the passing loops as shaded bands
    ctx.fillStyle = col.dark ? "rgba(255,201,60,0.07)" : "rgba(199,119,0,0.07)";
    for (const [a, b] of [[8, 13], [18, 23]]) ctx.fillRect(left, Y(a), w - left - right, Y(b) - Y(a));
    ctx.fillStyle = col.muted;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText("west", left - 6, Y(1));
    ctx.fillText("east", left - 6, Y(28));
    ctx.fillText("loop", left - 6, Y(10.5));
    ctx.fillText("loop", left - 6, Y(20.5));
    for (const ln of lines) {
      if (!ln.pts.length) continue;
      ctx.strokeStyle = ln.color;
      ctx.lineWidth = ln.width || 2;
      ctx.setLineDash(ln.dash || []);
      ctx.globalAlpha = ln.alpha || 1;
      // staircase: a train holds cell c from its entry until the next entry; time spent in a loop is dotted
      const seg = (x0, y0, x1, y1, loop) => {
        ctx.setLineDash(loop ? [2, 3] : ln.dash || []);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
      };
      ln.pts.forEach(([t, c, loop], i) => {
        if (i === 0) return;
        const [pt, pc, ploop] = ln.pts[i - 1];
        seg(X(pt), Y(pc), X(t), Y(pc), ploop);
        seg(X(t), Y(pc), X(t), Y(c), ploop && loop);
      });
      if (ln.end !== undefined) {
        const [lt, lc, lloop] = ln.pts[ln.pts.length - 1];
        seg(X(lt), Y(lc), X(ln.end), Y(lc), lloop);
      }
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      const [t0, c0] = ln.pts[0];
      ctx.fillStyle = ln.color;
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      if (ln.label) ctx.fillText(ln.label, X(t0) + 3, Y(c0) - 2);
    }
    for (const [t, c] of opts.conflicts || []) {
      ctx.strokeStyle = col.bad;
      ctx.lineWidth = 2;
      const x = X(t), y = Y(c), q = 6;
      ctx.beginPath();
      ctx.moveTo(x - q, y - q);
      ctx.lineTo(x + q, y + q);
      ctx.moveTo(x + q, y - q);
      ctx.lineTo(x - q, y + q);
      ctx.stroke();
    }
    if (opts.cursor !== undefined) {
      ctx.strokeStyle = col.fg;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(X(opts.cursor), top);
      ctx.lineTo(X(opts.cursor), h - bottom);
      ctx.stroke();
    }
    return { X, Y };
  }
  const pathPts = (p) => p.map(([cfg, t]) => [t, cfg[1], cfg[0] === 2]);
  /* cells shared at the same time, or swapped in one step, by two independently planned paths */
  function conflicts(paths) {
    const occ = new Map(), out = [];
    paths.forEach((p, h) => {
      if (!p) return;
      for (let j = 0; j < p.length; j++) {
        const [cfg, t] = p[j];
        const t1 = j + 1 < p.length ? p[j + 1][1] - 1 : t;
        for (let s = t; s <= t1; s++) {
          const k = cfg[0] * 100 + cfg[1] + ":" + s;
          if (occ.has(k) && occ.get(k) !== h) out.push([s, cfg[1]]);
          occ.set(k, h);
        }
      }
    });
    return out.filter((x, i) => out.findIndex((y) => Math.abs(y[0] - x[0]) < 3 && y[1] === x[1]) === i).slice(0, 12);
  }

  function wSipp(root) {
    let order = [0, 1, 2, 3], alone = false;
    const map = plMap();
    const env = new FL.RailEnv(map);
    const infos = FL.agentInfos(env);
    const c = controls(root);
    c.appendChild(el("span", "fl-ctl__label", "Priority order (click a train to move it to the front)"));
    const ord = el("div", "fl-order");
    c.appendChild(ord);
    const c2 = controls(root);
    chips(c2, [[0, "plan in priority order (PP + SIPP)"], [1, "every train for itself"]], 0, (v) => ((alone = +v === 1), draw()));
    const cvMap = D.makeCanvas(root, 120);
    const cv = D.makeCanvas(root, 300);
    const out = el("div", "fl-readout");
    root.appendChild(out);
    function draw() {
      ord.innerHTML = "";
      order.forEach((h, i) => {
        const b = button(ord, `${i + 1}. ${PL_AGENTS[h].name}`, () => {
          order = [h].concat(order.filter((x) => x !== h));
          draw();
        }, "fl-chip");
        b.setAttribute("aria-pressed", String(i === 0));
      });
      const col = D.colors();
      const colors = [col.series[0], col.series[5], col.series[2], col.series[1]];
      let paths;
      if (alone) paths = PL_AGENTS.map((a, h) => FL.sipp(env.graph, new FL.ReservationTable(), h, a.start, a.ed + 1, a.k, 120));
      else {
        const [, ps] = FL.planAll(env.graph, infos, order, 120);
        paths = PL_AGENTS.map((a, h) => ps.get(h) || null);
      }
      drawToy(cvMap, env, { names: PL_AGENTS.map((a) => a.name), colors, stations: true });
      const cf = alone ? conflicts(paths) : [];
      drawTD(cv, paths.map((p, h) => ({ pts: p ? pathPts(p) : [], color: colors[h], label: PL_AGENTS[h].name + (p ? "" : " (unroutable)") })), { T: 90, conflicts: cf });
      let late = 0;
      let html = "<table><thead><tr><th>Train</th><th>Speed</th><th>Departs ≥</th><th>Due by</th><th>Planned arrival</th><th>Late</th></tr></thead><tbody>";
      paths.forEach((p, h) => {
        const a = PL_AGENTS[h], arr = p ? p[p.length - 1][1] : null;
        const l = arr === null ? null : Math.max(0, arr - a.la);
        late += l || 0;
        html += `<tr><td>${a.name}</td><td>1/${a.k}</td><td>${a.ed}</td><td>${a.la}</td><td>${arr === null ? "no path" : arr}</td><td>${l === null ? "–" : l}</td></tr>`;
      });
      html += `</tbody></table><p class="fl-status ${alone ? (cf.length ? "bad" : "") : late ? "" : "good"}">${alone ? `Planned alone, every train takes its fastest path: ${cf.length ? "the red crosses are cells two trains need at the same time. In flatland one of them would be stopped, and head-on meetings on single track end in deadlock." : "no conflicts."}` : `Total lateness ${late} steps. Each train's earliest path avoids every cell-time reserved by the trains planned before it; it waits in a loop where it must.`}</p>`;
      out.innerHTML = html;
    }
    draw();
    onTheme(() => (cv.fit(), cvMap.fit(), draw()));
    note(root, "Top: the corridor, two passing loops, four trains (A and C eastbound, B and D westbound; C is half speed). Bottom: time runs to the right, position along the corridor downwards (west at the top); a flat line is a train standing in a cell, the shaded bands are the loops. Dotted: a train in a passing loop. Lines that cross outside a loop are trains that would meet on single track.");
  }

  function wMcp(root) {
    let brkTrain = 0, brkAt = 6, brkLen = 12, ordered = true, t = 0, timer = null;
    const col0 = D.colors();
    let run = null;
    function simulate() {
      const map = plMap();
      const env = new FL.RailEnv(map, { extraMalfunctions: brkLen ? [[brkAt, brkTrain, brkLen]] : [] });
      const pl = ordered ? new FL.Planner({ orderings: ["speed"], randomOrderings: [] }) : new NaivePlanner({ orderings: ["speed"], randomOrderings: [] });
      pl.reset(env);
      const plan = [...pl.paths.entries()];
      const hist = PL_AGENTS.map(() => []);
      const frames = [env.snapshot()];
      let dlAt = -1;
      while (!env.done) {
        env.step(pl.act(env));
        pl.observe(env);
        frames.push(env.snapshot());
        env.agents.forEach((a, i) => {
          if (a.cfg) {
            const h = hist[i];
            if (!h.length || h[h.length - 1][1] !== a.cfg[1] || h[h.length - 1][2] !== (a.cfg[0] === 2)) h.push([env.elapsed, a.cfg[1], a.cfg[0] === 2]);
          }
        });
        if (dlAt < 0 && FL.findDeadlocked(env).size) dlAt = env.elapsed;
      }
      run = { env, plan, hist, frames, dlAt };
    }
    const c = controls(root);
    c.appendChild(el("span", "fl-ctl__label", "Execution"));
    chips(c, [[1, "ordered execution (keep every cell's planned order)"], [0, "go at the planned time"]], 1, (v) => ((ordered = +v === 1), simulate(), (t = Math.min(t, run.frames.length - 1)), draw()));
    const c2 = controls(root);
    c2.appendChild(el("span", "fl-ctl__label", "Breakdown"));
    chips(c2, PL_AGENTS.map((a, i) => [i, "train " + a.name]), brkTrain, (v) => ((brkTrain = +v), simulate(), draw()));
    slider(c2, "at t", 1, 40, 1, brkAt, (v) => String(v), (v) => ((brkAt = v), simulate(), draw()));
    slider(c2, "for", 0, 30, 1, brkLen, (v) => v + " steps", (v) => ((brkLen = v), simulate(), draw()));
    const c3 = controls(root);
    const play = button(c3, "Play", () => toggle(), "fl-btn fl-btn--on");
    const ts = slider(c3, "t", 0, 120, 1, 0, (v) => String(v), (v) => ((t = v), draw()));
    const cvMap = D.makeCanvas(root, 120);
    const cv = D.makeCanvas(root, 300);
    const out = el("div", "fl-status");
    root.appendChild(out);
    function toggle() {
      if (timer) stop();
      else {
        if (t >= run.frames.length - 1) t = 0;
        timer = setInterval(() => {
          if (t >= run.frames.length - 1) return stop();
          t++;
          ts.set(t);
          draw();
        }, 140);
        play.textContent = "Pause";
      }
    }
    function stop() {
      clearInterval(timer);
      timer = null;
      play.textContent = "Play";
    }
    function draw() {
      const col = D.colors();
      const colors = [col.series[0], col.series[5], col.series[2], col.series[1]];
      // the map at time t
      const snap = run.frames[Math.min(t, run.frames.length - 1)];
      const ghost = { rail: run.env.rail, graph: run.env.graph, agents: snap.map((s, i) => Object.assign({}, run.env.agents[i], { state: s.state, cfg: s.cfg })) };
      drawToy(cvMap, ghost, { names: PL_AGENTS.map((a) => a.name), colors });
      const lines = [];
      for (const [h, p] of run.plan) lines.push({ pts: pathPts(p), color: colors[h], dash: [4, 4], width: 1.4, alpha: 0.7 });
      run.hist.forEach((pts, h) => {
        const a = run.env.agents[h];
        lines.push({ pts, color: colors[h], width: 2.6, label: PL_AGENTS[h].name, end: a.arrival !== null ? undefined : run.frames.length - 1 });
      });
      drawTD(cv, lines, { T: 90, cursor: t });
      const env = run.env;
      const delay = env.agents.reduce((s, a, i) => {
        const p = run.plan.find(([h]) => h === i);
        return s + (a.arrival !== null && p ? Math.max(0, a.arrival - p[1][p[1].length - 1][1]) : 0);
      }, 0);
      if (run.dlAt >= 0) {
        out.className = "fl-status bad";
        out.textContent = `Deadlock at t = ${run.dlAt}: two trains met head-on on single track. Without the order, a late train and an on-time train both believed they had the segment.`;
      } else {
        out.className = "fl-status good";
        out.textContent = `No deadlock. ${env.arrived()} of 4 arrived; together they arrive ${delay} steps later than planned (dashed lines: the plan; solid: what happened).`;
      }
    }
    simulate();
    draw();
    onTheme(() => (cv.fit(), cvMap.fit(), draw()));
    note(root, "The plan is fixed before the breakdown (dashed). Ordered execution lets a train into a cell only after every train planned through it earlier has left, so a late train delays the others but never meets them head-on. Going at the planned time instead ignores who is actually where.");
  }
  // "go at the planned time": the planner's paths, but a train only checks the clock, not the order
  class NaivePlanner extends FL.Planner {
    _willMoveInner(h, now) {
      const a = this.env.agents[h], p = this.paths.get(h), nxt = this.progress.get(h) + 1;
      if (nxt >= p.length || a.state === ST.DONE || a.malf > 0) return false;
      if (now + 1 < p[nxt][1]) return false;
      if (FL.offMap(a.state)) return (a.state === ST.READY_TO_DEPART || a.state === ST.MALFUNCTION_OFF_MAP) && now >= a.ed;
      return this.env.isCellExit(a, 1);
    }
  }

  // ------------------------------------------------------------------ 7. reward
  function wReward(root) {
    const S = { N: 30, T: 600, late: 25, sp: 120, left: 40, stuckLA: 560 };
    const c = controls(root);
    slider(c, "Trains N", 10, 100, 10, S.N, (v) => String(v), (v) => ((S.N = v), draw()));
    slider(c, "Horizon T", 200, 2000, 50, S.T, (v) => String(v), (v) => ((S.T = v), draw()));
    const c2 = controls(root);
    slider(c2, "Train 1 arrives late by", 0, 400, 5, S.late, (v) => v + " steps", (v) => ((S.late = v), draw()));
    slider(c2, "Train 2 never departs; its trip is", 20, 600, 10, S.sp, (v) => v + " steps", (v) => ((S.sp = v), draw()));
    slider(c2, "Train 3 is stuck this far from its target", 0, 200, 5, S.left, (v) => v + " steps", (v) => ((S.left = v), draw()));
    const out = el("div", "fl-readout");
    root.appendChild(out);
    function draw() {
      const r1 = -S.late, r2 = -S.sp, r3 = Math.min(0, S.stuckLA - S.T - S.left);
      const cap = (r) => Math.max(r, -S.T);
      const sum = cap(r1) + cap(r2) + cap(r3);
      const score = sum / (S.N * S.T) + 1;
      out.innerHTML = `<table><thead><tr><th>Train</th><th>What happened</th><th>Flatland reward</th><th>Capped at −T</th></tr></thead><tbody>
        <tr><td>1</td><td>arrived ${S.late} steps after its latest arrival</td><td>${r1}</td><td>${cap(r1)}</td></tr>
        <tr><td>2</td><td>never left (cancellation: minus its shortest-path travel time)</td><td>${r2}</td><td>${cap(r2)}</td></tr>
        <tr><td>3</td><td>on the map at the horizon; due by ${S.stuckLA}, ${S.left} steps of travel left</td><td>${r3}</td><td>${cap(r3)}</td></tr>
        <tr><td>the other ${S.N - 3}</td><td>arrived on time</td><td>0</td><td>0</td></tr></tbody></table>
        <p>Normalised reward = Σ max(r<sub>i</sub>, −T) / (N · T) + 1 = ${sum} / (${S.N} · ${S.T}) + 1 = <b>${score.toFixed(4)}</b></p>
        <p class="fl-note">One train cancelled on a 120-step trip costs 30 trains on a 600-step horizon ${(120 / (30 * 600)).toFixed(4)}: the score moves in the third decimal, which is why the results pages report arrival rates next to it.</p>`;
    }
    draw();
  }

  // ------------------------------------------------------------------ mount
  const WIDGETS = { cells: wCells, drive: wDrive, speeds: wSpeeds, deadlock: wDeadlock, sipp: wSipp, mcp: wMcp, reward: wReward };
  function mountAll() {
    document.querySelectorAll(".fl-widget[data-widget]:not([data-mounted])").forEach((root) => {
      const f = WIDGETS[root.dataset.widget];
      if (!f) return;
      root.dataset.mounted = "1";
      try {
        f(root);
      } catch (e) {
        root.appendChild(document.createTextNode("This widget could not start: " + e.message));
        console.error(e);
      }
    });
  }
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(mountAll);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mountAll);
  else mountAll();
})();
