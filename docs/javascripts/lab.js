/* The Scheduling Lab: flatland in the browser, on the 40 held-out small and medium maps of this repo.
 *
 *   recorded presets   replay the actions a policy took in flatland-rl, through the JavaScript port of the env
 *                      (flatland-core.js), which reproduces flatland step for step (scripts/check_lab.mjs)
 *   live presets       the OR planner runs here, on the same port; "you" dispatch on top of it with the TADA
 *                      clearances (tada-core.js). Breakdowns you add and clearances you issue re-run the
 *                      episode from the start, deterministically.
 *
 * Mounts on <div class="fl-widget" data-widget="lab">. State is mirrored in the URL:
 *   ?map=medium-1003-m&preset=or&t=120&train=4&brk=4@60x20&you=118:7:1&order=slack
 */
(function () {
  "use strict";
  const FL = window.FL, D = window.FLDraw;
  if (!FL || !D) return;
  const ST = FL.ST;
  const SCRIPT_URL = document.currentScript ? document.currentScript.src : location.href;
  const DATA = new URL("../assets/lab/", SCRIPT_URL);
  const jcache = {};
  const getJSON = (p) => (jcache[p] = jcache[p] || fetch(new URL(p, DATA)).then((r) => r.json()));

  // ------------------------------------------------------------------ presets
  const PRESETS = {
    or: { group: "plan", label: "OR planner", live: true, story: "Main's OR reference, running here: prioritized planning with SIPP gives every train a timed path, and ordered execution keeps every cell's planned visiting order. Breakdowns make trains late; the order keeps them out of deadlock." },
    you: { group: "you", label: "You dispatch", live: true, story: "The OR planner runs, and you are the dispatcher on top of it: pause, look at the window of trains that matter, and issue HOLD, YIELD_TO or REROUTE. Every clearance is planned against the live reservation table first; illegal ones are greyed out." },
    tada: { group: "learned", label: "TADA dispatcher", live: false, story: "The learned dispatcher from page 8, recorded in flatland: the same planner underneath, plus the clearances its policy chose (marked on the timeline)." },
    ppo: { group: "learned", label: "PPO (main)", live: false, story: "Main's best learned policy: PPO with a compact observation, one decision per train at switches, no planner. Watch for trains facing each other on a single track." },
    shortest_path: { group: "none", label: "Shortest path", live: false, story: "Every train departs at its earliest time and follows its shortest path. Nobody waits for anybody." },
    reactive_avoid: { group: "none", label: "Reactive rule", live: false, story: "A train enters the next segment only if no opposing train is on it: a local rule without a plan." },
  };
  const GROUPS = [
    ["plan", "Plan and repair", "a plan fixes every cell's visiting order; runs live here"],
    ["you", "You", "dispatch on top of the planner"],
    ["learned", "Learned", "recorded in flatland, replayed exactly"],
    ["none", "No coordination", "recorded in flatland, replayed exactly"],
  ];
  const KIND = ["PROCEED", "HOLD", "YIELD_TO", "REROUTE"];

  // ------------------------------------------------------------------ small DOM helpers
  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html !== undefined) e.innerHTML = html;
    return e;
  }
  function btn(parent, text, onClick, cls = "fl-btn") {
    const b = el("button", cls, text);
    b.type = "button";
    b.addEventListener("click", onClick);
    parent.appendChild(b);
    return b;
  }
  function panel(parent, title, extraCls) {
    const p = el("section", "fl-lab-panel" + (extraCls ? " " + extraCls : ""));
    const h = el("div", "fl-lab-panel__title", title);
    p.appendChild(h);
    parent.appendChild(p);
    return p;
  }
  const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));

  // ------------------------------------------------------------------ simulation
  /* one episode -> frames. Each frame: Int16Array(n * 5): state, row, col, heading, malfunction counter */
  function packFrame(env) {
    const f = new Int16Array(env.n * 5);
    env.agents.forEach((a, i) => {
      f[i * 5] = a.state;
      f[i * 5 + 1] = a.cfg ? a.cfg[0] : -1;
      f[i * 5 + 2] = a.cfg ? a.cfg[1] : -1;
      f[i * 5 + 3] = a.cfg ? a.cfg[2] : -1;
      f[i * 5 + 4] = a.malf;
    });
    return f;
  }

  function simulate(map, preset, opts) {
    const live = PRESETS[preset].live;
    const env = new FL.RailEnv(map, { extraMalfunctions: live ? opts.brk : [] });
    const run = { preset, map, T: env.T, n: env.n, frames: [packFrame(env)], dl: [[]], windows: [], clear: [], plan: null, tried: null, chosen: null };
    let ctrl = null, next = null;
    const t0 = performance.now();
    if (preset === "or") {
      ctrl = new FL.Planner(opts.order && opts.order !== "best" ? { orderings: FL.ORDERINGS[opts.order] ? [opts.order] : [], randomOrderings: opts.order.startsWith("random") ? [map.orderings[+opts.order.slice(6) - 1]] : [] } : {});
      ctrl.reset(env);
    } else if (preset === "you") {
      ctrl = new FL.TadaExecutor({ holdSteps: opts.hold });
      ctrl.reset(env);
    } else next = FL.decodeActions(map.episodes[preset].actions, env.n);
    if (ctrl) {
      run.plan = new Map([...ctrl.paths].map(([h, p]) => [h, p.map((x) => [x[0], x[1]])]));
      run.tried = ctrl.tried;
      run.chosen = ctrl.chosen;
    }
    const youAt = new Map();
    for (const c of opts.you || []) {
      if (!youAt.has(c[0])) youAt.set(c[0], []);
      youAt.get(c[0]).push(c);
    }
    const tadaWin = preset === "tada" ? map.episodes.tada.window : null;
    let wi = 0, wcur = null;
    while (!env.done) {
      const now = env.elapsed;
      let actions;
      if (preset === "you") {
        const w = FL.buildWindow(ctrl, { M: opts.M });
        run.windows[now] = { members: w.members, choosable: w.rows.filter((r) => r.choosable).map((r) => r.train), reasons: w.rows.map((r) => r.reasons.join("")) };
        for (const [, h, kind, partner] of youAt.get(now) || []) {
          let plan = null;
          try {
            plan = ctrl.planClearance({ train: h, kind, partner }, ctrl.liveTable(now), now, 3);
          } catch (e) {
            plan = null;
          }
          if (plan) ctrl.commit(plan, now);
          run.clear.push([now, h, kind, partner, plan ? 1 : 0]);
        }
        actions = ctrl.act(env);
      } else if (preset === "or") actions = ctrl.act(env);
      else {
        if (tadaWin) {
          while (wi < tadaWin.length && tadaWin[wi][0] <= now) wcur = tadaWin[wi++];
          if (wcur) run.windows[now] = { members: wcur[1], choosable: wcur[2], reasons: null };
        }
        actions = next(now + 1);
      }
      env.step(actions);
      if (ctrl) ctrl.observe(env);
      run.frames.push(packFrame(env));
      let dl;
      if (ctrl) {
        const r = FL.tadaDeadlock(env, ctrl);
        dl = r.deadlock ? [...r.flagged] : [];
      } else dl = [...FL.findDeadlocked(env)];
      run.dl.push(dl);
    }
    if (preset === "tada") run.clear = map.episodes.tada.clearances;
    run.env = env;
    run.ctrl = ctrl;
    run.events = env.events;
    run.ms = performance.now() - t0;
    const late = env.agents.filter((a) => a.state === ST.DONE && a.arrival > a.la).length;
    run.outcome = { arrived: env.arrived(), n: env.n, nr: env.normalizedReward(), deadlocked: run.dl[run.dl.length - 1].length, late, steps: env.elapsed };
    // arrivals per frame
    run.arr = run.frames.map((f) => {
      let k = 0;
      for (let i = 0; i < run.n; i++) if (f[i * 5] === ST.DONE) k++;
      return k;
    });
    return run;
  }

  /* the executor's state at time t of a "you" run (re-simulated; used to compute legal clearances) */
  function stateAt(map, opts, t) {
    const env = new FL.RailEnv(map, { extraMalfunctions: opts.brk });
    const ex = new FL.TadaExecutor({ holdSteps: opts.hold });
    ex.reset(env);
    const youAt = new Map();
    for (const c of opts.you || []) {
      if (!youAt.has(c[0])) youAt.set(c[0], []);
      youAt.get(c[0]).push(c);
    }
    while (!env.done && env.elapsed < t) {
      const now = env.elapsed;
      FL.buildWindow(ex, { M: opts.M });
      for (const [, h, kind, partner] of youAt.get(now) || []) {
        let plan = null;
        try {
          plan = ex.planClearance({ train: h, kind, partner }, ex.liveTable(now), now, 3);
        } catch (e) {}
        if (plan) ex.commit(plan, now);
      }
      env.step(ex.act(env));
      ex.observe(env);
    }
    if (env.done) return null;
    const w = FL.buildWindow(ex, { M: opts.M });
    const issued = (youAt.get(env.elapsed) || []).length;
    // apply clearances already issued at t, so the menus show what is still legal after them
    for (const [, h, kind, partner] of youAt.get(env.elapsed) || []) {
      const plan = ex.planClearance({ train: h, kind, partner }, ex.liveTable(env.elapsed), env.elapsed, 3);
      if (plan) ex.commit(plan, env.elapsed);
    }
    const w2 = issued ? FL.buildWindow(ex, { M: opts.M }) : w;
    const now = env.elapsed, rt = ex.liveTable(now);
    const legal = w2.rows.map((r) => {
      const out = { train: r.train, slack: r.slack, reasons: r.reasons, atDecision: r.atDecision, hold: false, reroute: false, yieldTo: [] };
      if (!r.choosable) return out;
      if (r.legal[FL.HOLD]) out.hold = !!ex.planClearance({ train: r.train, kind: FL.HOLD }, rt, now, 3);
      if (r.legal[FL.REROUTE]) out.reroute = !!ex.planClearance({ train: r.train, kind: FL.REROUTE }, rt, now, 3);
      if (r.legal[FL.YIELD_TO]) out.yieldTo = r.partnersAfter.filter((p) => !!ex.planClearance({ train: r.train, kind: FL.YIELD_TO, partner: p }, rt, now, 3));
      return out;
    });
    return { now, legal, issued };
  }

  // ------------------------------------------------------------------ the widget
  function wLab(root) {
    root.classList.add("fl-lab");
    const S = { mapId: null, map: null, preset: "or", t: 0, playing: false, speed: 10, sel: null, brk: [], you: [], order: "best", hold: 5, M: 8, run: null, compare: null, scen: "medium", seed: 1003, malf: true };
    // URL state
    const q = new URLSearchParams(location.search);
    if (q.get("map")) {
      const m = q.get("map").match(/^(small|medium)-(\d+)-([nm])$/);
      if (m) Object.assign(S, { scen: m[1], seed: +m[2], malf: m[3] === "m" });
    }
    if (q.get("preset") && PRESETS[q.get("preset")]) S.preset = q.get("preset");
    if (q.get("order")) S.order = q.get("order");
    if (q.get("hold")) S.hold = +q.get("hold");
    if (q.get("brk"))
      S.brk = q
        .get("brk")
        .split(",")
        .map((s) => s.match(/^(\d+)@(\d+)x(\d+)$/))
        .filter(Boolean)
        .map((m) => [+m[2], +m[1], +m[3]]);
    if (q.get("you"))
      S.you = q
        .get("you")
        .split(",")
        .map((s) => s.split(":").map(Number))
        .filter((x) => x.length >= 3)
        .map((x) => [x[0], x[1], x[2], x.length > 3 ? x[3] : null]);
    const startT = q.get("t") ? +q.get("t") : null;
    const startSel = q.get("train") !== null && q.get("train") !== "" ? +q.get("train") : null;
    const autoplay = q.get("play") === "1";

    // ---- layout
    const top = el("div", "fl-lab-top");
    root.appendChild(top);
    const mapRow = el("div", "fl-lab-maprow");
    top.appendChild(mapRow);
    const presetRow = el("div", "fl-lab-presets");
    top.appendChild(presetRow);
    const story = el("div", "fl-lab-story");
    top.appendChild(story);
    const grid = el("div", "fl-lab-grid");
    root.appendChild(grid);
    const side = el("aside", "fl-lab-side");
    grid.appendChild(side);
    const main = el("div", "fl-lab-main");
    grid.appendChild(main);

    // map picker
    mapRow.appendChild(el("span", "fl-lab-label", "Map"));
    const scenBtns = {};
    for (const sc of ["small", "medium"])
      scenBtns[sc] = btn(mapRow, sc === "small" ? "small · 30×30 · 10 trains" : "medium · 50×50 · 30 trains", () => {
        S.scen = sc;
        loadMap();
      }, "fl-chip");
    const seedSel = el("select", "fl-lab-select");
    for (let s = 1000; s < 1010; s++) seedSel.appendChild(new Option("seed " + s, s));
    seedSel.addEventListener("change", () => {
      S.seed = +seedSel.value;
      loadMap();
    });
    mapRow.appendChild(seedSel);
    const malfBtn = btn(mapRow, "malfunctions", () => {
      S.malf = !S.malf;
      loadMap();
    }, "fl-chip");

    // presets
    const presetBtns = {};
    for (const [g, name, sub] of GROUPS) {
      const box = el("div", "fl-lab-group fl-lab-group--" + g);
      box.appendChild(el("div", "fl-lab-group__head", `<b>${name}</b> <span>${sub}</span>`));
      const row = el("div", "fl-lab-group__row");
      box.appendChild(row);
      for (const [id, p] of Object.entries(PRESETS))
        if (p.group === g)
          presetBtns[id] = btn(row, p.label, () => {
            S.preset = id;
            resim(true);
          }, "fl-chip");
      presetRow.appendChild(box);
    }

    // ---- side pane
    const sPlay = el("div", "fl-lab-sec");
    side.appendChild(sPlay);
    sPlay.appendChild(el("div", "fl-lab-sec__title", "Playback"));
    const playRow = el("div", "fl-lab-row");
    sPlay.appendChild(playRow);
    const playBtn = btn(playRow, "Play", () => setPlaying(!S.playing), "fl-btn fl-btn--on");
    btn(playRow, "⏮", () => setT(0));
    btn(playRow, "−1", () => setT(S.t - 1));
    btn(playRow, "+1", () => setT(S.t + 1));
    const speedSel = el("select", "fl-lab-select");
    for (const v of [2, 5, 10, 20, 40]) speedSel.appendChild(new Option(v + " steps/s", v));
    speedSel.value = String(S.speed);
    speedSel.addEventListener("change", () => (S.speed = +speedSel.value));
    playRow.appendChild(speedSel);
    const tRow = el("label", "fl-lab-trow");
    sPlay.appendChild(tRow);
    const tSlider = el("input");
    tSlider.type = "range";
    tSlider.min = 0;
    tSlider.step = 1;
    const tOut = el("output", "fl-lab-tout", "t = 0");
    tRow.append(tSlider, tOut);
    tSlider.addEventListener("input", () => setT(+tSlider.value));
    const strip = D.makeCanvas(sPlay, 46, "fl-lab-strip");
    strip.title = "Events: breakdowns (red), deadlocks (dark red), clearances (amber), arrivals (green). Click to jump.";
    strip.addEventListener("click", (e) => {
      const r = strip.getBoundingClientRect();
      if (S.run) setT(Math.round(((e.clientX - r.left) / r.width) * (S.run.frames.length - 1)));
    });

    const sCtl = el("div", "fl-lab-sec");
    side.appendChild(sCtl);
    sCtl.appendChild(el("div", "fl-lab-sec__title", "Controller"));
    const ctlBody = el("div");
    sCtl.appendChild(ctlBody);

    const sDist = el("div", "fl-lab-sec");
    side.appendChild(sDist);
    sDist.appendChild(el("div", "fl-lab-sec__title", "Disturbance"));
    const distBody = el("div");
    sDist.appendChild(distBody);

    const sOut = el("div", "fl-lab-sec");
    side.appendChild(sOut);
    sOut.appendChild(el("div", "fl-lab-sec__title", "Outcome"));
    const outBody = el("div", "fl-lab-out");
    sOut.appendChild(outBody);

    // ---- main panels
    const pMap = panel(main, "The network, live");
    const mapHead = el("div", "fl-lab-legend");
    pMap.appendChild(mapHead);
    const mapCv = D.makeCanvas(pMap, (w) => Math.min(620, Math.round(w * 0.78)), "fl-lab-map");
    const mapInfo = el("div", "fl-lab-info");
    pMap.appendChild(mapInfo);
    const pDisp = panel(main, "The dispatcher's window", "fl-lab-panel--disp");
    const dispBody = el("div");
    pDisp.appendChild(dispBody);
    const pGantt = panel(main, "Every train through the episode");
    pGantt.appendChild(el("div", "fl-note", "One row per train. Grey: not departed yet (darker once its earliest departure has passed); amber: moving; slate: stopped; red: broken down; dark red: deadlocked; a green tick at arrival. Ticks above each row mark the earliest departure and latest arrival. Click a row to select the train."));
    const ganttCv = D.makeCanvas(pGantt, 260, "fl-lab-gantt");
    const pLine = panel(main, "Time–space diagram along the selected train's route");
    const lineNote = el("div", "fl-note");
    pLine.appendChild(lineNote);
    const lineCv = D.makeCanvas(pLine, 300, "fl-lab-line");
    const pCmp = panel(main, "Trains arrived, every controller on this map");
    const cmpCv = D.makeCanvas(pCmp, 230, "fl-lab-cmp");
    const cmpNote = el("div", "fl-note");
    pCmp.appendChild(cmpNote);
    const pPlan = panel(main, "What the planner tried", "fl-lab-panel--plan");
    const planBody = el("div");
    pPlan.appendChild(planBody);

    // ---- state helpers
    const mapId = () => `${S.scen}-${S.seed}-${S.malf ? "m" : "n"}`;
    function syncURL() {
      const p = new URLSearchParams(location.search);
      p.set("map", mapId());
      p.set("preset", S.preset);
      p.set("t", S.t);
      if (S.sel !== null) p.set("train", S.sel);
      else p.delete("train");
      if (S.brk.length && PRESETS[S.preset].live) p.set("brk", S.brk.map(([t, h, d]) => `${h}@${t}x${d}`).join(","));
      else p.delete("brk");
      if (S.you.length && S.preset === "you") p.set("you", S.you.map((c) => c.filter((x) => x !== null).join(":")).join(","));
      else p.delete("you");
      if (S.order !== "best" && S.preset === "or") p.set("order", S.order);
      else p.delete("order");
      p.delete("play");
      history.replaceState(null, "", location.pathname + "?" + p.toString() + location.hash);
    }
    function setPlaying(v) {
      S.playing = v && !!S.run;
      playBtn.textContent = S.playing ? "Pause" : "Play";
      if (S.playing) {
        if (S.t >= S.run.frames.length - 1) setT(0);
        last = performance.now();
        acc = 0;
        requestAnimationFrame(tick);
      } else {
        renderDispatcher();
        syncURL();
      }
    }
    let last = 0, acc = 0;
    function tick(now) {
      if (!S.playing) return;
      acc += ((now - last) / 1000) * S.speed;
      last = now;
      if (acc >= 1) {
        const k = Math.floor(acc);
        acc -= k;
        if (S.t + k >= S.run.frames.length - 1) {
          setT(S.run.frames.length - 1, true);
          setPlaying(false);
          return;
        }
        setT(S.t + k, true);
      }
      requestAnimationFrame(tick);
    }
    function setT(t, fromPlay) {
      if (!S.run) return;
      S.t = Math.max(0, Math.min(S.run.frames.length - 1, t));
      tSlider.value = S.t;
      tOut.textContent = `t = ${S.t} / ${S.run.T}`;
      if (S._breakBtn && S.sel !== null) S._breakBtn.textContent = `Break train ${S.sel} at t ${S.t + 1}`;
      drawMap();
      drawGanttCursor();
      drawLine();
      drawStrip();
      drawCompare();
      if (!fromPlay) {
        renderDispatcher();
        syncURL();
      } else renderDispatcherLight();
    }

    // ---- loading
    async function loadMap() {
      const id = mapId();
      for (const [sc, b] of Object.entries(scenBtns)) b.setAttribute("aria-pressed", String(sc === S.scen));
      malfBtn.setAttribute("aria-pressed", String(S.malf));
      malfBtn.textContent = S.malf ? "malfunctions on" : "malfunctions off";
      seedSel.value = String(S.seed);
      story.innerHTML = '<span class="fl-note">Loading the map…</span>';
      const map = await getJSON(`maps/${id}.json`);
      S.mapId = id;
      S.map = map;
      if (S.sel !== null && S.sel >= map.agents.length) S.sel = null;
      S.brk = S.brk.filter(([, h]) => h < map.agents.length);
      S.you = S.you.filter(([, h]) => h < map.agents.length);
      S.compare = null;
      resim(false);
      computeCompare();
    }
    function resim(keepT) {
      if (!S.map) return;
      for (const [id, b] of Object.entries(presetBtns)) b.setAttribute("aria-pressed", String(id === S.preset));
      const tPrev = S.t;
      S.run = simulate(S.map, S.preset, { brk: S.brk, you: S.you, order: S.order, hold: S.hold, M: S.M });
      tSlider.max = S.run.frames.length - 1;
      renderControls();
      renderStory();
      renderOutcome();
      renderPlan();
      drawGantt();
      if (startT !== null && !S._started) {
        S._started = true;
        S.t = startT;
        if (startSel !== null) S.sel = startSel;
      } else if (!keepT) S.t = 0;
      else S.t = Math.min(tPrev, S.run.frames.length - 1);
      setT(S.t);
      if (autoplay && !S._autoplayed) {
        S._autoplayed = true;
        setPlaying(true);
      }
    }
    function computeCompare() {
      // every controller on this map (recorded ones replayed, the planner live, without your changes)
      const out = {};
      for (const id of Object.keys(PRESETS)) {
        if (id === "you") continue;
        out[id] = simulate(S.map, id, { brk: [], you: [], order: "best", hold: 5, M: 8 }).arr;
      }
      S.compare = out;
      drawCompare();
    }

    // ---- controls per preset
    function renderControls() {
      ctlBody.innerHTML = "";
      distBody.innerHTML = "";
      const p = PRESETS[S.preset];
      const badge = el("div", "fl-lab-badge fl-lab-badge--" + p.group, p.live ? "live on this page" : "recorded in flatland-rl");
      ctlBody.appendChild(badge);
      if (S.preset === "or") {
        const row = el("label", "fl-lab-field", "<span>Priority order</span>");
        const sel = el("select", "fl-lab-select");
        for (const [v, l] of [
          ["best", "best of 8 (main's planner)"],
          ["speed", "fast trains first"],
          ["slack", "least slack first"],
          ["departure", "earliest departure first"],
          ["short", "shortest trip first"],
          ["random1", "random order 1"],
          ["random2", "random order 2"],
          ["random3", "random order 3"],
          ["random4", "random order 4"],
        ])
          sel.appendChild(new Option(l, v));
        sel.value = S.order;
        sel.addEventListener("change", () => {
          S.order = sel.value;
          resim(true);
        });
        row.appendChild(sel);
        ctlBody.appendChild(row);
        ctlBody.appendChild(el("p", "fl-note", "Prioritized planning plans one train at a time, earliest-arrival path around the cell-times already reserved. A bad order can leave trains unroutable or late; main's planner tries eight orders and keeps the best."));
      } else if (S.preset === "you") {
        const row = el("label", "fl-lab-field", "<span>HOLD length</span>");
        const sel = el("select", "fl-lab-select");
        for (const v of [3, 5, 10, 20]) sel.appendChild(new Option(v + " steps", v));
        sel.value = String(S.hold);
        sel.addEventListener("change", () => {
          S.hold = +sel.value;
          resim(true);
        });
        row.appendChild(sel);
        ctlBody.appendChild(row);
        ctlBody.appendChild(el("p", "fl-note", "Pause at any step and use the window panel. Your clearances are kept in the URL; at most two per step, as in the brief."));
        if (S.you.length) {
          const lst = el("div", "fl-lab-list");
          S.you.forEach((c, i) => {
            const it = el("div", "fl-lab-list__item", `t ${c[0]}: ${KIND[c[2]]} train ${c[1]}${c[2] === FL.YIELD_TO ? " → " + c[3] : ""}`);
            btn(it, "×", () => {
              S.you.splice(i, 1);
              resim(true);
            }, "fl-btn fl-lab-x");
            lst.appendChild(it);
          });
          ctlBody.appendChild(lst);
        }
      } else {
        ctlBody.appendChild(el("p", "fl-note", "These are the actions the policy took in flatland-rl, replayed exactly. They are open loop here: adding a breakdown would not change them, so breakdowns are only for the live presets."));
      }
      // disturbance
      if (p.live) {
        const sel = S.sel;
        const row = el("div", "fl-lab-row");
        const dur = el("select", "fl-lab-select");
        for (const v of [5, 10, 20, 30, 50]) dur.appendChild(new Option(v + " steps", v));
        dur.value = "20";
        row.appendChild(dur);
        const b = btn(row, sel === null ? "Select a train to break it" : `Break train ${sel} at t ${S.t}`, () => {
          if (S.sel === null) return;
          S.brk.push([Math.max(1, S.t + 1), S.sel, +dur.value]);
          resim(true);
        });
        b.disabled = sel === null;
        b.classList.add("fl-lab-break");
        S._breakBtn = b;
        distBody.appendChild(row);
        distBody.appendChild(el("p", "fl-note", `Recorded breakdowns of this map: ${S.map.malf.length ? S.map.malf.map(([t, h, d]) => `train ${h} at t ${t} for ${d}`).slice(0, 6).join("; ") + (S.map.malf.length > 6 ? "; …" : "") : "none"}.`));
        if (S.brk.length) {
          const lst = el("div", "fl-lab-list");
          S.brk.forEach(([t, h, d], i) => {
            const it = el("div", "fl-lab-list__item", `train ${h} breaks at t ${t} for ${d}`);
            btn(it, "×", () => {
              S.brk.splice(i, 1);
              resim(true);
            }, "fl-btn fl-lab-x");
            lst.appendChild(it);
          });
          distBody.appendChild(lst);
        }
      } else distBody.appendChild(el("p", "fl-note", "Recorded runs keep the breakdowns flatland drew for this map: " + (S.map.malf.length ? S.map.malf.length + " this episode." : "none on this map.")));
    }

    // ---- story
    function renderStory() {
      const r = S.run, p = PRESETS[S.preset];
      const items = [];
      const firstDl = r.dl.findIndex((x) => x.length);
      for (const e of r.events.filter((e) => e.type === "malfunction").slice(0, 4)) items.push([e.t, `train ${e.train} breaks down for ${e.steps} steps`]);
      if (firstDl >= 0) items.push([firstDl, `deadlock: trains ${r.dl[firstDl].join(", ")} can never move again`]);
      const arr = r.events.filter((e) => e.type === "arrival");
      if (arr.length) {
        const lateOnes = arr.filter((e) => e.late > 0);
        if (lateOnes.length) items.push([lateOnes[0].t, `first late arrival: train ${lateOnes[0].train}, ${lateOnes[0].late} steps after its latest arrival`]);
        items.push([arr[arr.length - 1].t, `last arrival (train ${arr[arr.length - 1].train})`]);
      }
      if (r.clear.length) items.push([r.clear[0][0], `first clearance: ${KIND[r.clear[0][2]]} train ${r.clear[0][1]}${r.clear[0][3] !== null && r.clear[0][3] !== undefined ? " → " + r.clear[0][3] : ""} (${r.clear.length} in total)`]);
      items.sort((a, b) => a[0] - b[0]);
      story.innerHTML = "";
      story.appendChild(el("p", null, p.story));
      const ul = el("div", "fl-lab-events");
      for (const [t, txt] of items) {
        const it = el("div", "fl-lab-event");
        const b = btn(it, `t = ${t}`, () => setT(t), "fl-chip fl-lab-tchip");
        b.title = "Jump to this step";
        it.appendChild(el("span", null, txt));
        ul.appendChild(it);
      }
      const o = r.outcome;
      const end = el("div", "fl-lab-event");
      end.appendChild(btn(end, `t = ${o.steps}`, () => setT(o.steps), "fl-chip fl-lab-tchip"));
      end.appendChild(el("span", null, `end: ${o.arrived} of ${o.n} trains arrived, normalised reward ${fmt(o.nr)}${o.deadlocked ? `, ${o.deadlocked} deadlocked` : ""}`));
      ul.appendChild(end);
      story.appendChild(ul);
    }

    // ---- outcome
    function renderOutcome() {
      const o = S.run.outcome, rec = S.map.episodes[S.preset] ? S.map.episodes[S.preset].summary : S.map.episodes.or.summary;
      outBody.innerHTML = "";
      const tiles = [
        ["arrived", `${o.arrived}/${o.n}`],
        ["norm. reward", fmt(o.nr)],
        ["deadlocked", String(o.deadlocked)],
        ["late", String(o.late)],
      ];
      const tg = el("div", "fl-lab-tiles");
      for (const [k, v] of tiles) tg.appendChild(el("div", "fl-lab-tile", `<b>${v}</b><span>${k}</span>`));
      outBody.appendChild(tg);
      const unchanged = !(PRESETS[S.preset].live && (S.brk.length || (S.preset === "or" && S.order !== "best") || (S.preset === "you" && S.you.length)));
      if (S.preset !== "you" && unchanged) {
        const same = rec.arrived === o.arrived && Math.abs(rec.normalized_reward - o.nr) < 1e-9;
        outBody.appendChild(el("p", "fl-note", `flatland-rl recorded ${rec.arrived}/${rec.n_agents} and ${fmt(rec.normalized_reward)} for this ${PRESETS[S.preset].live ? "planner" : "policy"} on this map${same ? ": identical." : "."}`));
      } else if (S.preset === "you" || !unchanged) {
        const base = S.map.episodes.or.summary;
        outBody.appendChild(el("p", "fl-note", `The planner alone, unchanged: ${base.arrived}/${base.n_agents}, ${fmt(base.normalized_reward)}.`));
      }
      outBody.appendChild(el("p", "fl-note", `Simulated in ${Math.round(S.run.ms)} ms.`));
    }

    // ---- planner panel
    function renderPlan() {
      planBody.innerHTML = "";
      const r = S.run;
      pPlan.style.display = r.tried ? "" : "none";
      if (!r.tried) return;
      const tb = el("table", "fl-lab-table");
      tb.innerHTML = "<thead><tr><th>Priority order</th><th>Trains routed</th><th>Total lateness</th><th>Sum of arrival times</th></tr></thead>";
      const body = el("tbody");
      for (const x of r.tried) {
        const tr = el("tr", x.name === r.chosen ? "fl-lab-chosen" : "");
        tr.innerHTML = `<td>${x.name}${x.name === r.chosen ? " ✓" : ""}</td><td>${x.routed}/${r.n}</td><td>${x.late}</td><td>${x.arrival}</td>`;
        body.appendChild(tr);
      }
      tb.appendChild(body);
      planBody.appendChild(tb);
      planBody.appendChild(el("p", "fl-note", "Orders are tried in this sequence until one routes every train with no lateness; the best by (routed, −lateness, −arrival) is kept. The four random orders are the same permutations main's planner draws (numpy seed 0)."));
    }

    // ---- dispatcher window panel
    function renderDispatcherLight() {
      const r = S.run;
      if (!(S.preset === "you" || S.preset === "tada")) return;
      const w = r.windows[S.t];
      const head = dispBody.querySelector(".fl-lab-disp-now");
      if (head) head.textContent = w ? `t = ${S.t}: ${w.members.length} trains in the window, ${w.choosable.length} at a decision instant` : `t = ${S.t}`;
    }
    function renderDispatcher() {
      const r = S.run;
      dispBody.innerHTML = "";
      pDisp.style.display = S.preset === "you" || S.preset === "tada" ? "" : "none";
      if (pDisp.style.display === "none") return;
      const w = r.windows[S.t];
      dispBody.appendChild(el("div", "fl-lab-disp-now", w ? `t = ${S.t}: ${w.members.length} trains in the window, ${w.choosable.length} at a decision instant` : `t = ${S.t}`));
      const issuedHere = r.clear.filter((c) => c[0] === S.t);
      if (issuedHere.length)
        dispBody.appendChild(el("p", "fl-note", "Issued at this step: " + issuedHere.map((c) => `${KIND[c[2]]} train ${c[1]}${c[2] === FL.YIELD_TO ? " → " + c[3] : ""}${c[4] ? "" : " (rejected)"}`).join("; ")));
      if (S.preset === "tada") {
        if (w) {
          const tb = el("table", "fl-lab-table");
          tb.innerHTML = "<thead><tr><th>Rank</th><th>Train</th><th>At a decision instant</th></tr></thead>";
          const body = el("tbody");
          w.members.forEach((h, i) => {
            const tr = el("tr", h === S.sel ? "fl-lab-chosen" : "");
            tr.innerHTML = `<td>${i + 1}</td><td>${h}</td><td>${w.choosable.includes(h) ? "yes" : ""}</td>`;
            tr.addEventListener("click", () => select(h));
            body.appendChild(tr);
          });
          tb.appendChild(body);
          dispBody.appendChild(tb);
        }
        dispBody.appendChild(el("p", "fl-note", `The window as the learned dispatcher saw it in flatland, least slack first. It issued ${r.clear.length} clearances in this episode (amber marks on the event strip).`));
        return;
      }
      // you: legal clearances at this step
      if (S.playing) return;
      const st = stateAt(S.map, { brk: S.brk, you: S.you, hold: S.hold, M: S.M }, S.t);
      if (!st) {
        dispBody.appendChild(el("p", "fl-note", "The episode is over."));
        return;
      }
      const left = 2 - st.issued;
      const tb = el("table", "fl-lab-table fl-lab-disp");
      tb.innerHTML = "<thead><tr><th>Train</th><th>Slack</th><th>Why in the window</th><th>Clearances you can issue now</th></tr></thead>";
      const body = el("tbody");
      const why = { a: "conflict ahead", b: "late on plan", c: "contested track ahead", d: "departure blocked" };
      for (const row of st.legal) {
        const tr = el("tr", row.train === S.sel ? "fl-lab-chosen" : "");
        tr.appendChild(el("td", null, `<button type="button" class="fl-lab-link">train ${row.train}</button>`));
        tr.firstChild.firstChild.addEventListener("click", () => select(row.train));
        tr.appendChild(el("td", null, String(Math.round(row.slack))));
        tr.appendChild(el("td", null, row.reasons.map((x) => why[x]).join(", ")));
        const td = el("td", "fl-lab-actions");
        if (!row.atDecision) td.appendChild(el("span", "fl-note", "not at a decision instant"));
        else if (left <= 0) td.appendChild(el("span", "fl-note", "budget used at this step"));
        else {
          const add = (label, ok, kind, partner = null) => {
            const b = btn(td, label, () => {
              S.you.push([st.now, row.train, kind, partner]);
              S.you.sort((a, b) => a[0] - b[0]);
              resim(true);
            }, "fl-btn");
            b.disabled = !ok;
            if (!ok) b.title = "No safe plan exists for this clearance now";
          };
          add("HOLD", row.hold, FL.HOLD);
          add("REROUTE", row.reroute, FL.REROUTE);
          for (const p of row.yieldTo) add("YIELD_TO " + p, true, FL.YIELD_TO, p);
        }
        tr.appendChild(td);
        body.appendChild(tr);
      }
      tb.appendChild(body);
      dispBody.appendChild(tb);
      dispBody.appendChild(el("p", "fl-note", "HOLD keeps the train where it is for a few more steps and re-plans it; YIELD_TO lets the partner go first through the cells they share; REROUTE forbids the planned branch at the next facing switch. Each is a speculative SIPP search against the live reservation table, so only safe clearances are offered."));
    }

    // ---- selection
    function select(h) {
      S.sel = S.sel === h ? null : h;
      renderControls();
      drawMap();
      drawGantt();
      drawLine();
      renderDispatcher();
      syncURL();
    }

    // ---- drawing: map
    let view = null;
    function frameAgent(f, i) {
      return { state: f[i * 5], cfg: f[i * 5 + 1] < 0 ? null : [f[i * 5 + 1], f[i * 5 + 2], f[i * 5 + 3]], malf: f[i * 5 + 4] };
    }
    function drawMap() {
      const r = S.run, col = D.colors(), ctx = mapCv.getContext("2d");
      const w = mapCv._w, h = mapCv._h;
      ctx.clearRect(0, 0, w, h);
      const env = r.env;
      view = D.fitView(env.rail, w, h, 8);
      D.drawRail(ctx, env.rail, env.graph, view, col);
      const f = r.frames[S.t], dl = new Set(r.dl[S.t]);
      const win = r.windows[S.t];
      const winSet = new Set(win ? win.members : []);
      // targets of trains still travelling (selected one highlighted)
      for (let i = 0; i < r.n; i++) {
        const a = frameAgent(f, i);
        if (a.state === ST.DONE) continue;
        const tg = S.map.agents[i].targets[0];
        if (S.sel === i || S.sel === null) D.drawStation(ctx, view, tg[0], tg[1], S.sel === i ? col.accent : col.muted, S.sel === i ? "target " + i : undefined);
      }
      // the selected train's route ahead
      if (S.sel !== null) {
        const a = frameAgent(f, S.sel);
        let ahead = [];
        if (r.ctrl && r.ctrl.paths.has(S.sel)) ahead = r.ctrl.paths.get(S.sel).map((x) => x[0]);
        else {
          for (let t = S.t; t < r.frames.length; t++) {
            const b = frameAgent(r.frames[t], S.sel);
            if (b.cfg && (!ahead.length || ahead[ahead.length - 1][0] !== b.cfg[0] || ahead[ahead.length - 1][1] !== b.cfg[1])) ahead.push(b.cfg);
          }
        }
        if (ahead.length) {
          ctx.save();
          ctx.strokeStyle = col.accent;
          ctx.globalAlpha = 0.55;
          ctx.setLineDash([3, 3]);
          ctx.lineWidth = Math.max(1.5, view.s * 0.18);
          ctx.beginPath();
          ahead.forEach((c, k) => {
            const x = view.x0 + (c[1] + 0.5) * view.s, y = view.y0 + (c[0] + 0.5) * view.s;
            k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
          });
          ctx.stroke();
          ctx.restore();
        }
        if (!a.cfg && a.state !== ST.DONE) {
          const st = S.map.agents[S.sel].start;
          D.drawStation(ctx, view, st[0], st[1], col.accent, "start " + S.sel);
        }
      }
      // trains
      for (let i = 0; i < r.n; i++) {
        const a = frameAgent(f, i);
        if (!a.cfg) continue;
        D.drawTrain(ctx, view, a.cfg, D.stateColor(a.state, col), {
          ring: S.sel === i ? col.fg : winSet.has(i) ? col.accent : null,
          label: i,
          labelColor: col.fg,
          cross: dl.has(i) ? col.bad : null,
          outline: col.dark ? "rgba(0,0,0,0.6)" : "rgba(0,0,0,0.45)",
        });
      }
      // info
      let onm = 0, br = 0, arr = 0, wait = 0;
      for (let i = 0; i < r.n; i++) {
        const s = f[i * 5];
        if (FL.onMap(s)) onm++;
        if (s === ST.MALFUNCTION || s === ST.MALFUNCTION_OFF_MAP) br++;
        if (s === ST.DONE) arr++;
        if (FL.offMap(s)) wait++;
      }
      let sel = "";
      if (S.sel !== null) {
        const a = frameAgent(f, S.sel), m = S.map.agents[S.sel];
        sel = `<span><b>train ${S.sel}</b> ${FL.STATE_NAMES[a.state]}${a.malf ? ` (${a.malf} steps left)` : ""} · speed 1/${m.k} · departs ≥ ${m.ed} · due by ${m.la}${a.state === ST.DONE ? "" : ""}</span>`;
      }
      mapInfo.innerHTML = `<span><b>t = ${S.t}</b></span><span>${onm} on the map</span><span>${wait} not departed</span><span>${arr} arrived</span><span class="${br ? "fl-lab-bad" : ""}">${br} broken down</span><span class="${dl.size ? "fl-lab-bad" : ""}">${dl.size} deadlocked</span>${sel}`;
      mapHead.innerHTML = `<span class="fl-lab-key"><i style="background:${col.accent}"></i>moving</span><span class="fl-lab-key"><i style="background:${col.stopped}"></i>stopped</span><span class="fl-lab-key"><i style="background:${col.bad}"></i>broken down</span><span class="fl-lab-key"><i class="fl-lab-key--x"></i>deadlocked</span><span class="fl-lab-key"><i class="fl-lab-key--d" style="background:${col.sw}"></i>switch</span>${win ? '<span class="fl-lab-key"><i class="fl-lab-key--o"></i>in the window</span>' : ""}<span class="fl-note">Click a train to select it.</span>`;
    }
    mapCv.addEventListener("click", (e) => {
      if (!S.run || !view) return;
      const rect = mapCv.getBoundingClientRect();
      const x = e.clientX - rect.left, y = e.clientY - rect.top;
      const f = S.run.frames[S.t];
      let best = null, bd = Infinity;
      for (let i = 0; i < S.run.n; i++) {
        const a = frameAgent(f, i);
        if (!a.cfg) continue;
        const cx = view.x0 + (a.cfg[1] + 0.5) * view.s, cy = view.y0 + (a.cfg[0] + 0.5) * view.s;
        const d = Math.hypot(cx - x, cy - y);
        if (d < bd) (bd = d), (best = i);
      }
      if (best !== null && bd < Math.max(14, view.s * 1.2)) select(best);
    });

    // ---- drawing: gantt (static layer + cursor)
    let ganttImg = null;
    function ganttGeom() {
      const r = S.run;
      const w = ganttCv._w;
      const left = 30, right = 8, top = 6;
      const rowH = Math.max(8, Math.min(14, Math.floor(240 / r.n)));
      return { left, right, top, rowH, w, x: (t) => left + (t / r.T) * (w - left - right) };
    }
    function drawGantt() {
      const r = S.run, col = D.colors();
      const g0 = ganttGeom();
      ganttCv.fit();
      const ctx = ganttCv.getContext("2d");
      const H = g0.top + r.n * g0.rowH + 22;
      if (Math.abs(ganttCv._h - H) > 1) {
        ganttCv.style.height = H + "px";
        const dpr = window.devicePixelRatio || 1;
        ganttCv.height = Math.round(H * dpr);
        ganttCv._h = H;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }
      const g = ganttGeom();
      ctx.clearRect(0, 0, g.w, ganttCv._h);
      ctx.font = D.FONT_SMALL;
      for (let i = 0; i < r.n; i++) {
        const y = g.top + i * g.rowH;
        if (S.sel === i) {
          ctx.fillStyle = col.dark ? "rgba(255,201,60,0.12)" : "rgba(199,119,0,0.1)";
          ctx.fillRect(0, y - 1, g.w, g.rowH);
        }
        ctx.fillStyle = col.muted;
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        if (g.rowH >= 8) ctx.fillText(String(i), g.left - 4, y + g.rowH / 2);
        const ed = S.map.agents[i].ed, la = S.map.agents[i].la;
        let prev = null, start = 0;
        const flush = (t1) => {
          if (prev === null) return;
          let c;
          if (prev === "dl") c = col.dark ? "#7a1f1f" : "#8b1a1a";
          else if (prev === ST.WAITING) c = col.waiting;
          else if (prev === ST.READY_TO_DEPART) c = col.dark ? "#7b7f88" : "#a29d92";
          else if (prev === ST.DONE) c = null;
          else c = D.stateColor(prev, col);
          if (c) {
            ctx.fillStyle = c;
            ctx.fillRect(g.x(start), y + 1, Math.max(1, g.x(t1) - g.x(start)), g.rowH - 2);
          }
        };
        for (let t = 0; t < r.frames.length; t++) {
          let s = r.frames[t][i * 5];
          if (r.dl[t].includes(i)) s = "dl";
          if (s !== prev) {
            flush(t);
            prev = s;
            start = t;
          }
        }
        flush(r.frames.length);
        const ag = r.env.agents[i];
        if (ag.arrival !== null) {
          ctx.strokeStyle = col.ok;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(g.x(ag.arrival), y);
          ctx.lineTo(g.x(ag.arrival), y + g.rowH);
          ctx.stroke();
        }
        ctx.fillStyle = col.fg;
        ctx.fillRect(g.x(ed) - 0.5, y, 1, 3);
        ctx.fillStyle = col.bad;
        ctx.fillRect(g.x(la) - 0.5, y, 1.5, 3);
      }
      // clearances
      ctx.fillStyle = col.accent;
      for (const c of r.clear) {
        const y = g.top + c[1] * g.rowH;
        ctx.beginPath();
        ctx.arc(g.x(c[0]), y + g.rowH / 2, Math.max(2, g.rowH * 0.3), 0, 2 * Math.PI);
        ctx.fill();
      }
      // axis
      const yb = g.top + r.n * g.rowH + 4;
      ctx.fillStyle = col.muted;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const step = r.T > 600 ? 100 : 50;
      for (let t = 0; t <= r.T; t += step) {
        ctx.fillRect(g.x(t), yb, 1, 4);
        ctx.fillText(String(t), g.x(t), yb + 5);
      }
      ganttImg = ctx.getImageData(0, 0, ganttCv.width, ganttCv.height);
      drawGanttCursor();
    }
    function drawGanttCursor() {
      if (!ganttImg) return;
      const ctx = ganttCv.getContext("2d"), g = ganttGeom(), col = D.colors();
      ctx.putImageData(ganttImg, 0, 0);
      ctx.strokeStyle = col.fg;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(g.x(S.t), 0);
      ctx.lineTo(g.x(S.t), ganttCv._h - 18);
      ctx.stroke();
    }
    ganttCv.addEventListener("click", (e) => {
      if (!S.run) return;
      const rect = ganttCv.getBoundingClientRect(), g = ganttGeom();
      const y = e.clientY - rect.top, x = e.clientX - rect.left;
      const i = Math.floor((y - g.top) / g.rowH);
      if (x < g.left) {
        if (i >= 0 && i < S.run.n) select(i);
        return;
      }
      if (i >= 0 && i < S.run.n) {
        if (S.sel !== i) select(i);
        setT(Math.round(((x - g.left) / (g.w - g.left - g.right)) * S.run.T));
      }
    });

    // ---- drawing: time-space along the selected train's route
    function drawLine() {
      const r = S.run, col = D.colors(), ctx = lineCv.getContext("2d");
      const w = lineCv._w, h = lineCv._h;
      ctx.clearRect(0, 0, w, h);
      if (S.sel === null) {
        lineNote.textContent = "Select a train (on the map or in the timeline) to see every train that uses its route, cell by cell over time: the railway's classic train graph.";
        ctx.fillStyle = col.muted;
        ctx.font = D.FONT;
        ctx.textAlign = "center";
        ctx.fillText("no train selected", w / 2, h / 2);
        return;
      }
      // the route: the plan (live presets) or the cells the train actually visited (recorded)
      let route = [];
      const planned = r.ctrl && r.ctrl.paths.has(S.sel) ? r.ctrl.paths.get(S.sel) : null;
      if (planned) route = planned.map((x) => x[0]);
      else {
        for (const f of r.frames) {
          const a = frameAgent(f, S.sel);
          if (a.cfg && (!route.length || route[route.length - 1][0] !== a.cfg[0] || route[route.length - 1][1] !== a.cfg[1])) route.push(a.cfg);
        }
        if (!route.length) route = r.env.graph.shortestPath(S.sel, S.map.agents[S.sel].start);
      }
      const idx = new Map();
      route.forEach((c, k) => {
        if (!idx.has(c[0] * 4096 + c[1])) idx.set(c[0] * 4096 + c[1], k);
      });
      const left = 34, right = 10, top = 8, bottom = 22;
      const T = r.T, n = route.length;
      const X = (t) => left + (t / T) * (w - left - right);
      const Y = (k) => top + (k / Math.max(1, n - 1)) * (h - top - bottom);
      // frame
      ctx.strokeStyle = col.grid;
      ctx.lineWidth = 1;
      ctx.font = D.FONT_SMALL;
      ctx.fillStyle = col.muted;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const step = T > 600 ? 100 : 50;
      for (let t = 0; t <= T; t += step) {
        ctx.beginPath();
        ctx.moveTo(X(t), top);
        ctx.lineTo(X(t), h - bottom);
        ctx.stroke();
        ctx.fillText(String(t), X(t), h - bottom + 4);
      }
      ctx.save();
      ctx.translate(10, (top + h - bottom) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillText("start → target", 0, 0);
      ctx.restore();
      // planned (dashed) for the selected train
      if (planned) {
        const base = r.plan && r.plan.get(S.sel);
        if (base) {
          ctx.save();
          ctx.setLineDash([4, 3]);
          ctx.strokeStyle = col.accent;
          ctx.globalAlpha = 0.7;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          base.forEach(([c, t], k) => {
            const kk = idx.get(c[0] * 4096 + c[1]);
            if (kk === undefined) return;
            const y = Y(kk);
            k ? ctx.lineTo(X(t), y) : ctx.moveTo(X(t), y);
          });
          ctx.stroke();
          ctx.restore();
        }
      }
      // every train's occupancy of the route's cells
      for (let i = 0; i < r.n; i++) {
        const isSel = i === S.sel;
        ctx.strokeStyle = isSel ? col.fg : col.series[i % 7];
        ctx.globalAlpha = isSel ? 1 : 0.75;
        ctx.lineWidth = isSel ? 2.5 : 1.4;
        ctx.beginPath();
        let prev = null, labelled = false;
        for (let t = 0; t < r.frames.length; t++) {
          const a = frameAgent(r.frames[t], i);
          const k = a.cfg ? idx.get(a.cfg[0] * 4096 + a.cfg[1]) : undefined;
          if (k === undefined) {
            prev = null;
            continue;
          }
          const x = X(t), y = Y(k);
          if (prev && Math.abs(prev[2] - k) <= 1) ctx.lineTo(x, y);
          else ctx.moveTo(x, y);
          prev = [x, y, k];
          if (!labelled && !isSel) {
            labelled = true;
            ctx.save();
            ctx.fillStyle = col.series[i % 7];
            ctx.globalAlpha = 1;
            ctx.font = D.FONT_SMALL;
            ctx.fillText(String(i), x + 6, y - 6);
            ctx.restore();
          }
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      // cursor
      ctx.strokeStyle = col.fg;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(X(S.t), top);
      ctx.lineTo(X(S.t), h - bottom);
      ctx.stroke();
      lineNote.innerHTML = `Vertical axis: the ${route.length} cells of train ${S.sel}'s ${planned ? "planned" : "actual"} route, start at the top. Every train that uses one of those cells is drawn where and when it is there; a flat stretch is a train standing still. ${planned ? "Dashed: train " + S.sel + "'s original plan." : ""} Lines that cross while running in opposite directions on single track are the conflicts a plan has to order.`;
    }

    // ---- drawing: event strip
    function drawStrip() {
      const r = S.run, col = D.colors(), ctx = strip.getContext("2d");
      const w = strip._w, h = strip._h;
      ctx.clearRect(0, 0, w, h);
      const X = (t) => 2 + (t / Math.max(1, r.frames.length - 1)) * (w - 4);
      ctx.fillStyle = col.grid;
      ctx.fillRect(0, h / 2 - 0.5, w, 1);
      for (const e of r.events) {
        if (e.type === "malfunction") {
          ctx.fillStyle = col.bad;
          ctx.fillRect(X(e.t), 4, Math.max(2, X(e.t + e.steps) - X(e.t)), 6);
        } else if (e.type === "arrival") {
          ctx.fillStyle = col.ok;
          ctx.fillRect(X(e.t) - 0.5, h - 12, 1.5, 8);
        }
      }
      ctx.fillStyle = col.accent;
      for (const c of r.clear) ctx.fillRect(X(c[0]) - 1, h / 2 - 5, 2, 10);
      const dlFirst = r.dl.findIndex((x) => x.length);
      if (dlFirst >= 0) {
        ctx.fillStyle = col.dark ? "#a02828" : "#8b1a1a";
        ctx.fillRect(X(dlFirst), h / 2 - 3, w - X(dlFirst), 6);
      }
      ctx.fillStyle = col.fg;
      ctx.fillRect(X(S.t) - 1, 0, 2, h);
    }

    // ---- drawing: arrivals, every controller
    function drawCompare() {
      if (!S.compare || !S.run) return;
      const col = D.colors(), ctx = cmpCv.getContext("2d");
      const w = cmpCv._w, h = cmpCv._h;
      ctx.clearRect(0, 0, w, h);
      const left = 34, right = 120, top = 10, bottom = 22;
      const T = S.run.T, n = S.run.n;
      const X = (t) => left + (t / T) * (w - left - right), Y = (v) => top + (1 - v / n) * (h - top - bottom);
      ctx.strokeStyle = col.grid;
      ctx.font = D.FONT_SMALL;
      ctx.fillStyle = col.muted;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (const v of [0, Math.round(n / 2), n]) {
        ctx.beginPath();
        ctx.moveTo(left, Y(v));
        ctx.lineTo(w - right, Y(v));
        ctx.stroke();
        ctx.fillText(String(v), left - 4, Y(v));
      }
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const step = T > 600 ? 100 : 50;
      for (let t = 0; t <= T; t += step) ctx.fillText(String(t), X(t), h - bottom + 4);
      const order = ["or", "tada", "reactive_avoid", "ppo", "shortest_path"];
      const labels = [];
      order.forEach((id, k) => {
        const arr = S.compare[id];
        if (!arr) return;
        const c = col.series[k];
        ctx.strokeStyle = c;
        ctx.lineWidth = S.preset === id ? 3 : 1.6;
        ctx.globalAlpha = S.preset === id || S.preset === "you" ? 1 : 0.8;
        ctx.beginPath();
        arr.forEach((v, t) => (t ? ctx.lineTo(X(t), Y(v)) : ctx.moveTo(X(t), Y(v))));
        ctx.lineTo(X(T), Y(arr[arr.length - 1]));
        ctx.stroke();
        labels.push([Y(arr[arr.length - 1]), PRESETS[id].label + " " + arr[arr.length - 1], c]);
      });
      if (S.preset === "you") {
        const arr = S.run.arr;
        ctx.strokeStyle = col.fg;
        ctx.lineWidth = 2.5;
        ctx.setLineDash([5, 3]);
        ctx.beginPath();
        arr.forEach((v, t) => (t ? ctx.lineTo(X(t), Y(v)) : ctx.moveTo(X(t), Y(v))));
        ctx.stroke();
        ctx.setLineDash([]);
        labels.push([Y(arr[arr.length - 1]), "You " + arr[arr.length - 1], col.fg]);
      }
      ctx.globalAlpha = 1;
      // direct labels, nudged apart
      labels.sort((a, b) => a[0] - b[0]);
      for (let i = 1; i < labels.length; i++) if (labels[i][0] - labels[i - 1][0] < 12) labels[i][0] = labels[i - 1][0] + 12;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      for (const [y, txt, c] of labels) {
        ctx.fillStyle = c;
        ctx.fillText(txt, w - right + 6, y);
      }
      ctx.strokeStyle = col.fg;
      ctx.beginPath();
      ctx.moveTo(X(S.t), top);
      ctx.lineTo(X(S.t), h - bottom);
      ctx.stroke();
      const o = S.map.episodes;
      cmpNote.textContent = `Recorded in flatland-rl on this map: ${Object.entries(o)
        .map(([k, e]) => `${PRESETS[k] ? PRESETS[k].label : k} ${e.summary.arrived}/${e.summary.n_agents}`)
        .join(", ")}. The curves are this page's replays of the same episodes (and the planner run live), so they end at the same numbers.`;
    }

    // ---- keyboard
    root.tabIndex = 0;
    root.addEventListener("keydown", (e) => {
      if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
      if (e.key === " ") {
        e.preventDefault();
        setPlaying(!S.playing);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setT(S.t + (e.shiftKey ? 10 : 1));
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        setT(S.t - (e.shiftKey ? 10 : 1));
      }
    });
    // ---- theme and resize
    const redrawAll = () => {
      if (!S.run) return;
      mapCv.fit();
      lineCv.fit();
      cmpCv.fit();
      strip.fit();
      drawGantt();
      setT(S.t, true);
    };
    new MutationObserver(redrawAll).observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
    let rz = null;
    window.addEventListener("resize", () => {
      clearTimeout(rz);
      rz = setTimeout(redrawAll, 150);
    });
    S.sel = startSel;
    loadMap();
  }

  // ------------------------------------------------------------------ mount
  function mount() {
    document.querySelectorAll('.fl-widget[data-widget="lab"]:not([data-mounted])').forEach((root) => {
      root.dataset.mounted = "1";
      try {
        wLab(root);
      } catch (e) {
        root.appendChild(document.createTextNode("The Lab could not start: " + e.message));
        console.error(e);
      }
    });
  }
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(mount);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
})();
