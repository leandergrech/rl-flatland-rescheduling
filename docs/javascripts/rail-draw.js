/* rail-draw: shared drawing helpers for the Scheduling Lab and the "How it works" widgets.
 * Colours come from the page theme (light / dark), read from CSS variables at draw time. No dependencies. */
(function (root) {
  "use strict";
  const FL = root.FL;

  // ------------------------------------------------------------------ theme
  const isDark = () => document.body && document.body.getAttribute("data-md-color-scheme") === "slate";
  function colors() {
    const dark = isDark();
    return {
      dark,
      bg: dark ? "#0f1115" : "#f8f6f1",
      panel: dark ? "#14161b" : "#ffffff",
      fg: dark ? "#ece9e0" : "#161614",
      muted: dark ? "#a39f95" : "#5c5a54",
      grid: dark ? "rgba(236,233,224,0.08)" : "rgba(22,22,20,0.07)",
      rail: dark ? "#6b6e76" : "#9a958a",
      railHi: dark ? "#b9b4a8" : "#4f4b43",
      sw: dark ? "#ffc93c" : "#9b5a00",
      accent: dark ? "#ffc93c" : "#c77700",
      ok: "#1baf7a",
      bad: dark ? "#ff6b6b" : "#d03b3b",
      warn: dark ? "#ffb347" : "#e07b00",
      stopped: dark ? "#9aa0ab" : "#6f7480",
      waiting: dark ? "#55595f" : "#c3bfb5",
      // categorical slots for per-train identity where needed (fixed order, never cycled past 7)
      series: dark ? ["#c98500", "#d55181", "#199e70", "#9085e9", "#d95926", "#3987e5", "#008300"] : ["#eda100", "#e87ba4", "#1baf7a", "#4a3aa7", "#eb6834", "#2a78d6", "#008300"],
    };
  }
  const FONT = '12px "IBM Plex Sans", Inter, "Helvetica Neue", Arial, sans-serif';
  const FONT_SMALL = '11px "IBM Plex Sans", Inter, "Helvetica Neue", Arial, sans-serif';
  const MONO = '11px "IBM Plex Mono", "Space Mono", monospace';

  // ------------------------------------------------------------------ canvas with device-pixel scaling
  function makeCanvas(parent, height, cls) {
    const cv = document.createElement("canvas");
    if (cls) cv.className = cls;
    parent.appendChild(cv);
    const fit = () => {
      const w = Math.max(200, parent.clientWidth || 600);
      const h = typeof height === "function" ? height(w) : height;
      const dpr = window.devicePixelRatio || 1;
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
      cv.style.width = w + "px";
      cv.style.height = h + "px";
      const ctx = cv.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      cv._w = w;
      cv._h = h;
    };
    fit();
    cv.fit = fit;
    return cv;
  }

  // ------------------------------------------------------------------ geometry of one cell
  // side midpoints of a cell in unit coordinates: N, E, S, W
  const SIDE = [
    [0.5, 0],
    [1, 0.5],
    [0.5, 1],
    [0, 0.5],
  ];
  /* unique track pieces of a cell: [entrySide, exitSide] pairs (an entry side is where a train comes from).
   * A train heading d entered through side (d + 2) % 4; an exit towards nd leaves through side nd. */
  function cellPieces(rail, r, c) {
    const seen = new Set(), out = [];
    for (let d = 0; d < 4; d++) {
      const t = rail.trans(r, c, d);
      for (let nd = 0; nd < 4; nd++) {
        if (!t[nd]) continue;
        const a = (d + 2) % 4, b = nd;
        if (a === b) {
          // dead end: a U-turn back out of the side it came in through
          const k = "u" + a;
          if (!seen.has(k)) seen.add(k), out.push([a, a]);
          continue;
        }
        const k = Math.min(a, b) + "-" + Math.max(a, b);
        if (!seen.has(k)) seen.add(k), out.push([a, b]);
      }
    }
    return out;
  }

  function pieceTo(ctx, x0, y0, s, a, b) {
    const [ax, ay] = SIDE[a], [bx, by] = SIDE[b];
    ctx.moveTo(x0 + ax * s, y0 + ay * s);
    if (a === b) {
      // dead end: short stub with a cap
      const mx = x0 + (ax * 0.55 + 0.5 * 0.45) * s, my = y0 + (ay * 0.55 + 0.5 * 0.45) * s;
      ctx.lineTo(mx, my);
      return;
    }
    if ((a + 2) % 4 === b) ctx.lineTo(x0 + bx * s, y0 + by * s);
    else ctx.quadraticCurveTo(x0 + 0.5 * s, y0 + 0.5 * s, x0 + bx * s, y0 + by * s);
  }

  /* drawRail(ctx, rail, graph, view): view = {x0, y0, s (cell size px), switches: bool, decisions: Set of cell keys} */
  function drawRail(ctx, rail, graph, view, col) {
    const { x0, y0, s } = view;
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = col.rail;
    ctx.lineWidth = Math.max(1.2, s * 0.16);
    ctx.beginPath();
    for (let r = 0; r < rail.H; r++)
      for (let c = 0; c < rail.W; c++) {
        if (!rail.code(r, c)) continue;
        for (const [a, b] of cellPieces(rail, r, c)) pieceTo(ctx, x0 + c * s, y0 + r * s, s, a, b);
      }
    ctx.stroke();
    if (view.switches !== false && graph) {
      ctx.fillStyle = col.sw;
      ctx.globalAlpha = 0.85;
      for (let r = 0; r < rail.H; r++)
        for (let c = 0; c < rail.W; c++)
          if (graph.isSwitchCell(r, c)) {
            const cx = x0 + (c + 0.5) * s, cy = y0 + (r + 0.5) * s, q = Math.max(1.6, s * 0.16);
            ctx.beginPath();
            ctx.moveTo(cx, cy - q);
            ctx.lineTo(cx + q, cy);
            ctx.lineTo(cx, cy + q);
            ctx.lineTo(cx - q, cy);
            ctx.closePath();
            ctx.fill();
          }
    }
    ctx.restore();
  }

  function drawStation(ctx, view, r, c, color, label) {
    const { x0, y0, s } = view;
    const x = x0 + c * s, y = y0 + r * s, m = s * 0.14;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1, s * 0.09);
    ctx.strokeRect(x + m, y + m, s - 2 * m, s - 2 * m);
    if (label !== undefined && s >= 14) {
      ctx.fillStyle = color;
      ctx.font = MONO;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(label, x + s / 2, y - 1);
    }
    ctx.restore();
  }

  /* a train: an arrow in its heading, inside its cell; fill by state, ring if selected */
  function drawTrain(ctx, view, cfg, fill, opts = {}) {
    const { x0, y0, s } = view;
    const cx = x0 + (cfg[1] + 0.5) * s, cy = y0 + (cfg[0] + 0.5) * s;
    const ang = [-Math.PI / 2, 0, Math.PI / 2, Math.PI][cfg[2]];
    const L = s * 0.46, Wd = s * 0.3;
    ctx.save();
    ctx.translate(cx, cy);
    if (opts.ring) {
      ctx.beginPath();
      ctx.arc(0, 0, s * 0.62, 0, 2 * Math.PI);
      ctx.strokeStyle = opts.ring;
      ctx.lineWidth = Math.max(1.5, s * 0.1);
      ctx.stroke();
    }
    ctx.rotate(ang);
    ctx.beginPath();
    ctx.moveTo(L, 0);
    ctx.lineTo(-L * 0.75, Wd);
    ctx.lineTo(-L * 0.45, 0);
    ctx.lineTo(-L * 0.75, -Wd);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.lineWidth = Math.max(0.8, s * 0.05);
    ctx.strokeStyle = opts.outline || "rgba(0,0,0,0.55)";
    ctx.stroke();
    ctx.restore();
    if (opts.label !== undefined && s >= 11) {
      ctx.save();
      ctx.font = s >= 18 ? FONT : FONT_SMALL;
      ctx.fillStyle = opts.labelColor || "#000";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(String(opts.label), cx + s * 0.32, cy + s * 0.2);
      ctx.restore();
    }
    if (opts.cross) {
      ctx.save();
      ctx.strokeStyle = opts.cross;
      ctx.lineWidth = Math.max(1.5, s * 0.12);
      const q = s * 0.42;
      ctx.beginPath();
      ctx.moveTo(cx - q, cy - q);
      ctx.lineTo(cx + q, cy + q);
      ctx.moveTo(cx + q, cy - q);
      ctx.lineTo(cx - q, cy + q);
      ctx.stroke();
      ctx.restore();
    }
  }

  function stateColor(state, col) {
    const ST = FL.ST;
    if (state === ST.MOVING) return col.accent;
    if (state === ST.STOPPED) return col.stopped;
    if (state === ST.MALFUNCTION || state === ST.MALFUNCTION_OFF_MAP) return col.bad;
    if (state === ST.DONE) return col.ok;
    return col.waiting;
  }

  /* fit a map into a box: returns view {x0, y0, s} and the used bounding box of rail cells (crop empty margins) */
  function fitView(rail, w, h, pad = 6) {
    let r0 = rail.H, r1 = -1, c0 = rail.W, c1 = -1;
    for (let r = 0; r < rail.H; r++)
      for (let c = 0; c < rail.W; c++)
        if (rail.code(r, c)) {
          r0 = Math.min(r0, r);
          r1 = Math.max(r1, r);
          c0 = Math.min(c0, c);
          c1 = Math.max(c1, c);
        }
    if (r1 < 0) (r0 = 0), (r1 = rail.H - 1), (c0 = 0), (c1 = rail.W - 1);
    r0 = Math.max(0, r0 - 1);
    c0 = Math.max(0, c0 - 1);
    r1 = Math.min(rail.H - 1, r1 + 1);
    c1 = Math.min(rail.W - 1, c1 + 1);
    const nr = r1 - r0 + 1, nc = c1 - c0 + 1;
    const s = Math.max(2, Math.min((w - 2 * pad) / nc, (h - 2 * pad) / nr));
    return { x0: pad + (w - 2 * pad - nc * s) / 2 - c0 * s, y0: pad + (h - 2 * pad - nr * s) / 2 - r0 * s, s, r0, r1, c0, c1, nr, nc };
  }

  /* toy rail builder: tracks = list of polylines of [r, c] cells (adjacent cells, 4-neighbour);
   * every polyline is bidirectional track. Ends of a polyline that touch nothing else become dead ends. */
  function buildRail(H, W, tracks, opts = {}) {
    const grid = new Uint16Array(H * W);
    const dirOf = (a, b) => (b[0] < a[0] ? 0 : b[1] > a[1] ? 1 : b[0] > a[0] ? 2 : 3);
    const add = (r, c, heading, exit) => {
      grid[r * W + c] |= 1 << ((3 - heading) * 4 + (3 - exit));
    };
    for (const line of tracks)
      for (let i = 0; i < line.length; i++) {
        const cur = line[i];
        const prev = line[i - 1], next = line[i + 1];
        if (prev && next) {
          const hin = dirOf(prev, cur), out = dirOf(cur, next);
          add(cur[0], cur[1], hin, out); // forward
          add(cur[0], cur[1], (out + 2) % 4, (hin + 2) % 4); // backward
        }
      }
    // the two ends of every polyline: a train can start there facing along the track (heading `out`),
    // and a train arriving there (heading away from the track) turns back (flatland's dead-end cell)
    // (only ends that no other polyline passes through: a branch that joins a line is not a dead end)
    const uses = new Map();
    for (const line of tracks) for (const c of line) uses.set(c[0] * 4096 + c[1], (uses.get(c[0] * 4096 + c[1]) || 0) + 1);
    if (opts.deadEnds !== false)
      for (const line of tracks)
        for (const i of [0, line.length - 1]) {
          const cur = line[i], nb = line[i === 0 ? 1 : line.length - 2];
          if (uses.get(cur[0] * 4096 + cur[1]) > 1) continue;
          const out = dirOf(cur, nb);
          add(cur[0], cur[1], out, out);
          add(cur[0], cur[1], (out + 2) % 4, out);
        }
    return Array.from(grid);
  }

  root.FLDraw = { colors, isDark, makeCanvas, drawRail, drawStation, drawTrain, stateColor, fitView, cellPieces, buildRail, FONT, FONT_SMALL, MONO, SIDE };
})(typeof self !== "undefined" ? self : this);
