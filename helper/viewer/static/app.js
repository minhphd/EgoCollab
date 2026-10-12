// Episode viewer: plays an ego clip with gaze and tracks node completion in its task graph.
const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), timeline = $("timeline"), actionsBar = $("actions");
const TRAIL_SEC = 0.6, PAD = 12;
const BOX = {
  subgoal: { w: 150, h: 46, gapX: 12, gapY: 26, chars: 19, lines: 3 },
  action: { w: 132, h: 36, gapX: 10, gapY: 20, chars: 18, lines: 2 },
};

const DATASETS = { hd_epic: "HD-EPIC (kitchen)", hd_epic_task: "HD-EPIC tasks (subgoal graphs)", ego_exo4d: "Ego-Exo4D (cooking)", industreal: "IndustReal (tabletop assembly)" };

let all = [], list = [], key = null, ep = null, layout = null, truth = new Set(), pred = null, predIdx = -2, gaze = null, preds = {}, nodeColor = {}, lastKey = "";

async function init() {
  all = await (await fetch("/api/episodes")).json();
  $("dataset").innerHTML = [...groupBy(all, (e) => e.dataset)].map(([d, eps]) =>
    `<option value="${esc(d)}">${esc(DATASETS[d] ?? d)} · ${eps.length}</option>`).join("");
  $("dataset").onchange = () => select(all.find((e) => e.dataset === $("dataset").value));
  $("group").onchange = () => select(list.find((e) => key(e) === $("group").value));
  $("episode").onchange = () => select(list.find((e) => e.path === $("episode").value));
  $("prev").onclick = () => step(-1);
  $("next").onclick = () => step(1);
  const fromHash = all.find((e) => e.path === decodeURIComponent(location.hash.slice(1)));
  if (all.length) select(fromHash ?? all[0]);
  else $("sg-title").textContent = "No episodes found. Run a dataset build first.";
}

// Points the dataset, group and episode selectors at episode e and loads it. Groups are videos
// when videos hold several episodes (HD-EPIC), else tasks (one episode per video).
function select(e) {
  list = all.filter((x) => x.dataset === e.dataset);
  const byVideo = list.length > new Set(list.map((x) => x.video_id)).size;
  key = byVideo ? (x) => x.video_id : (x) => x.task.split(" · ")[0];
  list.sort((x, y) => key(x).localeCompare(key(y)));
  $("dataset").value = e.dataset;
  const option = ([k, eps]) =>
    `<option value="${esc(k)}">${esc(byVideo ? `${k} · ${eps[0].task.split(" · ")[0]}` : k)} · ${eps.length}</option>`;
  const groups = [...groupBy(list, key)];
  $("group").innerHTML = byVideo  // videos under their participant, e.g. P01
    ? [...groupBy(groups, ([k]) => k.split("-")[0])].map(([p, gs]) =>
      `<optgroup label="${esc(p)}">${gs.map(option).join("")}</optgroup>`).join("")
    : groups.map(option).join("");
  $("group").value = key(e);
  $("episode").innerHTML = list.filter((x) => key(x) === key(e)).map((x) =>
    `<option value="${esc(x.path)}">${esc(byVideo ? `${fmt(x.start)}  ${x.title}` : x.title === key(x) ? x.video_id : `${x.video_id} · ${x.title}`)}</option>`).join("");
  $("episode").value = e.path;
  load(e.path);
}

function step(d) {
  const i = list.findIndex((e) => e.path === $("episode").value) + d;
  if (i >= 0 && i < list.length) select(list[i]);
}

async function load(path) {
  location.hash = path;
  [ep, gaze] = await Promise.all(["episode.json", "gaze.json"].map((f) => fetch(`/data/${path}/${f}`).then((r) => r.json())));
  preds = Object.fromEntries(ep.nodes.map((n) => [n.id, []]));
  ep.edges.forEach(([u, v]) => preds[v]?.push(u));
  nodeColor = Object.fromEntries(ep.nodes.map((n, i) => [n.id, `hsl(${(i * 137.5) % 360} 60% 55%)`]));
  $("graph-title").textContent = ep.node_kind === "action" ? "Action graph" : "Task graph";
  $("graph-source").textContent = ep.graph_source;
  $("actions-row").hidden = !ep.actions.length;
  $("episode-pos").textContent = `${list.findIndex((e) => e.path === path) + 1} / ${list.length}`;
  lastKey = "";
  video.src = `/data/${path}/video.mp4`;
  drawGraph();
  const names = all.find((e) => e.path === path)?.preds ?? [];
  $("pred-col").hidden = !names.length;
  document.body.classList.toggle("with-pred", names.length > 0);
  $("pred-model").innerHTML = names.map((n) => `<option>${esc(n)}</option>`).join("");
  $("pred-model").onchange = () => loadPred(path, $("pred-model").value);
  await loadPred(path, names[0]);
}

// ---------- clip window ----------
const clip = () => ep.clip ?? [0, video.duration || ep.video.duration];

video.addEventListener("loadedmetadata", () => { video.currentTime = clip()[0]; render(); tick(); });
video.addEventListener("seeked", render);
video.addEventListener("timeupdate", () => {
  const [lo, hi] = clip();
  if (video.currentTime >= hi) { video.pause(); video.currentTime = hi; }
  else if (video.currentTime < lo - 0.05) video.currentTime = lo;
});

// ---------- state at time t ----------
function stateAt(t) {
  const done = new Set(ep.done_before), active = new Set(), prep = new Set();
  for (const s of ep.segments) {
    if (s.kind === "step" && s.end <= t) done.add(s.node);
    if (s.start <= t && t < s.end) (s.kind === "step" ? active : prep).add(s.node);
  }
  const status = {};
  for (const n of ep.nodes) {
    status[n.id] = active.has(n.id) ? "active" : prep.has(n.id) ? "prep" : done.has(n.id) ? "done"
      : preds[n.id].every((p) => done.has(p)) ? "ready" : "blocked";
  }
  return { status, active, prep, done };
}

// ---------- gaze overlay ----------
function gazeAt(t) {
  const i = Math.round((t - (gaze.t0 || 0)) * gaze.fps);
  return i >= 0 && i < gaze.points.length ? gaze.points[i] : null;
}

function drawOverlay(t) {
  const r = video.getBoundingClientRect(), vw = video.videoWidth, vh = video.videoHeight;
  const dpr = window.devicePixelRatio || 1;
  overlay.width = r.width * dpr; overlay.height = r.height * dpr;
  overlay.style.width = `${r.width}px`; overlay.style.height = `${r.height}px`;
  const ctx = overlay.getContext("2d");
  ctx.scale(dpr, dpr);
  if (!vw || !$("show-gaze").checked) return;
  // The video is letterboxed inside its element; map normalised gaze into the drawn image.
  const s = Math.min(r.width / vw, r.height / vh), ox = (r.width - vw * s) / 2, oy = (r.height - vh * s) / 2;
  const toPx = (p) => [ox + p[0] * vw * s, oy + p[1] * vh * s];
  const color = getComputedStyle(document.documentElement).getPropertyValue("--gaze");
  if ($("show-trail").checked) {
    const n = Math.round(TRAIL_SEC * gaze.fps);
    let prev = null;
    for (let k = n; k >= 0; k--) {
      const p = gazeAt(t - k / gaze.fps);
      if (p && prev) {
        ctx.globalAlpha = 0.6 * (1 - k / n);
        ctx.strokeStyle = color; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(...toPx(prev)); ctx.lineTo(...toPx(p)); ctx.stroke();
      }
      prev = p;
    }
  }
  const p = gazeAt(t);
  if (!p) return;
  const [x, y] = toPx(p);
  ctx.globalAlpha = 1; ctx.strokeStyle = color; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(x, y, 16, 0, 2 * Math.PI); ctx.stroke();
  ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, 3, 0, 2 * Math.PI); ctx.fill();
}

// ---------- graph ----------
function wrap(text, maxChars, maxLines) {
  const lines = [""];
  for (const w of text.split(/\s+/)) {
    const cur = lines[lines.length - 1];
    if ((cur + " " + w).trim().length <= maxChars) lines[lines.length - 1] = (cur + " " + w).trim();
    else if (lines.length < maxLines) lines.push(w);
    else { lines[lines.length - 1] = cur.slice(0, maxChars - 1) + "…"; break; }
  }
  return lines;
}

function layoutGroup(ids, predsOf, succsOf, first) {
  // Longest-path layering, then sources move down to sit just above their first consumer.
  const inGroup = new Set(ids), layer = {};
  const inPreds = (id) => predsOf(id).filter((p) => inGroup.has(p));
  const inSuccs = (id) => succsOf(id).filter((v) => inGroup.has(v));
  const depth = (id) => layer[id] ?? (layer[id] = Math.max(-1, ...inPreds(id).map(depth)) + 1);
  ids.forEach(depth);
  ids.filter((id) => !inPreds(id).length && inSuccs(id).length)
    .forEach((id) => (layer[id] = Math.min(...inSuccs(id).map((v) => layer[v])) - 1));
  const rows = [];
  ids.forEach((id) => (rows[layer[id]] ??= []).push(id));
  // Order each row by the mean slot of its predecessors (barycenter), ties by first appearance in time.
  const slot = {};
  return rows.filter(Boolean).map((row) => {
    const key = (id) => {
      const ps = inPreds(id).filter((p) => p in slot);
      return ps.length ? ps.reduce((a, p) => a + slot[p], 0) / ps.length : Infinity;
    };
    row.sort((a, b) => key(a) - key(b) || first(a) - first(b));
    row.forEach((id, i) => (slot[id] = i - (row.length - 1) / 2));
    return row;
  });
}

function drawGraph() {
  const el = $("graph"), box = BOX[ep.node_kind] ?? BOX.subgoal;
  const succs = Object.fromEntries(ep.nodes.map((n) => [n.id, []]));
  ep.edges.forEach(([u, v]) => succs[u]?.push(v));
  const first = (id) => Math.min(Infinity, ...ep.segments.filter((s) => s.node === id).map((s) => s.start));
  // Layers wider than the panel wrap onto extra rows; nodes within one layer share no edges.
  const perRow = Math.max(2, Math.floor((el.clientWidth - PAD * 2 + box.gapX) / (box.w + box.gapX)));
  const chunk = (row) => Array.from({ length: Math.ceil(row.length / perRow) }, (_, i) => row.slice(i * perRow, (i + 1) * perRow));
  const groups = ep.node_kind === "action" ? [""] : [...new Set(ep.nodes.map((n) => n.group))];
  const laid = groups.map((g) => [g, layoutGroup(
    ep.nodes.filter((n) => groups.length === 1 || n.group === g).map((n) => n.id),
    (id) => preds[id], (id) => succs[id], first).flatMap(chunk)]);
  const rowW = (r) => r.length * (box.w + box.gapX) - box.gapX;
  const width = Math.max(300, ...laid.flatMap(([, rows]) => rows.map((r) => PAD * 2 + rowW(r))));
  const pos = {};
  let y = PAD, titles = "";
  for (const [g, rows] of laid) {
    if (g) { titles += `<text class="group-title" x="${PAD}" y="${y + 12}">${esc(g)}</text>`; y += 22; }
    for (const row of rows) {
      const x0 = (width - rowW(row)) / 2;
      row.forEach((id, i) => (pos[id] = { x: x0 + i * (box.w + box.gapX), y }));
      y += box.h + box.gapY;
    }
    y += 6;
  }
  layout = { pos, width, height: y, titles, box };
  truth = closure(ep.edges);
  el.innerHTML = graphSvg(ep.edges.map(([u, v], i) => ({ u, v, cls: "edge", note: ep.edge_notes?.[i] })), "node");
  el.querySelectorAll(".node").forEach((g) => (g.onclick = () => seekToNode(g.dataset.id)));
  predIdx = -2;
}

// One graph as SVG on the shared layout; node ids are <prefix>-<id>.
function graphSvg(edges, prefix) {
  const { pos, width, height, titles, box } = layout;
  const label = Object.fromEntries(ep.nodes.map((n) => [n.id, n.label]));
  const edgeSvg = edges.map(({ u, v, cls, note }) => {
    if (!pos[u] || !pos[v]) return "";
    const a = pos[u], b = pos[v], x1 = a.x + box.w / 2, y1 = a.y + box.h, x2 = b.x + box.w / 2, y2 = b.y;
    const my = (y1 + y2) / 2, ya = a.y + box.h / 2, yb = b.y + box.h / 2, bow = Math.min(a.x, b.x) - 40;
    const d = b.y > a.y ? `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2 - 2}`  // upward edges arc round the left
      : `M${a.x},${ya} C${bow},${ya} ${bow},${yb} ${b.x - 2},${yb}`;
    return `<g class="edge-g"><title>${esc(`${label[u]} → ${label[v]}${note ? `: ${note}` : ""}`)}</title>
      <path class="edge-hit" d="${d}"/><path class="${cls}" marker-end="url(#arrow-${prefix})" d="${d}"/></g>`;
  }).join("");
  const nodeSvg = ep.nodes.map((n) => {
    const { x, y } = pos[n.id], lines = wrap(n.label, box.chars, box.lines);
    const tspans = lines.map((l, i) =>
      `<tspan x="${x + 7}" y="${y + box.h / 2 + (i - (lines.length - 1) / 2) * 13 + 4}">${esc(l)}</tspan>`).join("");
    return `<g class="node" id="${prefix}-${cssId(n.id)}" data-id="${esc(n.id)}"><title>${esc(n.label)}</title>
      <rect x="${x}" y="${y}" width="${box.w}" height="${box.h}"/><text>${tspans}</text></g>`;
  }).join("");
  return `<svg width="${width}" height="${height}">
    <defs><marker id="arrow-${prefix}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0,0 L10,5 L0,10 z" fill="currentColor" opacity=".6"/></marker></defs>${titles}${edgeSvg}${nodeSvg}</svg>`;
}

// ---------- inferred graph ----------
// Pairs (u, v) with a path u -> ... -> v, as "u>v" strings.
function closure(edges) {
  const succ = {};
  edges.forEach(([u, v]) => (succ[u] ??= []).push(v));
  const out = new Set();
  for (const { id } of ep.nodes) {
    const seen = new Set(), stack = [...(succ[id] ?? [])];
    while (stack.length) { const m = stack.pop(); if (!seen.has(m)) { seen.add(m); stack.push(...(succ[m] ?? [])); } }
    seen.forEach((m) => m !== id && out.add(`${id}>${m}`));
  }
  return out;
}

const statedEdges = (est) => est.edges.filter(([, , p]) => p >= 0.5);

// Edge F1 on transitive closure and progress accuracy of one estimate, against the truth at its time.
function scoreOf(rec) {
  const g = closure(statedEdges(rec.estimate)), hit = [...g].filter((e) => truth.has(e)).length;
  const p = hit / Math.max(1, g.size), r = hit / Math.max(1, truth.size);
  const done = stateAt(rec.t).done;
  const progress = ep.nodes.filter((n) => rec.estimate.done.includes(n.id) === done.has(n.id)).length / ep.nodes.length;
  return { f1: p + r ? (2 * p * r) / (p + r) : 0, progress };
}

async function loadPred(path, name) {
  pred = name ? await fetch(`/data/${path}/pred_${name}.json`).then((r) => r.json()) : null;
  pred?.records.forEach((r) => { r.estimate ??= { done: [], current: null, next: [], edges: [] }; r.score = scoreOf(r); });
  predIdx = -2;
  render();
}

// The estimate on screen is the last one whose answer has arrived by t (observed time + latency).
function drawPred(t) {
  if (!pred || !layout) return;
  let i = -1;
  while (i + 1 < pred.records.length && pred.records[i + 1].t_avail <= t) i++;
  if (i !== predIdx) { predIdx = i; drawPredGraph(i >= 0 ? pred.records[i] : null); }
  drawCurve(t);
}

function drawPredGraph(rec) {
  const est = rec?.estimate ?? { done: [], current: null, next: [], edges: [] };
  const stated = statedEdges(est), g = closure(stated);
  const shown = stated.map(([u, v, p]) => ({ u, v, cls: truth.has(`${u}>${v}`) ? "edge e-hit" : "edge e-extra", note: `p = ${p}` }));
  const missed = ep.edges.filter(([u, v]) => !g.has(`${u}>${v}`)).map(([u, v]) => ({ u, v, cls: "edge e-miss", note: "missed" }));
  $("pred-graph").innerHTML = graphSvg([...missed, ...shown], "pnode");
  for (const n of ep.nodes) {
    const st = est.done.includes(n.id) ? "done" : est.current === n.id ? "active" : est.next.includes(n.id) ? "ready" : "blocked";
    document.getElementById(`pnode-${cssId(n.id)}`)?.setAttribute("class", `node ${st}`);
  }
  const right = shown.filter((e) => e.cls.includes("hit")).length;
  $("pred-meta").textContent = !rec ? `${pred.model} · waiting for the first estimate`
    : `${pred.model} · ${rec.prior ? "prior from the task description only" : `frames up to ${fmt(rec.t)}, shown at ${fmt(rec.t_avail)} (latency ${rec.latency.toFixed(1)} s)`}`
      + ` · edges: ${right} right, ${shown.length - right} wrong, ${missed.length} of ${ep.edges.length} true missed · edge F1 ${rec.score.f1.toFixed(2)}`;
}

function drawCurve(t) {
  const c = $("pred-curve"), dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = 46;
  if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = h * dpr; }
  const g = c.getContext("2d"), css = getComputedStyle(document.documentElement);
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const [lo, hi] = clip(), x = (s) => ((s - lo) / (hi - lo)) * w, y = (v) => h - 4 - v * (h - 8);
  for (const [key, color] of [["progress", "--done"], ["f1", "--ready"]]) {
    g.strokeStyle = css.getPropertyValue(color);
    g.lineWidth = 1.5;
    g.beginPath();
    pred.records.forEach((r, i) => {
      g.lineTo(x(r.t_avail), y(r.score[key]));
      g.lineTo(x(pred.records[i + 1]?.t_avail ?? hi), y(r.score[key]));
    });
    g.stroke();
  }
  g.strokeStyle = css.getPropertyValue("--active");
  g.beginPath(); g.moveTo(x(t), 0); g.lineTo(x(t), h); g.stroke();
}

function seekToNode(id) {
  const seg = ep.segments.filter((s) => s.node === id).sort((a, b) => a.start - b.start)[0];
  if (seg) video.currentTime = seg.start;
}

function updateGraph(state) {
  const key = JSON.stringify(state.status);
  if (key === lastKey) return;
  lastKey = key;
  const el = $("graph");
  for (const [id, st] of Object.entries(state.status)) {
    document.getElementById(`node-${cssId(id)}`)?.setAttribute("class", `node ${st}`);
  }
  const focus = el.querySelector(".node.active, .node.prep");
  if (focus) {  // scroll inside the panel only, never the page
    const r = focus.getBoundingClientRect(), c = el.getBoundingClientRect();
    if (r.top < c.top || r.bottom > c.bottom) el.scrollTop += r.top - c.top - c.height / 2 + r.height / 2;
  }
}

// ---------- card ----------
function updateCard(t, state) {
  const byId = Object.fromEntries(ep.nodes.map((n) => [n.id, n]));
  const cur = [...state.active].map((id) => byId[id]), pre = [...state.prep].map((id) => byId[id]);
  const complete = state.done.size === ep.nodes.length;
  if (ep.node_kind === "action") {
    $("sg-kind").textContent = `Subgoal · ${complete ? "complete" : "in progress"} · ${ep.task}`;
    $("sg-title").textContent = ep.title;
    const st = ep.stats ?? {};
    $("sg-group").textContent = [ep.video_id,
      st.head_displacement_m != null ? `head moves at most ${st.head_displacement_m.toFixed(2)} m` : ""].filter(Boolean).join(" · ");
  } else {
    const shown = cur.length ? cur : pre;
    $("sg-kind").textContent = cur.length ? "Current subgoal · executing" : pre.length ? "Current subgoal · preparing" : "Between subgoals";
    $("sg-title").textContent = shown.length ? shown.map((n) => n.label).join("  |  ") : "No subgoal in progress";
    $("sg-group").textContent = [...new Set(shown.map((n) => n.group))].join(", ");
  }
  $("subgoal").classList.toggle("active", cur.length > 0 && !complete);
  $("subgoal").classList.toggle("complete", complete);
  $("sg-current").textContent = cur.length ? cur.map((n) => n.label).join("; ") : "-";
  $("sg-next").textContent = few(ep.nodes.filter((n) => state.status[n.id] === "ready").map((n) => n.label));
  const act = ep.actions.find((a) => a.start <= t && t < a.end);
  $("sg-action").textContent = act ? act.label : "-";
  const [lo, hi] = clip();
  $("sg-progress").textContent = `${state.done.size} / ${ep.nodes.length} ${ep.node_kind === "action" ? "actions" : "subgoals"} done`
    + ` · ${fmt(Math.max(0, t - lo))} / ${fmt(hi - lo)}`;
}

const few = (xs) => (xs.length ? xs.slice(0, 3).join("; ") + (xs.length > 3 ? ` (+${xs.length - 3})` : "") : "-");

// ---------- timelines ----------
function drawBar(canvas, h, items, colorOf, t) {
  const w = canvas.clientWidth, dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr; canvas.height = h * dpr; canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const [lo, hi] = clip(), X = (s) => ((s - lo) / (hi - lo)) * w;
  for (const it of items) {
    const prep = it.kind === "prep";
    ctx.fillStyle = colorOf(it);
    ctx.globalAlpha = prep ? 0.4 : 0.9;
    ctx.fillRect(X(it.start), prep ? h * 0.55 : 0, Math.max(1, X(it.end) - X(it.start)), prep ? h * 0.45 : h);
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--ink");
  ctx.fillRect(X(t) - 1, 0, 2, h);
}

function drawTimelines(t) {
  drawBar(timeline, 28, ep.segments, (s) => nodeColor[s.node], t);
  if (ep.actions.length) drawBar(actionsBar, 14, ep.actions, () => "#8892a6", t);
}

function timeAt(c, e) {
  const r = c.getBoundingClientRect(), [lo, hi] = clip();
  return lo + ((e.clientX - r.left) / r.width) * (hi - lo);
}

for (const c of [timeline, actionsBar]) {
  c.addEventListener("click", (e) => (video.currentTime = timeAt(c, e)));
  c.addEventListener("mousemove", (e) => {
    const t = timeAt(c, e), items = c === timeline ? ep.segments : ep.actions;
    const hit = items.find((s) => s.start <= t && t < s.end);
    c.title = hit ? (hit.label ?? ep.nodes.find((n) => n.id === hit.node)?.label) : fmt(t - clip()[0]);
  });
}

// ---------- loop ----------
function render() {
  if (!ep) return;
  const t = video.currentTime, state = stateAt(t);
  drawOverlay(t);
  updateGraph(state);
  updateCard(t, state);
  drawTimelines(t);
  drawPred(t);
}

function tick() {
  render();
  if ("requestVideoFrameCallback" in video) video.requestVideoFrameCallback(tick);
  else requestAnimationFrame(tick);
}
window.addEventListener("resize", () => { if (ep) { drawGraph(); lastKey = ""; render(); } });
$("show-gaze").onchange = $("show-trail").onchange = render;

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT") return;
  if (e.key === "n" || e.key === "p") return step(e.key === "n" ? 1 : -1);
  const fps = ep?.video.fps || 30;
  const d = { " ": 0, ArrowLeft: -5, ArrowRight: 5, ",": -1 / fps, ".": 1 / fps }[e.key];
  if (d === undefined) return;
  e.preventDefault();
  if (e.key === " ") {
    if (video.currentTime >= clip()[1] - 0.05) video.currentTime = clip()[0];  // replay a finished clip
    video.paused ? video.play() : video.pause();
  } else { video.pause(); video.currentTime = Math.max(clip()[0], video.currentTime + d); }
});

function groupBy(xs, key) {
  const m = new Map();
  for (const x of xs) {
    if (!m.has(key(x))) m.set(key(x), []);
    m.get(key(x)).push(x);
  }
  return m;
}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const cssId = (s) => String(s).replace(/[^\w-]/g, "_");
const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

init();
