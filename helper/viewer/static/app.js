// Episode viewer: plays an ego video with gaze and every label of its dataset: steps, actions, label tracks,
// boxes on the video, other cameras in sync, and static info.
const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), timeline = $("timeline"), actionsBar = $("actions");
const TRAIL_SEC = 0.6, PAD = 12, BOX_SEC = 0.6, LANE_H = 7, MAX_LANES = 10;
const BOX = {
  subgoal: { w: 150, h: 46, gapX: 12, gapY: 26, chars: 19, lines: 3 },
  action: { w: 132, h: 36, gapX: 10, gapY: 20, chars: 18, lines: 2 },
};

const DATASETS = { hd_epic: "HD-EPIC (kitchen)", ego_exo4d: "Ego-Exo4D (cooking)", industreal: "IndustReal (tabletop assembly)", partnr: "PARTNR (simulated home)" };

let all = [], list = [], key = null, ep = null, layout = null, gaze = null, preds = {}, nodeColor = {}, lastKey = "";
let tracks = [], views = [], nowKey = "";

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
  $("graph-title").textContent = ep.edges.length ? "Task graph" : "Steps";
  $("graph-source").textContent = ep.graph_source;
  $("actions-row").hidden = !ep.actions.length;
  $("episode-pos").textContent = `${list.findIndex((e) => e.path === path) + 1} / ${list.length}`;
  lastKey = "";
  video.src = `/data/${path}/video.mp4`;
  drawGraph();
  buildTracks();
  buildViews(path);
  renderInfo();
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
      : !ep.edges.length ? "todo" : preds[n.id].every((p) => done.has(p)) ? "ready" : "blocked";
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
  if (!vw) return;
  // The video is letterboxed inside its element; map normalised coordinates into the drawn image.
  const s = Math.min(r.width / vw, r.height / vh), ox = (r.width - vw * s) / 2, oy = (r.height - vh * s) / 2;
  const toPx = (p) => [ox + p[0] * vw * s, oy + p[1] * vh * s];
  drawBoxes(ctx, t, toPx);
  if (!$("show-gaze").checked) return;
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
  el.innerHTML = graphSvg(ep.edges.map(([u, v], i) => ({ u, v, cls: "edge", note: ep.edge_notes?.[i] })), "node");
  el.querySelectorAll(".node").forEach((g) => (g.onclick = () => seekToNode(g.dataset.id)));
}

// The graph as SVG on the layout; node ids are <prefix>-<id>.
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
  const shown = cur.length ? cur : pre;
  $("sg-kind").textContent = cur.length ? "Current step · executing" : pre.length ? "Current step · preparing" : "Between steps";
  $("sg-title").textContent = shown.length ? shown.map((n) => n.label).join("  |  ") : "No step in progress";
  const segDetail = ep.segments.filter((g) => g.start <= t && t < g.end && g.detail && Object.keys(g.detail).length).map((g) => detailText(g.detail));
  $("sg-group").textContent = [[...new Set(shown.map((n) => n.group))].join(", "), ...segDetail].filter(Boolean).join(" · ");
  $("subgoal").classList.toggle("active", cur.length > 0 && !complete);
  $("subgoal").classList.toggle("complete", complete);
  $("sg-current").textContent = cur.length ? cur.map((n) => n.label).join("; ") : "-";
  const act = ep.actions.find((a) => a.start <= t && t < a.end);
  $("sg-action").textContent = act ? act.label : "-";
  $("sg-action-detail").textContent = act ? detailText(act.detail) : "";
  const [lo, hi] = clip();
  $("sg-progress").textContent = `${state.done.size} / ${ep.nodes.length} steps done`
    + ` · ${fmt(Math.max(0, t - lo))} / ${fmt(hi - lo)}`;
}


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
    c.title = hit ? [hit.label ?? ep.nodes.find((n) => n.id === hit.node)?.label, detailText(hit.detail)].filter(Boolean).join("\n")
      : fmt(t - clip()[0]);
  });
}

// ---------- label tracks, boxes, other cameras, info ----------
const labelColor = (label) => { let h = 0; for (const c of String(label)) h = (h * 31 + c.charCodeAt(0)) % 360; return `hsl(${h} 55% 55%)`; };
const detailText = (d) => Object.entries(d ?? {}).map(([k, v]) => `${k}: ${v}`).join(" · ");
const labelWithDetail = (it) => it.label + (Object.keys(it.detail ?? {}).length ? ` (${detailText(it.detail)})` : "");

function buildTracks() {
  const box = $("tracks");
  box.innerHTML = "";
  tracks = (ep.tracks ?? []).map((tr) => {
    const items = [...tr.items].sort((a, b) => a.start - b.start), ends = [];
    for (const it of items) {  // overlapping intervals stack in lanes
      let k = ends.findIndex((e) => e <= it.start);
      if (k < 0) { k = ends.length; ends.push(0); }
      ends[k] = Math.max(it.end, it.start + 0.3);
      it.lane = Math.min(k, MAX_LANES - 1);
    }
    const label = document.createElement("div"), canvas = document.createElement("canvas");
    label.className = "timeline-label";
    label.textContent = `${tr.name} · ${items.length}`;
    label.title = tr.source;
    canvas.className = "track";
    box.append(label, canvas);
    const view = { name: tr.name, items, canvas, h: Math.max(14, Math.min(ends.length, MAX_LANES) * LANE_H + 2) };
    canvas.addEventListener("click", (e) => (video.currentTime = timeAt(canvas, e)));
    canvas.addEventListener("mousemove", (e) => {
      const t = timeAt(canvas, e), tol = (4 * (clip()[1] - clip()[0])) / canvas.clientWidth;
      const hit = view.items.filter((it) => (it.end > it.start ? it.start <= t && t < it.end : Math.abs(it.start - t) <= tol));
      canvas.title = hit.length ? hit.map(labelWithDetail).join("\n") : fmt(t - clip()[0]);
    });
    return view;
  });
}

function drawTracks(t) {
  const [lo, hi] = clip(), ink = getComputedStyle(document.documentElement).getPropertyValue("--ink");
  for (const v of tracks) {
    const c = v.canvas, w = c.clientWidth, dpr = window.devicePixelRatio || 1;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(v.h * dpr)) {
      c.width = Math.round(w * dpr); c.height = Math.round(v.h * dpr); c.style.height = `${v.h}px`;
    }
    const g = c.getContext("2d"), X = (s) => ((s - lo) / (hi - lo)) * w;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, v.h);
    for (const it of v.items) {
      g.fillStyle = labelColor(it.label);
      if (it.end > it.start) { g.globalAlpha = 0.85; g.fillRect(X(it.start), 1 + it.lane * LANE_H, Math.max(1.5, X(it.end) - X(it.start)), LANE_H - 1); }
      else { g.globalAlpha = 1; g.fillRect(X(it.start) - 1, 0, 2, v.h); }  // an instant
    }
    g.globalAlpha = 1;
    g.fillStyle = ink;
    g.fillRect(X(t) - 1, 0, 2, v.h);
  }
}

// What every track says now: intervals around t, instants in the last 2 s.
function updateNow(t) {
  const rows = tracks.map((v) => {
    const hits = v.items.filter((it) => (it.end > it.start ? it.start <= t && t < it.end : it.start <= t && t - it.start < 2));
    if (!hits.length) return "";
    const shown = hits.slice(0, 4).map((it) => esc(it.label) + (Object.keys(it.detail ?? {}).length ? `<span class="detail">${esc(detailText(it.detail))}</span>` : ""));
    return `<div class="sg-row"><span class="k">${esc(v.name)}</span><span>${shown.join("")}${hits.length > 4 ? `<span class="detail">+${hits.length - 4} more</span>` : ""}</span></div>`;
  }).join("");
  if (rows !== nowKey) { $("now").innerHTML = rows; nowKey = rows; }
}

// Boxes are labelled only at certain frames (HD-EPIC: an object's pick-up and put-down); show them near those times.
function drawBoxes(ctx, t, toPx) {
  if (!$("show-boxes").checked || !ep.boxes?.length) return;
  ctx.font = "13px system-ui, sans-serif";
  for (const b of ep.boxes) {
    const d = Math.abs(b.t - t);
    if (d > BOX_SEC) continue;
    const [x1, y1] = toPx([b.box[0], b.box[1]]), [x2, y2] = toPx([b.box[2], b.box[3]]);
    ctx.globalAlpha = 1 - (0.7 * d) / BOX_SEC;
    ctx.strokeStyle = ctx.fillStyle = "#ffd400";
    ctx.lineWidth = 2.5;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillText(b.label, x1 + 3, Math.max(14, y1 - 4));
  }
  ctx.globalAlpha = 1;
}

function buildViews(path) {
  const box = $("views");
  box.innerHTML = "";
  views = (ep.views ?? []).map((name) => {
    const fig = document.createElement("figure"), v = document.createElement("video"), cap = document.createElement("figcaption");
    Object.assign(v, { src: `/data/${path}/view_${name}.mp4`, muted: true, preload: "auto", playsInline: true });
    cap.textContent = name;
    fig.append(v, cap);
    box.append(fig);
    return v;
  });
  box.hidden = !views.length;
}

function syncViews() {
  for (const v of views) if (!v.seeking && Math.abs(v.currentTime - video.currentTime) > 0.25) v.currentTime = video.currentTime;
}
video.addEventListener("play", () => views.forEach((v) => v.play().catch(() => {})));
video.addEventListener("pause", () => views.forEach((v) => v.pause()));
video.addEventListener("ratechange", () => views.forEach((v) => (v.playbackRate = video.playbackRate)));

function renderInfo() {
  const entries = Object.entries(ep.info ?? {});
  $("info-card").hidden = !entries.length;
  $("info").innerHTML = entries.map(([k, v]) => `<div class="info-k">${esc(k)}</div>`
    + (Array.isArray(v) ? `<ul>${v.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : `<div>${esc(v)}</div>`)).join("");
}

// ---------- loop ----------
function render() {
  if (!ep) return;
  const t = video.currentTime, state = stateAt(t);
  drawOverlay(t);
  updateGraph(state);
  updateCard(t, state);
  drawTimelines(t);
  drawTracks(t);
  updateNow(t);
  syncViews();
}

function tick() {
  render();
  if ("requestVideoFrameCallback" in video) video.requestVideoFrameCallback(tick);
  else requestAnimationFrame(tick);
}
window.addEventListener("resize", () => { if (ep) { drawGraph(); lastKey = ""; render(); } });
$("show-gaze").onchange = $("show-trail").onchange = $("show-boxes").onchange = render;

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
