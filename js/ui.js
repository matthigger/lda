// Shared helpers: plane geometry for both tabs (the square [-8, 8]^2),
// 2-d vector math, SVG arrows, ellipses and segments, pointer dragging,
// number formatting and the footer build stamp.

const PLANE = { w: 700, h: 700, x0: 46, x1: 686, y0: 14, y1: 654,
  dom: [-8, 8] };

/** Scales of plot box p: data to viewBox (sx, sy) and back (ix, iy). */
function scales(p) {
  const k = (p.x1 - p.x0) / (p.dom[1] - p.dom[0]);
  return {
    k,
    sx: v => p.x0 + (v - p.dom[0]) * k,
    sy: v => p.y1 - (v - p.dom[0]) * k,
    ix: px => p.dom[0] + (px - p.x0) / k,
    iy: py => p.dom[0] + (p.y1 - py) / k,
  };
}
const S = scales(PLANE);

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const mul = (a, s) => [a[0] * s, a[1] * s];
const norm = a => Math.hypot(a[0], a[1]);
const unit = t => [Math.cos(t), Math.sin(t)];
/** a turned 90 degrees counterclockwise. */
const perp = a => [-a[1], a[0]];
const DEG = Math.PI / 180;

/** Data point to "x,y" in viewBox units. */
function px(p) { return `${S.sx(p[0]).toFixed(1)},${S.sy(p[1]).toFixed(1)}`; }

/** SVG segment between data points a and c. */
function segSvg(a, c, attrs) {
  return `<line x1="${S.sx(a[0]).toFixed(1)}" y1="${S.sy(a[1]).toFixed(1)}"
    x2="${S.sx(c[0]).toFixed(1)}" y2="${S.sy(c[1]).toFixed(1)}" ${attrs}/>`;
}

/** SVG polygon through data points. */
function polySvg(pts, attrs) {
  return `<polygon points="${pts.map(px).join(" ")}" ${attrs}/>`;
}

/** SVG circle at data point q, radius r in viewBox units. */
function dotSvg(q, r, attrs) {
  return `<circle cx="${S.sx(q[0]).toFixed(1)}" cy="${S.sy(q[1]).toFixed(1)}"
    r="${r}" ${attrs}/>`;
}

/** SVG arrow from data point a to c; cls sets stroke and fill. */
function arrowSvg(a, c, cls, head = 14) {
  const x1 = S.sx(a[0]), y1 = S.sy(a[1]), x2 = S.sx(c[0]), y2 = S.sy(c[1]);
  const L = Math.hypot(x2 - x1, y2 - y1);
  if (L < 1) return "";
  const ux = (x2 - x1) / L, uy = (y2 - y1) / L, h = Math.min(head, L * .6);
  const bx = x2 - ux * h, by = y2 - uy * h;
  const pts = [[x2, y2], [bx - uy * h * .42, by + ux * h * .42],
    [bx + uy * h * .42, by - ux * h * .42]];
  return `<g class="arrow ${cls}"><line x1="${x1}" y1="${y1}" x2="${bx}"
    y2="${by}"/><polygon points="${pts.map(q => q.join(",")).join(" ")}"/>
    </g>`;
}

/** SVG path of the k-sd ellipse {x : (x − mu)ᵀM⁻¹(x − mu) = k²}. */
function ellipseSvg(mu, M, k, attrs) {
  const e = eig2(M.a, M.b, M.d), v1 = unit(isNaN(e.theta) ? 0 : e.theta);
  const v2 = perp(v1);
  const r1 = k * Math.sqrt(Math.max(e.l1, 0));
  const r2 = k * Math.sqrt(Math.max(e.l2, 0));
  let d = "";
  for (let t = 0; t <= 96; t++) {
    const a = 2 * Math.PI * t / 96;
    const q = add(mu, add(mul(v1, r1 * Math.cos(a)), mul(v2, r2 * Math.sin(a))));
    d += `${t ? "L" : "M"}${px(q)}`;
  }
  return `<path d="${d}Z" ${attrs}/>`;
}

/** Clip path over the plot box, then grid lines, ticks and axis labels. */
function axesSvg(clipId) {
  const p = PLANE;
  let a = `<defs><clipPath id="${clipId}"><rect x="${p.x0}" y="${p.y0}"
    width="${p.x1 - p.x0}" height="${p.y1 - p.y0}"/></clipPath></defs>`;
  for (let v = p.dom[0]; v <= p.dom[1]; v++) {
    const z = v === 0 ? " zero" : "";
    a += `<line class="grid${z}" x1="${S.sx(v)}" x2="${S.sx(v)}"
      y1="${p.y0}" y2="${p.y1}"/><line class="grid${z}" x1="${p.x0}"
      x2="${p.x1}" y1="${S.sy(v)}" y2="${S.sy(v)}"/>`;
    if (v % 2) continue;
    a += `<text class="tick" x="${S.sx(v)}" y="${p.y1 + 16}"
      text-anchor="middle">${fmt(v, 0)}</text><text class="tick"
      x="${p.x0 - 7}" y="${S.sy(v) + 4}" text-anchor="end">${fmt(v, 0)}</text>`;
  }
  a += `<text class="axis-label" x="${(p.x0 + p.x1) / 2}" y="${p.y1 + 38}"
    text-anchor="middle"><tspan class="sym">x</tspan>₀</text>
    <text class="axis-label" text-anchor="middle"
    transform="translate(16 ${(p.y0 + p.y1) / 2}) rotate(-90)"
    ><tspan class="sym">x</tspan>₁</text>`;
  return a;
}

/** Pointer position in the viewBox units of an SVG. */
function svgPoint(el, e) {
  const pt = el.createSVGPoint();
  pt.x = e.clientX;
  pt.y = e.clientY;
  return pt.matrixTransform(el.getScreenCTM().inverse());
}

/**
 * Route pointer drags on svg to handlers.
 *
 * Args:
 *   h.down (Function): (handle, q, e) -> bool, handle = data-h of the
 *     element under the pointer (or null), q the pointer in viewBox
 *     units; return true to start a drag
 *   h.move (Function): (q, e) during the drag
 *   h.up (Function | undefined): (e) at the end
 */
function dragOn(svg, h) {
  svg.addEventListener("pointerdown", e => {
    const t = e.target.closest("[data-h]");
    if (!h.down(t ? t.dataset.h : null, svgPoint(svg, e), e)) return;
    e.preventDefault();
    svg.setPointerCapture(e.pointerId);
    const move = ev => h.move(svgPoint(svg, ev), ev);
    const up = ev => {
      svg.removeEventListener("pointermove", move);
      svg.removeEventListener("pointerup", up);
      svg.removeEventListener("pointercancel", up);
      if (h.up) h.up(ev);
    };
    svg.addEventListener("pointermove", move);
    svg.addEventListener("pointerup", up);
    svg.addEventListener("pointercancel", up);
  });
}

/** Number with a true minus sign. */
function fmt(x, digits = 2) {
  const a = Math.abs(x).toFixed(digits);
  return (x < 0 && a !== (0).toFixed(digits) ? "−" : "") + a;
}

/** Vector [a, b] with true minus signs. */
function fvec(v, digits = 2) {
  return `[${fmt(v[0], digits)}, ${fmt(v[1], digits)}]`;
}

/** Percentage with one decimal. */
const pct = f => `${(100 * f).toFixed(1)}%`;

/** Readout row: label on the left, value on the right. */
function row(label, val) {
  return `<div class="row"><span>${label}</span><span class="val">${
    val}</span></div>`;
}

/** Single-choice segmented control; onPick(value) on click. */
function segmented(el, items, onPick) {
  el.innerHTML = items.map(([v, label]) =>
    `<button role="radio" data-v="${v}">${label}</button>`).join("");
  for (const b of el.querySelectorAll("button")) {
    b.onclick = () => onPick(b.dataset.v);
  }
}

/** Input text for v: at most 2 decimals, no trailing zeros. */
const inTxt = v => String(Number(v.toFixed(2)));

/** Set an input's value unless the student is typing in it. */
function setIn(el, v) {
  if (document.activeElement !== el) el.value = v;
}

/**
 * Wire number inputs for a symmetric matrix [[a, b], [b, d]].
 *
 * Args:
 *   ids (string[]): the four inputs, row-major; the two off-diagonal
 *     inputs both edit b
 *   get (Function): () -> the current {a, b, d}
 *   set (Function): (M) -> called with the edited matrix on each input
 *   done (Function): () -> called when an input is committed
 *
 * Returns:
 *   (Function): (M) -> show M in the inputs
 */
function matrixInputs(ids, get, set, done) {
  const els = ids.map(id => document.getElementById(id));
  els.forEach((el, i) => {
    el.addEventListener("input", () => {
      const v = parseFloat(el.value);
      if (!Number.isFinite(v)) return;
      const M = { ...get() };
      if (i === 0) M.a = v;
      else if (i === 3) M.d = v;
      else M.b = v;
      set(M);
    });
    el.addEventListener("change", () => { el.blur(); done(); });
  });
  return M => [M.a, M.b, M.b, M.d].forEach((v, i) => setIn(els[i], inTxt(v)));
}

/**
 * Wire two number inputs for a vector; same contract as matrixInputs.
 */
function vectorInputs(ids, get, set, done) {
  const els = ids.map(id => document.getElementById(id));
  els.forEach((el, i) => {
    el.addEventListener("input", () => {
      const v = parseFloat(el.value);
      if (!Number.isFinite(v)) return;
      const u = [...get()];
      u[i] = clamp(v, -7.5, 7.5);
      set(u);
    });
    el.addEventListener("change", () => { el.blur(); done(); });
  });
  return u => u.forEach((v, i) => setIn(els[i], inTxt(v)));
}

/** Pair a range slider with a number input; onSet(v) on either. */
function sliderPair(rangeId, numId, lo, hi, onSet) {
  const r = document.getElementById(rangeId), n = document.getElementById(numId);
  const go = (src, twin) => {
    const v = parseFloat(src.value);
    if (!Number.isFinite(v)) return;
    twin.value = v;
    onSet(clamp(v, lo, hi));
  };
  r.addEventListener("input", () => go(r, n));
  n.addEventListener("input", () => go(n, r));
  n.addEventListener("change", () => n.blur());
  return v => { setIn(r, v); setIn(n, inTxt(v)); };
}

// Footer build stamp: the deployed commit (linked) and build time.
(function () {
  const el = document.getElementById("build");
  if (!el) return;
  const repo = "https://github.com/matthigger/lda";
  if (!BUILD) { el.textContent = "local copy"; return; }
  const when = new Date(BUILD.time).toLocaleString("en-US", {
    dateStyle: "medium", timeStyle: "short" });
  el.innerHTML = `build <a href="${repo}/commit/${BUILD.sha}">${
    BUILD.sha.slice(0, 7)}</a>, ${when}`;
})();
