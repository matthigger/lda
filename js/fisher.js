// Projection tab: two classes of FN samples each, built so class k has
// exactly mean μ_k and covariance Σ (with 1/n), a draggable unit direction
// w through the midpoint m = (μ₀ + μ₁)/2, the scores t_i = wᵀ(x_i − m)
// histogrammed by class beside w's line, and a draggable threshold τ on
// the scores: predict class 1 on μ₁'s side of τ.
//
// Fisher's objective J(w) = (wᵀΔ)² / wᵀΣw, Δ = μ₁ − μ₀: the squared
// projected mean difference over the within-class scatter along w. It
// peaks at w ∝ Σ⁻¹Δ with height J* = ΔᵀΣ⁻¹Δ = D², the squared Mahalanobis
// distance between the means (Fisher 1936).

const FN = 40;
const FMAX = 12;

// "a bit of both": the largest mean difference (Δ = [5, 0], J = 25/3) and
// the smallest within-class scatter (Σ's short axis [1, −1], J = 12.5) each
// lose to w ∝ Σ⁻¹Δ = [3, −2], which reaches J* = 15.
const FPRESETS = [
  ["tilt", "tilted", { mu: [[0, -1], [0.5, 1]], S: { a: 4.6, b: 2.4, d: 1.9 } }],
  ["both", "a bit of both", { mu: [[-2.5, 0], [2.5, 0]], S: { a: 3, b: 2, d: 3 } }],
  ["aligned", "aligned", { mu: [[2, 2], [-2, -2]], S: { a: 2, b: 1, d: 2 } }],
  ["round", "round Σ", { mu: [[2, -2], [2, 2]], S: { a: 1, b: 0, d: 1 } }],
  ["skinny", "skinny", { mu: [[0, 0], [1, 1]], S: { a: 10, b: 0, d: 0.1 } }],
  ["back", "tilted back", { mu: [[0, 0], [2, 3]], S: { a: 3, b: -1, d: 4 } }],
];

const F = {
  mu: [[0, -1], [0.5, 1]],
  S: { a: 4.6, b: 2.4, d: 1.9 },
  phi: 150 * DEG,
  tau: 0,
  seed: 3,
  z: null,
  show: { ell: true, drop: false, side: true },
  spin: null,
  hside: 1,   // which side of w's line the histograms sit on: ±perp(w)
};

/** Fresh white draws for both classes. */
function fDraw() {
  F.z = [0, 1].map(k => whiten(normals(2 * F.seed + k, FN)));
}

const fX = () => F.z.map((z, k) => cloud(z, F.mu[k], F.S));
const fMid = () => mul(add(F.mu[0], F.mu[1]), 0.5);
const fD = () => sub(F.mu[1], F.mu[0]);

function fJ(w) {
  const g = dot(w, fD());
  return g * g / quad(F.S, w);
}
const fJstar = () => quad(inv2(F.S), fD());

/** Angle of Fisher's direction Σ⁻¹(μ₁ − μ₀). */
function fBest() {
  const v = fisherDir(F.mu[0], F.mu[1], F.S);
  return Math.atan2(v[1], v[0]);
}

/** Angle of Σ's short axis: the smallest within-class scatter. */
function fShort() {
  const e = eig2(F.S.a, F.S.b, F.S.d);
  return (isNaN(e.theta) ? 0 : e.theta) + 90 * DEG;
}

/** Angle of μ₁ − μ₀. */
function fNaive() {
  const d = fD();
  return Math.atan2(d[1], d[0]);
}

/**
 * Angle of PCA's v₁ for both classes pooled. With equal class sizes their
 * covariance is exactly Σ + ¼(μ₁ − μ₀)(μ₁ − μ₀)ᵀ.
 */
function fPca() {
  const d = fD();
  const T = mix2(F.S, 1, { a: d[0] * d[0], b: d[0] * d[1], d: d[1] * d[1] },
    0.25);
  const e = eig2(T.a, T.b, T.d);
  return isNaN(e.theta) ? 0 : e.theta;
}

/** +1 if class 1 projects above the midpoint along w, else −1. */
const fSide = w => dot(w, fD()) >= 0 ? 1 : -1;

// ------------------------------------------------------------- drawing

const pPlane = document.getElementById("p-plane");
pPlane.setAttribute("viewBox", `0 0 ${PLANE.w} ${PLANE.h}`);

function fPlane() {
  const X = fX(), w = unit(F.phi), m = fMid(), pw = perp(w), L = 30;
  const sg = fSide(w), b0 = add(m, mul(w, F.tau));
  const t = X.map(Xk => Xk.map(x => dot(w, sub(x, m))));
  let s = axesSvg("p-clip") + `<g clip-path="url(#p-clip)">`;
  if (F.show.side) {
    for (const k of [0, 1]) {
      const o = mul(w, (k ? sg : -sg) * L);
      s += polySvg([add(b0, mul(pw, L)), add(add(b0, mul(pw, L)), o),
        add(add(b0, mul(pw, -L)), o), add(b0, mul(pw, -L))],
        `class="halfplane c${k}"`);
    }
  }
  if (F.show.ell) {
    F.mu.forEach((mu, k) => { s += ellipseSvg(mu, F.S, 2, `class="ell c${k}"`); });
  }
  s += segSvg(add(m, mul(w, -L)), add(m, mul(w, L)), `class="wline"`);
  s += segSvg(add(m, mul(w, -L)), add(m, mul(w, L)), `class="hit" data-h="line"`);
  s += segSvg(add(b0, mul(pw, -L)), add(b0, mul(pw, L)), `class="bound"`);
  s += fHist(X, t, w, m);
  if (F.show.drop) {
    X.forEach((Xk, k) => Xk.forEach((x, i) => {
      s += segSvg(x, add(m, mul(w, t[k][i])), `class="drop c${k}"`);
    }));
  }
  X.forEach((Xk, k) => Xk.forEach((x, i) => {
    s += dotSvg(add(m, mul(w, t[k][i])), 3.2, `class="proj c${k}"`);
  }));
  let miss = 0;
  X.forEach((Xk, k) => Xk.forEach((x, i) => {
    const bad = (sg * (t[k][i] - F.tau) > 0 ? 1 : 0) !== k;
    miss += bad;
    s += dotSvg(x, 5.5, `class="pt c${k}${bad ? " miss" : ""}"`);
  }));
  s += "</g>";
  F.mu.forEach((mu, k) => {
    s += dotSvg(mu, 9, `class="mean c${k}" data-h="m${k}"`);
    s += `<text class="lbl c${k}" x="${S.sx(mu[0]) + 12}"
      y="${S.sy(mu[1]) - 10}"><tspan class="sym">μ</tspan>${
      k ? "₁" : "₀"}</text>`;
  });
  s += arrowSvg(m, add(m, w), "w", 12);
  s += dotSvg(add(m, w), 11, `class="tip" data-h="w"`);
  const o = add(m, mul(w, 1.6));
  s += `<text class="lbl violet" x="${S.sx(o[0]) - 7}" y="${S.sy(o[1]) + 6}"
    ><tspan class="vec">w</tspan></text>`;
  s += `<text class="lbl small ink" x="${S.sx(m[0]) + 8}"
    y="${S.sy(m[1]) + 18}">m</text>`;
  pPlane.innerHTML = s;
  return miss;
}

/**
 * Histograms of the scores t by class, pulled off to the side of w's
 * line: both share one baseline parallel to w, just past the farthest
 * sample on the less crowded side, and the bars grow away from the
 * clouds. The offset is capped so the bars stay on the plot. The curves
 * are the normals N(wᵀ(μ_k − m), wᵀΣw) each class projects to. The
 * threshold τ is a tick on the baseline.
 */
function fHist(X, t, w, m) {
  const H = 2.4, pw = perp(w), all = X[0].concat(X[1]);
  const reach = sg => Math.max(0, ...all.map(x => sg * dot(pw, sub(x, m)))) +
    0.6;
  // Flip sides only for a clear gain, so the histograms don't flicker.
  if (reach(-F.hside) < reach(F.hside) - 0.4) F.hside = -F.hside;
  const n = mul(pw, F.hside);
  let room = Infinity;
  for (const j of [0, 1]) {
    if (Math.abs(n[j]) > 1e-9) {
      room = Math.min(room, ((n[j] > 0 ? 7.7 : -7.7) - m[j]) / n[j]);
    }
  }
  const off = Math.min(reach(F.hside), Math.max(room - H, 0.6));
  const v = quad(F.S, w), c = F.mu.map(mu => dot(w, sub(mu, m)));
  const bw = 0.5, lo = -16, nb = 64;
  const cnt = [0, 1].map(k => {
    const a = new Array(nb).fill(0);
    for (const tk of t[k]) a[clamp(Math.floor((tk - lo) / bw), 0, nb - 1)]++;
    return a;
  });
  const peak = FN * bw / Math.sqrt(2 * Math.PI * v);
  const pdf = (k, u) => peak * Math.exp(-((u - c[k]) ** 2) / (2 * v));
  const hk = H / Math.max(8, ...cnt[0], ...cnt[1], Math.min(peak, FN));
  const at = (u, h) => add(m, add(mul(w, u), mul(n, off + h)));
  const ts = t[0].concat(t[1], [F.tau]);
  const t0 = Math.max(Math.min(...ts) - 0.6, lo);
  const t1 = Math.min(Math.max(...ts) + 0.6, -lo);
  let s = segSvg(at(t0, 0), at(t1, 0), `class="hbase"`);
  for (const k of [0, 1]) {
    cnt[k].forEach((q, b) => {
      if (!q) return;
      const u0 = lo + b * bw, u1 = u0 + bw, h = q * hk;
      s += polySvg([at(u0, 0), at(u1, 0), at(u1, h), at(u0, h)],
        `class="hbar c${k}"`);
    });
  }
  for (const k of [0, 1]) {
    let d = "";
    for (let u = t0; u <= t1 + 1e-9; u += 0.1) {
      d += `${d ? "L" : "M"}${px(at(u, Math.min(pdf(k, u), FN) * hk))}`;
    }
    s += `<path class="curve c${k}" d="${d}"/>`;
  }
  s += segSvg(at(F.tau, -0.4), at(F.tau, H + 0.4), `class="tau"`);
  s += dotSvg(at(F.tau, 0), 8, `class="tauh" data-h="tau"`);
  const q = at(F.tau, H + 0.75);
  s += `<text class="lbl small ink" x="${S.sx(q[0]) - 4}"
    y="${S.sy(q[1]) + 5}"><tspan class="sym">τ</tspan></text>`;
  return s;
}

// --------------------------------------------------------- J(θ) chart

const CH = { w: 440, h: 233, x0: 46, x1: 428, y0: 59, y1: 197 };
const pChart = document.getElementById("p-chart");
pChart.setAttribute("viewBox", `0 0 ${CH.w} ${CH.h}`);

/** Angle in degrees folded into [0, 180): w and −w score the same J. */
const fold180 = a => ((a / DEG) % 180 + 180) % 180;

function fChart() {
  const js = fJstar(), top = js * 1.15;
  const xs = th => CH.x0 + th / 180 * (CH.x1 - CH.x0);
  const ys = j => CH.y1 - j / top * (CH.y1 - CH.y0);
  let s = `<line class="grid zero" x1="${CH.x0}" x2="${CH.x1}"
    y1="${CH.y1}" y2="${CH.y1}"/><line class="total" x1="${CH.x0}"
    x2="${CH.x1}" y1="${ys(js)}" y2="${ys(js)}"/>`;
  for (const th of [0, 45, 90, 135, 180]) {
    s += `<line class="grid" x1="${xs(th)}" x2="${xs(th)}" y1="${CH.y0}"
      y2="${CH.y1}"/><text class="tick" x="${xs(th)}" y="${CH.y1 + 14}"
      text-anchor="middle">${th}°</text>`;
  }
  s += `<text class="tick" x="${CH.x0 - 6}" y="${ys(js) + 4}"
    text-anchor="end">J*</text><text class="tick" x="${CH.x0 - 6}"
    y="${CH.y1 + 4}" text-anchor="end">0</text>
    <text class="axis-label sm" x="${(CH.x0 + CH.x1) / 2}" y="${CH.h - 4}"
    text-anchor="middle">angle of <tspan class="vec">w</tspan></text>`;
  let d = "";
  for (let th = 0; th <= 180; th += 0.5) {
    d += `${d ? "L" : "M"}${xs(th).toFixed(1)},${ys(fJ(unit(th * DEG))).toFixed(1)}`;
  }
  s += `<path class="c-J" d="${d}"/>`;
  const marks = [[fBest(), "best", "Σ⁻¹Δ"], [fNaive(), "naive", "Δ"],
    [fShort(), "short", "Σ short axis"], [fPca(), "pca", "v₁ (PCA)"]].map(([a, cls, lab]) =>
    ({ x: xs(fold180(a)), y: ys(fJ(unit(a))), cls, lab }))
    .sort((p, q) => p.x - q.x)
    .reduce((acc, mk) => {
      // Coincident markers share one label: "a = b".
      const prev = acc[acc.length - 1];
      if (prev && Math.abs(mk.x - prev.x) < 1.5) prev.lab += ` = ${mk.lab}`;
      else acc.push(mk);
      return acc;
    }, []);
  // Labels sit in rows in the band above the curve (J* is drawn below
  // it), so they never cover a marker; a faint leader drops to each one.
  let lastX = -1e9, row = 0;
  for (const mk of marks) {
    row = mk.x - lastX < 70 ? row + 1 : 0;
    lastX = mk.x;
    const ly = CH.y0 - 2 - 15 * row, hw = 3.4 * mk.lab.length;
    const lx = clamp(mk.x, hw + 2, CH.w - hw - 2);
    s += `<line class="grid" x1="${mk.x}" x2="${mk.x}" y1="${ly + 3}"
      y2="${mk.y - 10}"/><path class="mk ${mk.cls}"
      d="M${mk.x},${mk.y - 1}l-5,-9h10z"/><text class="zl ${mk.cls}"
      x="${lx}" y="${ly}" text-anchor="middle">${mk.lab}</text>`;
  }
  const xn = xs(fold180(F.phi));
  s += `<line class="c-now" x1="${xn}" x2="${xn}" y1="${CH.y0 - 8}"
    y2="${CH.y1}"/><circle class="kdot" cx="${xn}"
    cy="${ys(fJ(unit(F.phi)))}" r="5"/>`;
  pChart.innerHTML = s;
}

dragOn(pChart, {
  down(h, q) { fStopSpin(); this.move(q); return true; },
  move(q) {
    F.phi = clamp((q.x - CH.x0) / (CH.x1 - CH.x0), 0, 1) * 180 * DEG;
    fRender();
  },
});

// ------------------------------------------------------------ readouts

const fWarn = document.getElementById("p-warn");
const fSetMu = [0, 1].map(k => vectorInputs([`p-m${k}0`, `p-m${k}1`],
  () => F.mu[k], u => { F.mu[k] = u; fRender(); }, fRender));
const fSetS = matrixInputs(["p-s00", "p-s01", "p-s10", "p-s11"], () => F.S,
  M => {
    const why = covProblem(M, "&Sigma;", FMAX);
    fWarn.hidden = !why;
    fWarn.innerHTML = why;
    if (!why) { F.S = M; fRender(); }
  }, fRender);

function fFields() {
  fSetMu.forEach((f, k) => f(F.mu[k]));
  fSetS(F.S);
  for (const b of document.querySelectorAll("#p-presets button")) {
    const P = FPRESETS.find(p => p[0] === b.dataset.v)[2];
    b.setAttribute("aria-checked", matDiff(P.S, F.S) < 1e-6 &&
      norm(sub(P.mu[0], F.mu[0])) + norm(sub(P.mu[1], F.mu[1])) < 1e-6);
  }
}

function fRead(miss) {
  const w = unit(F.phi), gap = dot(w, fD()), v = quad(F.S, w);
  const J = gap * gap / v, js = fJstar();
  document.getElementById("p-read").innerHTML =
    row("<b>w</b> (unit)", `${fvec(w)} at ${fmt(((F.phi / DEG) % 360 + 360) %
      360, 0)}°`) +
    row("projected mean difference <b>w</b><sup>T</sup>&Delta;",
      `<span class="gap-t">${fmt(gap, 3)}</span>`) +
    row("within-class scatter <b>w</b><sup>T</sup>&Sigma;<b>w</b>",
      `<span class="spr-t">${fmt(v, 3)}</span>`) +
    row("<i>J</i>(<b>w</b>) = (<b>w</b><sup>T</sup>&Delta;)<sup>2</sup> / <b>w</b><sup>T</sup>&Sigma;<b>w</b>",
      `<span class="kept-t">${fmt(J, 3)}</span>`) +
    row("best: <i>J</i>* = &Delta;<sup>T</sup>&Sigma;<sup>&minus;1</sup>&Delta; = <i>D</i><sup>2</sup>",
      fmt(js, 3));
  const f = js > 0 ? clamp(J / js, 0, 1) : 1;
  document.getElementById("p-bar").innerHTML =
    `<div class="bar2"><span class="k" style="width:${(100 * f).toFixed(1)}%"
    ></span></div><div class="muted"><b>w</b> reaches ${pct(f)} of the best
    <i>J</i></div>`;
  document.getElementById("p-tread").innerHTML =
    row("threshold <i>&tau;</i> on <i>t</i> = <b>w</b><sup>T</sup>(<b>x</b> &minus; <b>m</b>)",
      fmt(F.tau)) +
    row("training errors", `<span class="lost-t">${miss}</span> of ${2 * FN}
      (${pct(miss / (2 * FN))})`);
}

function fRender() {
  const miss = fPlane();
  fChart();
  fRead(miss);
  fFields();
}

// ------------------------------------------------------------- editing

segmented(document.getElementById("p-presets"),
  FPRESETS.map(p => [p[0], p[1]]), v => {
    const P = FPRESETS.find(p => p[0] === v)[2];
    F.mu = P.mu.map(u => [...u]);
    F.S = { ...P.S };
    fWarn.hidden = true;
    fRender();
  });

document.getElementById("p-resample").onclick = () => {
  F.seed++;
  fDraw();
  fRender();
};

for (const [id, key] of [["p-ell", "ell"], ["p-drop", "drop"],
  ["p-side", "side"]]) {
  const el = document.getElementById(id);
  el.checked = F.show[key];
  el.onchange = () => { F.show[key] = el.checked; fRender(); };
}

function fStopSpin() {
  if (!F.spin) return;
  cancelAnimationFrame(F.spin);
  F.spin = null;
  document.getElementById("p-spin").textContent = "Sweep w";
}

for (const [id, ang] of [["p-wd", fNaive], ["p-ws", fShort], ["p-wf", fBest],
  ["p-wp", fPca]]) {
  document.getElementById(id).onclick = () => {
    fStopSpin();
    F.phi = ang();
    fRender();
  };
}
document.getElementById("p-tau0").onclick = () => { F.tau = 0; fRender(); };
document.getElementById("p-spin").onclick = () => {
  if (F.spin) { fStopSpin(); return; }
  document.getElementById("p-spin").textContent = "Stop";
  let last = performance.now();
  const step = now => {
    F.phi += (now - last) / 1000 * 30 * DEG;
    last = now;
    fRender();
    F.spin = requestAnimationFrame(step);
  };
  F.spin = requestAnimationFrame(step);
};

let fDrag = null;
dragOn(pPlane, {
  down(h, q) {
    fStopSpin();
    fDrag = ["m0", "m1", "tau"].includes(h) ? h : "w";
    if (fDrag === "w") this.move(q);
    return true;
  },
  move(q) {
    const p = [S.ix(q.x), S.iy(q.y)];
    if (fDrag === "w") {
      const r = sub(p, fMid());
      if (norm(r) < 0.15) return;
      F.phi = Math.atan2(r[1], r[0]);
    } else if (fDrag === "tau") {
      F.tau = clamp(dot(unit(F.phi), sub(p, fMid())), -14, 14);
    } else {
      F.mu[+fDrag[1]] = p.map(v => clamp(v, -7.5, 7.5));
    }
    fRender();
  },
  up() { fDrag = null; },
});

fDraw();

/** Show the projection tab (called by main.js). */
function fisherShow() { fRender(); }
