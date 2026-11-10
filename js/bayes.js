// Bayes boundary tab: the student sets two true Gaussians (μ_k, Σ_k) and
// the prior π₁; n training samples are drawn from them (n_k = π_k n, at
// least 3 per class). LDA (one pooled Σ̂) and QDA (a Σ̂_k per class) are
// fit to the training samples by maximum likelihood, and each boundary
// g(x) = 0 is drawn: the true Bayes boundary, LDA's line and QDA's conic.
// Test error is over BT fresh draws per class, weighted by the true
// priors. The shading is the true posterior P(y = 1 | x).
//
// The draws are fixed standard normals z mapped by x = μ_k + Σ_k^½ z, so
// editing a mean or covariance moves the samples smoothly; New samples
// redraws the training z.

const BN_MAX = 400;
const BT = 5000;
const BMAX = 12;
const BG = 120;   // shading / contour grid cells per side

const BPRESETS = [
  ["shared", "shared Σ", { mu: [[-2, -1], [2, 1.5]], S: [{ a: 3, b: 1.2, d: 2 },
    { a: 3, b: 1.2, d: 2 }], shared: true }],
  ["rot", "rotated", { mu: [[-2, 0], [2, 0]], S: [{ a: 4, b: 1.8, d: 2 },
    { a: 2, b: -1.8, d: 4 }], shared: false }],
  ["wide", "one wider", { mu: [[-1.5, 0], [1.5, 0.5]], S: [
    { a: 0.8, b: 0, d: 0.8 }, { a: 6, b: 1, d: 5 }], shared: false }],
  ["cross", "crossed", { mu: [[0, 0], [0, 0]], S: [{ a: 6, b: 0, d: 0.5 },
    { a: 0.5, b: 0, d: 6 }], shared: false }],
  ["ring", "same mean", { mu: [[0, 0], [0, 0]], S: [{ a: 1, b: 0, d: 1 },
    { a: 7, b: 0, d: 7 }], shared: false }],
];

const B = {
  mu: [[-2, -1], [2, 1.5]],
  S: [{ a: 3, b: 1.2, d: 2 }, { a: 3, b: 1.2, d: 2 }],
  shared: true,
  pi1: 0.5,
  n: 40,
  seed: 1,
  show: { shade: true, ell: true, bayes: true, lda: true, qda: true },
  zTr: null,
  zTe: [normals(9001, BT), normals(9002, BT)],
  hover: null,
  truth: null,
};

function bDraw() {
  B.zTr = [0, 1].map(k => normals(100 + 2 * B.seed + k, BN_MAX));
}

/** Σ₁ in use: Σ₀ when the classes share one covariance. */
const bS = k => (B.shared ? B.S[0] : B.S[k]);

/** Training samples per class. */
function bCounts() {
  return [Math.max(3, Math.round((1 - B.pi1) * B.n)),
    Math.max(3, Math.round(B.pi1 * B.n))];
}

// ------------------------------------------------------------- drawing

const bPlane = document.getElementById("b-plane");
bPlane.setAttribute("viewBox", `0 0 ${PLANE.w} ${PLANE.h}`);
const bCanvas = document.createElement("canvas");
bCanvas.width = bCanvas.height = BG;

/** Data URL of the true posterior, blue (class 0) to orange (class 1). */
function bShade(md) {
  const cx = bCanvas.getContext("2d"), im = cx.createImageData(BG, BG);
  const lo = PLANE.dom[0], h = (PLANE.dom[1] - lo) / BG;
  const C = [[37, 99, 235], [234, 88, 12]];
  for (let j = 0; j < BG; j++) {
    for (let i = 0; i < BG; i++) {
      const p = posterior(md, [lo + (i + 0.5) * h, PLANE.dom[1] - (j + 0.5) * h]);
      const c = C[p >= 0.5 ? 1 : 0], o = 4 * (j * BG + i);
      im.data[o] = c[0];
      im.data[o + 1] = c[1];
      im.data[o + 2] = c[2];
      im.data[o + 3] = Math.round(255 * 0.26 * Math.abs(2 * p - 1));
    }
  }
  cx.putImageData(im, 0, 0);
  return bCanvas.toDataURL();
}

/**
 * SVG path of the zero set of f on a BG x BG grid over the plot
 * (marching squares, linear interpolation along cell edges; a saddle
 * cell is resolved by the sign of its center). Cell segments are chained
 * through the grid edges they share into polylines, so dashes run on.
 */
function zeroPath(f) {
  const lo = PLANE.dom[0], h = (PLANE.dom[1] - lo) / BG, M = BG + 1;
  const v = new Float64Array(M * M);
  for (let j = 0; j < M; j++) {
    for (let i = 0; i < M; i++) v[j * M + i] = f([lo + i * h, lo + j * h]);
  }
  const P = (i, j) => [lo + i * h, lo + j * h];
  const cut = (pa, va, pb, vb) => add(pa, mul(sub(pb, pa), va / (va - vb)));
  const at = new Map(), adj = new Map(), segs = [];
  for (let j = 0; j < BG; j++) {
    for (let i = 0; i < BG; i++) {
      // Corners 00, 10, 11, 01 counterclockwise from the bottom left.
      const a = v[j * M + i], b = v[j * M + i + 1];
      const c = v[(j + 1) * M + i + 1], e = v[(j + 1) * M + i];
      const sa = a >= 0, sb = b >= 0, sc = c >= 0, se = e >= 0;
      if (sa === sb && sb === sc && sc === se) continue;
      // Edges: 0 bottom, 1 right, 2 top, 3 left.
      const E = [
        sa !== sb ? cut(P(i, j), a, P(i + 1, j), b) : null,
        sb !== sc ? cut(P(i + 1, j), b, P(i + 1, j + 1), c) : null,
        se !== sc ? cut(P(i, j + 1), e, P(i + 1, j + 1), c) : null,
        sa !== se ? cut(P(i, j), a, P(i, j + 1), e) : null,
      ];
      const on = [0, 1, 2, 3].filter(k => E[k]);
      const pairs = on.length === 2 ? [on] :
        ((a + b + c + e) / 4 >= 0) === sa ? [[0, 1], [2, 3]] : [[0, 3], [1, 2]];
      const id = [`h${i},${j}`, `v${i + 1},${j}`, `h${i},${j + 1}`, `v${i},${j}`];
      for (const [p, q] of pairs) {
        at.set(id[p], E[p]);
        at.set(id[q], E[q]);
        for (const k of [id[p], id[q]]) {
          if (!adj.has(k)) adj.set(k, []);
          adj.get(k).push(segs.length);
        }
        segs.push([id[p], id[q]]);
      }
    }
  }
  const used = new Uint8Array(segs.length);
  /** Walk from edge k away from segment s until the chain ends. */
  const walk = (k, s) => {
    const out = [];
    for (;;) {
      const nx = adj.get(k).find(t => t !== s && !used[t]);
      if (nx === undefined) return out;
      used[nx] = 1;
      k = segs[nx][0] === k ? segs[nx][1] : segs[nx][0];
      s = nx;
      out.push(k);
    }
  };
  let d = "";
  segs.forEach((sg, s) => {
    if (used[s]) return;
    used[s] = 1;
    const ids = walk(sg[0], s).reverse().concat(sg, walk(sg[1], s));
    d += ids.map((k, t) => `${t ? "L" : "M"}${px(at.get(k))}`).join("");
  });
  return d;
}

function bPlaneRender(md, fits, Xtr) {
  const p = PLANE;
  let s = axesSvg("b-clip") + `<g clip-path="url(#b-clip)">`;
  if (B.show.shade) {
    s += `<image href="${bShade(md)}" x="${p.x0}" y="${p.y0}"
      width="${p.x1 - p.x0}" height="${p.y1 - p.y0}"
      preserveAspectRatio="none"/>`;
  }
  if (B.show.ell) {
    B.mu.forEach((mu, k) => { s += ellipseSvg(mu, bS(k), 2, `class="ell c${k}"`); });
  }
  Xtr.forEach((Xk, k) => Xk.forEach(x => {
    s += dotSvg(x, 4.8, `class="pt c${k}"`);
  }));
  for (const [key, m] of [["qda", fits.qda], ["lda", fits.lda],
    ["bayes", md]]) {
    if (B.show[key]) {
      s += `<path class="bnd ${key}" d="${zeroPath(x => logOdds(m, x))}"/>`;
    }
  }
  s += `</g><g id="b-hov"></g>`;
  B.mu.forEach((mu, k) => {
    s += dotSvg(mu, 9, `class="mean c${k}" data-h="m${k}"`);
    s += `<text class="lbl c${k}" x="${S.sx(mu[0]) + 12}"
      y="${S.sy(mu[1]) - 10 + 22 * k * (norm(sub(B.mu[0], B.mu[1])) < 0.8)}"
      ><tspan class="sym">μ</tspan>${k ? "₁" : "₀"}</text>`;
  });
  bPlane.innerHTML = s;
}

// ------------------------------------------------------------ readouts

/** Pointer marker: the point and dashed lines to both means. */
function bHoverRender() {
  const g = document.getElementById("b-hov"), el = document.getElementById("b-at");
  const md = B.truth, x = B.hover;
  if (!x) {
    if (g) g.innerHTML = "";
    el.innerHTML = `<p class="muted">Move the pointer over the plot (or tap
      it) to see the log-odds at that point.</p>`;
    return;
  }
  g.innerHTML = B.mu.map((mu, k) => segSvg(x, mu, `class="to c${k}"`)).join("") +
    dotSvg(x, 6, `class="here"`);
  const d0 = maha2(md, 0, x), d1 = maha2(md, 1, x);
  const ldet = -0.5 * (md.ld[1] - md.ld[0]), gx = logOdds(md, x);
  const post = posterior(md, x);
  el.innerHTML =
    row("<b>x</b>", fvec(x)) +
    row("<i>d</i><sub>0</sub><sup>2</sup> = (<b>x</b> &minus; <b>&mu;</b><sub>0</sub>)<sup>T</sup>&Sigma;<sub>0</sub><sup>&minus;1</sup>(<b>x</b> &minus; <b>&mu;</b><sub>0</sub>)",
      `<span class="c0-t">${fmt(d0)}</span>`) +
    row("<i>d</i><sub>1</sub><sup>2</sup> = (<b>x</b> &minus; <b>&mu;</b><sub>1</sub>)<sup>T</sup>&Sigma;<sub>1</sub><sup>&minus;1</sup>(<b>x</b> &minus; <b>&mu;</b><sub>1</sub>)",
      `<span class="c1-t">${fmt(d1)}</span>`) +
    row("&minus;&frac12; log(|&Sigma;<sub>1</sub>| / |&Sigma;<sub>0</sub>|)", fmt(ldet)) +
    row("log(<i>&pi;</i><sub>1</sub> / <i>&pi;</i><sub>0</sub>)", fmt(md.lp)) +
    row("<i>g</i>(<b>x</b>) = &frac12;<i>d</i><sub>0</sub><sup>2</sup> &minus; &frac12;<i>d</i><sub>1</sub><sup>2</sup> + &hellip;",
      `<b>${fmt(gx)}</b>`) +
    row("<i>P</i>(<i>y</i> = 1 | <b>x</b>) = 1 / (1 + e<sup>&minus;<i>g</i></sup>)",
      fmt(post, 3)) +
    row("Bayes predicts", `<span class="c${gx >= 0 ? 1 : 0}-t">class ${
      gx >= 0 ? 1 : 0}</span>`);
}

function bReadRender(md, fits, Xtr, Xte) {
  const nk = bCounts();
  const cell = v => `<td>${pct(v)}</td>`;
  const line = (key, name, m) => `<tr><td><i class="ln ${key}"></i>${name}</td>${
    cell(errRate(m, Xtr))}${cell(errRate(m, Xte, md.pi))}</tr>`;
  document.getElementById("b-errs").innerHTML =
    `<table class="errs"><tr><th></th><th>train</th><th>test</th></tr>${
    line("bayes", "Bayes (true Gaussians)", md)}${
    line("lda", "LDA fit", fits.lda)}${line("qda", "QDA fit", fits.qda)}
    </table><p class="muted">Training: ${nk[0]} + ${nk[1]} samples. Test:
    ${BT} fresh draws per class, weighted by <i>&pi;</i><sub>0</sub>,
    <i>&pi;</i><sub>1</sub>. No classifier beats Bayes on test, up to the
    noise of the draws.</p>`;
  const fl = linearForm(fits.lda);
  let r = "";
  if (B.shared) {
    const tl = linearForm(md);
    r += row("true <b>w</b> = &Sigma;<sup>&minus;1</sup>(<b>&mu;</b><sub>1</sub> &minus; <b>&mu;</b><sub>0</sub>), <i>b</i>",
      `${fvec(tl.w)}, ${fmt(tl.b)}`);
  }
  r += row("LDA fit <b>w&#770;</b>, <i>b&#770;</i>", `${fvec(fl.w)}, ${fmt(fl.b)}`);
  document.getElementById("b-lin").innerHTML = r;
  document.getElementById("b-lin-note").hidden = B.shared;
}

// ------------------------------------------------------------- editing

const bWarn = document.getElementById("b-warn");

/** Try Σ_k = M; warn and keep the old one if it is not a covariance. */
function bTrySigma(k, M) {
  const why = covProblem(M, `&Sigma;<sub>${k}</sub>`, BMAX);
  bWarn.hidden = !why;
  bWarn.innerHTML = why;
  if (why) return;
  B.S[k] = M;
  if (B.shared) B.S[1 - k] = { ...M };
  bRender();
}

const bSetMu = [0, 1].map(k => vectorInputs([`b-m${k}0`, `b-m${k}1`],
  () => B.mu[k], u => { B.mu[k] = u; bRender(); }, bRender));
const bSetS = [0, 1].map(k => matrixInputs(
  [`b-s${k}00`, `b-s${k}01`, `b-s${k}10`, `b-s${k}11`], () => B.S[k],
  M => bTrySigma(k, M), bRender));
const bSetPi = sliderPair("b-pir", "b-pi", 0.1, 0.9, v => { B.pi1 = v; bRender(); });
const bSetN = sliderPair("b-nr", "b-n", 10, BN_MAX, v => {
  B.n = Math.round(v);
  bRender();
});

function bFields() {
  bSetMu.forEach((f, k) => f(B.mu[k]));
  bSetS.forEach((f, k) => f(bS(k)));
  bSetPi(B.pi1);
  bSetN(B.n);
  document.getElementById("b-share").checked = B.shared;
  for (const id of ["b-s100", "b-s101", "b-s110", "b-s111"]) {
    document.getElementById(id).disabled = B.shared;
  }
  for (const b of document.querySelectorAll("#b-presets button")) {
    const P = BPRESETS.find(p => p[0] === b.dataset.v)[2];
    b.setAttribute("aria-checked", P.shared === B.shared &&
      [0, 1].every(k => matDiff(P.S[k], bS(k)) < 1e-6 &&
        norm(sub(P.mu[k], B.mu[k])) < 1e-6));
  }
}

function bRender() {
  const S0 = bS(0), S1 = bS(1);
  const md = model(B.mu[0], B.mu[1], S0, S1, B.pi1);
  const nk = bCounts();
  const Xtr = [0, 1].map(k => cloud(B.zTr[k].slice(0, nk[k]), B.mu[k], bS(k)));
  const Xte = [0, 1].map(k => cloud(B.zTe[k], B.mu[k], bS(k)));
  const fits = { lda: fitModel(Xtr[0], Xtr[1], true),
    qda: fitModel(Xtr[0], Xtr[1], false) };
  B.truth = md;
  bPlaneRender(md, fits, Xtr);
  bReadRender(md, fits, Xtr, Xte);
  bHoverRender();
  bFields();
}

segmented(document.getElementById("b-presets"),
  BPRESETS.map(p => [p[0], p[1]]), v => {
    const P = BPRESETS.find(p => p[0] === v)[2];
    B.mu = P.mu.map(u => [...u]);
    B.S = P.S.map(M => ({ ...M }));
    B.shared = P.shared;
    bWarn.hidden = true;
    bRender();
  });

document.getElementById("b-share").onchange = e => {
  B.shared = e.target.checked;
  if (B.shared) B.S[1] = { ...B.S[0] };
  bRender();
};

document.getElementById("b-resample").onclick = () => {
  B.seed++;
  bDraw();
  bRender();
};

for (const key of Object.keys(B.show)) {
  const el = document.getElementById(`b-show-${key}`);
  el.checked = B.show[key];
  el.onchange = () => { B.show[key] = el.checked; bRender(); };
}

/** Pointer in data units, or null off the plot. */
function bAt(q) {
  const p = [S.ix(q.x), S.iy(q.y)];
  return p.every(v => v >= PLANE.dom[0] && v <= PLANE.dom[1]) ? p : null;
}

let bDrag = null;
dragOn(bPlane, {
  down(h, q) {
    if (h !== "m0" && h !== "m1") {
      B.hover = bAt(q);
      bHoverRender();
      return false;
    }
    bDrag = +h[1];
    return true;
  },
  move(q) {
    const p = [S.ix(q.x), S.iy(q.y)];
    B.mu[bDrag] = p.map(v => clamp(v, -7.5, 7.5));
    B.hover = null;
    bRender();
  },
  up() { bDrag = null; },
});
bPlane.addEventListener("pointermove", e => {
  if (bDrag !== null) return;
  B.hover = bAt(svgPoint(bPlane, e));
  bHoverRender();
});
// A tap leaves its readout up; a mouse leaving the plot clears it.
bPlane.addEventListener("pointerleave", e => {
  if (bDrag !== null || e.pointerType === "touch") return;
  B.hover = null;
  bHoverRender();
});

bDraw();

/** Show the Bayes boundary tab (called by main.js). */
function bayesShow() { bRender(); }
