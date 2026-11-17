// 3D view of the Bayes boundary tab: the plane of x is the floor and a
// surface rises over it, under an orthographic orbit camera (drag to
// orbit, wheel to zoom). The height is one of
//   dens: P(y = k) p(x | y = k), one surface per class;
//   log:  its ln, −½D_k² + const, an upside-down bowl per class;
//   g:    the log-odds g(x), their difference, with a gray sheet at 0.
// Bayes predicts the class whose surface is on top, so the boundary
// g = 0 is where the two surfaces cross (or where g crosses its sheet).
// The surfaces come from the true Gaussians or from the LDA / QDA fit.
// The floor carries the 2D view's shading, samples, ellipses and
// boundaries; the crossing is also drawn lifted onto the surfaces.
//
// The mesh is painted back to front by quad depth, so the lifted
// crossing, drawn last, shows through any surface in front of it.

const M3 = 30;
const C3 = [[37, 99, 235], [234, 88, 12]];
// World height of the highest surface point (the floor is [-1, 1]^2).
const ZTOP = 0.85;
// g's height scale is at least ±G_LO, so a nearly flat g stays flat.
const G_LO = 2;

/** World (X, Y, Z) to screen { x, y, depth }; depth grows away. */
function b3Proj([X, Y, Z]) {
  const { theta, phi, zoom } = B.cam, s = 232 * zoom;
  const xr = X * Math.cos(theta) - Y * Math.sin(theta);
  const yr = X * Math.sin(theta) + Y * Math.cos(theta);
  return {
    x: PLANE.w / 2 + s * xr,
    y: PLANE.h * 0.6 - s * (Z * Math.cos(phi) + yr * Math.sin(phi)),
    depth: yr * Math.cos(phi) - Z * Math.sin(phi),
  };
}

/** Data point x lifted to height Z, in world units. */
const b3W = (x, Z = 0) => [x[0] / PLANE.dom[1], x[1] / PLANE.dom[1], Z];

/** "x,y" on screen of a world point. */
function b3xy(w) {
  const r = b3Proj(w);
  return `${r.x.toFixed(1)},${r.y.toFixed(1)}`;
}

/** SVG polyline through world points. */
const b3Line = (ws, attrs) =>
  `<polyline points="${ws.map(b3xy).join(" ")}" ${attrs}/>`;

// Light from the viewer's upper left; a quad's shade is 0.7 + 0.3 cos.
const LIGHT3 = (() => {
  const l = [-0.4, -0.6, 1], n = Math.hypot(...l);
  return l.map(c => c / n);
})();

/** Lambert light factor of the world quad a, b, c, d. */
function b3Light(a, b, c, d) {
  const p = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const q = [d[0] - b[0], d[1] - b[1], d[2] - b[2]];
  const n = [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2],
    p[0] * q[1] - p[1] * q[0]];
  const cos = Math.abs(n[0] * LIGHT3[0] + n[1] * LIGHT3[1] + n[2] * LIGHT3[2])
    / (Math.hypot(...n) || 1);
  return 0.7 + 0.3 * cos;
}

/**
 * The surfaces for the chosen height over model m.
 *
 * Returns:
 *   (object): fs (Function[]), one height function per surface (two
 *     for dens / log, one for g); z (Function), height value to world
 *     Z; zero (number | null), world Z of g's zero sheet
 */
function b3Surf(m) {
  const lo = PLANE.dom[0], h = (PLANE.dom[1] - lo) / M3;
  /** [min, max] of f over the mesh nodes. */
  const range = f => {
    let a = Infinity, b = -Infinity;
    for (let j = 0; j <= M3; j++) {
      for (let i = 0; i <= M3; i++) {
        const v = f([lo + i * h, lo + j * h]);
        a = Math.min(a, v);
        b = Math.max(b, v);
      }
    }
    return [a, b];
  };
  if (B.h3 === "g") {
    const G = Math.max(range(x => Math.abs(logOdds(m, x)))[1], G_LO);
    return { fs: [x => logOdds(m, x)], zero: ZTOP / 2,
      z: v => ZTOP / 2 * (1 + v / G) };
  }
  const lj = [0, 1].map(k => x => logJoint(m, k, x));
  // The upper envelope (the class Bayes picks) spans the full height;
  // the lower surface may dip under the floor, where it is cut off.
  const [bot, top] = range(x => Math.max(lj[0](x), lj[1](x)));
  if (B.h3 === "dens") {
    // Heights relative to the top, so the tallest point is 1.
    return { fs: lj.map(f => x => Math.exp(f(x) - top)), zero: null,
      z: v => ZTOP * Math.min(v, 1) };
  }
  return { fs: lj, zero: null,
    z: v => ZTOP * Math.max(v - bot, 0) / (top - bot) };
}

/** Name of the height, for the vertical axis. */
function b3ZName() {
  return { dens: "P(y = k) p(x | y = k)", log: "ln P(y = k) p(x | y = k)",
    g: "g(x)" }[B.h3];
}

/** The plot's caption under the 2D or 3D view. */
function b3Hint() {
  if (B.view !== "3d") {
    return `Drag the means <b>&mu;</b><sub>0</sub> and
      <b>&mu;</b><sub>1</sub>. Hover (or tap) anywhere for the log-odds
      there. Shown: the training samples and the true 2-sd ellipses.`;
  }
  const from = { truth: "the true Gaussians", lda: "the LDA fit (one pooled "
    + "&Sigma;&#770;)", qda: "the QDA fit (a &Sigma;&#770;<sub>k</sub> per "
    + "class)" }[B.m3];
  const what = {
    dens: `Blue is <i>P</i>(<i>y</i> = 0) <i>p</i>(<b>x</b> | <i>y</i> =
      0), orange the same for class 1, each faint where the other is on
      top. Bayes predicts the class on top, so the boundary is where they
      cross (black).`,
    log: `Each surface is ln <i>P</i>(<i>y</i> = <i>k</i>)
      <i>p</i>(<b>x</b> | <i>y</i> = <i>k</i>) = &minus;&frac12;<i>D<sub>k</sub></i><sup>2</sup>
      + const, an upside-down bowl shaped by &Sigma;<sub><i>k</i></sub>.
      The boundary is where the bowls cross (black).`,
    g: `The surface is the log-odds <i>g</i>(<b>x</b>), the difference of
      the two ln surfaces, orange where <i>g</i> &gt; 0; the gray sheet is
      <i>g</i> = 0. The boundary is where they meet (black).`,
  }[B.h3];
  return `${what} Surfaces from ${from}; the floor is the 2D view. Drag
    to orbit, scroll to zoom.`;
}

/** Draw the 3D view from one render's results (see bRender). */
function b3Render(last) {
  const { md, Xtr, models, lines, shade } = last;
  const m = models[B.m3 === "truth" ? "bayes" : B.m3];
  const sf = b3Surf(m);
  const lo = PLANE.dom[0], h = (PLANE.dom[1] - lo) / M3, N1 = M3 + 1;
  const node = (i, j) => [lo + i * h, lo + j * h];
  let s = "";

  // Floor: border, grid, shading, ellipses, boundaries, samples, means.
  const D = PLANE.dom[1];
  s += `<polygon class="floor3" points="${[[-D, -D], [D, -D], [D, D], [-D, D]]
    .map(x => b3xy(b3W(x))).join(" ")}"/>`;
  for (let v = lo + 2; v < D; v += 2) {
    s += b3Line([b3W([v, lo]), b3W([v, D])], `class="grid"`);
    s += b3Line([b3W([lo, v]), b3W([D, v])], `class="grid"`);
  }
  if (shade) {
    // The image's unit square maps affinely onto the floor: (0, 0) at
    // the top-left corner (lo, D), u along x₀, v down along x₁.
    const o = b3Proj(b3W([lo, D])), u = b3Proj(b3W([D, D]));
    const v = b3Proj(b3W([lo, lo]));
    s += `<image href="${shade}" width="1" height="1"
      preserveAspectRatio="none" transform="matrix(${u.x - o.x} ${u.y - o.y}
      ${v.x - o.x} ${v.y - o.y} ${o.x} ${o.y})"/>`;
  }
  if (B.show.ell) {
    B.mu.forEach((mu, k) => {
      s += b3Line(ellipsePts(mu, bS(k), 2).map(x => b3W(x)), `class="ell c${k}"`);
    });
  }
  for (const key of ["qda", "lda", "bayes"]) {
    if (!B.show[key]) continue;
    for (const l of lines[key]) s += b3Line(l.map(x => b3W(x)), `class="bnd ${key}"`);
  }
  Xtr.forEach((Xk, k) => Xk.forEach(x => {
    const p = b3Proj(b3W(x));
    s += `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="3"
      class="pt c${k}"/>`;
  }));
  B.mu.forEach((mu, k) => {
    const p = b3Proj(b3W(mu));
    s += `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="6"
      class="mean c${k}"/>`;
  });
  for (const [x, lab] of [[[0, lo * 1.16], "x₀"], [[lo * 1.16, 0], "x₁"]]) {
    const p = b3Proj(b3W(x));
    s += `<text class="axis-label" x="${p.x.toFixed(1)}"
      y="${(p.y + 5).toFixed(1)}" text-anchor="middle"><tspan
      class="sym">x</tspan>${lab.slice(1)}</text>`;
  }
  // Height axis at the floor corner farthest from the camera.
  const far = [[lo, lo], [D, lo], [D, D], [lo, D]]
    .map(x => ({ x, d: b3Proj(b3W(x)).depth })).sort((a, b) => b.d - a.d)[0].x;
  const zt = b3Proj(b3W(far, ZTOP + 0.06));
  s += b3Line([b3W(far), b3W(far, ZTOP + 0.06)], `class="zaxis"`) +
    `<text class="axis-label sm" x="${zt.x.toFixed(1)}"
    y="${(zt.y - 8).toFixed(1)}" text-anchor="middle">${b3ZName()}</text>`;

  // Mesh: every surface quad (and g's zero sheet), back to front.
  const vals = sf.fs.map(f => {
    const a = new Float64Array(N1 * N1);
    for (let j = 0; j <= M3; j++) {
      for (let i = 0; i <= M3; i++) a[j * N1 + i] = sf.z(f(node(i, j)));
    }
    return a;
  });
  const quads = [];
  const push = (w, rgb, op) => {
    const pr = w.map(b3Proj);
    quads.push({ pr, rgb, op, light: b3Light(...w),
      depth: pr.reduce((t, p) => t + p.depth, 0) });
  };
  for (let j = 0; j < M3; j++) {
    for (let i = 0; i < M3; i++) {
      const c = node(i + 0.5, j + 0.5), on1 = logOdds(m, c) >= 0;
      const ij = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]];
      vals.forEach((a, k) => {
        const w = ij.map(([p, q]) => b3W(node(p, q), a[q * N1 + p]));
        // A lower ln surface cut off at the floor: skip what lies flat.
        if (B.h3 === "log" && w.every(p => p[2] <= 0)) return;
        if (sf.zero !== null) push(w, C3[on1 ? 1 : 0], 0.88);
        else push(w, C3[k], on1 === (k === 1) ? 0.88 : 0.26);
      });
      if (sf.zero !== null) {
        push(ij.map(([p, q]) => b3W(node(p, q), sf.zero)), [110, 118, 132], 0.2);
      }
    }
  }
  quads.sort((a, b) => b.depth - a.depth);
  s += `<g class="mesh3">`;
  for (const q of quads) {
    const rgb = q.rgb.map(c => Math.round(c * q.light)).join(",");
    s += `<polygon points="${q.pr.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`)
      .join(" ")}" fill="rgb(${rgb})" fill-opacity="${q.op}"/>`;
  }
  s += `</g>`;

  // The crossing lifted onto the surfaces: there both have one height.
  const key = B.m3 === "truth" ? "bayes" : B.m3;
  const lift = x => b3W(x, sf.zero !== null ? sf.zero : sf.z(sf.fs[1](x)));
  for (const l of lines[key]) {
    const ws = l.map(lift);
    s += b3Line(ws, `class="cross3 halo"`) + b3Line(ws, `class="cross3"`);
  }
  bPlane.innerHTML = s;
}
