// 2 x 2 symmetric matrices M = {a, b, d} = [[a, b], [b, d]]: the
// closed-form eigenproblem, matrix functions (the square root), inverse,
// determinant and quadratic forms.

/**
 * Eigen-decompose the symmetric 2 x 2 matrix [[a, b], [b, d]].
 *
 * Returns:
 *   {l1, l2, theta}: eigenvalues l1 >= l2; theta (radians, in
 *     (-pi/2, pi/2]) is the angle of the unit eigenvector v1 of l1,
 *     v1 = [cos theta, sin theta]. theta is NaN when l1 = l2 (every
 *     direction is an eigenvector).
 */
function eig2(a, b, d) {
  const m = (a + d) / 2, r = Math.hypot((a - d) / 2, b);
  let theta = r < 1e-12 ? NaN : 0.5 * Math.atan2(2 * b, a - d);
  if (theta <= -Math.PI / 2) theta += Math.PI;
  return { l1: m + r, l2: m - r, theta };
}

/** M = λ₁v₁v₁ᵀ + λ₂v₂v₂ᵀ with v₁ = [cos th, sin th], v₂ ⟂ v₁. */
function matFromEig(e) {
  const c = Math.cos(e.th), s = Math.sin(e.th);
  return { a: e.l1 * c * c + e.l2 * s * s, b: (e.l1 - e.l2) * c * s,
    d: e.l1 * s * s + e.l2 * c * c };
}

/** f(M) = R diag(f(λ)) Rᵀ for a symmetric M. */
function matFn(M, f) {
  const e = eig2(M.a, M.b, M.d);
  return matFromEig({ l1: f(Math.max(e.l1, 0)), l2: f(Math.max(e.l2, 0)),
    th: isNaN(e.theta) ? 0 : e.theta });
}

/** M v. */
const mv = (M, v) => [M.a * v[0] + M.b * v[1], M.b * v[0] + M.d * v[1]];
const det2 = M => M.a * M.d - M.b * M.b;
/** vᵀ M v. */
const quad = (M, v) => dot(v, mv(M, v));

function inv2(M) {
  const D = det2(M);
  return { a: M.d / D, b: -M.b / D, d: M.a / D };
}

/** p A + q B. */
function mix2(A, p, B, q) {
  return { a: p * A.a + q * B.a, b: p * A.b + q * B.b, d: p * A.d + q * B.d };
}

/** Max abs entry difference of two matrices. */
function matDiff(A, B) {
  return Math.max(Math.abs(A.a - B.a), Math.abs(A.b - B.b),
    Math.abs(A.d - B.d));
}

/** Angle th folded into (−90°, 90°]. */
function foldTh(th) {
  let t = ((th + Math.PI / 2) % Math.PI + Math.PI) % Math.PI - Math.PI / 2;
  if (t <= -Math.PI / 2 + 1e-12) t += Math.PI;
  return t;
}

/**
 * Why M cannot be a covariance here, or "" if it can.
 *
 * Args:
 *   M ({a, b, d}): proposed covariance
 *   name (string): its HTML name, e.g. "&Sigma;"
 *   lmax (number): largest eigenvalue allowed, so clouds fit on the plot
 */
function covProblem(M, name, lmax) {
  if (![M.a, M.b, M.d].every(Number.isFinite)) return "Enter a number.";
  const e = eig2(M.a, M.b, M.d);
  if (e.l2 < 0.01) {
    return `This ${name} has eigenvalues ${fmt(e.l1)} and ${fmt(e.l2)}.
      A variance <b>w</b><sup>T</sup>${name}<b>w</b> must be positive in
      every direction for the Gaussian to exist: both diagonal entries
      &gt; 0 and the off-diagonal smaller than &radic;(product of the
      diagonal).`;
  }
  if (e.l1 > lmax + 1e-9) {
    return `Keep the eigenvalues at most ${lmax} (this ${name} has
      <i>&lambda;</i><sub>1</sub> = ${fmt(e.l1)}) so the cloud fits on
      the plot.`;
  }
  return "";
}
