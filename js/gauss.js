// Samples and Gaussian class models, shared by both tabs and checked
// against sklearn / scipy in test/.
//
// A model holds two classes, k = 0, 1, with means mu[k], covariances S[k]
// and priors pi[k]. Its log-odds
//   g(x) = log π₁N(x; μ₁, Σ₁) − log π₀N(x; μ₀, Σ₀)
//        = ½d₀² − ½d₁² − ½ log(|Σ₁| / |Σ₀|) + log(π₁ / π₀),
// d_k² = (x − μ_k)ᵀΣ_k⁻¹(x − μ_k), is > 0 where class 1 is more likely.
// With Σ₀ = Σ₁ = Σ the quadratic terms cancel: g(x) = wᵀx + b with
// w = Σ⁻¹(μ₁ − μ₀), a line.

/** Seeded uniform [0, 1) generator (mulberry32). */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** n standard normal 2-vectors (Box-Muller). */
function normals(seed, n) {
  const r = rng(seed), z = [];
  for (let i = 0; i < n; i++) {
    const u = 1 - r(), v = r(), rad = Math.sqrt(-2 * Math.log(u));
    z.push([rad * Math.cos(2 * Math.PI * v), rad * Math.sin(2 * Math.PI * v)]);
  }
  return z;
}

/** Sample mean and covariance (with 1/n). */
function meanCov(X) {
  const n = X.length;
  const m = X.reduce((s, q) => add(s, q), [0, 0]).map(v => v / n);
  const C = { a: 0, b: 0, d: 0 };
  for (const q of X) {
    const e = sub(q, m);
    C.a += e[0] * e[0] / n;
    C.b += e[0] * e[1] / n;
    C.d += e[1] * e[1] / n;
  }
  return { m, C };
}

/** z re-centered and re-scaled to mean exactly 0 and covariance I. */
function whiten(z) {
  const { m, C } = meanCov(z), W = matFn(C, l => 1 / Math.sqrt(l));
  return z.map(q => mv(W, sub(q, m)));
}

/** Samples mu + M^½ z: mean mu and covariance M if z is white. */
function cloud(z, mu, M) {
  const R = matFn(M, Math.sqrt);
  return z.map(q => add(mu, mv(R, q)));
}

/** A model with its inverses and log determinants cached. */
function model(mu0, mu1, S0, S1, pi1) {
  const Sg = [S0, S1];
  return { mu: [mu0, mu1], S: Sg, Si: Sg.map(inv2),
    ld: Sg.map(M => Math.log(det2(M))), pi: [1 - pi1, pi1],
    lp: Math.log(pi1 / (1 - pi1)) };
}

/** Squared Mahalanobis distance from x to class k's mean. */
function maha2(md, k, x) { return quad(md.Si[k], sub(x, md.mu[k])); }

/** Log-odds g(x) of class 1 over class 0. */
function logOdds(md, x) {
  return 0.5 * maha2(md, 0, x) - 0.5 * maha2(md, 1, x) -
    0.5 * (md.ld[1] - md.ld[0]) + md.lp;
}

/** P(y = 1 | x). */
function posterior(md, x) { return 1 / (1 + Math.exp(-logOdds(md, x))); }

/** Shift M by a tiny ridge if it is (numerically) singular. */
function ridge(M) {
  return det2(M) > 1e-12 ? M : { a: M.a + 1e-6, b: M.b, d: M.d + 1e-6 };
}

/**
 * Fit a Gaussian per class by maximum likelihood.
 *
 * Args:
 *   X0, X1 (number[][]): samples of class 0 and class 1
 *   shared (bool): LDA, one pooled covariance (n₀Σ̂₀ + n₁Σ̂₁) / n, as
 *     sklearn's LinearDiscriminantAnalysis (lsqr / eigen solvers);
 *     otherwise QDA, each class its own Σ̂_k (1/n_k)
 *
 * Returns:
 *   (object): model with the class frequencies as priors
 */
function fitModel(X0, X1, shared) {
  const a = meanCov(X0), b = meanCov(X1);
  const n0 = X0.length, n1 = X1.length, n = n0 + n1;
  let S0 = a.C, S1 = b.C;
  if (shared) S0 = S1 = mix2(a.C, n0 / n, b.C, n1 / n);
  return model(a.m, b.m, ridge(S0), ridge(S1), n1 / n);
}

/** g(x) = wᵀx + b for a model whose classes share Σ. */
function linearForm(md) {
  const w = mv(md.Si[0], sub(md.mu[1], md.mu[0]));
  return { w, b: -0.5 * dot(add(md.mu[0], md.mu[1]), w) + md.lp };
}

/** Fisher's direction Σ⁻¹(μ₁ − μ₀) (not normalized). */
function fisherDir(mu0, mu1, M) { return mv(inv2(M), sub(mu1, mu0)); }

/**
 * Fraction of samples misclassified by md (predict 1 where g ≥ 0).
 *
 * Args:
 *   X (number[][][]): [class 0 samples, class 1 samples]
 *   wt (number[] | undefined): per-class weights (e.g. the true priors);
 *     omitted, every sample counts once
 */
function errRate(md, X, wt) {
  const miss = X.map((Xk, k) => Xk.reduce((s, x) =>
    s + ((logOdds(md, x) >= 0 ? 1 : 0) !== k ? 1 : 0), 0));
  if (wt) return wt[0] * miss[0] / X[0].length + wt[1] * miss[1] / X[1].length;
  return (miss[0] + miss[1]) / (X[0].length + X[1].length);
}
