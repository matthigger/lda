# lda

Interactive teaching demo of linear (and quadratic) discriminant analysis,
a companion to the other "visually" demos (e.g.
[PCA](https://github.com/matthigger/pca),
[SVMs](https://github.com/matthigger/svm)). Notation: class means **μ**₀,
**μ**₁, mean difference Δ = **μ**₁ − **μ**₀, shared covariance Σ (or Σ₀,
Σ₁), priors P(y = k), squared Mahalanobis distance D², unit direction
**w**, log-odds g(**x**) in nats. Two tabs:

- **Projection onto a line.** 40 samples per class, each class with
  exactly mean **μ**ₖ and covariance Σ (with 1/n). Drag **w** (or its
  line) through the midpoint **m** and the scores t = **w**ᵀ(**x** − **m**)
  of each class are histogrammed beside the line, with the normal each
  class projects to. The readout splits Fisher's
  J(**w**) = (**w**ᵀΔ)² / **w**ᵀΣ**w** into the projected mean difference
  and the within-class scatter, and the J-by-angle chart marks Σ⁻¹Δ (the
  peak), Δ (largest mean difference), Σ's short axis (smallest
  within-class scatter) and PCA's **v**₁. Drag the means, edit Σ, and
  drag a threshold τ on the histograms' baseline to see the boundary and
  the training errors.
- **Bayes boundary: LDA → QDA.** Set two true Gaussians (or share one Σ),
  the prior P(y = 1) and the training n. The plot shades the true
  posterior P(y = 1 | **x**) and draws the Bayes boundary g = 0 with the
  LDA and QDA boundaries fit to the training samples (maximum
  likelihood); train and test error (5000 fresh draws per class) for all
  three. With a shared Σ the boundary is the line **w**ᵀ**x** + b,
  **w** = Σ⁻¹Δ; otherwise a conic. Hover for the Mahalanobis distances
  and log-odds at a point. The 3D view (`js/bayes3d.js`) lifts the plane
  into surfaces, from the true Gaussians or either fit: P(y = k)
  p(**x** | y = k) per class, its ln (an upside-down bowl,
  −½D²ₖ + const), or g itself against a g = 0 sheet. The boundary is
  where the surfaces cross; drag to orbit, scroll to zoom.

Plain HTML/CSS/JS with SVG: no build step and no dependencies, so it
also works offline from `file://`. The model math is `js/gauss.js`;
boundaries are zero contours on a 120 × 120 grid (marching squares).

## Testing

```sh
~/venv_ds4400/bin/python -I build/make_ref.py      # rewrites test/ref.js
~/venv_ds4400/bin/python test/run_headless.py "file://$PWD/test/index.html" \
    --wait '/PASS|FAIL/.test(document.title)' \
    --eval 'document.getElementById("out").textContent'
```

compares `js/gauss.js` with sklearn's `LinearDiscriminantAnalysis`
(lsqr solver: decision function, coef and intercept), with scipy log
densities for QDA's maximum likelihood fit (sklearn's QDA divides by
n − 1 instead), for set models and for the 3D view's ln P(y = k)
p(**x** | y = k), and checks that the projection tab's
clouds have exactly the mean and covariance on screen.
`run_headless.py` also takes `--shot` screenshots; it drives playwright's
Chromium (`~/.cache/ms-playwright`), so run it outside the sandbox.

## Run locally

Open `index.html` in a browser, or serve it:

```sh
python3 -m http.server 8000   # then visit http://localhost:8000
```

Link straight to a tab with `index.html#projection` or `index.html#bayes`.

## Publish on GitHub Pages

```sh
gh repo create matthigger/lda --public --source . --push
gh api -X POST repos/matthigger/lda/pages -f build_type=workflow
```

`.github/workflows/pages.yml` then deploys each push to main. The site
lives at <https://matthigger.github.io/lda/>.
