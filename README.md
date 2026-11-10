# lda

Interactive teaching demo of linear (and quadratic) discriminant analysis
for DS 4400 lesson 20, a companion to the other "visually" demos (e.g.
[PCA](https://github.com/matthigger/pca),
[SVMs](https://github.com/matthigger/svm)). Notation follows the lesson:
class means **μ**₀, **μ**₁, shared covariance Σ (or Σ₀, Σ₁), priors π₀,
π₁, unit direction **w**, log-odds g(**x**). Two tabs:

- **Projection onto a line.** 40 samples per class, each class with
  exactly mean **μ**ₖ and covariance Σ (with 1/n). Drag **w** (or its
  line) through the midpoint **m** and the scores t = **w**ᵀ(**x** − **m**)
  of each class are histogrammed beside the line, with the normal each
  class projects to. The readout splits Fisher's
  J(**w**) = (**w**ᵀ(**μ**₁ − **μ**₀))² / **w**ᵀΣ**w** into gap and spread,
  and the J-by-angle chart marks Σ⁻¹(**μ**₁ − **μ**₀) (the peak),
  **μ**₁ − **μ**₀ and PCA's **v**₁. Drag the means, edit Σ, and drag a
  threshold τ on the histograms' baseline to see the boundary and the
  training errors. Presets include the `lda_demo.ipynb` ICA parts.
- **Bayes boundary: LDA → QDA.** Set two true Gaussians (or share one Σ),
  the prior π₁ and the training n. The plot shades the true posterior
  P(y = 1 | **x**) and draws the Bayes boundary g = 0 with the LDA and
  QDA boundaries fit to the training samples (maximum likelihood); train
  and test error (5000 fresh draws per class) for all three. With a
  shared Σ the boundary is the line **w**ᵀ**x** + b, **w** =
  Σ⁻¹(**μ**₁ − **μ**₀); otherwise a conic. Hover for the Mahalanobis
  distances and log-odds at a point.

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
n − 1 instead) and for set models, and checks that the projection tab's
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
