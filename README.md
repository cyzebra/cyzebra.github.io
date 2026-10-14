# Muon research blog — animated edition

**What Does the Edge of Stability Mean for Muon?**

This package contains a GitHub Pages/Jekyll homepage and English article, two opening animations, seven numbered figures, five tables, and an offline preview outside the site folder.

The article separates fixed-point perturbation stability, empirical training near an edge, conditional one-step loss balance (T1), and consecutive-update alignment (T2). Wu–Ma–E (2018) supplies the fixed-point stability background; Cohen et al. (2021) supplies the empirical neural-network EoS background. All technical material stays in the main text.

## Files to edit

| Content | File |
| --- | --- |
| Homepage text and blog entry | `index.md` |
| Article, headings, formulas, tables and ordinary figure captions | `blog/muon-eos-en.md` |
| Layout, colours, spacing, mobile and print styles | `assets/style.css` |
| Animation playback, exact toy dynamics and reveal controls | `assets/animations.js` |
| Scalar stability animation and its labels | `_includes/edge-animation.html` |
| Guided 130M figure reveal and its captions | `_includes/reading-animation.html` |
| Standard figure component | `_includes/figure.html` |
| Grouped diagnostic figure | `_includes/diagnostic-grid.html` |
| Grouped learning-rate figure | `_includes/intervention-grid.html` |
| MathJax and animation script loading | `_includes/head-custom.html` |
| Navigation and footer | `_layouts/default.html` |
| Site URL, repository base path and title | `_config.yml` |
| Measured figure images and original source figures | `assets/images/muon/` |

## Math and figures

Kramdown inline math uses same-line `$$...$$`. Display equations put the opening and closing `$$` on separate lines, with blank lines around the equation. The shared head translates Kramdown's generated math scripts for MathJax 3.

Use `{% include figure.html src="/assets/images/muon/FILE.png" alt="Description" caption="**Figure N · Title.** Explanation." %}` for a single figure. Grouped figures and animations are in the named includes above. Internal paths use `relative_url` so they work with a configured repository `baseurl`.

Keep `permalink: /blog/muon-eos-en/`, `math: true`, and `animations: true` in the article front matter.

## What the animations do

1. Exact scalar GD on `L(x)=x²/2`, with `x₀=1`, fixed curvature, and ηλ in {0.8, 1.8, 2, 2.2}. Six updates show contraction, oscillatory contraction, marginal reversal and divergence. Curvature is fixed: this is not a demonstration of neural-network self-organization at an edge.
2. A reading guide that reveals three unchanged panels of paper Figure 1. It animates reading order, not training iterations. No experimental values were simulated, interpolated or digitized.

Both provide play, pause, replay and scrubbing. They pause when the page becomes hidden or the component leaves view. They do not autoplay. Reduced-motion preferences prevent replay from starting automatically. With JavaScript disabled, labels and the measured panels remain visible; controls remain hidden.

## Upload

Upload the **contents** of `github-pages/` to the repository root. Merge `index.md` with your personalized homepage instead of replacing its personal content blindly. GitHub Pages should build from `main` and `/(root)`; do not add `.nojekyll`.

The offline `Muon_Blog_Preview.html` embeds images, formula SVGs and animation code; open it locally to read and interact. It need not be uploaded. Published formulas use MathJax and can differ slightly in font from the preview.

The Chinese article is absent. Delete an old `blog/muon-eos-zh.md` separately if it remains in the remote repository. Uploading the package does not delete remote files.
