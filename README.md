# Muon blog — split edge edition

A complete GitHub Pages / Jekyll source package for **How Muon Reshapes the Edge of Stability**.

The article keeps its URL: `/blog/muon-eos-en/`. It contains 11 numbered figures, three tables, five expandable mathematical derivations, one scalar-GD animation and one singular-geometry animation. Experimental plots and numbers were preserved.

## Edit map

| Content | File |
| --- | --- |
| Homepage | `index.md` |
| Article, formulas, tables, derivations | `blog/muon-eos-en.md` |
| Layout, mobile styling, disclosures | `assets/style.css` |
| Both animations and image viewer | `assets/animations.js` |
| Muon geometry illustration | `_includes/mechanism-geometry.html` |
| Direction / parameter-position diagram | `_includes/mechanism-directions.html` |
| Exact matrix counterexample | `_includes/mechanism-matrix.html` |
| Editable direction and spectrum SVGs | `assets/images/muon/directions-*.svg`, `matrix-modes-*.svg` |
| Scalar-GD animation | `_includes/edge-animation.html` |
| Grouped experimental plots | `_includes/hero-results.html`, `intervention-grid.html`, `diagnostic-grid.html` |
| Generic plot component | `_includes/figure.html` |
| Math and script loading | `_includes/head-custom.html` |
| Header / footer | `_layouts/default.html` |
| Site URL and base path | `_config.yml` |

## Math and expandable derivations

Keep `math: true` and `animations: true`. Kramdown inline formulas use same-line `$$...$$`; display formulas put `$$` on separate lines with blank lines around them.

Derivations use native `<details class="technical" markdown="1">` with a descriptive `<summary>`. Leave blank lines around Markdown paragraphs and display equations inside the disclosure. Key definitions, boundaries and conclusions remain in the main text. The five derivations expand automatically for printing and restore their prior state afterward.

## Illustrations

The new Figure 3 compares a gradient with singular values 4 and 1, global Frobenius normalization, and the exact polar target. All panels use the same geometric scale. The slider interpolates singular values; it does not simulate Newton–Schulz iterations. No autoplay is used.

New Figure 4 (formerly Figure 5) matches every common-origin direction pair to its own parameter-position path. Equal-length schematic displacements illustrate reinforcement, orthogonality, partial cancellation and reversal. The reversed arrows are offset only for visibility.

Figure 5 shows three independent quadratic modes in each state of the exact diagonal matrix construction. Signed residual coordinates start at 0.8/0.8/0.8 or 0.1/1.1/1.2 times eta. The same polar step subtracts eta in all three modes. All modes cross zero in A, but only one in B; actual next-step arrows explain cosines −1 and 1/3. The modewise parabolas are not projections of a full matrix-loss trajectory. Both states have the same effective curvature and the same one-step loss decrement, despite different initial losses.

The old Figure 4, T1 balance bars, has been removed. EOC / Edge of Cosine is not introduced.

## Preview and upload

Open the outer `Muon_Blog_Preview.html` for an offline preview. Images, interaction scripts, styles and native MathML formulas are embedded. Use a current Chrome, Edge, Firefox or Safari browser. The live Jekyll page uses MathJax, so equation fonts can differ slightly.

Upload the **contents** of `github-pages/` to the repository root. Keep your personalized homepage and merge the article link as needed. Do not upload the outer preview or ZIP. GitHub Pages should build from `main` and `/(root)`; do not add `.nojekyll`.

Files absent from this ZIP are not automatically deleted when using GitHub's upload interface. The old `_includes/mechanism-balance.html`, `assets/images/muon/balance-*.svg`, old `geometry-*.svg`, `direction-*.svg`, and `matrix-state-*.svg` are no longer referenced and may be removed from the remote repository. New direction SVG filenames start with `directions-` (plural). Delete any obsolete Chinese article separately.

Paragraphs and headings now share a centered reading column; paragraph vertical spacing preserves horizontal auto margins.
