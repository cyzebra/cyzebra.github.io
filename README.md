# Muon research blog — geometry refinement

Upload this folder's contents to the GitHub Pages repository root. Keep the Jekyll build; do not add `.nojekyll`. Article URL: `/blog/muon-eos-en/`.

Edit `blog/muon-eos-en.md` for prose and derivations, `assets/style.css` for layout, `_includes/*.html` for figure composition, and `assets/geometry.js` for the exact matrix-map/NS-5 interaction. `assets/animations.js` handles scalar GD and image viewing. All scripts are buildless.

Figure 3 includes mathematical identities and a paused interactive mapping; the NS supplement computes actual finite matrix iterations. Part 3 uses a two-coordinate diagnostic diagram. Figure 5 emphasizes equal effective curvature and different cosine. Figure 7 presents exact event order. Figure 10 retains actual measurements and original intervals.

The outer package contains a fully offline MathML preview, source records/PDFs, editable vector exports, and reproduction scripts. No external demo images or code are embedded. The online page uses MathJax. Keep any personal homepage changes when merging.

## Advisor revision

The article is titled “How Muon Sublates the Edge of Stability”. The layout loads `_includes/article-toc.html` as a sticky left sidebar on desktop. Figure 1 uses responsive desktop/mobile GIFs with an always-visible first-500-update inset inside the middle panel. `assets/article.js` controls GIF playback, reduced motion, printing and active-section navigation. The inline “Return to Figure 1” links to `#fig-1`. Upload the entire assets folder, including both GIFs and both SVG posters.

## Momentum and AdamW addition

The new subsection in section 4 loads `_includes/optimizer-grid.html`. Figure 8 adds `optimizer-global.svg` and `optimizer-layers.svg`; the remaining figures are renumbered 9–12. The sidebar includes a direct link to `#optimizer-memory`. The complete source records and plot script are in `reproduction/data/optimizer-controls/` and `reproduction/plot_optimizer_controls.py`.
