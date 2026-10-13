# Yanzhe Chen — academic homepage and English research blog

Blog: **Does Muon Operate at the Edge of Stability in LLM Pretraining?**

This revised visual edition includes 10 illustrations and 3 tables. The geometry introduction comes before the main diagnostic comparison. Muon is introduced through its SVD-polar formula, all technical material appears directly in the article, and one combined reference replaces the former repeated comparison/diagnostic tables. The 130M module heatmap is grouped with block trajectories and probe-batch coherence from the same peak-rate baseline.

## Edit content

- `index.md`: homepage introduction, research entries and blog entry.
- `blog/muon-eos-en.md`: English article, headings, formulas, tables and captions.
- `_config.yml`: site title and description.
- `assets/images/muon/`: the ten referenced PNG/SVG illustrations and one original heatmap retained as a source for the grouped figure. SVG files can be edited as vector sources.
- `_layouts/default.html`: navigation, footer and shared page structure.
- `_includes/head-custom.html`: CSS styling and MathJax configuration.

For everyday writing, edit the first two files. Keep `permalink: /blog/muon-eos-en/` unchanged to preserve existing links.

## Remove the old Chinese page

The Chinese article is no longer included. Uploading these files does not delete old remote files. Delete `blog/muon-eos-zh.md` in GitHub separately. This package removes its links from the homepage and English article.

## Publish

Upload this folder's contents to the root of `cyzebra.github.io`. Merge your personal homepage edits before replacing `index.md`.

Settings → Pages → Deploy from a branch → main → /(root).

- Home: https://cyzebra.github.io/
- Blog: https://cyzebra.github.io/blog/muon-eos-en/

The published page loads MathJax from jsDelivr. The standalone preview embeds its figures and pre-rendered SVG formulas.

There are no folded `details` blocks. The local-linearization discussion is an intuitive explanation; LLM T1/T2 probes do not measure perturbation stability. The former polar-spectrum diagram and learning-rate bar chart have been removed. The learning-rate probe numbers remain in a table.
