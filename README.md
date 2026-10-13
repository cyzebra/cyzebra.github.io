# Yanzhe Chen — academic homepage and English research blog

Blog: **Does Muon Operate at the Edge of Stability in LLM Pretraining?**

This visual edition includes 12 illustrations and 5 tables (one inside an expandable technical note). Seven Muon-paper figure crops and a credited classical EoS image provide experimental context. Three SVG diagrams explain geometry, split diagnostics and future schedule research; one SVG chart replots the paper's Table 2 and Appendix C.3 values. Conceptual diagrams are labeled separately from experimental measurements.

## Edit content

- `index.md`: homepage introduction, research entries and blog entry.
- `blog/muon-eos-en.md`: English article, headings, formulas, tables and captions.
- `_config.yml`: site title and description.
- `assets/images/muon/`: the twelve PNG/SVG illustrations referenced by the article. SVG files can be edited as vector sources.
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

Technical definitions and a local-linearization perspective are folded into two HTML `details` blocks. Preserve `<details markdown="1">`, `<summary>` and `</details>` when editing those blocks. The optional perturbation discussion is explanatory; the LLM runs do not measure perturbation stability.
