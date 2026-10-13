# Yanzhe Chen — academic homepage and English research blog

This edition restructures the article into six sections: GD intuition, Muon geometry, two diagnostics, experiments, learning-rate interventions, and outlook. It uses eight numbered figures and four tables. All technical discussion remains in the main article.

## Edit the site

| Content | File |
| --- | --- |
| Homepage introduction and research/blog entries | `index.md` |
| Article text, formulas, tables, and single-figure captions | `blog/muon-eos-en.md` |
| Fonts, widths, spacing, colours, mobile and print styles | `assets/style.css` |
| Shared figure markup | `_includes/figure.html` |
| Three-panel diagnostic figure and its caption | `_includes/diagnostic-grid.html` |
| Learning-rate figure and its caption | `_includes/intervention-grid.html` |
| MathJax configuration | `_includes/head-custom.html` |
| Navigation and footer | `_layouts/default.html` |
| Site name, URL and repository subpath | `_config.yml` |
| Figure files | `assets/images/muon/` |

Use `$$...$$` on one line for inline math in Jekyll/Kramdown; put the opening and closing `$$` on their own lines for display equations. Keep blank lines around display equations, tables and includes. The MathJax configuration handles Kramdown's generated math scripts.

Single figures use the `figure.html` include with `src`, `alt` and Markdown `caption` parameters. Internal links and figure paths use `relative_url`, so they also work with a repository `baseurl`. Images can be opened at full resolution by clicking them. The two grouped figures stack vertically on narrow screens; tables scroll within their own containers.

## Publish on GitHub Pages

Upload the **contents** of `github-pages/` into the repository root. Keep your existing personal homepage content when merging `index.md`. Pages should build from `main` and `/(root)`; do not add `.nojekyll`. Existing article permalinks are retained.

The Chinese article is absent from this package. If an old `blog/muon-eos-zh.md` remains in your repository, delete it separately: uploading a new package does not delete remote files.

The preview outside this folder is for local reading and should not be uploaded. It embeds figures and renders formulas as static SVG; the published site uses MathJax. Original source figures are retained for future edits. Numerical results and plots have not been re-estimated.
