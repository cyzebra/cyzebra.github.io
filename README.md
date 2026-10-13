# Yanzhe Chen — academic homepage and Muon blog

Blog title: **Does Muon Operate at the Edge of Stability in LLM Pretraining?**

Chinese title: **Muon 在大模型预训练中也处于稳定性边缘吗？**

The article follows a question-driven research story inspired by Fanghui Liu's supplied announcement: the answer is sublation, loss balance separates from update reversal, experiments extend to 1B directional measurements, and the diagnostics motivate future learning-rate and batch-size schedules. Claims remain scoped to the no-momentum setting studied in the paper.

Upload the contents of this folder to the root of the `cyzebra.github.io` repository. Preserve every folder name, including the leading underscores in `_layouts` and `_includes`.

Publishing settings: Settings → Pages → Deploy from a branch → main → /(root) → Save.

Expected URLs:

- Home: https://cyzebra.github.io/
- Chinese blog: https://cyzebra.github.io/blog/muon-eos-zh/
- English blog: https://cyzebra.github.io/blog/muon-eos-en/

Edit the articles in `blog/`. Keep `permalink` unchanged to preserve links. Both language versions share five PNG figures: four extracted from the Muon paper and one classical EoS image supplied by the user and credited to Litman (2026), Figure 1, CC BY 4.0. The custom layout provides a wider reading area than the initial Minimal theme; no additional theme installation is required.

MathJax is loaded from jsDelivr when a page has `math: true`. The supplied draft preview uses pre-rendered SVG formulas for checking formulas without relying on that CDN.

The article is based on *Muon Sublates the Edge of Stability in LLM Pretraining*, arXiv:2609.34915, by Yanzhe Chen, Qifang Zhao, Xiaoxiao Xu, and Fanghui Liu. In particular, the 1B run does not measure T1, and the adaptive scheduling discussion is future work.
