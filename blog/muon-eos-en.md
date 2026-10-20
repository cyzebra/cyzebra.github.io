---
layout: default
title: "How Muon Sublates the Edge of Stability"
description: "A loss‑neutral edge survives under no-momentum Muon, but its classical coupling to update reversal breaks."
permalink: /blog/muon-eos-en/
lang: en
math: true
article: true
animations: true
---

<div class="article-header" markdown="1">
<p class="eyebrow">Optimization · Muon · LLM pretraining</p>

# How Muon Sublates the Edge of Stability

<p class="byline">Yanzhe Chen · With Qifang Zhao, Xiaoxiao Xu, and Fanghui Liu</p>
<p class="standfirst">A loss‑neutral edge survives. Its classical coupling to update reversal does not.</p>
<div class="article-links"><a href="https://arxiv.org/abs/2609.34915">Paper ↗</a><a href="https://github.com/cyzebra/Muon-Sublates-the-Edge-of-Stability-in-LLM-Pretraining">Code ↗</a></div>
</div>

**Does language-model training with Muon still happen near an edge of stability?** Changing the optimizer changes what this question means. In a scalar GD mode, loss neutrality, equal-magnitude reversal and marginal stability coincide. Muon's matrix geometry breaks that coupling.

{% include hero-results.html %}

<div class="takeaway" markdown="1">

The three panels show the puzzle: **loss-boundary tracking, weak negative direction alignment, and continued learning coexist.**

</div>

 Our paper explains the split through two diagnostics: **T1**, the conditional loss‑neutral boundary, and **T2**, the temporal-orthogonality boundary. The contribution is a characterization of what remains coupled—and what no longer does—when updates become matrix‑aware.

<h2 id="stability">1. What is the Edge of Stability?</h2>

The **Edge of Stability (EoS)** describes a striking training regime: curvature stays near a learning-rate-dependent stability boundary, while short-term loss fluctuations coexist with progress over longer horizons. In full-batch neural-network gradient descent, the largest Hessian eigenvalue often hovers just above $$2/\eta$$. Training can keep improving even though individual steps do not decrease loss monotonically, as documented by [Cohen et al. (2021)](https://arxiv.org/abs/2103.00065).

{% include figure.html src="/assets/images/muon/classical-eos-litman2026.png" alt="Classical GD: maximum curvature approaches the learning-rate scale while loss oscillates and improves." caption="**Figure 2 · The classical EoS training pattern.** Full-batch GD, three-layer MLP, CIFAR-10, learning rate 0.5. Curvature approaches the learning-rate scale while loss fluctuates and improves. Reproduced unchanged from [Litman (2026), Figure 1](https://arxiv.org/html/2604.20446v1#S1.F1), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)." %}

### Where does the scale 2/η come from?

For one scalar quadratic mode, $$L(x)=\lambda x^2/2$$, GD gives

$$
x_{k+1}=(1-\eta\lambda)x_k.
$$

The multiplier $$1-\eta\lambda$$ determines whether this mode contracts or grows. For $$0<\eta\lambda<2$$, its magnitude shrinks; above $$2$$, it grows. At $$\eta\lambda=2$$, the multiplier is $$-1$$: the loss stays constant, consecutive updates reverse with equal magnitude, and the mode is marginally stable. This simple calculation explains the classical boundary scale.

{% include edge-animation.html %}

Direction changes begin earlier, above $$\eta\lambda=1$$. The next gradient vanishes exactly at that point, so its cosine is undefined there. Thus negative alignment alone does not identify the loss-neutral or marginal-stability boundary.

### A boundary calculation and a training observation

The quadratic calculation identifies a **local stability boundary**: it tells us when perturbations contract or grow. The EoS training phenomenon asks a further question: **does the evolving trajectory actually approach and remain near such a boundary while learning?** A fixed quadratic illustrates the boundary, but cannot demonstrate the changing curvature of neural-network training.

[Wu, Ma and E (2018)](https://proceedings.neurips.cc/paper/2018/file/6651526b6fb8f29a00507de6a49ce30f-Paper.pdf) study which minima can be selected through linearized stochastic stability. Their criterion bounds perturbation second moments and depends on both average sampled curvature and its variation. They also observe selected GD minima near the permitted sharpness bound in some settings. Cohen et al. study the related behavior along neural-network training trajectories. The distinction is between characterizing a boundary and establishing how training evolves relative to it.

Our Muon question brings these ideas together: **which parts of the classical EoS picture survive when update geometry changes?** We characterize the conditional loss-neutral boundary and its relationship to temporal alignment, then examine boundary tracking and direction behavior along training. The scalar GD coincidence above provides the reference mechanism; it does not require full-network updates to reverse whenever maximum Hessian curvature approaches $$2/\eta$$.

<h2 id="muon">2. Muon changes relative update strengths</h2>

For a sampled matrix gradient, exact no-momentum Muon uses its compact SVD:

$$
\widehat G_k=U_r\Sigma_rV_r^\top,
\qquad P_k=\operatorname{Pol}(\widehat G_k)=U_rV_r^\top,
\qquad W_{k+1}=W_k-\eta P_k.
$$

It keeps the active left and right singular vectors and replaces their nonzero singular values by one. Multiplying the whole gradient by one scalar would leave those values in the same proportions. **Muon changes the proportions.** Zero singular directions remain zero.

{% include mechanism-geometry.html %}

The same geometry has a steepest-descent interpretation: with a spectral-norm step budget, the polar direction maximizes first-order descent,

$$
P\in\operatorname*{arg\,max}_{\|D\|_2\leq1}\langle\widehat G,D\rangle_F,
\qquad \langle\widehat G,P\rangle_F=\|\widehat G\|_*.
$$

Here $$\|D\|_2$$ is the matrix spectral norm, and $$\|G\|_*$$ is the nuclear norm. This is why a nuclear norm appears in the loss identity below. Our controlled experiments use SVD-polar directions; LLM runs use the practical **five-step Newton–Schulz approximation, NS-5**, without momentum.

<details class="technical" markdown="1">
<summary>Why does the nuclear norm appear?</summary>

Write $$G=U_r\Sigma_rV_r^\top$$. For any $$\|D\|_2\leq1$$, each active singular pair satisfies $$|u_i^\top Dv_i|\leq1$$. Consequently,

$$
\langle G,D\rangle_F=\sum_{i=1}^{r}\sigma_i u_i^\top Dv_i
\leq\sum_{i=1}^{r}\sigma_i=\|G\|_*.
$$

Taking $$D=U_rV_r^\top$$ attains equality. The linearized loss change of $$-\eta D$$ is therefore minimized by the polar direction under this norm budget. This argument concerns the exact no-momentum direction; an approximate transform does not automatically inherit every exact-polar property. See [Bernstein's derivation](https://jeremybernste.in/writing/deriving-muon) for the broader layer-geometry perspective.

For a differentiable optimizer map $$F(W)=W-\eta T(\nabla L(W))$$,

$$
DF(W)=I-\eta\,DT(G)\circ\nabla^2L(W).
$$

GD has $$T=\operatorname{Id}$$. A nonlinear matrix transform adds its own local response, with care required at rank changes. This explains why the GD mechanism cannot simply be transferred; it is not a stochastic stability theorem for Muon.

</details>

<h2 id="diagnostics">3. Why the classical coupling splits</h2>

**The loss change uses the current polar direction; temporal alignment also uses how that direction changes next.** These are different pieces of geometry.

{% include coupling-map.html %}

### T1: the conditional loss‑neutral boundary

Hold $$W_k$$ fixed and resample minibatches of size $$b$$. With nonzero full gradient $$G_k=\nabla L(W_k)$$, define coherence

$$
\rho_b=\frac{\mathbb E_b\langle G_k,P_k\rangle_F}{\|G_k\|_*}.
$$

It measures how much first-order descent is retained after sampling and polar normalization. An unbiased sampled gradient does not guarantee an unbiased polar direction. Exact-polar duality gives $$\rho_b\in[-1,1]$$.

The finite-step effective curvature $$s_b^M$$ is defined from the loss remainder. These quantities give the exact identity

$$
\mathbb E_b[L(W_{k+1})-L(W_k)\mid W_k]
=\frac{\eta^2\|G_k\|_*}{2}\left(s_b^M-\frac{2\rho_b}{\eta}\right).
$$

**Read the bracket:** negative means expected descent, zero means balance, positive means an expected increase. T1 is therefore $$s_b^M=2\rho_b/\eta$$. Full-batch exact-polar Muon has $$\rho_{\mathrm{full}}=1$$ and recovers the scale $$2/\eta$$. The effective curvature is a normalized directional loss remainder, not the maximum Hessian eigenvalue.

<details class="technical" markdown="1">
<summary>Deriving the conditional loss identity</summary>

Let the finite-step remainder be

$$
r_k(P)=L(W_k-\eta P)-L(W_k)+\eta\langle G_k,P\rangle_F.
$$

Define

$$
s_b^M(W_k;\eta)=\frac{2\mathbb E_b r_k(P_k)}{\eta^2\|G_k\|_*}.
$$

Rearranging the remainder and taking conditional expectations gives

$$
\mathbb E_b\Delta L=-\eta\mathbb E_b\langle G_k,P_k\rangle_F
+\mathbb E_b r_k(P_k)
=-\eta\|G_k\|_*\rho_b+\frac{\eta^2\|G_k\|_*}{2}s_b^M.
$$

Factoring yields the displayed identity. It requires continuous differentiability, nonzero $$G_k$$ and finite expectations, not a second-order Taylor approximation. With additional second-order smoothness, the small-step limit at a fixed state is

$$
s_b^M\longrightarrow\frac{\mathbb E_b\langle P_k,H_k[P_k]\rangle_F}{\|G_k\|_*}.
$$

The equality defines the conditional boundary; it does not prove that training stays there. In practical multi-block probes, numerical direction normalization and the finite reference objective follow the paper's experimental protocol rather than an ideal single-matrix calculation.

</details>

### T2: the temporal-orthogonality boundary

For two nonzero consecutive directions,

$$
c_k^{\mathrm{pol}}=\frac{\langle P_k,P_{k+1}\rangle_F}{\|P_k\|_F\|P_{k+1}\|_F}.
$$

T2 is the sign boundary $$c_k^{\mathrm{pol}}=0$$. Positive means acute alignment; negative means obtuse alignment and partial cancellation relative to same-direction steps. A discrete trajectory can cross the sign boundary without attaining zero. **A negative cosine alone proves neither a two-step return nor local perturbation stability.**

{% include mechanism-directions.html %}

For GD the direction-sign boundary is tied to a directional secant response of $$1/\eta$$. For exact full-batch Muon, the corresponding value is **geometry-dependent**, $$\vartheta_k/\eta$$. T1 and T2 are loss and direction diagnostics, not two interchangeable linear-stability criteria.

<details class="technical" markdown="1">
<summary>From direction cosine to two-step motion and the secant boundary</summary>

Let $$\Delta_k=W_{k+1}-W_k=-\eta_kP_k$$. Then

$$
\|W_{k+2}-W_k\|_F^2=\|\Delta_k\|_F^2+\|\Delta_{k+1}\|_F^2
+2\|\Delta_k\|_F\|\Delta_{k+1}\|_F c_k^{\mathrm{pol}}.
$$

At $$c=-1$$, Cauchy–Schwarz gives proportional opposite directions. Exact nonzero polar factors both have operator norm one, so the proportionality factor is one. At fixed rate this yields $$P_{k+1}=-P_k$$ and $$W_{k+2}=W_k$$. For implemented NS-5 directions or unequal rates, opposite directions alone need not imply equal displacement lengths.

For full-batch exact-polar Muon, define

$$
\bar s_k^M=-\frac{\langle P_k,G_{k+1}-G_k\rangle_F}{\eta\|G_k\|_*},
\qquad a_k=\frac{\|G_{k+1}\|_*}{\|G_k\|_*},\qquad r_k=\operatorname{rank}(G_k).
$$

Separate the next gradient into its mean singular strength and residual:

$$
R_{k+1}=G_{k+1}-\frac{\|G_{k+1}\|_*}{r_{k+1}}P_{k+1},
\qquad \vartheta_k=1-\frac{\langle P_k,R_{k+1}\rangle_F}{\|G_k\|_*}.
$$

Using $$\|P_k\|_F^2=r_k$$ gives

$$
\bar s_k^M=\frac{\vartheta_k-a_k\sqrt{r_k/r_{k+1}}\,c_k^{\mathrm{pol}}}{\eta}.
$$

Thus $$c_k^{\mathrm{pol}}=0$$ iff $$\bar s_k^M=\vartheta_k/\eta$$, and negative cosine corresponds to a larger secant response. The coefficient $$\vartheta_k$$ need not be one. This is an exact relation for a given step, not a universal curvature-only predictor. Loss curvature $$s_k^M$$ and secant response $$\bar s_k^M$$ agree for constant Hessian, but generally weight curvature along the update segment differently. See paper Appendix A.3.

</details>

### A counterexample you can see in three quadratic modes

To see the geometry, take the paper's $$X=I_3$$ example and restrict $$W-W_{\min}$$ to a diagonal matrix with signed coordinates $$e_1,e_2,e_3$$. The loss becomes $$L=\tfrac12\sum_i e_i^2$$, so the three modes have identical quadratic bowls. Initially all coordinates are positive: exact polar Muon therefore takes the same step, $$\Delta_k=-\eta I_3$$, in both states.

**Each mode moves left by η, regardless of its initial strength.** In A, every coordinate overshoots zero and the next step reverses all three directions. In B, just one coordinate overshoots; the other two keep moving left. This makes the difference in alignment visible.

**The key comparison is equal effective curvature with different update alignment.** T1 depends on the **sum of the initial strengths**, or trace. Alignment depends on **how many modes cross zero**. Two initial points on the same plane $$e_1+e_2+e_3=2.4\eta$$ can therefore have the same effective curvature and different next directions.

{% include mechanism-matrix.html %}

<details class="technical" markdown="1">
<summary>Working through the matrix counterexample</summary>

Consider $$L(W)=\tfrac12\|WX-Y\|_F^2$$ with $$XX^\top\succ0$$, $$m\geq d$$, a full-column-rank gradient and nonsingular $$S_k-\eta XX^\top$$, where $$G_k=P_kS_k$$ and $$S_k=(G_k^\top G_k)^{1/2}$$. The affine gradient gives

$$
G_{k+1}=G_k-\eta P_kXX^\top=P_k(S_k-\eta XX^\top).
$$

Because $$P_k^\top P_k=I_d$$, its next stretch and polar factors are

$$
S_{k+1}=|S_k-\eta XX^\top|,
\qquad P_{k+1}=P_k\operatorname{sign}(S_k-\eta XX^\top).
$$

The absolute value and sign act eigenvaluewise. The quadratic loss expansion yields

$$
\Delta L=-\eta\operatorname{tr}(S_k)+\frac{\eta^2}{2}\|X\|_F^2,
\qquad s_k^M=\frac{\|X\|_F^2}{\operatorname{tr}(S_k)}.
$$

If $$n_-$$ counts negative eigenvalues of $$S_k-\eta XX^\top$$,

$$
c_k^{\mathrm{pol}}=\frac{\operatorname{tr}\operatorname{sign}(S_k-\eta XX^\top)}{d}
=1-\frac{2n_-}{d}.
$$

Take $$X=I_3$$ and either $$S_k=\eta\operatorname{diag}(0.8,0.8,0.8)$$ or $$S_k=\eta\operatorname{diag}(0.1,1.1,1.2)$$. Both traces are $$2.4\eta$$, so both have $$s_k^M=5/(4\eta)$$ and $$\Delta L=-0.9\eta^2$$. Their shifted spectra have three and one negative eigenvalues, giving $$c=-1$$ and $$c=1/3$$. Equal loss curvature therefore does not determine alignment. See paper §2.3 and Appendix A.2.

</details>

### What theory says about approaching the loss boundary

There is also a trajectory constraint. For full-batch exact-polar Muon at fixed rate on a lower-bounded objective, if the cumulative active-step weight $$\sum_k\eta^2\|G_k\|_*$$ diverges, effective curvature cannot eventually stay a fixed distance below $$2/\eta$$:

$$
\limsup_{k\to\infty:G_k\ne0}s_k^M\geq\frac{2}{\eta}.
$$

This constrains **curvature visits**. Persistent tracking under stochastic NS-5 training remains an empirical finding to examine next.

<details class="technical" markdown="1">
<summary>Why curvature cannot stay uniformly below the boundary</summary>

Suppose all sufficiently late active steps satisfy $$s_k^M\leq2/\eta-\delta$$ for some $$\delta>0$$. Full-batch coherence is one, so the loss identity gives

$$
L(W_{k+1})-L(W_k)\leq-\frac{\delta}{2}\eta^2\|G_k\|_*.
$$

Summing these increments would send loss to negative infinity if the cumulative weight diverges, contradicting the lower bound. Hence the limsup bound. It does not establish attraction, persistent tracking, or a temporal cosine condition. See paper Appendix A.1.

</details>

<h2 id="experiments">4. Do the diagnostics separate along training?</h2>

The matrix example proves that loss curvature need not determine temporal alignment. We first test how batch size affects that separation, then ask whether a related pattern persists in language-model training.

### Controlled networks: batch size matters, and the order can swap

{% include controlled-grid.html %}

Small batches remain closer to orthogonality; larger batches produce more negative alignment in these runs. The displayed full-batch global measurements approach coherent reversal without reaching $$-1$$.

{% include event-grid.html %}

In the displayed event panels, T2 precedes T1 at batch 64 in the MLP and CNN; at batch 4096, T1 precedes T2. This illustrates their different responses along training. Unequal event times alone are not the proof of separation: even a scalar GD mode changes direction before loss balance. The matrix counterexample above supplies that proof.

### LLMs: a loss boundary with weak negative alignment

The LLM runs use no-momentum NS-5 on Muon blocks and AdamW on auxiliary parameters. Their empirical diagnostics complement the exact-polar analysis; they do not make the approximation identical to the theoretical update.

<div class="table-wrap" markdown="1">

| Experiment | Configuration | What to look for |
| :--- | :--- | :--- |
| 22M checkpoint branches | Batch 16 for rate interventions | Loss-change signs can switch while cosines stay negative |
| 130M training | Batch 32; sequence length 4096; about 5.23B tokens | Conditional boundary tracking and weak negative global alignment |
| 1B training | Batch 512; sequence length 4096; about 20B tokens | More negative alignment with improving loss; T1 was not measured |

</div>

Return to [**Figure 1**](#fig-1) at the top. It shows the coupled empirical picture we want to explain: conditional loss-boundary tracking, weak negative temporal alignment, and continued validation improvement.



Across four peak-rate runs, full-horizon cosine medians range from approximately $$-0.048$$ to $$-0.032$$. The central empirical pattern is **sustained negative alignment close to orthogonality**, together with continued learning.

The loss measurements have different scopes. Boundary tracking comes from finite-reference, **Muon-only checkpoint probes**. Validation improvement comes from complete **Muon-plus-AdamW training**. A near-zero conditional probe increment does not mean that the full run has stopped learning.

<h3 id="optimizer-memory">What changes with momentum and AdamW?</h3>

Does the weak negative alignment persist when the optimizer carries memory? Our additional 130M runs show a different directional picture: **Muon with momentum 0.95 and AdamW both maintain positive alignment between consecutive updates.**

Over the common window from updates 500 to 36,827, their mean global cosines are approximately **0.712** and **0.891**, respectively. Every recorded global cosine is positive in both runs. The positive sign also persists in all 12 layers and all 84 individual matrix blocks, so it is not an artifact of averaging positive and negative blocks together.

{% include optimizer-grid.html %}

The intuition is that **a momentum update inherits directional information from earlier steps**. Consecutive updates reuse part of the same gradient history, making persistent positive alignment natural even as fresh minibatches change. AdamW also carries a moving average of past gradients. This contrasts with the weakly negative, nearly orthogonal directions of the no-momentum Muon run in [Figure 1](#fig-1): optimizer memory adds temporal persistence to the directional picture.

<h2 id="interventions">5. Change the learning rate at the same checkpoint</h2>

A more direct test is to change the learning rate while starting from one shared state. We branch from 0.04 to **0.02, 0.04 or 0.08**, keeping each model's batch size and auxiliary AdamW schedule fixed. The branch points are step 4000 for 22M and step 7000 for 130M.

For 22M, the one-step conditional probes at step 4000 give:

<div class="table-wrap" markdown="1">

| Muon rate | Conditional loss increment | 95% probe interval | Later mean cosine |
| :--- | ---: | :--- | ---: |
| 0.02 | −0.01967 | [−0.01999, −0.01936] | −0.064 |
| 0.04 | −0.00026 | [−0.00138, 0.00086] | −0.100 |
| 0.08 | +0.15219 | [0.14650, 0.15789] | −0.117 |

</div>

**The expected loss response crosses from decrease to increase, while later alignment remains negative in all three branches.** The middle interval contains zero; the smaller and larger rates fall on opposite sides of the loss boundary.

The columns refer to different horizons. Intervals come from eight paired one-step probes at one state, not across-seed uncertainty. Cosine means summarize 280 recorded pairs per branch during steps 5000–6399, with no reported intervals. The 130M rate branches have no conditional T1 probes.

{% include intervention-grid.html %}

Rate changes initially shift alignment, which then tends back toward its pre-change value. Both displayed continuations also enter learning-rate decay, so the later response reflects the schedule as well as the intervention. Some larger-rate branches also improve later validation loss. These observations motivate schedule research, but they do not establish a rule to raise the rate whenever cosine is near zero.

<h2 id="scale">6. What do noise, layers and scale add?</h2>

The two-threshold picture also suggests where to look beyond a global average: sampling affects coherence and directional dispersion, while layers and model scale reveal heterogeneous temporal behavior.

### A global average hides structure

{% include diagnostic-grid.html %}

The module medians locate stronger negative alignment; the temporal heatmap shows its evolution; the coherence curves below show how probe sampling affects the loss-side signal. Middle-layer key/value matrices differ from many query matrices, and larger probe batches generally strengthen reference coherence. A global cosine near zero should therefore be read alongside the underlying layers and modules.

### Is near orthogonality just minibatch randomness?

A frozen-state test helps distinguish sampling dispersion from temporal feedback. At a fixed checkpoint, compare independent minibatch directions without advancing the model. Then compare this with consecutive directions along the training trajectory.

{% include frozen-grid.html %}

This supports sampling dispersion as a source of weak alignment and changes in the model state as a source of negative bias. Both Muon and AdamW change that state, so this test cannot assign the feedback to either optimizer alone.

### At 1B, loss still improves despite stronger cancellation

{% include scale-grid.html %}

The larger batch is consistent with the controlled trend, but the 130M-to-1B comparison changes several factors and does not isolate batch size. **No T1 probes were collected at 1B**; this run supplies directional evidence, not a test of conditional loss neutrality.


<h2 id="outlook">7. What survives—and why “sublates”?</h2>

**Muon preserves a coherence-corrected loss‑neutral edge and changes its relation to temporal reversal.** The loss identity defines T1. Polar geometry makes T2 depend on a different response. The matrix example proves separation; controlled training and LLM experiments show how it appears in practice.

At 130M scale, loss-boundary tracking coexists with weak negative alignment. The studied 1B configuration has stronger partial cancellation, still far from coherent reversal. Layerwise measurements reveal structure hidden by a global cosine. These findings give a split EoS picture: a surviving loss boundary has no universal temporal-direction signature.

That is what *sublates* means here: preserving an essential part of the classical picture while changing its structure. A complete stochastic perturbation-stability characterization is a different mathematical task; our loss and direction results specify the signatures actually tested along training.

For schedule design, the two signals answer complementary questions:

- **T1:** Does a candidate rate and batch place a conditional step below, near or above loss neutrality?
- **T2 and cosine magnitude:** How much do successive directions reinforce or partially cancel?

Their joint response may inform WSD learning-rate choices and a data-driven critical batch size. Cosine alone does not justify increasing the rate; conditional loss response, validation loss and token budgets also matter. An optimized adaptive scheduler remains future work.

The loss-boundary analysis centers on **Muon without momentum**. The additional momentum-Muon and AdamW runs complement it with a different directional regime: updates retain information from their shared history and show persistent positive alignment.

<div class="article-sources" markdown="1">

### Paper, code and further reading

Yanzhe Chen, Qifang Zhao, Xiaoxiao Xu, and Fanghui Liu. [*Muon Sublates the Edge of Stability in LLM Pretraining*](https://arxiv.org/abs/2609.34915), arXiv v1. [Code](https://github.com/cyzebra/Muon-Sublates-the-Edge-of-Stability-in-LLM-Pretraining).

Background: [Wu, Ma and E (2018)](https://proceedings.neurips.cc/paper/2018/file/6651526b6fb8f29a00507de6a49ce30f-Paper.pdf); [Cohen et al. (2021)](https://arxiv.org/abs/2103.00065); [Jordan et al., Muon](https://kellerjordan.github.io/posts/muon/); [Bernstein, Deriving Muon](https://jeremybernste.in/writing/deriving-muon).

For additional geometric demos: [Neel Mishra, Muon Part 1](https://neelmishra.github.io/blog/dl/optimizers/muon.html) and [Newton–Schulz Part 2](https://neelmishra.github.io/blog/dl/optimizers/newton-schulz.html). Our new diagrams and geometric interpolation are independently implemented; no external demo assets are embedded. Reading structure takes inspiration from [Tri Dao's ReplaySSM blog](https://tridao.me/blog/2026/replayssm/).

Figures 1 and 6–12 use the supplied measurements and vector PDF plots. Plots with supplied CSV/JSONL files are redrawn from those records; other line plots retain the original plotted vertices, calibrated against the PDF tick marks. The heatmaps retain the source cells. Frozen-state marker centers and interval endpoints are recovered from the supplied PDFs without changing the intervals. No experimental values or uncertainty bands are invented. Figure 2 has its own attribution. Figures 3–5 are exact or explicitly labeled explanatory constructions. The scalar animation holds curvature fixed and illustrates the GD threshold. The intervention table follows paper Table 2 and Appendix C.3. Detailed assumptions and protocols are in the paper.

[Back to top](#top) · [Home]({{ '/' | relative_url }})

</div>
