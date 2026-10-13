---
layout: default
title: "Does Muon Operate at the Edge of Stability in LLM Pretraining?"
description: "What survives of classical EoS under no-momentum Muon—and why loss balance and update reversal separate."
permalink: /blog/muon-eos-en/
lang: en
math: true
article: true
---

<div class="article-header" markdown="1">
<p class="eyebrow">Optimization · Research notes</p>

# Does Muon Operate at the Edge of Stability in LLM Pretraining?

<p class="byline">Yanzhe Chen · With Qifang Zhao, Xiaoxiao Xu, and Fanghui Liu</p>

<p class="standfirst">Our answer is “sublate” (from Hegelian philosophy)：It separates loss balance from update reversal (up to 1B model) .</p>

<div class="article-links">
<a href="https://arxiv.org/abs/2609.34915">Read the paper ↗</a>
<a href="https://github.com/cyzebra/Muon-Sublates-the-Edge-of-Stability-in-LLM-Pretraining">Source code ↗</a>
</div>
</div>


<nav class="article-toc" aria-label="Article contents">
<span>In this article</span>
<a href="#gd">01 · GD intuition</a>
<a href="#muon">02 · Muon geometry</a>
<a href="#diagnostics">03 · Two diagnostics</a>
<a href="#experiments">04 · Experiments</a>
<a href="#interventions">05 · Rate changes</a>
<a href="#outlook">06 · Outlook</a>
</nav>

The word *sublates* in our title means preserving part of an idea while changing its structure. Here, the part that survives is a loss-neutral boundary. What changes is its relationship to temporal reversal. We study this in exact-polar models and in practical language-model runs up to 1B parameters, always focusing on **Muon without momentum**.

<h2 id="gd">1. Why GD brings the classical picture together</h2>

For gradient descent (GD), the edge of stability is commonly associated with curvature approaching the learning-rate scale $$2/\eta$$. Training loss can oscillate over short intervals while continuing to decrease over longer horizons.

{% include figure.html src="/assets/images/muon/classical-eos-litman2026.png" alt="Classical GD: curvature tracks the learning-rate boundary while loss oscillates and decreases." caption="**Figure 1 · The classical picture.** Full-batch GD on a three-layer MLP, CIFAR-10, learning rate 0.5. Read curvature on the left together with loss on the right. Reproduced unchanged from [Litman (2026), Figure 1](https://arxiv.org/html/2604.20446v1#S1.F1), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)." %}

The connection is easiest to see in a single quadratic mode. If $$L(x)=\lambda x^2/2$$, then

$$
x_{k+1}=(1-\eta\lambda)x_k.
$$

When $$\eta\lambda=2$$, the multiplier is $$-1$$. The parameter changes sign without changing magnitude; the loss is unchanged; consecutive updates reverse exactly. For this scalar mode, loss balance, exact reversal and marginal linear stability meet at the same point. The direction sign changes earlier, at $$\eta\lambda=1$$; at that exact value the next update is zero and its cosine is undefined.

**Why can GD couple these phenomena so tightly?** A perturbation changes the gradient, and GD passes that gradient change directly into the update. Muon inserts a different transformation between those two steps.

<h2 id="muon">2. Muon changes the geometry of an update</h2>

For a minibatch matrix gradient with compact SVD, exact no-momentum Muon uses

$$
\widehat G_k=U_r\Sigma_rV_r^\top,
\qquad P_k=\operatorname{Pol}(\widehat G_k)=U_rV_r^\top,
\qquad W_{k+1}=W_k-\eta P_k.
$$

Muon keeps the active singular vectors and normalizes their singular values to one. It changes the spectral structure of the update, rather than simply scaling the entire gradient by one number. The theory below isolates this exact polar update; our LLM experiments use the practical five-step Newton–Schulz approximation, **NS-5**.

### The geometric reason to expect a separation

Write a deterministic or fixed-batch optimizer step as $$F(W)=W-\eta T(\nabla L(W))$$. Where the derivatives exist,

$$
DF(W)=I-\eta\,DT(G)\circ\nabla^2L(W).
$$

The local response depends on both the Hessian and the optimizer transformation. GD uses $$T=\operatorname{Id}$$, so the operator reduces to $$I-\eta H$$. For Muon, the transformation contributes its own geometry. There is therefore no reason for a loss-neutral step to impose the same temporal signature as in GD.

This is an intuition about local perturbations, not a perturbation-stability measurement in our LLM experiments. The derivative formula also needs care where the polar map is not differentiable, including at rank changes.

<h2 id="diagnostics">3. Two diagnostics, two different questions</h2>

We separate **T1**, the conditional loss-balance boundary, from **T2**, the zero-cosine boundary between reinforcement and partial cancellation. Neither should be used as a substitute for the other.

<div class="table-wrap" markdown="1">

| Question | GD: one scalar quadratic mode | No-momentum Muon |
| :--- | :--- | :--- |
| When is the step loss-neutral? | $$\lambda=2/\eta$$ | $$s_b^M=2\rho_b/\eta$$ (**T1**) |
| When does temporal alignment change sign? | At $$\lambda=1/\eta$$; cosine is undefined at the zero update | At $$c_k^{\mathrm{pol}}=0$$ (**T2**), when both directions are nonzero |
| Does loss neutrality require exact reversal? | At $$2/\eta$$, in this mode | No; loss balance and direction can separate |
| What controls local perturbations? | $$I-\eta H$$ | $$I-\eta\,DT(G)\circ H$$, where differentiable |

</div>

### T1: does the next step decrease loss in expectation?

Fix the state $$W_k$$ and resample minibatches of size $$b$$. Let $$G_k=\nabla L(W_k)\ne0$$ be the full gradient, and let $$P_k$$ be the sampled polar direction. Define

$$
\rho_b(W_k)=\frac{\mathbb{E}_b\langle G_k,P_k\rangle_F}{\|G_k\|_*},
$$

$$
s_b^M(W_k;\eta)=
\frac{2\mathbb{E}_b\!\left[L(W_k-\eta P_k)-L(W_k)+\eta\langle G_k,P_k\rangle_F\right]}
{\eta^2\|G_k\|_*}.
$$

The nuclear norm $$\|G_k\|_*$$ is the sum of singular values. **Coherence** $$\rho_b\in[-1,1]$$ measures the retained first-order descent signal. The quantity $$s_b^M$$ measures the normalized finite-step loss remainder. An unbiased minibatch gradient does not necessarily yield an unbiased polar direction.

These definitions give the exact conditional identity

$$
\mathbb{E}_b\!\left[L(W_{k+1})-L(W_k)\mid W_k\right]
=\frac{\eta^2\|G_k\|_*}{2}
\left(s_b^M-\frac{2\rho_b}{\eta}\right).
$$

<div class="takeaway" markdown="1">
Below the boundary, the expected loss increment is negative; on it, zero; above it, positive. Full-batch exact-polar Muon has $$\rho_{\mathrm{full}}=1$$, recovering the scale $$2/\eta$$.
</div>

This effective Muon curvature is not GD's maximum Hessian eigenvalue. Under additional second-order smoothness, its small-step limit is a normalized directional curvature.

### T2: do consecutive directions reinforce or cancel?

For two nonzero consecutive directions, measure

$$
c_k^{\mathrm{pol}}=
\frac{\langle P_k,P_{k+1}\rangle_F}{\|P_k\|_F\|P_{k+1}\|_F}.
$$

A positive cosine indicates reinforcement; zero means orthogonality; a negative cosine indicates partial cancellation. Only $$-1$$ means exact reversal of direction. A negative cosine alone establishes neither a two-step parameter return nor a loss-neutral step.

The paper makes the separation concrete with a matrix quadratic. At the same learning rate and full coherence, two states have the same effective loss curvature but opposite alignment signs:

<div class="table-wrap compact" markdown="1">

| State | Effective loss curvature | Consecutive cosine | Loss response |
| :--- | :--- | :--- | :--- |
| A | $$5/(4\eta)$$ | $$-1$$ | Decrease |
| B | $$5/(4\eta)$$ | $$1/3$$ | Decrease |

</div>

Both states lie below $$2/\eta$$. Their loss response agrees, but their temporal directions differ: the curvature involves a spectral trace, while alignment involves an eigenvalue sign count. The construction and assumptions are in paper §2.3.

<h2 id="experiments">4. From controlled networks to LLM pretraining</h2>

The theory predicts that the two boundaries can separate. We first test how batch size affects that separation, then ask whether a related pattern persists in language-model training.

### Controlled networks: batch size matters, and the order can swap

{% include figure.html src="/assets/images/muon/figure1-controlled.png" alt="Controlled Transformer: loss, loss balance and update cosine at several batch sizes." caption="**Figure 2 · Follow one batch across all three panels.** Exact SVD-polar Transformer at fixed learning rate 0.007. The right panel shows increasingly negative alignment at larger batches. From the top row of paper Figure 1." %}

Small batches remain closer to orthogonality; larger batches produce more negative alignment in these runs. The displayed full-batch global measurements approach coherent reversal without reaching $$-1$$.

{% include figure.html src="/assets/images/muon/figure3-event-order.png" alt="First recorded loss-rise, T1 and T2 events in MLP, CNN and Transformer experiments." caption="**Figure 3 · T1 and T2 have no universal order.** Compare the red T1 and blue T2 bars within each batch configuration. In the MLP, T2 comes first at batch 64, whereas T1 comes first at batch 4096. From paper Figure 3; event detection uses the paper's recorded tolerances." %}

The event order is not fixed across settings. This supports treating loss balance and temporal alignment as two measurements, rather than two names for one transition.

### LLMs: a loss boundary with weak negative alignment

The LLM runs use no-momentum NS-5 on Muon blocks and AdamW on auxiliary parameters. Their empirical diagnostics complement the exact-polar analysis; they do not make the approximation identical to the theoretical update.

<div class="table-wrap" markdown="1">

| Experiment | Configuration | What to look for |
| :--- | :--- | :--- |
| 22M checkpoint branches | Batch 16 for rate interventions | Loss-change signs can switch while cosines stay negative |
| 130M training | Batch 32; sequence length 4096; about 5.23B tokens | Conditional boundary tracking and weak negative global alignment |
| 1B training | Batch 512; sequence length 4096; about 20B tokens | More negative alignment with improving loss; T1 was not measured |

</div>

{% include figure.html src="/assets/images/muon/figure1-130m.png" alt="130M run: training and validation loss, conditional loss balance, and global update cosine." caption="**Figure 4 · The two diagnostics coexist at 130M.** Peak rate 0.02. Conditional effective curvature repeatedly approaches the corrected boundary, while global directions remain far from exact reversal. From the bottom row of paper Figure 1." %}

Across four peak-rate runs, full-horizon cosine medians range from approximately $$-0.048$$ to $$-0.032$$. The central empirical pattern is **sustained negative alignment close to orthogonality**, together with continued learning.

The loss measurements have different scopes. Boundary tracking comes from finite-reference, **Muon-only checkpoint probes**. Validation improvement comes from complete **Muon-plus-AdamW training**. A near-zero conditional probe increment does not mean that the full run has stopped learning.

### A global average hides structure

{% include diagnostic-grid.html %}

The module medians locate stronger negative alignment; the temporal heatmap shows its evolution; the coherence curves show how probe sampling affects the loss-side signal. Middle-layer key/value matrices differ from many query matrices, and larger probe batches generally strengthen reference coherence. A global cosine near zero should therefore be read alongside the underlying layers and modules.

### Is near orthogonality just minibatch randomness?

A frozen-state test helps distinguish sampling dispersion from temporal feedback. At a fixed checkpoint, compare independent minibatch directions without advancing the model. Then compare this with consecutive directions along the training trajectory.

{% include figure.html src="/assets/images/muon/figure11-frozen-vs-time.png" alt="Two Tiny Transformer seeds: frozen-state minibatch comparisons versus actual temporal cosines." caption="**Figure 6 · Hold the state fixed, then compare with training.** Orange: independent minibatch directions at a frozen state, with 95% delete-one-batch jackknife intervals. Blue: consecutive-step directions. Frozen-state averages are small and positive; stable-stage temporal averages are negative. From paper Figure 11, batch eight." %}

This supports sampling dispersion as a source of weak alignment and changes in the model state as a source of negative bias. Both Muon and AdamW change that state, so this test cannot assign the feedback to either optimizer alone.

### At 1B, loss still improves despite stronger cancellation

{% include figure.html src="/assets/images/muon/figure8-1b.png" alt="1B training and validation loss, global cosine, and layer-level cosine trajectories." caption="**Figure 7 · A larger-scale directional pattern.** Loss improves while global alignment is more negative, and layers follow different trajectories. The run remains far from exact reversal. From paper Figure 8." %}

The larger batch is consistent with the controlled trend, but the 130M-to-1B comparison changes several factors and does not isolate batch size. **No T1 probes were collected at 1B**; this run supplies directional evidence, not a test of conditional loss neutrality.

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

Rate changes initially shift alignment, which then tends back toward its pre-change value. Some larger-rate branches also improve later validation loss. These observations motivate schedule research, but they do not establish a rule to raise the rate whenever cosine is near zero.

<h2 id="outlook">6. What survives—and what this suggests</h2>

The loss-neutral edge survives, with stochastic coherence correcting its scale. The universal reversal signature does not: conditional loss balance and temporal alignment can move independently. This is the sense in which no-momentum Muon *sublates* the classical edge of stability.

For schedule design, the two diagnostics supply complementary information:

- **T1 asks about one-step loss cost.** Does a candidate rate or batch place the conditional step below, near or above its loss boundary?
- **Temporal alignment asks about directional cancellation.** How much do successive updates reinforce or oppose one another?

Their joint response may inform WSD learning-rate choices and a data-driven critical batch size, alongside validation loss and token budgets. Turning that idea into an optimized adaptive scheduler remains future work.

<div class="takeaway" markdown="1">
**Scope.** These results concern Muon without momentum. Momentum introduces temporal filtering and optimizer state, so the conclusions should not be transferred unchanged to that setting.
</div>

<div class="article-sources" markdown="1">


[Back to top](#top) · [Back to home]({{ '/' | relative_url }})

</div>
