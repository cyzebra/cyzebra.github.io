---
layout: default
title: "What Does the Edge of Stability Mean for Muon?"
description: "Stability, training at an edge, and the separation of loss balance from update reversal under no-momentum Muon."
permalink: /blog/muon-eos-en/
lang: en
math: true
article: true
animations: true
---

<div class="article-header" markdown="1">
<p class="eyebrow">Optimization · Muon · LLM pretraining</p>

# What Does the Edge of Stability Mean for Muon?

<p class="byline">Yanzhe Chen · With Qifang Zhao, Xiaoxiao Xu, and Fanghui Liu</p>

<p class="standfirst">Loss can track a boundary while successive updates remain almost orthogonal. Muon asks us to separate the edge we measure from the stability we analyze.</p>

<div class="article-links">
<a href="https://arxiv.org/abs/2609.34915">Read the paper ↗</a>
<a href="https://github.com/cyzebra/Muon-Sublates-the-Edge-of-Stability-in-LLM-Pretraining">Source code ↗</a>
</div>
</div>

**Does the classical edge-of-stability picture survive a different optimizer geometry?** In our no-momentum Muon experiments, conditional loss balance and consecutive-update alignment separate: a loss-side boundary can be tracked while directions stay weakly negative and far from exact reversal.

There are two questions to keep apart. One asks whether a small perturbation near a fixed point grows. The other asks where an actual training trajectory spends its time. Our loss and direction probes then ask two more specific questions. The opening animations explain these distinctions before the formulas.

{% include edge-animation.html %}

{% include reading-animation.html %}

<div class="takeaway" markdown="1">
**Our contribution in one sentence.** Exact-polar theory identifies a conditional loss-balance boundary; training experiments probe how that boundary and temporal alignment behave separately. Observing either diagnostic is not, by itself, a proof of perturbation stability.
</div>

<nav class="article-toc" aria-label="Article contents">
<span>In this article</span>
<a href="#stability">01 · Stability ≠ edge tracking</a>
<a href="#muon">02 · Muon geometry</a>
<a href="#diagnostics">03 · What we measure</a>
<a href="#experiments">04 · Training evidence</a>
<a href="#interventions">05 · Rate changes</a>
<a href="#scale">06 · Noise and scale</a>
<a href="#outlook">Outlook</a>
</nav>

<h2 id="stability">1. Stability and training at an edge are different questions</h2>

### A local stability question

[Wu, Ma and E (2018)](https://proceedings.neurips.cc/paper/2018/file/6651526b6fb8f29a00507de6a49ce30f-Paper.pdf) study fixed-point selection through linearized stochastic dynamics. Their stability definition bounds the second moment of perturbations. In a scalar quadratic example with i.i.d. sampled curvatures $$a_\xi$$, write $$\bar a=\mathbb{E}[a_\xi]$$ and $$\sigma_a^2=\operatorname{Var}(a_\xi)$$. The second-moment multiplier is

$$
\mathbb{E}[(1-\eta a_\xi)^2]=(1-\eta\bar a)^2+\eta^2\sigma_a^2.
$$

Bounded second moments require this multiplier to be at most one. Thus average curvature alone does not characterize this stochastic stability notion. This example is about perturbations around a fixed point; it does not imply that training approaches the boundary.

### A trajectory question

[Cohen et al. (2021)](https://arxiv.org/abs/2103.00065) report a different phenomenon: in full-batch neural-network GD, the maximum Hessian eigenvalue hovers just above $$2/\eta$$, while loss is non-monotone locally but decreases over longer horizons. The observation concerns a changing training trajectory, not a frozen quadratic.

{% include figure.html src="/assets/images/muon/classical-eos-litman2026.png" alt="Classical neural-network GD: maximum curvature tracks the learning-rate scale while loss oscillates and improves." caption="**Figure 1 · A measured trajectory, rather than the toy animation.** Full-batch GD on a three-layer MLP, CIFAR-10, learning rate 0.5. Curvature and loss evolve during training. Reproduced unchanged from [Litman (2026), Figure 1](https://arxiv.org/html/2604.20446v1#S1.F1), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)." %}

The boundary calculation explains a local scale. Sustained proximity to it is an additional empirical fact. The toy animation shows why neither stability nor oscillation alone establishes such proximity.

<div class="table-wrap" markdown="1">

| Question | Object being studied | Evidence needed |
| :--- | :--- | :--- |
| Is a fixed point locally stable? | Growth of small perturbations | Linearized dynamics and an explicit stability notion |
| Does training take place near an edge? | A time-varying training trajectory | Sustained tracking of a specified boundary |
| Is the conditional step loss-neutral? | Expected one-step loss change at a fixed state | The conditional loss identity and probes (**T1**) |
| Do consecutive updates reinforce or cancel? | Two nonzero temporal directions | Their cosine (**T2**) |

</div>

**Our blog keeps these four questions separate.** In particular, conditional loss neutrality is not a general substitute for stochastic perturbation stability. The distinction becomes especially useful when changing from GD to Muon.

<h2 id="muon">2. Muon changes the geometry of an update</h2>

For a minibatch matrix gradient with compact SVD, exact no-momentum Muon uses

$$
\widehat G_k=U_r\Sigma_rV_r^\top,
\qquad P_k=\operatorname{Pol}(\widehat G_k)=U_rV_r^\top,
\qquad W_{k+1}=W_k-\eta P_k.
$$

Muon keeps the active singular vectors and normalizes their singular values to one. It changes the spectral structure of the update, rather than simply scaling the entire gradient by one number. The theory below isolates this exact polar update; our LLM experiments use the practical five-step Newton–Schulz approximation, **NS-5**.

### Why GD links several thresholds in a scalar mode

For $$L(x)=\lambda x^2/2$$, GD has multiplier $$1-\eta\lambda$$. At $$\eta\lambda=2$$, it is $$-1$$: loss is unchanged and consecutive updates reverse exactly. At $$\eta\lambda=1$$, the next update is zero, so its cosine is undefined; on either side, the temporal direction sign changes.

This coincidence at $$2/\eta$$ belongs to the scalar quadratic mode. It does not say that whole-network consecutive updates must have cosine $$-1$$ whenever maximum Hessian curvature approaches that scale.

### The optimizer transformation adds another geometry

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
**Read the bracket.** Below the boundary, the expected loss increment is negative; on it, zero; above it, positive. Full-batch exact-polar Muon has $$\rho_{\mathrm{full}}=1$$, recovering the scale $$2/\eta$$.
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

<h2 id="experiments">4. Do the diagnostics separate along training?</h2>

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

Return to **Animation 02** at the top: its three original panels pair full training loss with conditional loss balance and global temporal cosine. Read the measurements together, while keeping their scopes separate.



Across four peak-rate runs, full-horizon cosine medians range from approximately $$-0.048$$ to $$-0.032$$. The central empirical pattern is **sustained negative alignment close to orthogonality**, together with continued learning. We call this temporal regime the **Edge of Cosine (EOC)**. The name describes consecutive-update alignment; it is not a perturbation-stability criterion.

The loss measurements have different scopes. Boundary tracking comes from finite-reference, **Muon-only checkpoint probes**. Validation improvement comes from complete **Muon-plus-AdamW training**. A near-zero conditional probe increment does not mean that the full run has stopped learning.

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

<h2 id="scale">6. What do noise, layers and scale add?</h2>

A global measurement can conceal structure. These additional experiments clarify the observed directional regime, without turning it into a local-stability test.

### A global average hides structure

{% include diagnostic-grid.html %}

The module medians locate stronger negative alignment; the temporal heatmap shows its evolution; the coherence curves show how probe sampling affects the loss-side signal. Middle-layer key/value matrices differ from many query matrices, and larger probe batches generally strengthen reference coherence. A global cosine near zero should therefore be read alongside the underlying layers and modules.

### Is near orthogonality just minibatch randomness?

A frozen-state test helps distinguish sampling dispersion from temporal feedback. At a fixed checkpoint, compare independent minibatch directions without advancing the model. Then compare this with consecutive directions along the training trajectory.

{% include figure.html src="/assets/images/muon/figure11-frozen-vs-time.png" alt="Two Tiny Transformer seeds: frozen-state minibatch comparisons versus actual temporal cosines." caption="**Figure 6 · Hold the state fixed, then compare with training.** Orange: independent minibatch directions at a frozen state, with 95% delete-one-batch jackknife intervals. Blue: consecutive-step directions. Frozen-state averages are small and positive; later-stage temporal averages are negative. From paper Figure 11, batch eight." %}

This supports sampling dispersion as a source of weak alignment and changes in the model state as a source of negative bias. Both Muon and AdamW change that state, so this test cannot assign the feedback to either optimizer alone.

### At 1B, loss still improves despite stronger cancellation

{% include figure.html src="/assets/images/muon/figure8-1b.png" alt="1B training and validation loss, global cosine, and layer-level cosine trajectories." caption="**Figure 7 · A larger-scale directional pattern.** Loss improves while global alignment is more negative, and layers follow different trajectories. The run remains far from exact reversal. From paper Figure 8." %}

The larger batch is consistent with the controlled trend, but the 130M-to-1B comparison changes several factors and does not isolate batch size. **No T1 probes were collected at 1B**; this run supplies directional evidence, not a test of conditional loss neutrality.

<h2 id="outlook">What survives—and what this suggests</h2>

An exact conditional loss-balance boundary is available for polar Muon, with stochastic coherence correcting its scale. Our experiments show tracking of that loss-side boundary alongside a different temporal regime: weak negative alignment rather than complete reversal. This separation is the sense in which no-momentum Muon *sublates* the classical picture.

These results concern an identity, measured boundary tracking and temporal directions. **They do not establish a general perturbation-stability boundary for stochastic NS-5 Muon-plus-AdamW training.** That question requires a separate analysis of the optimizer dynamics.

For schedule design, the two diagnostics supply complementary information:

- **T1 asks about one-step loss cost.** Does a candidate rate or batch place the conditional step below, near or above its loss boundary?
- **Temporal alignment asks about directional cancellation.** How much do successive updates reinforce or oppose one another?

Their joint response may inform WSD learning-rate choices and a data-driven critical batch size, alongside validation loss and token budgets. Turning that idea into an optimized adaptive scheduler remains future work.

<div class="takeaway" markdown="1">
**Scope.** These results concern Muon without momentum. Momentum introduces temporal filtering and optimizer state, so the conclusions should not be transferred unchanged to that setting.
</div>

<div class="article-sources" markdown="1">

### Paper, code and sources

Yanzhe Chen, Qifang Zhao, Xiaoxiao Xu, and Fanghui Liu. [*Muon Sublates the Edge of Stability in LLM Pretraining*](https://arxiv.org/abs/2609.34915), arXiv v1. [Source code](https://github.com/cyzebra/Muon-Sublates-the-Edge-of-Stability-in-LLM-Pretraining).

Background: [Wu, Ma and E (2018), *How SGD Selects the Global Minima in Over-parameterized Learning: A Dynamical Stability Perspective*](https://proceedings.neurips.cc/paper/2018/file/6651526b6fb8f29a00507de6a49ce30f-Paper.pdf); [Cohen et al. (2021), *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability*](https://arxiv.org/abs/2103.00065); [Jordan et al. on Muon](https://kellerjordan.github.io/posts/muon/).

Animation 01 is an exact scalar toy calculation. Animation 02 reveals unchanged measured panels from paper Figure 1; it does not animate raw training logs. Figures 2–7 reproduce or regroup the paper's results; Figure 1 is credited separately. The learning-rate table follows paper Table 2 and Appendix C.3. Full assumptions and protocols are in the paper.

[Back to top](#top) · [Back to home]({{ '/' | relative_url }})

</div>
